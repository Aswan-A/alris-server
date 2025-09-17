import torch
import open_clip
import numpy as np
from PIL import Image
from io import BytesIO
import psycopg2
from app.config.db import get_db_connection
from app.config.settings import settings

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "ViT-B-32-quickgelu"
pretrained = "laion400m_e32"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained
)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device).eval()

# Define labels and departments
labels = [
    "garbage dumping",
    "open pothole",
    "damaged road",
    "abandoned vehicle",
    "illegal construction",
    "street light not working",
    "water logging",
    "stray animals",
    "fallen tree",
    "fire hazard",
    "public urination",
    "open manhole",
    "unauthorized parking",
    "faded zebra crossing",
    "unsafe electrical pole",
    "broken footpath",
    "blocked drainage"
]

label_to_department = {
    "garbage dumping": "Municipal Waste Management",
    "open pothole": "Public Works Department",
    "damaged road": "Road Maintenance",
    "abandoned vehicle": "Traffic Police",
    "illegal construction": "Town Planning",
    "street light not working": "Electricity Department",
    "water logging": "Drainage Board",
    "stray animals": "Animal Control",
    "fallen tree": "Disaster Management",
    "fire hazard": "Fire Department",
    "public urination": "Sanitation Department",
    "open manhole": "Sewerage Board",
    "unauthorized parking": "Traffic Police",
    "faded zebra crossing": "Road Safety",
    "unsafe electrical pole": "Electricity Department",
    "broken footpath": "Urban Development",
    "blocked drainage": "Drainage Board"
}

text_tokens = tokenizer(labels).to(device)

def classify_clip(image_bytes: bytes):
    """Classify an image using CLIP model"""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits_per_image = (image_features @ text_features.T) * 100.0
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    best_idx = probs.argmax()
    best_label = labels[best_idx]
    responsible_authority = label_to_department.get(best_label, "General Department")
    
    return {
        "label": best_label,
        "confidence": float(probs[best_idx]),
        "probabilities": {label: float(prob) for label, prob in zip(labels, probs)},
        "department": responsible_authority
    }

def extract_clip_embedding(image_bytes: bytes) -> list:
    """Extract CLIP embedding from image"""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    embedding_list = embedding.squeeze(0).cpu().tolist()
    return embedding_list

def merge_model(image_bytes: bytes, latitude: float, longitude: float):
    """Check for duplicate issues using CLIP embeddings"""
    print(f"\n🔍 Checking location: ({latitude}, {longitude})")
    embedding = extract_clip_embedding(image_bytes)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Step 1: Find nearby issues
            cursor.execute("""
                SELECT id
                FROM issues
                WHERE ST_DWithin(
                    location,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    10
                );
            """, (longitude, latitude))
            
            nearby_issues = cursor.fetchall()
            if not nearby_issues:
                print("🆕 No nearby issues — new issue.")
                return {
                    "embedding": embedding,
                    "issue_id": None,  # Will be generated later
                    "is_duplicate": False,
                    "duplicate_of_id": None
                }
            
            nearby_issue_ids = [row[0] for row in nearby_issues]
            
            # Step 2: Check for similar uploads
            cursor.execute("""
                SELECT id, report_id, embedding <=> %s::vector AS distance
                FROM report_uploads
                WHERE report_id IN (
                    SELECT id FROM reports WHERE issue_id = ANY(%s::uuid[])
                )
                AND embedding <=> %s::vector < 0.2
                ORDER BY distance ASC
                LIMIT 1;
            """, (embedding, nearby_issue_ids, embedding))
            
            match = cursor.fetchone()
            if match:
                match_id, report_id, distance = match
                # Get the issue_id from the report
                cursor.execute("""
                    SELECT issue_id FROM reports WHERE id = %s
                """, (report_id,))
                issue_id = cursor.fetchone()[0]
                
                print(f"✅ Match found! Issue ID: {issue_id}")
                print(f"Cosine Distance: {distance:.5f}")
                return {
                    "embedding": embedding,
                    "issue_id": issue_id,
                    "is_duplicate": True,
                    "duplicate_of_id": match_id,
                    "distance": distance
                }
            
            # No match found
            print("🆕 New issue — no similar upload found.")
            return {
                "embedding": embedding,
                "issue_id": None,  # Will be generated later
                "is_duplicate": False,
                "duplicate_of_id": None
            }
    except Exception as e:
        conn.rollback()
        print("❌ Error:", e)
        return {
            "error": str(e),
            "embedding": None,
            "issue_id": None,
            "is_duplicate": False
        }
    finally:
        conn.close()