import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import psycopg2
from app.config.db import get_db_connection
from app.config.settings import settings

JINA_API_URL = 'https://api.jina.ai/v1/classify'
JINA_EMBEDDINGS_URL = 'https://api.jina.ai/v1/embeddings'
JINA_API_KEY = settings.JINA_API_KEY

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

def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def classify_clip(image_bytes: bytes):
    """Classify an image using Jina CLIP v2 API"""
    try:
        # Convert image to base64
        image_b64 = image_bytes_to_base64(image_bytes)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        
        data = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}
            ],
            "labels": labels
        }
        
        response = requests.post(JINA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract classification results
        classification_data = result['data'][0]
        predictions = classification_data['predictions']
        
        # Find best prediction
        best_prediction = max(predictions, key=lambda x: x['score'])
        best_label = best_prediction['label']
        best_confidence = best_prediction['score']
        
        # Create probabilities dict
        probabilities = {pred['label']: pred['score'] for pred in predictions}
        
        responsible_authority = label_to_department.get(best_label, "General Department")
        
        return {
            "label": best_label,
            "confidence": float(best_confidence),
            "probabilities": probabilities,
            "department": responsible_authority
        }
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return {
            "error": f"API request failed: {str(e)}",
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "department": "General Department"
        }
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            "error": f"Classification failed: {str(e)}",
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "department": "General Department"
        }

def extract_clip_embedding(image_bytes: bytes) -> list:
    """Extract CLIP embedding from image using Jina API"""
    try:
        # Convert image to base64
        image_b64 = image_bytes_to_base64(image_bytes)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        
        data = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}
            ]
        }
        
        response = requests.post(JINA_EMBEDDINGS_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract embedding
        embedding = result['data'][0]['embedding']
        return embedding
        
    except requests.exceptions.RequestException as e:
        print(f"Embedding API request error: {e}")
        return None
    except Exception as e:
        print(f"Embedding extraction error: {e}")
        return None

def merge_model(image_bytes: bytes, latitude: float, longitude: float):
    """Check for duplicate issues using CLIP embeddings"""
    print(f"\n🔍 Checking location: ({latitude}, {longitude})")
    embedding = extract_clip_embedding(image_bytes)
    
    if embedding is None:
        print("❌ Failed to extract embedding")
        return {
            "error": "Failed to extract embedding",
            "embedding": None,
            "issue_id": None,
            "is_duplicate": False
        }
    
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