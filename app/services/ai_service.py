import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import psycopg2
import logging
import json
from typing import Optional, Dict, Any, List
from app.config.db import get_db_connection
from app.config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def validate_and_process_image(image_bytes: bytes) -> Optional[bytes]:
    """
    Validate and process image to ensure it works with Jina API
    - Converts to RGB if needed
    - Resizes if too large
    - Compresses if file size is too big
    """
    try:
        # Open image
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"Original image: {image.format}, {image.size}, {image.mode}")
        
        # Convert to RGB if needed (RGBA, P, etc. can cause issues)
        if image.mode not in ['RGB']:
            if image.mode == 'RGBA':
                # Create a white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            else:
                image = image.convert('RGB')
            logger.info(f"Converted image to RGB from {image.mode}")
        
        # Resize if too large (Jina recommends smaller images for better performance)
        max_size = 800
        if max(image.size) > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to: {image.size}")
        
        # Convert back to bytes with good quality
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=90, optimize=True)
        processed_bytes = output_buffer.getvalue()
        
        # Check final file size (Jina has limits around 5MB)
        max_file_size = 4 * 1024 * 1024  # 4MB to be safe
        if len(processed_bytes) > max_file_size:
            # Reduce quality progressively
            for quality in [75, 60, 50, 40]:
                output_buffer = BytesIO()
                image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                processed_bytes = output_buffer.getvalue()
                if len(processed_bytes) <= max_file_size:
                    logger.info(f"Compressed image to quality {quality}, size: {len(processed_bytes)} bytes")
                    break
        
        logger.info(f"Final processed image size: {len(processed_bytes)} bytes")
        return processed_bytes
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string (no data URI prefix)"""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 encoding error: {e}")
        return None

def make_jina_request(url: str, payload: dict) -> Dict[str, Any]:
    """Make request to Jina API with proper error handling"""
    
    if not JINA_API_KEY:
        return {"error": "JINA_API_KEY is not configured"}
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    
    try:
        logger.info(f"Making request to {url}")
        logger.debug(f"Payload keys: {list(payload.keys())}")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60  # Increased timeout for image processing
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"API Error {response.status_code}: {error_text}")
            
            # Try to parse error details
            try:
                error_json = response.json()
                error_message = error_json.get('detail', error_json.get('message', error_text))
            except:
                error_message = error_text
            
            return {
                "error": f"API Error {response.status_code}: {error_message}",
                "status_code": response.status_code
            }
        
        result = response.json()
        logger.info("API request successful")
        return {"success": True, "data": result}
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return {"error": "Request timeout - please try again"}
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return {"error": "Failed to connect to Jina API"}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return {"error": f"Request failed: {str(e)}"}
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def classify_clip(image_bytes: bytes) -> Dict[str, Any]:
    """Classify an image using Jina CLIP v2 API"""
    
    try:
        # Step 1: Process image
        logger.info("Processing image for classification...")
        processed_image_bytes = validate_and_process_image(image_bytes)
        if not processed_image_bytes:
            return {
                "error": "Failed to process image",
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
                "department": "General Department"
            }
        
        # Step 2: Convert to base64
        logger.info("Converting image to base64...")
        image_b64 = image_bytes_to_base64(processed_image_bytes)
        if not image_b64:
            return {
                "error": "Failed to encode image",
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
                "department": "General Department"
            }
        
        # Step 3: Prepare payload (matching the curl example format exactly)
        payload = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}  # Direct base64 string, no data URI prefix
            ],
            "labels": labels
        }
        
        # Step 4: Make API request
        result = make_jina_request(JINA_API_URL, payload)
        
        if "error" in result:
            return {
                "error": result["error"],
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
                "department": "General Department"
            }
        
        # Step 5: Parse response
        api_data = result["data"]
        
        if 'data' not in api_data or not api_data['data']:
            return {
                "error": "Invalid response format from API",
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
                "department": "General Department"
            }
        
        classification_data = api_data['data'][0]
        predictions = classification_data.get('predictions', [])
        
        if not predictions:
            return {
                "error": "No predictions returned",
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
                "department": "General Department"
            }
        
        # Find best prediction
        best_prediction = max(predictions, key=lambda x: x.get('score', 0))
        best_label = best_prediction.get('label', 'unknown')
        best_confidence = best_prediction.get('score', 0.0)
        
        # Create probabilities dict
        probabilities = {
            pred.get('label', f'unknown_{i}'): pred.get('score', 0.0) 
            for i, pred in enumerate(predictions)
        }
        
        responsible_authority = label_to_department.get(best_label, "General Department")
        
        logger.info(f"Classification successful: {best_label} (confidence: {best_confidence:.3f})")
        
        return {
            "label": best_label,
            "confidence": float(best_confidence),
            "probabilities": probabilities,
            "department": responsible_authority
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            "error": f"Classification failed: {str(e)}",
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "department": "General Department"
        }

def extract_clip_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """Extract CLIP embedding from image using Jina API"""
    
    try:
        # Step 1: Process image
        logger.info("Processing image for embedding extraction...")
        processed_image_bytes = validate_and_process_image(image_bytes)
        if not processed_image_bytes:
            logger.error("Failed to process image for embedding")
            return None
        
        # Step 2: Convert to base64
        image_b64 = image_bytes_to_base64(processed_image_bytes)
        if not image_b64:
            logger.error("Failed to encode image for embedding")
            return None
        
        # Step 3: Prepare payload (matching the curl example format exactly)
        payload = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}  # Direct base64 string
            ]
        }
        
        # Step 4: Make API request
        result = make_jina_request(JINA_EMBEDDINGS_URL, payload)
        
        if "error" in result:
            logger.error(f"Embedding API error: {result['error']}")
            return None
        
        # Step 5: Parse response
        api_data = result["data"]
        
        if 'data' not in api_data or not api_data['data']:
            logger.error("Invalid embedding response format")
            return None
        
        embedding_data = api_data['data'][0]
        embedding = embedding_data.get('embedding', [])
        
        if not embedding:
            logger.error("No embedding returned from API")
            return None
        
        logger.info(f"Successfully extracted embedding (dimension: {len(embedding)})")
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        return None

def merge_model(image_bytes: bytes, latitude: float, longitude: float) -> Dict[str, Any]:
    """Check for duplicate issues using CLIP embeddings"""
    
    logger.info(f"🔍 Checking for duplicates at location: ({latitude}, {longitude})")
    
    # Extract embedding
    embedding = extract_clip_embedding(image_bytes)
    if embedding is None:
        logger.error("❌ Failed to extract embedding")
        return {
            "error": "Failed to extract embedding",
            "embedding": None,
            "issue_id": None,
            "is_duplicate": False
        }
    
    conn = None
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
            # Step 1: Find nearby issues (within 10 meters)
            cursor.execute("""
                SELECT id FROM issues 
                WHERE ST_DWithin(
                    location, 
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326), 
                    10
                );
            """, (longitude, latitude))
            
            nearby_issues = cursor.fetchall()
            
            if not nearby_issues:
                logger.info("🆕 No nearby issues found - this is a new issue")
                return {
                    "embedding": embedding,
                    "issue_id": None,
                    "is_duplicate": False,
                    "duplicate_of_id": None
                }
            
            nearby_issue_ids = [row[0] for row in nearby_issues]
            logger.info(f"Found {len(nearby_issue_ids)} nearby issues")
            
            # Step 2: Check for similar uploads using cosine similarity
            # Using 0.2 as threshold (lower values = more similar)
            cursor.execute("""
                SELECT id, report_id, embedding <=> %s::vector AS distance 
                FROM report_uploads 
                WHERE report_id IN (
                    SELECT id FROM reports 
                    WHERE issue_id = ANY(%s::uuid[])
                ) 
                AND embedding IS NOT NULL
                AND embedding <=> %s::vector < 0.2 
                ORDER BY distance ASC 
                LIMIT 1;
            """, (embedding, nearby_issue_ids, embedding))
            
            match = cursor.fetchone()
            
            if match:
                match_id, report_id, distance = match
                
                # Get the issue_id from the matched report
                cursor.execute("""
                    SELECT issue_id FROM reports WHERE id = %s
                """, (report_id,))
                
                issue_result = cursor.fetchone()
                if not issue_result:
                    logger.error("Could not find issue_id for matched report")
                    return {
                        "embedding": embedding,
                        "issue_id": None,
                        "is_duplicate": False,
                        "duplicate_of_id": None
                    }
                
                issue_id = issue_result[0]
                
                logger.info(f" Duplicate found! Issue ID: {issue_id}")
                logger.info(f"   Cosine distance: {distance:.5f}")
                logger.info(f"   Matched upload ID: {match_id}")
                
                return {
                    "embedding": embedding,
                    "issue_id": issue_id,
                    "is_duplicate": True,
                    "duplicate_of_id": match_id,
                    "distance": float(distance)
                }
            
            # No similar upload found
            logger.info(" No similar uploads found - this is a new issue")
            return {
                "embedding": embedding,
                "issue_id": None,
                "is_duplicate": False,
                "duplicate_of_id": None
            }
            
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Database error in merge_model: {e}")
        return {
            "error": str(e),
            "embedding": embedding if 'embedding' in locals() else None,
            "issue_id": None,
            "is_duplicate": False
        }
    finally:
        if conn:
            conn.close()

# Helper function to test the API connection
def test_api_connection():
    """Test if the Jina API is accessible with current credentials"""
    
    if not JINA_API_KEY:
        return {"error": "JINA_API_KEY not set"}
    
    # Test with a simple text embedding request
    test_payload = {
        "model": "jina-clip-v2",
        "input": [
            {"text": "test connection"}
        ]
    }
    
    result = make_jina_request(JINA_EMBEDDINGS_URL, test_payload)
    
    if "error" in result:
        return {"error": f"API test failed: {result['error']}"}
    
    return {"success": True, "message": "API connection successful"}

# Test function for development (remove in production)
def test_with_sample_image():
    """Test the complete pipeline with a sample image"""
    
    try:
        # Create a simple test image
        from PIL import Image, ImageDraw
        import io
        
        # Create a 400x300 test image with some visual content
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes to make it more realistic
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=3)
        draw.ellipse([200, 100, 350, 200], fill='green', outline='black', width=3)
        draw.line([0, 0, 400, 300], fill='black', width=5)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()
        
        logger.info(" Testing API connection...")
        connection_test = test_api_connection()
        logger.info(f"Connection test result: {connection_test}")
        
        if "error" in connection_test:
            return connection_test
        
        logger.info(" Testing image classification...")
        classification_result = classify_clip(img_bytes)
        logger.info(f"Classification result: {classification_result}")
        
        logger.info(" Testing embedding extraction...")
        embedding = extract_clip_embedding(img_bytes)
        logger.info(f"Embedding extracted: {embedding is not None}")
        if embedding:
            logger.info(f"Embedding dimension: {len(embedding)}")
        
        return {
            "connection_test": connection_test,
            "classification": classification_result,
            "embedding_extracted": embedding is not None,
            "embedding_dimension": len(embedding) if embedding else 0
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {"error": f"Test failed: {str(e)}"}

