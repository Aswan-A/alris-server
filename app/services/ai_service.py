import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import psycopg2
import logging
import json
import time
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
    Process image to work optimally with Jina API
    The classification API seems to have stricter requirements than embeddings
    """
    try:
        # Open image
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"Original image: {image.format}, {image.size}, {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
            logger.info(f"Converted image to RGB from {image.mode}")
        
        # For classification API, use smaller images (it seems more sensitive)
        max_size = 512  # Smaller size for classification
        if max(image.size) > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to: {image.size}")
        
        # Use high quality but ensure small file size
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=85, optimize=True)
        processed_bytes = output_buffer.getvalue()
        
        # Ensure file is under 2MB for classification API
        max_file_size = 2 * 1024 * 1024  # 2MB
        if len(processed_bytes) > max_file_size:
            for quality in [70, 60, 50, 40]:
                output_buffer = BytesIO()
                image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                processed_bytes = output_buffer.getvalue()
                if len(processed_bytes) <= max_file_size:
                    logger.info(f"Compressed image to quality {quality}")
                    break
        
        logger.info(f"Final processed image size: {len(processed_bytes)} bytes")
        return processed_bytes
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

def validate_and_process_image_for_embeddings(image_bytes: bytes) -> Optional[bytes]:
    """
    Process image specifically for embeddings API (can handle larger images)
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"Original image for embedding: {image.format}, {image.size}, {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Embeddings can handle larger images
        max_size = 800
        if max(image.size) > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized embedding image to: {image.size}")
        
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=90, optimize=True)
        processed_bytes = output_buffer.getvalue()
        
        logger.info(f"Final embedding image size: {len(processed_bytes)} bytes")
        return processed_bytes
        
    except Exception as e:
        logger.error(f"Embedding image processing error: {e}")
        return None

def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 encoding error: {e}")
        return None

def make_jina_request(url: str, payload: dict, max_retries: int = 2) -> Dict[str, Any]:
    """Make request to Jina API with retry logic for 500 errors"""
    
    if not JINA_API_KEY:
        return {"error": "JINA_API_KEY is not configured"}
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying request (attempt {attempt + 1}) after {wait_time}s...")
                time.sleep(wait_time)
            
            logger.info(f"Making request to {url} (attempt {attempt + 1})")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=90  # Longer timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("API request successful")
                return {"success": True, "data": result}
            
            elif response.status_code == 500 and attempt < max_retries:
                # Retry on 500 errors
                error_text = response.text
                logger.warning(f"500 error on attempt {attempt + 1}, will retry: {error_text}")
                continue
            
            else:
                # Don't retry on other errors or final attempt
                error_text = response.text
                logger.error(f"API Error {response.status_code}: {error_text}")
                
                try:
                    error_json = response.json()
                    error_message = error_json.get('detail', error_json.get('message', error_text))
                except:
                    error_message = error_text
                
                return {
                    "error": f"API Error {response.status_code}: {error_message}",
                    "status_code": response.status_code
                }
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Request timeout on attempt {attempt + 1}, will retry...")
                continue
            logger.error("Request timeout - all retries exhausted")
            return {"error": "Request timeout - please try again"}
        
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                logger.warning(f"Connection error on attempt {attempt + 1}, will retry...")
                continue
            logger.error("Connection error - all retries exhausted")
            return {"error": "Failed to connect to Jina API"}
        
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                continue
            return {"error": f"Unexpected error: {str(e)}"}
    
    return {"error": "All retry attempts exhausted"}

def classify_clip_fallback(image_bytes: bytes) -> Dict[str, Any]:
    """
    Fallback classification using a simple heuristic based on image characteristics
    Used when the Jina classification API is having issues
    """
    try:
        # Analyze image characteristics
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Simple heuristic classification based on color analysis
        # This is a basic fallback - you might want to train a local model
        
        # Calculate average color values
        avg_color = np.mean(img_array, axis=(0, 1))
        r, g, b = avg_color
        
        # Calculate some basic features
        brightness = np.mean(avg_color)
        color_std = np.std(img_array)
        aspect_ratio = width / height
        
        # Simple rule-based classification (improve this based on your data)
        if brightness < 80:  # Dark images
            if color_std > 50:
                predicted_label = "street light not working"
            else:
                predicted_label = "open manhole"
        elif brightness > 180:  # Very bright images
            predicted_label = "water logging"
        elif r > g and r > b:  # Reddish images
            predicted_label = "fire hazard"
        elif g > r and g > b:  # Greenish images
            predicted_label = "stray animals"
        elif b > r and b > g:  # Bluish images
            predicted_label = "water logging"
        elif color_std < 30:  # Low variation - possibly road
            if aspect_ratio > 1.5:
                predicted_label = "damaged road"
            else:
                predicted_label = "open pothole"
        else:
            # Default classification
            predicted_label = "garbage dumping"  # Most common issue
        
        # Create mock probabilities
        probabilities = {label: 0.1 for label in labels}
        probabilities[predicted_label] = 0.7
        
        responsible_authority = label_to_department.get(predicted_label, "General Department")
        
        logger.warning(f"Used fallback classification: {predicted_label}")
        
        return {
            "label": predicted_label,
            "confidence": 0.7,
            "probabilities": probabilities,
            "department": responsible_authority,
            "fallback_used": True
        }
        
    except Exception as e:
        logger.error(f"Fallback classification error: {e}")
        return {
            "error": f"Fallback classification failed: {str(e)}",
            "label": "garbage dumping",  # Default fallback
            "confidence": 0.5,
            "probabilities": {label: 1.0/len(labels) for label in labels},
            "department": "General Department",
            "fallback_used": True
        }

def classify_clip(image_bytes: bytes) -> Dict[str, Any]:
    """Classify an image using Jina CLIP v2 API with fallback"""
    
    try:
        # Step 1: Process image for classification
        logger.info("Processing image for classification...")
        processed_image_bytes = validate_and_process_image(image_bytes)
        if not processed_image_bytes:
            logger.warning("Image processing failed, using fallback classification")
            return classify_clip_fallback(image_bytes)
        
        # Step 2: Convert to base64
        logger.info("Converting image to base64...")
        image_b64 = image_bytes_to_base64(processed_image_bytes)
        if not image_b64:
            logger.warning("Base64 encoding failed, using fallback classification")
            return classify_clip_fallback(image_bytes)
        
        # Step 3: Prepare payload
        payload = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}
            ],
            "labels": labels
        }
        
        # Step 4: Make API request with retries
        result = make_jina_request(JINA_API_URL, payload, max_retries=2)
        
        if "error" in result:
            logger.warning(f"Jina API failed: {result['error']}, using fallback classification")
            return classify_clip_fallback(image_bytes)
        
        # Step 5: Parse response
        api_data = result["data"]
        
        if 'data' not in api_data or not api_data['data']:
            logger.warning("Invalid API response format, using fallback classification")
            return classify_clip_fallback(image_bytes)
        
        classification_data = api_data['data'][0]
        predictions = classification_data.get('predictions', [])
        
        if not predictions:
            logger.warning("No predictions returned, using fallback classification")
            return classify_clip_fallback(image_bytes)
        
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
        logger.error(f"Classification error: {e}, using fallback")
        return classify_clip_fallback(image_bytes)

def extract_clip_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """Extract CLIP embedding from image using Jina API"""
    
    try:
        # Step 1: Process image specifically for embeddings
        logger.info("Processing image for embedding extraction...")
        processed_image_bytes = validate_and_process_image_for_embeddings(image_bytes)
        if not processed_image_bytes:
            logger.error("Failed to process image for embedding")
            return None
        
        # Step 2: Convert to base64
        image_b64 = image_bytes_to_base64(processed_image_bytes)
        if not image_b64:
            logger.error("Failed to encode image for embedding")
            return None
        
        # Step 3: Prepare payload
        payload = {
            "model": "jina-clip-v2",
            "input": [
                {"image": image_b64}
            ]
        }
        
        # Step 4: Make API request (embeddings seem more stable)
        result = make_jina_request(JINA_EMBEDDINGS_URL, payload, max_retries=1)
        
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
    """Check for duplicate issues using CLIP embeddings with better error handling"""
    
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
        
        # Test connection
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        
        logger.info("✅ Database connection successful")
        
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
                
                logger.info(f"✅ Duplicate found! Issue ID: {issue_id}")
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
            logger.info("🆕 No similar uploads found - this is a new issue")
            return {
                "embedding": embedding,
                "issue_id": None,
                "is_duplicate": False,
                "duplicate_of_id": None
            }
            
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Database connection error: {e}")
        # Return embedding even if DB fails, so the upload can still proceed
        return {
            "error": f"Database connection failed: {str(e)}",
            "embedding": embedding if 'embedding' in locals() else None,
            "issue_id": None,
            "is_duplicate": False,
            "db_error": True
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

# Test functions
def test_api_connection():
    """Test if the Jina API is accessible"""
    if not JINA_API_KEY:
        return {"error": "JINA_API_KEY not set"}
    
    test_payload = {
        "model": "jina-clip-v2",
        "input": [{"text": "test connection"}]
    }
    
    result = make_jina_request(JINA_EMBEDDINGS_URL, test_payload, max_retries=1)
    
    if "error" in result:
        return {"error": f"API test failed: {result['error']}"}
    
    return {"success": True, "message": "API connection successful"}

def test_database_connection():
    """Test database connection"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
        conn.close()
        return {"success": True, "message": f"Database connected: {version}"}
    except Exception as e:
        return {"error": f"Database connection failed: {str(e)}"}