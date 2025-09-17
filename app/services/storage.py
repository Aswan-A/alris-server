import os
import uuid
from fastapi import UploadFile
from app.config.settings import settings
from supabase import create_client, Client

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

async def upload_file_to_supabase(file: UploadFile, path: str) -> str:
    """
    Upload a file to Supabase storage
    Returns the URL of the uploaded file
    """
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        full_path = f"{path}/{unique_filename}"
        
        # Upload the file
        contents = await file.read()
        await file.seek(0)  # Reset file pointer
        
        response = supabase.storage.from_("reports").upload(
            path=full_path,
            file=contents
        )
        
        # Get the public URL
        public_url = supabase.storage.from_("reports").get_public_url(full_path)
        return public_url
    
    except Exception as e:
        print(f"Error uploading file to Supabase: {e}")
        return None