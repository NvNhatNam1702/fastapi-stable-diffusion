import boto3
from botocore.exceptions import NoCredentialsError
from urllib.parse import urlparse
import os
from config.settings import settings

def get_s3_client():
    """Helper function to create a boto3 client configured for our S3-compatible storage."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        region_name=settings.S3_REGION
    )

def download_file_from_storage(file_url: str, destination_folder: str = "/tmp") -> str:
    """Download a file from MinIO to a local temporary path."""
    parsed_url = urlparse(file_url)
    bucket_name = parsed_url.path.split('/')[1]
    object_key = '/'.join(parsed_url.path.split('/')[2:])
    
    local_filename = os.path.join(destination_folder, os.path.basename(object_key))
    s3_client = get_s3_client()
    
    print(f"Downloading s3://{bucket_name}/{object_key} from MinIO to {local_filename}")
    s3_client.download_file(bucket_name, object_key, local_filename)
    return local_filename

def upload_file_to_storage(file_name: str, object_name: str = None) -> str:
    """Upload a file to the MinIO bucket and return its public URL."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = get_s3_client()
    s3_client.upload_file(file_name, settings.S3_BUCKET_NAME, object_name)
    
    # Construct the URL for MinIO access (note: this URL is for external access)
    # For local testing, you might need to replace 'minio:9000' with 'localhost:9000' if accessing from your browser
    public_endpoint = settings.S3_ENDPOINT_URL.replace("minio", "localhost")
    url = f"{public_endpoint}/{settings.S3_BUCKET_NAME}/{object_name}"

    print(f"Upload to MinIO Successful: {url}")
    return url

