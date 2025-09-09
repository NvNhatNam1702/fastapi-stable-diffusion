# core/storage.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse
import os
from config.settings import settings

def download_file_from_s3(s3_url: str, destination_folder: str = "/tmp") -> str:
    """Download a file from an S3 URL to a local temporary path."""
    try:
        # Parse the S3 URL to get bucket and key
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        
        # Create a unique local filename
        local_filename = os.path.join(destination_folder, f"{os.path.basename(object_key)}")
        
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        
        print(f"Downloading s3://{bucket_name}/{object_key} to {local_filename}")
        s3_client.download_file(bucket_name, object_key, local_filename)
        
        return local_filename
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("The object does not exist in S3.")
        else:
            print(f"An S3 client error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during S3 download: {e}")
        return None

def upload_file_to_s3(file_name: str, object_name: str = None) -> str:
    """Upload a file to an S3 bucket and return its public URL."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    try:
        s3_client.upload_file(file_name, settings.S3_BUCKET_NAME, object_name)
        url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{object_name}"
        print(f"Upload Successful: {url}")
        return url
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
