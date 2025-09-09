import os
import uuid
from core.celery_app import celery_app
from tasks.image_generation import get_model_adapter # Reusing the model factory
from core.storage import upload_file_to_s3, download_file_from_s3

@celery_app.task(name="tasks.generate_3d_model_from_s3_image")
def generate_3d_model_from_s3_image(s3_image_url: str):
    """
    Celery task for the S3 Image-to-3D pipeline.
    1. Downloads an image from an S3 URL.
    2. Generates a 3D model using Hunyuan3D.
    3. Uploads the resulting 3D model back to S3.
    """
    local_image_path = None
    local_model_path = f"/tmp/{uuid.uuid4()}.glb"

    try:
        # --- Step 1: Download Image from S3 ---
        local_image_path = download_file_from_s3(s3_image_url)
        if not local_image_path:
            raise Exception(f"Failed to download image from S3 URL: {s3_image_url}")

        # --- Step 2: Generate 3D Model from the downloaded image ---
        hunyuan_adapter = get_model_adapter("hunyuan3d-v1")
        generated_model_file = hunyuan_adapter.predict(
            image_path=local_image_path,
            output_path=local_model_path
        )

        # --- Step 3: Upload the final 3D Model to S3 ---
        s3_model_url = upload_file_to_s3(generated_model_file)
        if not s3_model_url:
            raise Exception("Failed to upload 3D model to S3")

        return {
            "status": "SUCCESS",
            "source_s3_image_url": s3_image_url,
            "result_s3_model_url": s3_model_url
        }

    except Exception as e:
        print(f"Task failed: {e}")
        return {"status": "FAILURE", "error": str(e)}
    finally:
        # --- Step 4: Clean up local temporary files ---
        if local_image_path and os.path.exists(local_image_path):
            os.remove(local_image_path)
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
        print("Cleaned up temporary local files.")
