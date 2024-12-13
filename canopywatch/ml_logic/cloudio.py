import os
from google.cloud import storage
from canopywatch.params import BUCKET_NAME, DATA_PATH
from PIL import Image
import io

def upload_files_to_cloud_storage(subdir: str):
    """
    subdir should be for example 'images' or 'masks'
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    root_path = os.path.join(DATA_PATH, subdir)

    i = 0
    for image in os.listdir(root_path):
        if image.endswith(".jpg"):
            blob = bucket.blob(subdir + "/" + image)
            blob.upload_from_filename(os.path.join(root_path, image))
            print(f"Uploaded {image} to {BUCKET_NAME} imgNum={i}")
            i += 1

def read_images_from_cloud_write_local_machine(dest_root_path: str, subdir: str,
                                               max_images: int = -1):
    """
    Requires dest_root_path and dest_root_path/subdir to already exist
    subdir should be for example 'images' or 'masks'
    Give a max_images value if you first want to test
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=subdir)

    i = 0
    for blob in blobs:
        if blob.name.endswith(('.jpg')):
            img_data = blob.download_as_bytes()
            image = Image.open(io.BytesIO(img_data))
            local_file_path = dest_root_path + "/" + blob.name
            image.save(local_file_path, format='JPEG')
            print(f"Wrote {image} to {local_file_path} imgNum={i}")
            i += 1
            if max_images > -1 and i>= max_images:
                break
