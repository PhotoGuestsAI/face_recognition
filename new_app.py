import os
import shutil
import zipfile
import tempfile
import boto3
import json
from flask import Flask, request, jsonify
from app.utils.new_face_utils import find_matching_mapping
from pathlib import Path

app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

BUCKET_NAME = 'photoguests-events'

# Change from S3 paths to local paths
YOLO_WEIGHTS_PATH = 'app/weights/yolov8n-face.pt'
FACENET_WEIGHTS_PATH = 'app/weights/facenet512_weights.h5'

# Get port from environment variable or use default
PORT = int(os.environ.get('PORT', 5001))

def check_weights_exist():
    """Check if required model weights exist locally."""
    if not Path(YOLO_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS_PATH}, Consider cd face_recognition")
    if not Path(FACENET_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"FaceNet weights not found at {FACENET_WEIGHTS_PATH}")


def create_phone_to_images_mapping(processed_results):
    """Creates a mapping between phone numbers and their matching event photo names."""
    phone_mapping = {}
    for guest_photo, data in processed_results.items():
        phone_number = data['phone_number']
        matching_photos = [os.path.basename(match['event_photo']) for match in data['matches']]
        phone_mapping[phone_number] = matching_photos
    return phone_mapping


def save_phone_mappings_to_s3(phone_mapping, event_path):
    """Saves phone number to images mapping as JSON files in S3."""
    base_s3_path = f"{event_path}personalized_mapping"
    saved_paths = []
    
    # Create a temporary file for JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        for phone_number, images in phone_mapping.items():
            # Create the JSON content

            mapping_data = {
                "phone_number": phone_number,
                "matching_photos": images
            }
            
            # Write to temp file
            json.dump(mapping_data, temp_file)
            temp_file.flush()
            
            # Upload to S3
            s3_key = f"{base_s3_path}/{phone_number}/matches.json"
            s3_client.upload_file(temp_file.name, BUCKET_NAME, s3_key)
            saved_paths.append(s3_key)
            
            # Reset file for next write
            temp_file.seek(0)
            temp_file.truncate()
    
    # Clean up
    os.remove(temp_file.name)
    return saved_paths


@app.route('/process', methods=['POST'])
def process_album():
    """
    Endpoint to process the album and guest photo for face recognition.
    Expects:
    - event_album_s3_path: Path to event album directory on S3
    - event_path: Base S3 path where event data is stored

    Returns:
    - Processed results and phone number to images mapping
    """
    check_weights_exist()
    try:
        # Get data from request using the same parameter names as app.py
        data = request.json
        event_album_s3_path = data.get("event_album_s3_path")
        event_path = data.get("event_path")  # Example: PhotoGuestsAI/2025-04-09/tamar4/9ff481e7-5633...
        event_path += 'guest-submissions/'
        # Validate input
        if not event_album_s3_path or not event_path:
            return jsonify({"error": "Missing required parameters"}), 400

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory for event photos
            event_photos_dir = os.path.join(temp_dir, "event_photos")
            os.makedirs(event_photos_dir, exist_ok=True)

            # List and download all photos from the album directory
            event_photos_paths = []
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=event_album_s3_path)
            for obj in response.get('Contents', []):
                if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                    local_path = os.path.join(event_photos_dir, os.path.basename(obj['Key']))
                    s3_client.download_file(BUCKET_NAME, obj['Key'], local_path)
                    event_photos_paths.append(local_path)
                    # Store relative path for later use
                    rel_path = os.path.relpath(local_path, event_photos_dir)
                    print(f"Found event photo: {rel_path}")

            # List reference images in the specified S3 folder
            reference_images = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=event_path)
            guests_photos_paths = []
            for obj in reference_images.get('Contents', []):
                if obj['Key'].endswith(('.jpg', '.jpeg', '.png')):
                    guests_photos_paths.append(f"{BUCKET_NAME}/{obj['Key']}")

            results = find_matching_mapping(s3_client, 
                                        guest_photos_paths=guests_photos_paths,
                                        event_photos_paths=event_photos_paths,
                                        temp_dir=temp_dir,
                                        bucket_name=BUCKET_NAME)
            
            # Extract phone numbers and restructure results
            processed_results = {}
            for guest_photo, matches in results.items():
                # Extract phone number from filename
                filename = os.path.basename(guest_photo)
                phone_number = filename.split('_')[0]
                
                processed_results[guest_photo] = {
                    'phone_number': phone_number,
                    'matches': matches
                }
            
            # Create phone number to images mapping
            phone_mapping = create_phone_to_images_mapping(processed_results)
            
            # Save mappings to S3
            saved_paths = save_phone_mappings_to_s3(phone_mapping, event_path.replace('guest-submissions/', ''))
            
            return jsonify({
                "message": "Done", 
                "results": processed_results,
                "phone_mapping": phone_mapping,
                "saved_mapping_paths": saved_paths
            }), 200

        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_personalized_albums(results, temp_dir, event_path):
    """Creates personalized albums for all guests in S3."""
    base_s3_path = f"{event_path}personalized_albums"
    album_paths = [create_personalized_album(guest_photo, guest_data, temp_dir, base_s3_path) 
                  for guest_photo, guest_data in results.items()]
    # Filter out None results
    return [path for path in album_paths if path is not None]


def create_personalized_album(guest_photo_path, guest_data, temp_dir, base_s3_path):
    """Creates a personalized album ZIP file and uploads to S3."""
    phone_number = guest_data['phone_number']
    temp_zip_path = os.path.join(temp_dir, f"temp_{phone_number}.zip")
    
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)
    
    if not guest_data['matches']:
        print(f"Warning: No matches found for guest {phone_number}")
        return None
        
    with zipfile.ZipFile(temp_zip_path, 'w') as zipf:
        files_added = 0
        for match in guest_data['matches']:
            # Get the basename and search for it recursively
            photo_name = os.path.basename(match['event_photo'])
            found = False
            for root, _, files in os.walk(os.path.join(temp_dir, 'event_photos')):
                if photo_name in files:
                    event_photo = os.path.join(root, photo_name)
                    zipf.write(event_photo, photo_name)
                    files_added += 1
                    found = True
                    break
            if not found:
                print(f"Warning: Event photo not found anywhere: {photo_name}")
    
    if files_added == 0:
        print(f"Warning: No files were added to zip for guest {phone_number}")
        return None
        
    s3_key = f"{base_s3_path}/{phone_number}/personalized_album.zip"
    s3_client.upload_file(temp_zip_path, BUCKET_NAME, s3_key)
    os.remove(temp_zip_path)
    return s3_key


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
