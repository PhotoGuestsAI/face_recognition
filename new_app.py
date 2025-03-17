import os
import tempfile
import boto3
import json
from flask import Flask, request, jsonify
from app.utils.new_face_utils import find_matching_mapping
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor
import time


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

            # Measure time for downloading event photos
            start_event_download_time = time.time()
            event_photos_paths = []
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=event_album_s3_path)

            def download_file(obj_key):
                try:
                    local_path = os.path.join(event_photos_dir, os.path.basename(obj_key))
                    with open(local_path, 'wb') as f:
                        s3_client.download_fileobj(BUCKET_NAME, obj_key, f)
                    return local_path
                except Exception as e:
                    print(f"Error downloading {obj_key}: {e}")
                    return None  

            with ThreadPoolExecutor() as executor:
                future_to_key = {executor.submit(download_file, obj['Key']): obj['Key'] for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))}
                for future in future_to_key:
                    local_path = future.result()
                    event_photos_paths.append(local_path)
                    # Store relative path for later use
                    rel_path = os.path.relpath(local_path, event_photos_dir)
                    print(f"Found event photo: {rel_path}")
            event_download_time = time.time() - start_event_download_time

            # Measure time for downloading guest photos
            start_guest_download_time = time.time()
            reference_images = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=event_path)
            guests_photos_paths = []
            guest_photos_dir = os.path.join(temp_dir, "guest_photos")
            os.makedirs(guest_photos_dir, exist_ok=True)

            def download_guest_photo(obj_key):
                try:
                    local_path = os.path.join(guest_photos_dir, os.path.basename(obj_key))
                    with open(local_path, 'wb') as f:
                        s3_client.download_fileobj(BUCKET_NAME, obj_key, f)
                    return local_path
                except Exception as e:
                    print(f"Error downloading {obj_key}: {e}")
                    return None  

            with ThreadPoolExecutor() as executor:
                future_to_key = {executor.submit(download_guest_photo, obj['Key']): obj['Key'] for obj in reference_images.get('Contents', []) if obj['Key'].endswith(('.jpg', '.jpeg', '.png'))}
                for future in future_to_key:
                    local_path = future.result()
                    if local_path:
                        guests_photos_paths.append(local_path)
            guest_download_time = time.time() - start_guest_download_time

            # Now call the find_matching_mapping function with local paths
            results, timing_info = find_matching_mapping(s3_client, 
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
                "saved_mapping_paths": saved_paths,
                "timing_info": {
                    "event_download_time": event_download_time,
                    "guest_download_time": guest_download_time,
                    **timing_info  # Include timing information from find_matching_mapping
                }
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
