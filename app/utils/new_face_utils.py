import boto3
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
from app.utils.Architecture import load_facenet512d_model
from scipy.spatial.distance import cosine, cdist
import numpy as np
import time  # Add this import at the top of the file

# Add this with the other constants at the top of the file
BUCKET_NAME = 'photoguests-events'

# Change from S3 paths to local paths
YOLO_WEIGHTS_PATH = 'app/weights/yolov8n-face.pt'
FACENET_WEIGHTS_PATH = 'app/weights/facenet512_weights.h5'

MATCHING_DISTANCE_THRESHOLD = 0.7
FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.5

def check_weights_exist():
    """Check if required model weights exist locally."""
    if not Path(YOLO_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS_PATH}, Consider cd face_recognition")
    if not Path(FACENET_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"FaceNet weights not found at {FACENET_WEIGHTS_PATH}")

def extract_faces(s3_client, guest_photos_paths, event_photos_paths, temp_dir, bucket_name):
    """
    Extracts faces from the event album photos and guest reference photos.
    
    Args:
        s3_client: boto3 S3 client instance
        guest_photos_paths: List of S3 paths to guest reference photos
        event_photos_paths: List of S3 paths to event photos
        temp_dir: Path to temporary directory for processing
        bucket_name: Name of the S3 bucket
    
    Returns:
        dict: Dictionary containing extracted face paths for both guests and events
    """
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from pathlib import Path
    import tempfile
    import os
    
    # Use provided temp_dir instead of creating new one
    guest_faces_dir = Path(temp_dir) / "guest_faces"
    event_faces_dir = Path(temp_dir) / "event_faces"
    guest_faces_dir.mkdir(exist_ok=True)
    event_faces_dir.mkdir(exist_ok=True)
    
    # Use local model path instead of downloading from S3
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    
    def process_image_batch(photo_paths, output_dir, prefix):
        extracted_faces = []
        face_mapping = {}  # Maps face crops to original photos
        face_count = 0
        
        for photo_path in photo_paths:
            try:
                # Check if this is an S3 path or local path
                if str(photo_path).startswith(bucket_name):
                    # S3 path
                    temp_image_path = Path(temp_dir) / f"temp_{os.path.basename(photo_path)}"
                    s3_client.download_file(
                        Bucket=photo_path.split('/')[0],
                        Key='/'.join(photo_path.split('/')[1:]),
                        Filename=str(temp_image_path)
                    )
                    image = cv2.imread(str(temp_image_path))
                    # Clean up temporary image
                    temp_image_path.unlink()
                else:
                    # Local path
                    image = cv2.imread(str(photo_path))
                
                if image is None:
                    print(f"Could not read image: {photo_path}")
                    continue
                
                # Detect faces
                results = yolo_model.predict(image, conf=FACE_DETECTOR_CONFIDENCE_THRESHOLD)

                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for box in boxes: 
                        # Get face location
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Add padding (20% of face size)
                        height = y2 - y1
                        width = x2 - x1
                        padding_v = int(height * 0.2)
                        padding_h = int(width * 0.2)
                        
                        # Ensure padding doesn't go outside image bounds
                        y1 = max(0, y1 - padding_v)
                        y2 = min(image.shape[0], y2 + padding_v)
                        x1 = max(0, x1 - padding_h)
                        x2 = min(image.shape[1], x2 + padding_h)
                        
                        # Extract and resize face crop
                        face_crop = image[y1:y2, x1:x2]
                        face_crop = cv2.resize(face_crop, (160, 160))
                        
                        # Save face crop
                        face_path = output_dir / f"{prefix}_face_{face_count}.jpg"
                        cv2.imwrite(str(face_path), face_crop)
                        
                        # Store the mapping and face path
                        extracted_faces.append(str(face_path))
                        face_mapping[str(face_path)] = {
                            'original_photo': photo_path
                        }
                        face_count += 1
                
            except Exception as e:
                print(f"Error processing image {photo_path}: {str(e)}")
                continue
        
        return extracted_faces, face_mapping
    
    # Process both guest and event photos
    guest_faces, guest_mapping = process_image_batch(guest_photos_paths, guest_faces_dir, "guest")
    event_faces, event_mapping = process_image_batch(event_photos_paths, event_faces_dir, "event")
    
    return guest_faces, event_faces, guest_mapping, event_mapping

def calculating_embeddings(s3_client, guest_faces, event_faces, guest_mapping, event_mapping, facenet_model):
    """
    Calculates embeddings for the guest faces and event faces using the facenet-512 model.
    
    Args:
        s3_client: boto3 S3 client instance
        guest_faces: List of paths to guest face images
        event_faces: List of paths to event face images
        guest_mapping: Dictionary mapping guest face paths to original photos
        event_mapping: Dictionary mapping event face paths to original photos
        facenet_model: Loaded FaceNet model instance
    
    Returns:
        dict: Dictionary containing embeddings and mappings in a concise format
    """
    def preprocess_image(image):
        """Preprocess image for FaceNet model."""
        img = cv2.resize(image, (160, 160))
        img = np.expand_dims(img, axis=0)
        img = img[:, :, ::-1]  # RGB to BGR
        img = img / 127.5 - 1.0
        return img

    def get_face_embedding(image_path):
        """Get face embedding for a single image."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        preprocessed_face = preprocess_image(image)
        embedding = facenet_model.predict(preprocessed_face)
        return embedding[0]
    
    # Calculate embeddings for guest faces
    guest_embeddings = []
    for face_path in guest_faces:
        embedding = get_face_embedding(face_path)
        if embedding is not None:
            guest_embeddings.append({
                'embedding': embedding,
                'original_photo': guest_mapping[face_path]['original_photo']
            })
    
    # Calculate embeddings for event faces
    event_embeddings = []
    for face_path in event_faces:
        embedding = get_face_embedding(face_path)
        if embedding is not None:
            event_embeddings.append({
                'embedding': embedding,
                'original_photo': event_mapping[face_path]['original_photo']
            })
    
    return guest_embeddings, event_embeddings

def calculate_matching(guest_embeddings, event_embeddings):
    """
    Calculate similarities between guest reference faces and event album faces using vectorized operations.
    Ensures each event photo appears only once per guest, keeping the highest similarity score.
    
    Args:
        guest_embeddings: List of dictionaries containing guest face embeddings and original photo paths
        event_embeddings: List of dictionaries containing event face embeddings and original photo paths
    
    Returns:
        dict: Dictionary mapping guest photos to their matching event photos
    """
    
    matches = {}
    
    # Extract embedding arrays and photo paths
    guest_photos = [data['original_photo'] for data in guest_embeddings]
    event_photos = [data['original_photo'] for data in event_embeddings]
    
    # Convert embeddings to numpy arrays for vectorized computation
    guest_emb_array = np.array([data['embedding'] for data in guest_embeddings])
    event_emb_array = np.array([data['embedding'] for data in event_embeddings])
    
    # Calculate all similarities at once using cosine distance
    # This returns a matrix of shape (num_guests, num_events)
    distances = cdist(guest_emb_array, event_emb_array, metric='cosine')
    similarities = 1 - distances
    
    # Find matches above threshold
    for i, guest_photo in enumerate(guest_photos):
        # Get indices where similarity is above threshold
        match_indices = np.where(similarities[i] >= MATCHING_DISTANCE_THRESHOLD)[0]
        
        # If we have matches
        if len(match_indices) > 0:
            # Create a dictionary to store the highest similarity score for each event photo
            best_matches = {}
            
            # For each matching index
            for idx in match_indices:
                event_photo = event_photos[idx]
                similarity = float(similarities[i][idx])
                
                # If this event photo hasn't been seen before, or if this similarity is higher
                if event_photo not in best_matches or similarity > best_matches[event_photo]['similarity_score']:
                    best_matches[event_photo] = {
                        'event_photo': event_photo,
                        'similarity_score': similarity
                    }
            
            # Convert the dictionary values to a list and sort by similarity score
            current_matches = sorted(
                best_matches.values(),
                key=lambda x: x['similarity_score'],
                reverse=True
            )
            
            # Add to matches dictionary
            matches[guest_photo] = current_matches
    
    return matches

def find_matching_mapping(s3_client, guest_photos_paths, event_photos_paths, temp_dir, bucket_name='photoguests-events'):
    check_weights_exist()

    # Measure time for extracting faces
    start_extract_time = time.time()
    guest_faces, event_faces, guest_mapping, event_mapping = extract_faces(
        s3_client, 
        guest_photos_paths, 
        event_photos_paths, 
        temp_dir,
        bucket_name
    )
    extract_time = time.time() - start_extract_time

    # Print detailed statistics
    print("\n=== Face Detection Statistics ===")
    print(f"Guest Photos Processing:")
    print(f"- Total guest photos processed: {len(guest_photos_paths)}")
    print(f"- Total faces detected in guest photos: {len(guest_faces)}")
    print(f"- Average faces per guest photo: {len(guest_faces)/len(guest_photos_paths):.2f}")
    
    print(f"\nEvent Photos Processing:")
    print(f"- Total event photos processed: {len(event_photos_paths)}")
    print(f"- Total faces detected in event photos: {len(event_faces)}")
    print(f"- Average faces per event photo: {len(event_faces)/len(event_photos_paths):.2f}")
    
    # Print sample of detections
    print("\n=== Sample Detections ===")
    print("Guest Photos (first 3):")
    for face_path in guest_faces[:3]:
        print(f"- Face: {os.path.basename(face_path)}")
        print(f"  From: {os.path.basename(guest_mapping[face_path]['original_photo'])}")
    
    print("\nEvent Photos (first 3):")
    for face_path in event_faces[:3]:
        print(f"- Face: {os.path.basename(face_path)}")
        print(f"  From: {os.path.basename(event_mapping[face_path]['original_photo'])}")
    
    print("\n=========================")

    # Use local model path instead of downloading from S3
    facenet_model = load_facenet512d_model()

    # Measure time for calculating embeddings
    start_embedding_time = time.time()
    guest_embeddings, event_embeddings = calculating_embeddings(s3_client, guest_faces, event_faces, guest_mapping, event_mapping, facenet_model)
    embedding_time = time.time() - start_embedding_time


    # Measure time for calculating matches
    start_matching_time = time.time()
    matches = calculate_matching(guest_embeddings, event_embeddings)
    matching_time = time.time() - start_matching_time

    # Return the matching results along with timing information
    return matches, {
        "face_extraction_time": extract_time,
        "embedding_time": embedding_time,
        "finding_matches_time": matching_time
    }
