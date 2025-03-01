import os
import boto3
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
import requests
from typing import Dict, Tuple

CONFIDENCE_THRESHOLD = 0.8

# Fix the path to weights
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get tests directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Get project root
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "app", "weights", "yolov8n-face.pt")

print(f"Looking for weights at: {WEIGHTS_PATH}")  # Debug print

# Update these variables at the top of the file
BUCKET_NAME = "algorithm-v0"  # Replace with your actual bucket name

def download_and_process_s3_images(
    s3_event_prefix: str = "data/mock_event_photos",
    s3_reference_prefix: str = "data/mock_reference_photos",
    event_output_dir: str = "event_faces_crop",
    reference_output_dir: str = "reference_faces_crop"
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract faces from S3 wedding and reference photos, saving crops locally with mapping.
    
    Args:
        s3_event_prefix: S3 prefix for wedding photos
        s3_reference_prefix: S3 prefix for reference photos
        event_output_dir: Directory for wedding face crops
        reference_output_dir: Directory for reference face crops
    
    Returns:
        Tuple of two dictionaries mapping crop filenames to original image names
    """
    # Create output directories
    Path(event_output_dir).mkdir(parents=True, exist_ok=True)
    Path(reference_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Verify weights exist
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"YOLO weights not found at {WEIGHTS_PATH}")
    
    # Initialize S3 client and YOLO model
    print("Initializing YOLO model...")
    yolo_model = YOLO(WEIGHTS_PATH)
    s3_client = boto3.client('s3')

    # Initialize mapping dictionaries
    event_mapping = {}
    reference_mapping = {}
    
    def process_image(image_bytes) -> np.ndarray:
        """Convert image bytes to numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def extract_faces(
        prefix: str,
        output_dir: str,
        mapping: dict,
        single_face: bool = False
    ) -> None:
        """Extract faces from images in given S3 prefix."""
        paginator = s3_client.get_paginator('list_objects_v2')
        face_count = 0
        
        # Iterate through S3 objects
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in tqdm(page['Contents'], desc=f"Processing {prefix}"):
                try:
                    # Skip if not an image
                    if not obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    # Download image from S3
                    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                    image = process_image(response['Body'].read())
                    
                    if image is None:
                        print(f"Could not load image: {obj['Key']}")
                        continue
                    
                    # Detect faces
                    results = yolo_model.predict(image)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        
                        # For reference photos, only keep the face with highest confidence
                        if single_face:
                            best_box = max(boxes, key=lambda x: x.conf[0])
                            boxes = [best_box]
                        
                        for box in boxes:
                            if box.conf[0] < CONFIDENCE_THRESHOLD:
                                continue
                            
                            # Extract face with padding
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            height = y2 - y1
                            width = x2 - x1
                            padding_v = int(height * 0.2)
                            padding_h = int(width * 0.2)
                            
                            y1 = max(0, y1 - padding_v)
                            y2 = min(image.shape[0], y2 + padding_v)
                            x1 = max(0, x1 - padding_h)
                            x2 = min(image.shape[1], x2 + padding_h)
                            
                            face_crop = image[y1:y2, x1:x2]
                            face_crop = cv2.resize(face_crop, (160, 160))
                            
                            # Save face crop and update mapping
                            crop_filename = f"face_{face_count}.jpg"
                            face_path = Path(output_dir) / crop_filename
                            cv2.imwrite(str(face_path), face_crop)
                            
                            original_filename = Path(obj['Key']).name
                            mapping[crop_filename] = original_filename
                            
                            face_count += 1
                            
                            # For reference photos, we only need one face
                            if single_face:
                                break
                                
                except Exception as e:
                    print(f"Error processing {obj['Key']}: {str(e)}")
        
        print(f"Extracted {face_count} faces from {prefix}")
    
    # Process event photos
    extract_faces(s3_event_prefix, event_output_dir, event_mapping, single_face=False)
    
    # Process reference photos (only highest confidence face per image)
    extract_faces(s3_reference_prefix, reference_output_dir, reference_mapping, single_face=True)
    
    # Save mappings to JSON files
    with open('event_mapping.json', 'w') as f:
        json.dump(event_mapping, f, indent=2)
    
    with open('reference_mapping.json', 'w') as f:
        json.dump(reference_mapping, f, indent=2)
    
    return event_mapping, reference_mapping

if __name__ == "__main__":
    print("Script starting...")
    try:
        event_mapping, reference_mapping = download_and_process_s3_images()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
