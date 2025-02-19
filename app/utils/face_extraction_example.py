from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

CONFIDENCE_THRESHOLD = 0.8

def extract_face_crops(input_dir: str = "wedding_images", output_dir: str = "faces_crop"):
    """
    Extract face crops from images using YOLO face detection.
    
    Args:
        input_dir (str): Directory containing wedding images
        output_dir (str): Directory where face crops will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print("Loading YOLO model...")
    yolo_model = YOLO("weights/yolov8n-face.pt")
    
    # Get all valid image paths
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [
        f for f in Path(input_dir).glob('*') 
        if f.suffix.lower() in valid_extensions
    ]
    
    print(f"Found {len(image_paths)} images to process")
    face_count = 0
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
            
            # Detect faces using YOLO
            results = yolo_model.predict(image)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Check confidence score
                    if box.conf[0] < CONFIDENCE_THRESHOLD:
                        continue
                    
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
                    
                    # Extract face crop
                    face_crop = image[y1:y2, x1:x2]
                    # resize the face crop to 160x160
                    face_crop = cv2.resize(face_crop, (160, 160))
                    # Save face crop
                    face_path = Path(output_dir) / f"face_{face_count}.jpg"
                    cv2.imwrite(str(face_path), face_crop)
                    face_count += 1
                    
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Successfully extracted {face_count} faces")

if __name__ == "__main__":
    extract_face_crops()
