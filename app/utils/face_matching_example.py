import cv2
import numpy as np
from pathlib import Path
from Architecture import load_facenet512d_model
from scipy.spatial.distance import cosine
from tqdm import tqdm

def preprocess_image(image, transpose=True):
    """Preprocess image for FaceNet model."""
    img = cv2.resize(image, (160, 160))
    img = np.expand_dims(img, axis=0)
    if transpose:
        img = img[:, :, ::-1]  # RGB to BGR
    img = img / 127.5 - 1.0
    return img

def calculate_similarity(encoding1, encoding2):
    """Calculate cosine similarity between two face encodings."""
    return 1 - cosine(encoding1, encoding2)

def get_face_embedding(image_path, facenet_model):
    """Get face embedding for a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    preprocessed_face = preprocess_image(image)
    embedding = facenet_model.predict(preprocessed_face)
    return embedding[0]

def match_faces(reference_dir, faces_crop_dir, output_dir, similarity_threshold=0.7):
    """
    Match faces from faces_crop directory with reference faces and organize them into folders.
    A face can match with multiple reference faces if similarity is above threshold.
    
    Args:
        reference_dir: Directory containing reference face images
        faces_crop_dir: Directory containing face crops to be matched
        output_dir: Directory where matched faces will be organized
        similarity_threshold: Threshold for face matching (default: 0.7)
    """
    # Load FaceNet model
    print("Loading FaceNet model...")
    facenet_model = load_facenet512d_model()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get reference faces and their embeddings
    reference_faces = {}
    print("Processing reference faces...")
    for ref_path in tqdm(list(Path(reference_dir).glob('*.*'))):
        if ref_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            embedding = get_face_embedding(ref_path, facenet_model)
            if embedding is not None:
                reference_faces[ref_path.stem] = {
                    'embedding': embedding,
                    'path': ref_path
                }
                # Create output directory for this person
                (output_path / ref_path.stem).mkdir(exist_ok=True)
    
    # Process and match face crops
    print("Matching faces...")
    face_crops = list(Path(faces_crop_dir).glob('*.*'))
    for face_path in tqdm(face_crops):
        if face_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
            
        # Get embedding for face crop
        face_embedding = get_face_embedding(face_path, facenet_model)
        if face_embedding is None:
            continue
        
        # Find all matches above threshold
        face_image = cv2.imread(str(face_path))
        for person_id, ref_data in reference_faces.items():
            similarity = calculate_similarity(face_embedding, ref_data['embedding'])
            
            # If similarity is above threshold, copy face to corresponding folder
            if similarity >= similarity_threshold:
                output_file = output_path / person_id / face_path.name
                cv2.imwrite(str(output_file), face_image)

if __name__ == "__main__":
    # Directory paths
    REFERENCE_DIR = "reference_faces"
    FACES_CROP_DIR = "faces_crop"
    OUTPUT_DIR = "matched_faces"
    
    # Match faces
    match_faces(REFERENCE_DIR, FACES_CROP_DIR, OUTPUT_DIR)
    print("Face matching completed!")
