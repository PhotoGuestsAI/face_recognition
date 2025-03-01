import cv2
import numpy as np
from scipy.spatial.distance import cosine

def preprocess_image(image, transpose=True):
    """Preprocess image for FaceNet model."""
    img = cv2.resize(image, (160, 160))
    img = np.expand_dims(img, axis=0)
    if transpose:
        img = img[:, :, ::-1]  # RGB to BGR
    img = img / 127.5 - 1.0
    return img

def calculate_cosine_similarity(encoding1, encoding2):
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
