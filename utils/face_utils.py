from deepface import DeepFace


def match_faces(img1_path, img2_path, model_name='VGG-Face', distance_threshold=0.7):
    """
    Matches two faces using DeepFace.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        model_name (str): The model to use for face recognition.
        distance_threshold (float): Maximum distance for a match.

    Returns:
        bool: True if faces match, False otherwise.
    """
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name)
        return result["verified"] and result["distance"] <= distance_threshold
    except Exception as e:
        print(f"Error in DeepFace matching: {e}")
        return False


def find_matching_images(guest_photo_path, event_photos_paths, model_name='VGG-Face', distance_threshold=0.7):
    """
    Compares a guest photo against a list of event photos to find matches.

    Args:
        guest_photo_path (str): Path to the guest photo.
        event_photos_paths (list): List of paths to event photos.
        model_name (str): The model to use for face recognition.
        distance_threshold (float): Maximum distance for a match.

    Returns:
        list: List of matching image paths.
    """
    matching_images = []
    for photo_path in event_photos_paths:
        if match_faces(guest_photo_path, photo_path, model_name, distance_threshold):
            matching_images.append(photo_path)
    return matching_images
