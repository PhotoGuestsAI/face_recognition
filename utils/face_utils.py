from deepface import DeepFace
import os


def find_matching_images(guest_photo_path, extracted_album_dir):
    """
    Compare the guest photo with each photo in the extracted album directory using DeepFace.

    Args:
        guest_photo_path (str): Path to the guest photo.
        extracted_album_dir (str): Path to the directory where the album images are extracted.

    Returns:
        list: List of matching image filenames.
    """
    matching_images = []

    # Iterate over each image in the extracted album directory
    for root, _, files in os.walk(extracted_album_dir):
        for file in files:
            image_path = os.path.join(root, file)
            try:
                # Use DeepFace to verify the match
                result = DeepFace.verify(img1_path=guest_photo_path, img2_path=image_path, model_name="VGG-Face",
                                         enforce_detection=False)
                if result['verified']:
                    matching_images.append(file)
                    print(f"Match found: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return matching_images
