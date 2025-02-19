import os
import shutil
import zipfile
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.face_utils import find_matching_images

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_album():
    """
    Endpoint to process the album and guest photo for face recognition.
    Expects:
    - event_album: ZIP file containing the event album
    - guest_photo: JPG/PNG file containing the guest photo

    Returns:
    - Processed personalized album as a ZIP file
    """
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temp directory
            event_album = request.files['event_album']
            guest_photo = request.files['guest_photo']

            event_album_path = os.path.join(temp_dir, secure_filename(event_album.filename))
            guest_photo_path = os.path.join(temp_dir, secure_filename(guest_photo.filename))

            event_album.save(event_album_path)
            guest_photo.save(guest_photo_path)

            # Process the album and create a personalized version
            personalized_album_path = os.path.join(temp_dir, "personalized_album.zip")
            create_personalized_album(event_album_path, guest_photo_path, personalized_album_path)

            # Send the personalized album back as a response
            with open(personalized_album_path, 'rb') as f:
                return f.read(), 200, {
                    'Content-Type': 'application/zip',
                    'Content-Disposition': 'attachment; filename="personalized_album.zip"'
                }
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_personalized_album(album_path, guest_photo_path, output_path):
    """
    Process the event album and guest photo to create a personalized album.

    Args:
        album_path (str): Path to the event album zip file.
        guest_photo_path (str): Path to the guest's photo.
        output_path (str): Path to save the personalized album zip file.

    Returns:
        str: Path to the created personalized album zip file.
    """
    try:
        # Unzip the event album to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            album_extraction_path = os.path.join(temp_dir, "album")
            with zipfile.ZipFile(album_path, 'r') as zip_ref:
                zip_ref.extractall(album_extraction_path)

            # Gather paths for all event photos, excluding __MACOSX and hidden files
            event_photos_paths = []
            for root, _, files in os.walk(album_extraction_path):
                if "__MACOSX" in root:
                    continue
                for file in files:
                    if not file.startswith("._"):  # Exclude hidden files
                        event_photos_paths.append(os.path.join(root, file))

            # Find matching photos
            matching_photos = find_matching_images(guest_photo_path, event_photos_paths)

            # Create a directory for personalized images
            personalized_images_dir = os.path.join(temp_dir, "personalized")
            os.makedirs(personalized_images_dir, exist_ok=True)

            # Copy matching photos to personalized directory
            for photo_path in matching_photos:
                shutil.copy(photo_path, personalized_images_dir)

            # Zip the personalized album
            with zipfile.ZipFile(output_path, 'w') as zipf:
                for root, _, files in os.walk(personalized_images_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, personalized_images_dir)
                        zipf.write(file_path, arcname)

            print(f"Personalized album created at {output_path}")
            return output_path

    except Exception as e:
        print(f"Error creating personalized album: {e}")
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
