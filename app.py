import os
import shutil
import tempfile
import zipfile

from flask import Flask, request, jsonify

from app.utils.face_utils import find_matching_images
from app.utils.s3_utils import (
    download_file_from_s3, upload_file_to_s3, upload_images_to_s3, get_guest_list_from_s3
)

app = Flask(__name__)
BUCKET_NAME = os.getenv("EVENTS_BUCKET_NAME", "photoguests-events")


@app.route('/process', methods=['POST'])
def process_album():
    """
    Processes all guests in `guest_list.json` and generates personalized albums.
    Expects JSON with:
    - event_album_s3_path: Path to event album ZIP on S3.
    - event_path: Base S3 path where event data is stored.

    Returns:
    - JSON containing S3 URLs for each guest’s personalized album.
    """
    data = request.json
    event_album_s3_path = data.get("event_album_s3_path")
    event_path = data.get("event_path")  # Example: PhotoGuestsAI/2025-04-09/tamar4/9ff481e7-5633...

    # Validate input
    if not event_album_s3_path or not event_path:
        return jsonify({"error": "Missing required parameters"}), 400

    # Get guest list from S3
    guest_list = get_guest_list_from_s3(event_path)
    if not guest_list:
        return jsonify({"error": "No guests found in guest_list.json"}), 400

    # ✅ Use a persistent temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"✅ Created temp directory: {temp_dir}")

    try:
        album_zip_path = os.path.join(temp_dir, "event_album.zip")
        print(f"📥 Downloading event album: {event_album_s3_path}")
        download_file_from_s3(event_album_s3_path, album_zip_path)

        for guest in guest_list:
            guest_name = guest.get("name")
            guest_phone = guest.get("phone")
            guest_photo_s3_path = guest.get("photo_url").replace(
                f"https://{BUCKET_NAME}.s3.amazonaws.com/", ""
            )
            output_s3_path = f"{event_path}personalized-albums/{guest_phone}/"

            print(f"🟢 Processing guest: {guest_name} ({guest_phone})")

            # Download guest's photo
            guest_photo_path = os.path.join(temp_dir, f"{guest_phone}.jpg")
            print(f"📥 Downloading guest photo: {guest_photo_s3_path}")
            download_file_from_s3(guest_photo_s3_path, guest_photo_path)

            # Generate personalized album
            personalized_album_path, extracted_image_paths = create_personalized_album(
                album_zip_path, guest_photo_path, temp_dir
            )

            # Upload personalized album ZIP
            zip_s3_path = f"{output_s3_path}{guest_phone}.zip"
            print(f"📤 Uploading personalized album to S3: {zip_s3_path}")
            upload_file_to_s3(personalized_album_path, zip_s3_path, "application/zip")

            # Upload extracted images
            upload_images_to_s3(extracted_image_paths, zip_s3_path.rsplit("/", 1)[0])

        return jsonify({"success": "personalized albums created"}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Temporary directory deleted: {temp_dir}")


def create_personalized_album(album_path, guest_photo_path, temp_dir):
    """
    Matches guest photo with the event album to create a personalized album.

    Returns:
        - Path to personalized ZIP file
        - List of extracted matching image paths
    """
    try:
        album_extraction_path = os.path.join(temp_dir, "album")
        os.makedirs(album_extraction_path, exist_ok=True)

        print(f"📂 Extracting event album: {album_path}")
        with zipfile.ZipFile(album_path, 'r') as zip_ref:
            zip_ref.extractall(album_extraction_path)

        event_photos_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(album_extraction_path)
            for file in files if not file.startswith("._") and "__MACOSX" not in root
        ]

        print(f"🔍 Finding matching images for guest photo: {guest_photo_path}")
        matching_photos = find_matching_images(guest_photo_path, event_photos_paths)

        # Create a directory for personalized images
        personalized_images_dir = os.path.join(temp_dir, "personalized")
        os.makedirs(personalized_images_dir, exist_ok=True)

        # Copy matching photos
        for photo_path in matching_photos:
            shutil.copy(photo_path, personalized_images_dir)

        # Create ZIP file
        personalized_album_path = os.path.join(temp_dir, "personalized_album.zip")
        print(f"📦 Creating ZIP archive: {personalized_album_path}")
        with zipfile.ZipFile(personalized_album_path, 'w') as zipf:
            for root, _, files in os.walk(personalized_images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, personalized_images_dir))

        return personalized_album_path, [
            os.path.join(personalized_images_dir, file) for file in os.listdir(personalized_images_dir)
        ]

    except Exception as e:
        print(f"❌ Error creating personalized album: {e}")
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)