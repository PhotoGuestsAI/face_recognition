import zipfile
import os


def extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def create_zip(file_paths, output_zip_path):
    """Create a zip file containing the specified files."""
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))
