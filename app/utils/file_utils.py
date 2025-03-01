import os

def cleanup_directory(directory):
    """Delete all files and the directory itself."""
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
    os.rmdir(directory)
