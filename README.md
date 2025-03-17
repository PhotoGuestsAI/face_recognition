# Face Recognition
## Project Structure 📁

```
face_recognition/
├── app/
│   ├── core/        # Core ML functionality
│   ├── api/         # API endpoints
│   ├── utils/       # Utility functions
│   └── config/      # Configuration files
├── tests/           # Test suite
├── logs/            # Application logs
└── docker/          # Docker configuration
```
## How to run the project
### Option 1
1) clone the repo
2) make sure you have valid .env file
3) create 'weights' folder in face_recognition/app/
4) download for s3 the weights of facenet and yolo
5) run new_app.py

### Option 2
1) download fron google drive the docker image (face_algo.tar)
2) make it valid docker image by    docker load -i <path_to_your_tar_file>.tar
3) run docker run -p 5001:5001 -e PORT=5001

## How to Run the Project  

### Option 1: Run Locally  
1. Clone the repo and enter the directory: `git clone <repository_url> && cd <repository_name>`  
2. Ensure you have a valid `.env` file.  
3. Create a `weights` folder: `mkdir -p face_recognition/app/weights`  
4. Download FaceNet & YOLO weights from S3 into `weights/`.  
5. Run the app: `python new_app.py`  

### Option 2: Run with Docker  
1. Download `face_algo.tar` from Google Drive.  
2. Load the image: `docker load -i <path_to_your_tar_file>.tar`  
3. Run the container: `docker run -p 5001:5001 -e PORT=5001 face_algo`  

