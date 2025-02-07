# YOLOv8-Based Real-Time Detection System

## Project Overview
This project implements a **real-time object detection system** using **YOLOv8** to detect thermometers, used thermometers, and humans. The model has been trained with a curated dataset using **Roboflow** for augmentation and is deployed for inference on images, videos, and live camera feeds.

## Key Features
- **Live Camera Detection**: Detects objects in real-time using a webcam.
- **Image Processing**: Performs inference on static images and saves results.
- **Video Processing**: Processes local video files with detection overlays.
- **Efficient and Fast**: Uses **YOLOv8** for high-speed, high-accuracy detection.
- **Custom Model Training**: Trained with an augmented dataset for robust performance.

## Technologies Used
- **YOLOv8**: State-of-the-art object detection model.
- **Roboflow**: Data augmentation and dataset management.
- **OpenCV**: Image and video processing.
- **Python**: Main programming language.
- **Google Colab**: Cloud-based training and testing environment.

## Installation
Ensure you have the necessary dependencies installed:
```bash
pip install ultralytics==8.0.20
pip install roboflow opencv-python pillow matplotlib
```

## Dataset Preparation
1. Process and augment your dataset using **Roboflow**.
2. Download the dataset with:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your_workspace").project("your_project")
dataset = project.version(1).download("yolov8")
```

## Model Training
Train the YOLOv8 model with:
```python
from ultralytics import YOLO
model = YOLO("yolov8.yaml")
model.train(data="dataset.yaml", epochs=50, imgsz=640)
```

## Running Inference
### On an Image
```python
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")
image = Image.open("your_image.jpg")
results = model.predict(source=image, show=True, save=True, save_crop=True)
```

### On a Video
```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("your_video.mp4")
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
```

### Live Camera Detection
```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Live Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
```

## Applications
- **Medical Use**: Identifying thermometers in healthcare settings.
- **Security & Safety**: Monitoring people in restricted areas.
- **Automated Surveillance**: Enhancing object detection in security systems.

## Contributors
- **Mahmoud Hany Fathalla**
- **Haneen Hazem**
- **Nada Ahmed**

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Ultralytics** for YOLOv8.
- **Roboflow** for dataset augmentation.
- **Google Colab** for cloud-based training and testing.
