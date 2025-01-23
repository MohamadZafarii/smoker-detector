# Smoking Detection with YOLO11n

This project demonstrates fine-tuning a YOLO11n model for detecting smoking-related objects ("cigarette" and "face") using a custom dataset. The trained model can be used for inference on webcam feeds, videos, or static images.

## Project Overview

- **Task**: Object detection
- **Dataset**: Custom dataset named **"Smoking"**, containing two classes:
    - `cigarette`
    - `face`
- **Augmentations**:
    - Rotation
    - Noise
    - Blur
- **Model**: YOLO11n (fine-tuned)
- **Training Configuration**:
    - **Optimizer**: AdamW
    - **Epochs**: 50
    - **Image Size**: 640x640
- **Framework**: YOLO implementation

## Requirements

To run this project, ensure the required dependencies are installed:

```bash
pip install -r requirements.txt
```

## How to Use the Model

### 1. Run on a Webcam or a video
To use the model for real-time detection with a webcam or video, follow these steps:

1. Open the .env file and set the VIDEO_SOURCE variable:

    If you want to use a webcam, specify its index (e.g., 0 or 1).
    If you want to use a video file, provide the path to the file.
2. Once VIDEO_SOURCE is set, run the following command:

```bash
python smocker_detection.py
```


### 2. Run on an Image
To perform detection on a static image:

```bash
python detect_image.py --source path/to/image.jpg
```

### 4. Model Weights
Ensure the YOLO11n fine-tuned weights (`yolo11n.pt`) are placed in the project directory.

## Dataset Description
The "Smoking" dataset was provided by Roboflow and includes two classes:
- **Cigarette**: Bounding boxes around smoking-related objects.
- **Face**: Bounding boxes around faces.

### Augmentations Used:
- **Rotation**: Helps improve detection for different orientations.
- **Noise**: Improves robustness under noisy conditions.

## Training Details
- **Model**: YOLO11n (fine-tuned on the "Smoking" dataset).
- **Optimizer**: AdamW
- **Epochs**: 50
- **Image Size**: 640x640
- **Framework**: YOLO (ensure compatibility with YOLO11n).

## Files in the Repository
- **`requirements.txt`**: List of dependencies required to run the project.
- **`detect_webcam.py`**: Code for real-time object detection using a webcam.
- **`detect_image.py`**: Code for object detection on static images.
- **`best.pt`**: Fine-tuned YOLO11n model weights.



