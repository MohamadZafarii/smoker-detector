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

### 1. Run on a Webcam
To use the model for real-time detection with a webcam:

```bash
python detect_webcam.py
```

### 2. Run on a Video File
To use the model on a pre-recorded video:

```bash
python detect_video.py --source path/to/video.mp4
```

### 3. Run on an Image
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
- **`detect_video.py`**: Code for object detection on video files.
- **`detect_image.py`**: Code for object detection on static images.
- **`yolo11n.pt`**: Fine-tuned YOLO11n model weights (ensure this file is in the directory).



