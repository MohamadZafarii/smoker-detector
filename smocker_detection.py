import cv2 as cv
import cvzone
import random
import math
from ultralytics import YOLO
import os
from dotenv import load_dotenv


load_dotenv("./.env")
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE")
if VIDEO_SOURCE is not None:
    # Check if the value is numeric
    if VIDEO_SOURCE.isdigit():
        VIDEO_SOURCE = int(VIDEO_SOURCE)  # Convert to int for webcam index
    else:
        # Keep it as a string (assume it's a file path)
        VIDEO_SOURCE = VIDEO_SOURCE.strip()

class ObjectDetectionApp:
    """
    A class to handle real-time object detection using YOLO and OpenCV.

    Attributes:
        model (YOLO): The YOLO model loaded from the specified path.
        confidence_threshold (float): The minimum confidence required to consider a detection.
        cap (cv.VideoCapture): The video capture object for accessing the camera.
        colors (dict): A dictionary to store random colors for bounding boxes.
        frame_count (int): A counter for the number of processed frames.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.3, camera_index=VIDEO_SOURCE) -> None:
        """
        Initializes the ObjectDetectionApp with the given model, confidence threshold, and camera index.

        Args:
            model_path (str): Path to the YOLO model weights file.
            confidence_threshold (float): Minimum confidence for detections. Defaults to 0.3.
            camera_index (int): Index of the camera to use. Defaults to 0.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.cap = cv.VideoCapture(camera_index)
        self.colors = {}
        self.frame_count = 0

    @staticmethod
    def get_random_color() -> tuple[int, int, int]:
        """
        Generates a random color as an RGB tuple.

        Returns:
            tuple[int, int, int]: A random RGB color.
        """
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def process_frame(self, frame: cv.Mat) -> cv.Mat:
        """
        Processes a frame by flipping it and applying Gaussian blur.

        Args:
            frame (cv.Mat): The input frame to process.

        Returns:
            cv.Mat: The processed frame.
        """
        self.frame_count += 1
        frame = cv.flip(frame, 1)
        blurred_frame = cv.GaussianBlur(frame, (5, 5), 0.8)
        return blurred_frame

    def detect_objects(self, frame: cv.Mat) -> list[tuple[int, int, int, int, str]]:
        """
        Detects objects in the given frame using the YOLO model.

        Args:
            frame (cv.Mat): The input frame for object detection.

        Returns:
            list[tuple[int, int, int, int, str]]: A list of detections, each containing the bounding box coordinates
                                                 (x1, y1, w, h) and the label.
        """
        results = self.model(frame, stream=True)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                prob = math.ceil((box.conf[0] * 100)) / 100
                if prob < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                label = f"{self.model.names[int(box.cls[0])]} {prob:.2f}"
                detections.append((x1, y1, w, h, label))

        return detections

    def draw_detections(self, frame: cv.Mat, detections: list[tuple[int, int, int, int, str]]) -> None:
        """
        Draws bounding boxes and labels on the frame for each detection.

        Args:
            frame (cv.Mat): The frame to draw detections on.
            detections (list[tuple[int, int, int, int, str]]): A list of detections to draw.
        """
        for x1, y1, w, h, label in detections:
            color = self.get_random_color()
            cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(244, 64, 33), colorR=(87, 49, 200), l=9)
            cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1 - 10)), scale=0.8, thickness=1, offset=5)

    def run(self) -> None:
        """
        Runs the object detection application, capturing frames from the camera, processing them,
        detecting objects, and displaying the results in real time.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            detections = self.detect_objects(processed_frame)
            self.draw_detections(processed_frame, detections)

            cv.imshow("Object Detection", processed_frame)
            if cv.waitKey(1) == 27:  # Exit on 'Esc' key
                break

        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    app = ObjectDetectionApp(model_path="./weights/best.pt", confidence_threshold=0.3)
    app.run()
