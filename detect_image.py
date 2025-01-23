import argparse
import cv2
import os
from ultralytics import YOLO
import math

# Load the YOLO model
model = YOLO("./weights/best.pt")

def draw_detections(image, results):
    """
    Draw bounding boxes and labels on the image based on YOLO detection results.

    Args:
        image (numpy.ndarray): The image to draw on.
        results: The YOLO detection results.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            prob = math.ceil((box.conf[0] * 100)) / 100
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} {prob:.2f}"

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label above the bounding box
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def process_image(image_path):
    """
    Process the image by running YOLO detection and displaying the results.

    Args:
        image_path (str): The path to the image file.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return

    # Read the image
    img = cv2.imread(image_path)

    # Run YOLO detection
    results = model(img)

    # Draw detections on the image
    draw_detections(img, results)

    # Display the processed image
    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_path_from_user():
    """
    Prompt the user to enter the image path if not provided via command line.

    Returns:
        str or None: The valid image path or None if invalid.
    """
    image_path = input("Please enter the path to the image: ")
    if not os.path.exists(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return None
    return image_path

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Process an image.')
    parser.add_argument('--source', type=str, help='Path to the image file', default=None)

    # Parse arguments
    args = parser.parse_args()

    # If the user provides the image path via command line, use it. Otherwise, prompt them.
    if args.source:
        image_path = args.source
    else:
        print("No command-line argument provided.")
        image_path = get_image_path_from_user()

    # Proceed if the image path is valid
    if image_path:
        process_image(image_path)

if __name__ == '__main__':
    main()
