import cv2 as cv
import cvzone
import random
import math
from ultralytics import YOLO

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Load the YOLO model
model = YOLO("./weights/smoker_new.pt")
colors = {}
count = 0
CONFIDENCE_THRESHOLD = 0.3
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    count+=1
    frame = cv.flip(frame, 1)
    # frame = cv.GaussianBlur(frame, (5, 5), 0.5)
    bluer_frame = cv.GaussianBlur(frame , (5,5) , 0.8)
    if not ret:
        break

    results = model(bluer_frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            prob = math.ceil((box.conf[0] * 100)) / 100
            if prob < CONFIDENCE_THRESHOLD :
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Check if box.id is not None before accessing it
            # if box.id is not None:
            #     id = int(box.id[0])
            #
            #     # Get the color for this track_id, or assign a new random color
            #     if id not in colors:
            #         colors[id] = get_random_color()

            color = get_random_color()  # Use the assigned color
            label = f"{model.names[int(box.cls[0])]} {prob:.2f}"

            # Draw the bounding box with the assigned color
            cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(244,64,33), colorR=(87,49,200), l=9)

            # Display the label above the bounding box
            cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1 - 10)), scale=0.8, thickness=1, offset=5)


    cv.imshow("hi", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

