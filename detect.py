import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO


# Open the video file
vid_path=r"C:\sanikabhalge\test photos field\IMG_4720.mp4"
# vid_path=r"C:\pd_anomaly_detection\test video\2_steady_bag.mp4"
# vid_path = r"C:\pd_anomaly_detection\test video\Polite Luggage at Changi Airport.mp4"
cap = cv2.VideoCapture(vid_path)

# Set the width and height for video frames
# des_width = 1280
# des_height = 720
#
# # Set the resolution of the video capture
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO(r"C:\pd_anomaly_detection\PD___Anomaly_Detection\runs\detect\train3\weights\best.pt")
model.conf = 0.4  # Adjust confidence threshold for detection
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

coco_classes = [
    "background",
"blue_cone",
"large_orange_cone",
"orange_cone",
"unknown_cone",
"yellow_cone"
]



while True:
    # Read a frame from the video
    ret, img = cap.read()
    start_time=time.time()
    # if the frame is read
    if not ret:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Perform object detection using YOLO
    results = model.predict(img, stream=True)

    # Process detection results

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w, h = x2 - x1, y2 - y1
            class_index = int(box.cls[0])  # Access class labels directly
            confidence = box.conf[0]
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0))








    # Show the final image

    fps = 1 / (time.time()- start_time)
    start_time=time.time()

    # Display FPS on the image
    cv2.putText(img, f"{str(int(fps))}fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 3)
    resized = cv2.resize(img, (680, 420))

    cv2.imshow("Object Detection", resized)

    # Break the loop if the 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()