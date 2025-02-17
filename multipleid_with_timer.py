import cv2
import math
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# Initialize time variables
pr_time = 0
curr_time = 0
previous_time = int(time.time())
fps=1/time.time()


# Open the video file
#vid_path=r"C:\pd_anomaly_detection\test video\suitcase - Search Images and 4 more pages - Personal - Microsoft​ Edge 2024-04-16 23-57-56.mp4"
#vid_path=r"C:\pd_anomaly_detection\test video\2_steady_bag.mp4"
vid_path = r"C:\pd_anomaly_detection\test video\Polite Luggage at Changi Airport.mp4"
cap = cv2.VideoCapture(vid_path)

# Set the width and height for video frames
des_width = 1280
des_height = 720

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO("yolov5su.pt")
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model.to("cuda:0")
# COCO classes
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Initialize centroid tracker
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.zeros((len(objectIDs), len(inputCentroids)), dtype="float")
            for i in range(0, len(objectIDs)):
                for j in range(0, len(inputCentroids)):
                    D[i, j] = np.linalg.norm(objectCentroids[i] - inputCentroids[j])
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

# Initialize centroid tracker
ct = CentroidTracker()

# Threshold for considering an object abandoned (in seconds)
abandoned_threshold = 10

# Initialize dictionary to store timestamps of last movement for each object
last_movement = {}

while True:
    # Read a frame from the video
    ret, img = cap.read()

    # if the frame is read
    if not ret:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Perform object detection using YOLO
    results = model.predict(img, stream=True)

    # Process detection results
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_index = int(box.cls[0])  # Access class labels directly
            confidence = box.conf[0]

            # Check if the detected object is a suitcase
            if coco_classes[class_index] == "suitcase":
                # Highlight the bounding box
                print("suitcase detected!!!!")

                # Append bounding box to detections list
                detections.append((x1, y1, x2, y2))

                # Draw bounding box with label "suitcase" and time per second
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Suitcase - {time.strftime('%H:%M:%S', time.localtime())}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img,f"fps:{fps}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2)

                # Update last movement dictionary with current time
                for (objectID, _) in ct.objects.items():
                    last_movement[objectID] = time.time()-previous_time
                    curr_time=last_movement[objectID]
                    print("r")
                    print(last_movement[objectID])

    # Update centroid tracker with the detections
    objects = ct.update(detections)

    # Draw bounding boxes and centroids
    # Draw bounding boxes and centroids
    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Check if the object has been stationary for longer than a threshold
        if objectID in last_movement:
            print(objectID)
            current_time=time.time()
            elapsed_time=current_time-previous_time
            print(elapsed_time)
            if elapsed_time > abandoned_threshold:
                # Draw warning symbol if the suiqtcase is stationary
                cv2.putText(img, "ABANDONED", (centroid[0], centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                # Draw red box around the detected suitcase
                if len(detections) > objectID:
                    x1, y1, x2, y2 = detections[objectID]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the final image
    resized = cv2.resize(img, (680, 420))
    cv2.imshow("Object Detection", resized)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()