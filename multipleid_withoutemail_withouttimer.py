# import cv2
# import time
# import torch
# import numpy as np
# from ultralytics import YOLO
# from collections import OrderedDict
# from scipy.spatial import distance as dist
#
# # Open the video file
#  vid_path=r"C:\pd_anomaly_detection\test video\suitcase - Search Images and 4 more pages - Personal - Microsoft​ Edge 2024-04-16 23-57-56.mp4"
# # vid_path=r"C:\pd_anomaly_detection\test video\2_steady_bag.mp4"
# # vid_path = r"C:\pd_anomaly_detection\test video\Polite Luggage at Changi Airport.mp4"
# cap = cv2.VideoCapture(vid_path)
#
# # Set the width and height for video frames
# # des_width = 1280
# # des_height = 720
# #
# # # Set the resolution of the video capture
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)
#
# # Load YOLO model
# model = YOLO("yolov5su.pt")
# model.conf = 0.4  # Adjust confidence threshold for detection
# print(torch.cuda.is_available())
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')
#
# coco_classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
#     "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
#     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]
#
# # Initialize centroid tracker
# class CentroidTracker:
#     def __init__(self, maxDisappeared=50):
#         self.nextObjectID = 0
#         self.objects = OrderedDict()
#         self.disappeared = OrderedDict()
#         self.maxDisappeared = maxDisappeared
#
#     def register(self, centroid):
#         self.objects[self.nextObjectID] = centroid
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1
#
#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]
#
#     def update(self, rects):
#         if len(rects) == 0:
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.maxDisappeared:
#                     self.deregister(objectID)
#             return self.objects
#
#         inputCentroids = np.zeros((len(rects), 2), dtype="int")
#         for (i, (startX, startY, endX, endY)) in enumerate(rects):
#             cX = int((startX + endX) / 2.0)
#             cY = int((startY + endY) / 2.0)
#             inputCentroids[i] = (cX, cY)
#
#         if len(self.objects) == 0:
#             for i in range(0, len(inputCentroids)):
#                 self.register(inputCentroids[i])
#         else:
#             objectIDs = list(self.objects.keys())
#             objectCentroids = list(self.objects.values())
#             D = dist.cdist(np.array(objectCentroids), inputCentroids)
#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]
#             usedRows = set()
#             usedCols = set()
#             for (row, col) in zip(rows, cols):
#                 if row in usedRows or col in usedCols:
#                     continue
#                 objectID = objectIDs[row]
#                 self.objects[objectID] = inputCentroids[col]
#                 self.disappeared[objectID] = 0
#                 usedRows.add(row)
#                 usedCols.add(col)
#             unusedRows = set(range(0, D.shape[0])).difference(usedRows)
#             unusedCols = set(range(0, D.shape[1])).difference(usedCols)
#             if D.shape[0] >= D.shape[1]:
#                 for row in unusedRows:
#                     objectID = objectIDs[row]
#                     self.disappeared[objectID] += 1
#                     if self.disappeared[objectID] > self.maxDisappeared:
#                         self.deregister(objectID)
#             else:
#                 for col in unusedCols:
#                     self.register(inputCentroids[col])
#         return self.objects
#
# # Initialize centroid tracker
# ct = CentroidTracker()
#
# # Movement threshold to consider an object as moving
# movement_threshold =2
#
# # Dictionary to store the last centroid of each object
# last_movement = {}
#
# while True:
#     # Read a frame from the video
#     ret, img = cap.read()
#
#     # if the frame is read
#     if not ret:
#         break
#
#     # Flip the image horizontally
#     img = cv2.flip(img, 1)
#
#     # Perform object detection using YOLO
#     results = model.predict(img, stream=True)
#
#     # Process detection results
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_index = int(box.cls[0])  # Access class labels directly
#             confidence = box.conf[0]
#
#             # Check if the detected object is a suitcase with high confidence
#             if coco_classes[class_index] == "suitcase" and confidence > model.conf:
#                 # Append bounding box to detections list
#                 detections.append((x1, y1, x2, y2))
#
#     # Update centroid tracker with the detections
#     objects = ct.update(detections)
#
#     # Draw bounding boxes and centroids
#     for (objectID, centroid) in objects.items():
#         # Draw bounding box with label "suitcase"
#         color = (0, 0, 255)  # Red for steady suitcases
#         if objectID in last_movement:
#             prev_centroid = last_movement[objectID]
#             # Calculate displacement between centroids
#             displacement = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))
#             if displacement > movement_threshold:
#                 color = (0, 255, 0)  # Green for moving suitcases
#
#         cv2.rectangle(img, (centroid[0] - 20, centroid[1] - 20), (centroid[0] + 20, centroid[1] + 20), color, 2)
#         cv2.putText(img, f"Suitcase", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         cv2.putText(img, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#         # Update last centroid for the object
#         last_movement[objectID] = centroid
#
#     # Show the final image
#     resized = cv2.resize(img, (680, 420))
#     cv2.imshow("Object Detection", resized)
#
#     # Break the loop if the 'Esc' key is pressed
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#
# # Release the video capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import math
# import cvzone
import time
import torch


# Initialize time variables
pr_time = 0
curr_time = 0
current_time=0
previous_time=0



centre_points_current_frame=[]
centre_points_previous_frame=[]
# Open the video file
vid_path = r"C:\pd_anomaly_detection\test video\2_steady_bag.mp4"
cap = cv2.VideoCapture(vid_path)
tracking_objects={}
track_id=0

# Set the width and height for video frames
des_width = 1280
des_height = 720

trialpoints=[]
# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO("yolov8n.pt")
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# model.to("cuda:0")
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Check for available GPUs
    model.to("cuda:0")  # Move the model to the GPU

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

# def detect_abandoned(p1,p2):
#     diff=int(math.hypot(p1[0]-p2[0],p1[1]-p2[1]))
#     print("difference",diff)
#     if diff<30:
#         print("NON ATTENDED!!!!!!!!")
#     else:
#         print("attended")
previous_time = int(time.time())

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
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x2 - (w / 2), y2 - (h / 2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            class_index = int(box.cls[0])

            # Check if the detected object is a suitcase
            if coco_classes[class_index] == "suitcase":

                # Highlight the bounding box
                print("cell phone detected!!!!")
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                # Display class name
                cv2.putText(img,f'{coco_classes[class_index]}',(x2,y2),cv2.FONT_HERSHEY_PLAIN,0.5,(255,0,0),2,2)
                # print("obj_centre_points:",(int(cx),int(cy)))
                centre_points_current_frame.append((cx,cy))

                cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), 3)

                # cv2.putText(img, format(elapsed_time), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #             (0, 255, 0), 2, 2)
                # previous_time = int(time.time())
                for pt1 in centre_points_current_frame:
                    for pt2 in centre_points_previous_frame:
                        diff = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

                        # print("DIFFERENCE", diff)
                        if diff < 20:
                            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                            print("UNATTENDED>>>>>>>>>")
                            current_time = int(time.time())
                            elapsed_time = current_time - previous_time
                            print(elapsed_time)



                            if elapsed_time>15:
                                # previous_time = int(time.time())
                                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                                # trialpoints.append((cx, cy))
                                cv2.circle(img, (int(cx), int(cy)), 10, (255, 255, 255), 5, 1)
                                # print("######################", trialpoints)
                                # cv2.putText(img, f'Unattended for {elapsed_time} seconds', (x1, y1 - 20),
                                            # cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 1, 1)


                        else:

                            print("attended")
                            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                            cv2.circle(img, (int(cx), int(cy)), 10, (0, 0, 0), 5, 1)
            # else:
            #     elapsed_time = 0
            #     previous_time = int(time.time())
            #     # print("tracking_infoooooooooooooooooooo",tracking_objects)


    # Calculate frames per second (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - pr_time)
    pr_time = curr_time

    # Display FPS on the image
    cv2.putText(img, f"{str(int(fps))}fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 3)

    # Show the final image
    resized=cv2.resize(img,(1280,720))
    # print("current_centre", centre_points_current_frame)
    # print("previous_centre", centre_points_previous_frame)

    cv2.imshow("Object Detection", resized)
    centre_points_previous_frame=centre_points_current_frame.copy()



    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
# close all windows
cap.release()
cv2.destroyAllWindows()