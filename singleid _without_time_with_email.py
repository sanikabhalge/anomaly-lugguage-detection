import ssl
# run this on proxy free net and replace your mail with the unique token generated
# when you allow 2fa on your mail via google app password for third party apps
import cv2
from ultralytics import YOLO
import math
import cvzone
import time
import torch
import smtplib

# server = smtplib.SMTP("smtp.gmail.com", 587)
# server.starttls()
# server.login('UnknownUserVPB69@gmail.com', 'wiww ylou igsy ynmn')

# Initialize time variables
pr_time = 0
curr_time = 0
previous_time=int(time.time())

centre_points_current_frame=[]
centre_points_previous_frame=[]
# Open the video file
vid_path = r"C:\Users\bhalg\Videos\Captures\suitcase - Search Images and 4 more pages - Personal - Microsoft​ Edge 2024-04-16 23-57-56.mp4"
# vid_path1 = "tcp://100.80.24.66:4747"  # replace with your actual IP and port
cap = cv2.VideoCapture(vid_path)

# Set the width and height for video frames
des_width = 1280
des_height = 720

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO("yolov8n.pt")
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# model.to("cuda:0")
# # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#     # Check for available GPUs
#     model.to("cuda:0")  # Move the model to the GPU

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
            if coco_classes[class_index] == "cell phone":
                # Highlight the bounding box
                print("suitcase detected!!!!")

                current_time=int(time.time())
                elapsed_time=current_time-previous_time
                # Display class name
                cv2.putText(img,f'{coco_classes[class_index]}',(x2,y2),cv2.FONT_HERSHEY_PLAIN,0.5,(255,0,0),2,2)
                print("obj_centre_points:",(int(cx),int(cy)))
                centre_points_current_frame.append((cx,cy))

                cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), 3)
                cv2.putText(img, format(elapsed_time), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, 2)

                for pt1 in centre_points_current_frame:
                    for pt2 in centre_points_previous_frame:
                        diff=(math.hypot(pt1[0]-pt2[0],pt1[1]-pt2[1]))
                        # print("DIFFERENCE",diff)
                        if diff<10 :
                            print("UNATTENDED>>>>>>>>>")
                            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                            # server.sendmail('UnknownUserVPB69@gmail.com', 'Manthan.barhate70@gmail.com', 'Bag is Unattended !!!')


                        else:
                            print("attended")
                            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)



    # Calculate frames per second (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - pr_time)
    pr_time = curr_time

    # Display FPS on the image
    cv2.putText(img, "15 fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 3)

    # Show the final image
    resized=cv2.resize(img,(680,420))
    cv2.imshow("Object Detection", resized)
    centre_points_previous_frame=centre_points_current_frame.copy()

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
# close all windows
cap.release()
cv2.destroyAllWindows()




