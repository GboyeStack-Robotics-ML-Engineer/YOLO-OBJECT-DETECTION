import cv2
#from IPython import display
import numpy as np

#from IPython.display import display, Image
#import matplotlib.pyplot as plt
import time
import ultralytics
#ultralytics.checks()
from ultralytics import YOLO
import time

WEIGTH_PATH=r'Weights\FootBallAnalytics.pt'
VIDEO_PATH=r'Videos\VILLARREAL CF 1 - 5 FC BARCELONA _ RESUMEN LALIGA EA SPORTS.mp4'
model=YOLO(WEIGTH_PATH).to('cuda')
cap = cv2.VideoCapture(VIDEO_PATH)
text="{}"
if not cap.isOpened():
 print("Cannot open camera")
 exit()
 
#frames=np.hstack([frame for ret,frame in cap.read() if cap.read()[0]])
while True:

   ret, frame = cap.read()   
   if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
   
   detections=model.predict(frame)
   
   for detection in detections:
      #   print(detection.boxes.data)
      #   print(detection.boxes.cls)
      #   print(detection.boxes.conf)
        for box in detection.boxes.data:
            
            (startX, startY, endX, endY,conf,cls) = box.cpu().numpy().astype(int).tolist()
            
            if cls>=0.7:
            
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 128, 0), 1)
                cv2.putText(frame, text.format(model.names[cls]), (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                break
        #detection.show()
   cv2.imshow('Video Stream',frame)
 
   
   if cv2.waitKey(1) == ord('q'):
      break
   
