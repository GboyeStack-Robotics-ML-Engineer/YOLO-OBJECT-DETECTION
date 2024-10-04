import cv2
from IPython import display
import numpy as np

from IPython.display import display, Image
import matplotlib.pyplot as plt
import time

import ultralytics
#ultralytics.checks()
from ultralytics import YOLO
import time

import supervision as sv

WEIGTH_PATH=r'Weights\YoloV8CarDetection.pt'
VIDEO_PATH=r'Videos\Cars Moving On Road Stock Footage - Free Download.mp4'

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

model=YOLO(WEIGTH_PATH).to('cuda')
cap = cv2.VideoCapture(VIDEO_PATH)


text="{}"

if not cap.isOpened():
 print("Cannot open camera")
 exit()
 

while True:

   ret, frame = cap.read()   
   if not ret:
      print("Can't receive frame (stream end?) and end of frame reached. Exiting ...")
      break
  
   detections=model.predict(frame)[0]
   #print(detections)
   
   detections = sv.Detections.from_ultralytics(detections)
  
   labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
    ]
   
   annotated_image = box_annotator.annotate(
                                             scene=frame, detections=detections)
   annotated_image = label_annotator.annotate(
                                             scene=annotated_image, detections=detections, labels=labels)
   
   # for detection in detections:
      
   #      for box in detection.boxes.data:
            
   #          (startX, startY, endX, endY,conf,cls) = box.cpu().numpy().astype(int).tolist()
            
   #          cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 128, 0), 1)
   #          cv2.putText(frame, text.format(model.names[cls]), (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
   

   cv2.imshow('Video Stream',annotated_image)
 
   
   if cv2.waitKey(1) == ord('q'):
      break
   
