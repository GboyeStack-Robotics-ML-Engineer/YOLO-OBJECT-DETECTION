import cv2
from IPython import display
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt
import time
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import time

import supervision as sv
import warnings
import os
import torch 

WEIGTH_PATH=r'Weights\YoloV8CarDetection.pt'
VIDEO_PATH=r'Videos\Cars Moving On Road Stock Footage - Free Download.mp4'


device='cuda'if torch.cuda.is_available() else 'cpu'
model=YOLO(WEIGTH_PATH).to(device)
cap = cv2.VideoCapture(VIDEO_PATH)

text="{}"
if not cap.isOpened():
 print("Cannot open camera")
 exit()
 

class CarDetect:
   def __init__(self):
      self.frame=None
      self.ret=None
      self.detections=None
      self.video_path=None
      self.model=None
      self.box_annotator = sv.BoxAnnotator()
      self.label_annotator = sv.LabelAnnotator()
      
      
   def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model.predict(image_slice)[0]
    return sv.Detections.from_ultralytics(result)
      
   def InferenceSlicer(self,model,frame):
      
      slicer = sv.InferenceSlicer(callback =self.callback)
      
      detections=slicer(frame)
      
      return detections
   
   def get_default_save_path(self):
      
      default_dir=f"Results\{os.path.basename(__file__).split('.py')[0]}\Runs"
            
      if os.path.exists(default_dir):
         files_in_dir=os.listdir(default_dir)
         path=os.path.join(default_dir,f"Results_{len(files_in_dir)+1}.avi")
         
      elif not os.path.exists(default_dir):
         os.makedirs(default_dir)
         path=os.path.join(default_dir,f'Results_1.avi')
      
      return path
      
      
   def detect (self,video_path,model,use_slicer=False,track=False,tracker=sv.ByteTrack(),save_dir=None,save=True,display=False):
      
      self.model=model
      self.video_path=video_path
   
      cap = cv2.VideoCapture(self.video_path)
      
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
      if not cap.isOpened():
         print("Cannot open camera")
         exit()
      
      if save:
            
            size=(width,height)
            
            if save_dir is None:
               default_save_path=self.get_default_save_path()
               warnings.warn(f'No directory was specified for saving results.Results are saved to default path: {default_save_path}')
               result = cv2.VideoWriter(default_save_path,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size)
               
            elif save_dir is not None:
               if os.path.exists(save_dir):
                  files_in_dir=os.listdir(save_dir)
                  path=os.path.join(save_dir,f'Results_{len(files_in_dir)+1}.avi')
                  result = cv2.VideoWriter(path,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size)
               else:
                  raise(f'Provided path:{save_dir} does not exist')
         
      while True:

         ret, frame = cap.read()   
         
         if not ret:
            print("Can't receive frame (stream end?) and end of frame reached. Exiting ...")
            break
         
         if use_slicer:
            
            detections=self.InferenceSlicer(model,frame)
         
         detections=self.model.predict(frame)[0]
      
         detections = sv.Detections.from_ultralytics(detections)

         if track:   
                 
            if track and tracker is None:
            
               warnings.warn("The track argument was set to true and the tracker was not provide. The algorithm will use the defualt tracker 'sv.ByteTrack()' when inferencing")
            
               detections = tracker.update_with_detections(detections)
               
               
               
            elif track==True and tracker is not None:
               
               try:
                  detections = tracker.update_with_detections(detections)
               except:
                  message='There is an issue with the provided tracker.You can use sv.ByteTrack()'
               else:
                  detections = tracker.update_with_detections(detections)

            labels = [
                        f"#{tracker_id} {model.names[class_id]}"
                        for class_id, tracker_id
                        in zip(detections.class_id, detections.tracker_id)
                     ]
         
         labels = [
                     f"{class_name} {confidence:.2f}"
                     for class_name, confidence
                     in zip(detections['class_name'], detections.confidence)
                  ]

      
         annotated_image = self.box_annotator.annotate(
                                          scene=frame, detections=detections)
         annotated_image = self.label_annotator.annotate(
                                          scene=annotated_image, detections=detections, labels=labels)

         if save:
            result.write(annotated_image)        
         if display:
            cv2.imshow('Video Stream',annotated_image)
               
         if cv2.waitKey(1) == ord('q'):
            break
   
detector=CarDetect()
detector.detect(video_path=VIDEO_PATH,
                model=model,
                use_slicer=False,
                track=False,
                save_dir=None,
                save=True,
                display=False)