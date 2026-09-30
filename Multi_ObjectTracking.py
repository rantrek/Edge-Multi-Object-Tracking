from ultralytics import YOLO
from Vision_tasks import *

import os
os.environ["OMP_NUM_THREADS"] = "4"

def multiObjectTracking():

    #Load an official or custom model
    #model = YOLO("models/yolo26n.pt")  
    model = YOLO("models/yolo26n_openvino_model") 
    #model = YOLO("models/yolo26n_320.onnx")

    path = "assets/IMG_0895.MOV" #image or video path (if not using webcam)
    
    TARGET_CLASSES = [14]
    
    detectObjectsVideo(path, model, TARGET_CLASSES, task = 'track')

if __name__ == "__main__":
    multiObjectTracking()
