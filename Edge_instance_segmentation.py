from ultralytics import YOLO
from Vision_tasks import *

import os
os.environ["OMP_NUM_THREADS"] = "4"

def edgeInstanceSegmentation(mode,source):

    # Load an official or custom model
    #model = YOLO("yolo26n_ncnn_model")  
    #model = YOLO("models/yolo26n-seg.pt")
    model = YOLO("models/yolo26n-seg_openvino_model") 
    
    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 14 - bird
    TARGET_CLASSES = [14]

    if mode == 'image':
        detectObjectsImage(source, model, TARGET_CLASSES, task = 'segment')

    elif mode == 'video':
        detectObjectsVideo(source, model, TARGET_CLASSES, task = 'segment')

    else:
        print("Error! No choice inputted for mode")

if __name__ == "__main__":

    path = "assets/IMG_0895.MOV" #image or video path (if not using webcam)
    edgeInstanceSegmentation(mode = 'video', source = path)