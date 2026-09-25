# Multi-Object Tracking, Detection and Segmentation on Raspberry Pi using YOLO

## Objective

The purpose of this project is to implement YOLOv26n model on an Raspberry Pi to detect, track and segment objects in saved videos and real-time streaming. The model is designed to be lightweight, enabling it to perform efficiently on resource constrained edge devices with limited memory and computational power. 

## Performance

## Example Output

## Hardware 

1) Raspberry Pi 5 microprocessor, 4GB RAM
2) USB webcam (logitech C270 HD webcam)

## Source code

The code was developed in Python 3.13.6. There are three python files: 
1) Multi_ObjectTracking.py - tracks multiple objects across video frames, using YOLOv26 to detect the objects and ByteTrack to track them.
2) Edge_ObjectDetection.py - Detects objects in images and videos (saved files and real-time).
3) Edge_instance_segmentation.py - Detects and segments objects in images and videos (saved files and real-time).

## Techniques

   - Image/video processing
   - Edge deployment
   - Model inference
   - Object detection
   - Multi-object tracking
   - Instance segmentation

## Algorithms 

   - YOLO 
   - ByteTrack  

## Libraries

   - OpenCV
   - Ultralytics

