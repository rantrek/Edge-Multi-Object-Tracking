import cv2
from ultralytics import YOLO
from time import time
import os
os.environ["OMP_NUM_THREADS"] = "4"

def detectObjectsImage(image_path,model, classes):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image, classes = classes, conf = 0.25)  

    results[0].show()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox_coordinates = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
            inference_time = results[0].speed["inference"]

            print(f"Inference Time: {inference_time:.1f} ms")
            print(f"Detected {class_name} ({class_id}) with {confidence:.2f} confidence at {bbox_coordinates}")

    cv2.imshow("YOLOv26 Detector", image)

def detectObjectsVideo(source, model, classes):
    # 2. Setup video capture (using default webcam index 0)
    
        #Initializing variables for FPS
        fps_smooth = 0.0
        alpha = 0.9  # Smoothing factor for rolling average
        latency = 0
    
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
    
        print("Detecting... Press 'q' to exit.")
    
        while True:
            
            #start time
            tr_start = time()

            ret, frame = cap.read()
            if not ret:
                break

            results = model(source=frame, classes = classes, conf = 0.25, stream = True)

              # Display metrics on frame
            cv2.putText(
                    frame,
                    f"Latency: {latency:.1f}ms | FPS: {fps_smooth}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
           
            for result in results:


                 # 4. Generate annotated visual frame
                annotated_frame = result.plot()

                tr_end = time()
                                
                # Calculations for latency and fps
                latency = (tr_end - tr_start) * 1000.0
                fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
                fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * fps)
                fps_smooth = round(fps_smooth)

            print(f"Latency: {latency:.1f} ms")
            print(f"FPS: {fps_smooth} ")

            cv2.imshow("YOLOv26 Detector", annotated_frame)               
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

def edgeObjectDetection(mode,source):

    # Load an official or custom model
    #model = YOLO("models/yolo26n_ncnn_model")  
    #model = YOLO("models/yolo26n_320.onnx")
    model = YOLO("models/yolo26n_openvino_model") 
    
    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 14 - bird
    TARGET_CLASSES = [67,73]

    if mode == 'image':
        detectObjectsImage(source, model, TARGET_CLASSES)

    elif mode == 'video':
        detectObjectsVideo(source, model, TARGET_CLASSES)

    else:
        print("Error! No choice inputted for mode")



    
if __name__ == "__main__":

    path = "assets/IMG_0895.MOV" #image or video path (if not using webcam)
    edgeObjectDetection(mode = 'video', source = 0)