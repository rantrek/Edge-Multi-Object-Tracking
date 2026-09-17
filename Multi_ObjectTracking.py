import cv2
from ultralytics import YOLO
from time import time
import os
os.environ["OMP_NUM_THREADS"] = "4"

def multiObjectTracking():

    # Load an official or custom model
    #model = YOLO("models/yolo26n_ncnn_model")  
    model = YOLO("models/yolo26n_openvino_model") 
    #model = YOLO("models/yolo26n_320.onnx")
    
    # Setup video capture (using default webcam index 0)
    cap = cv2.VideoCapture("assets/IMG_0895.MOV")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps_smooth = 0.0
    alpha = 0.9  # Smoothing factor for rolling average
    latency_ms = 0.0
    frame_count = 0
    skip_frames = 3
    results = None

    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 14 - bird
    TARGET_CLASSES = [14]

    print("Tracking initiated... Press 'q' to exit.")

    while True:

        #start time
        tr_start = time()
        

        ret, frame = cap.read()
        if not ret:
            break

         # Display metrics on frame
        cv2.putText(
                    frame,
                    f"Latency: {latency_ms:.1f}ms | FPS: {fps_smooth}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

        frame_count+=1

        # Apply Multi-Tracking with strict class filtering
        # The 'classes' parameter guarantees the model discards other detected elements instantly
        if frame_count % (skip_frames + 1) == 0:

            results = model.track(
                source=frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                classes=TARGET_CLASSES,  
                conf=0.40,               
                verbose=False,
                stream = True,
                task = 'detect',
    
            )

    
        # Generate annotated visual frame
        if results is not None: 
            for result in results:

                annotated_frame = result.plot()
                
                # Extract specific tracking data for your filtered items
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                    class_ids = result.boxes.cls.int().cpu().tolist()
                    
                    # Map index integers back to friendly text names
                    for class_id, track_id in zip(class_ids, track_ids):
                        item_name = "bird" if class_id == 14 else "No detection."
                        print(f"Active Track -> {item_name} (ID: {track_id})")

                #Extract the inference time
                inference_time = result.speed["inference"]
                print(f"Inference Time: {inference_time:.1f} ms")
        else:
            annotated_frame = frame

        
        tr_end = time()
                            
        # Calculations for latency and fps 
        latency_ms = (tr_end - tr_start) * 1000.0
        fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * fps)
        fps_smooth = round(fps_smooth)
        
        print(f"Latency: {latency_ms:.1f} ms")
        print(f"FPS: {fps_smooth} ")
        

        # 6. Display pipeline output
        cv2.imshow("YOLOv26 Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiObjectTracking()
