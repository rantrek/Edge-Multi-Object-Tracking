import cv2
from ultralytics import YOLO
from time import time

def multiObjectTracking():

    # Load an official or custom model
    #model = YOLO("yolo26n_ncnn_model")  
    model = YOLO("models/yolo26n.pt")
    #model = YOLO("yolo26n-seg.pt")

    # 2. Setup video capture (using default webcam index 0)
    cap = cv2.VideoCapture("assets/IMG_0895.MOV")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps_smooth = 0.0
    alpha = 0.9  # Smoothing factor for rolling average

    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 15 - bird
    TARGET_CLASSES = [14]

    print("Tracking initiated for: Cell Phones and Books. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Apply Multi-Tracking with strict class filtering
        # The 'classes' parameter guarantees the model discards other detected elements instantly

        #start time
        tr_start = time()
        results = model.track(
            source=frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=TARGET_CLASSES,  
            conf=0.30,               
            verbose=False,
            task = 'detect',
        )

        tr_end = time()

         # Calculations for latency and fps
        latency_ms = (tr_end - tr_start) * 1000.0
        current_fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * current_fps)

        # Display metrics on frame
        cv2.putText(
            frame,
            f"Latency: {latency_ms:.1f}ms | FPS: {fps_smooth:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        # 4. Generate annotated visual frame
        annotated_frame = results[0].plot()

        # 5. Extract specific tracking data for your filtered items
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            # Map index integers back to friendly text names
            for class_id, track_id in zip(class_ids, track_ids):
                item_name = "bird" if class_id == 14 else "No detection."
                print(f"Active Track -> {item_name} (ID: {track_id})")

        
        # 6. Display pipeline output
        cv2.imshow("YOLOv26 Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiObjectTracking()
