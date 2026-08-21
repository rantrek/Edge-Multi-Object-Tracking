import cv2
from ultralytics import YOLO
from time import time

def segmentImage(image_path,model, classes):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image, classes = classes)  

    results[0].show()

    for result in results:
        # Access raw segmentation mask structures if you need to manipulate them
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                # Get mask coordinates as normalized or pixel coordinates
                polygon = mask.xy[0]  # Pixel coordinates (x, y) outlining the object
                
                # Get class ID and confidence score for this specific instance
                class_id = int(result.boxes.cls[i])
                class_name = model.names[class_id]
                confidence = float(result.boxes.conf[i])
                
                print(f"Detected {class_name} ({confidence:.2f}) with polygon length: {len(polygon)}")
    cv2.imshow("YOLOv26 Segmentation", image)

def segmentVideo(source, model, classes):
    # 2. Setup video capture (using default webcam index 0)
    
        #Initializing variables for FPS
        fps_smooth = 0.0
        alpha = 0.9  # Smoothing factor for rolling average
    
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    
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
            for result in results:
                annotated_frame = result.plot()
           

            cv2.imshow("YOLOv26 Detector", annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

def edgeInstanceSegmentation(mode,source):

    # Load an official or custom model
    #model = YOLO("yolo26n_ncnn_model")  
    model = YOLO("yolo26n-seg.pt")
    
    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 15 - bird
    TARGET_CLASSES = [32]

    if mode == 'image':
        segmentImage(source, model, TARGET_CLASSES)

    elif mode == 'video':
        segmentVideo(source, model, TARGET_CLASSES)

    else:
        print("Error! No choice inputted for mode")



    
if __name__ == "__main__":

    path = "assets/ball_closeup.mp4" #image or video path (if not using webcam)
    edgeInstanceSegmentation(mode = 'video', source = path)