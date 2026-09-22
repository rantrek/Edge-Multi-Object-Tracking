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
                inference_time = results[0].speed["inference"]
                
                print(f"Inference Time: {inference_time:.1f} ms")
                print(f"Detected {class_name} ({confidence:.2f}) with polygon length: {len(polygon)}")

    cv2.imshow("YOLOv26 Segmentation", image)

def segmentVideo(source, model, classes):
    # 2. Setup video capture (using default webcam index 0)
    
        #Initializing variables for FPS
        fps_smooth = 0.0
        alpha = 0.9  # Smoothing factor for rolling average
        latency = 0
        frame_count = 0
        skip_stride = 3
        results = None
    
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
    
        while True:
            #start time
            tr_start = time()

            ret, frame = cap.read()
            if not ret:
                break

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

            frame_count+=1

            if frame_count % skip_stride == 0:
    
                results = model(source=frame, classes = classes, conf = 0.25, stream = True)

            if results is not None:
                for result in results:
                    #Generate annotated visual frame
                    annotated_frame = result.plot()
                    print(f"Frame ID: {frame_count}") #print frame ID for detected frames

                    #Extract the inference time
                    inference_time = result.speed["inference"]
                    print(f"Inference Time: {inference_time:.1f} ms")

                    tr_end = time()
                                                    
                    # Calculations for latency and fps
                    latency = (tr_end - tr_start) * 1000.0
                    fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
                    fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * fps)
                    fps_smooth = round(fps_smooth)
        
                    print(f"Latency: {latency:.1f} ms")
                    print(f"FPS: {fps_smooth} ")

            else:
                annotated_frame = frame
           
            cv2.imshow("YOLOv26 Segmentation", annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()

def edgeInstanceSegmentation(mode,source):

    # Load an official or custom model
    #model = YOLO("yolo26n_ncnn_model")  
    #model = YOLO("models/yolo26n-seg.pt")
    model = YOLO("models/yolo26n-seg_openvino_model") 
    
    # COCO Class Mapping: 32 = sports ball, 67 = cell phone, 73 = book, 14 - bird
    TARGET_CLASSES = [14]

    if mode == 'image':
        segmentImage(source, model, TARGET_CLASSES)

    elif mode == 'video':
        segmentVideo(source, model, TARGET_CLASSES)

    else:
        print("Error! No choice inputted for mode")

if __name__ == "__main__":

    path = "assets/IMG_0895.MOV" #image or video path (if not using webcam)
    edgeInstanceSegmentation(mode = 'video', source = path)