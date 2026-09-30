import cv2
from time import time

def detectObjectsImage(image_path,model, classes, task):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image, classes = classes, conf = 0.25)  

    results[0].show()

    for result in results:
        if task == 'detect':
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox_coordinates = box.xyxy[0].tolist() 
                inference_time = results[0].speed["inference"]

                print(f"Inference Time: {inference_time:.1f} ms")
                print(f"Detected {class_name} ({class_id}) with {confidence:.2f} confidence at {bbox_coordinates}")

        elif task == 'segment':
            if result.masks is not None:
                for i, mask in enumerate(result.masks):
                    
                    polygon = mask.xy[0]  
                    class_id = int(result.boxes.cls[i])
                    class_name = model.names[class_id]
                    confidence = float(result.boxes.conf[i])
                    inference_time = results[0].speed["inference"]
                    
                    print(f"Inference Time: {inference_time:.1f} ms")
                    print(f"Detected {class_name} ({confidence:.2f}) with polygon length: {len(polygon)}")
    

    cv2.imshow("YOLOv26 Detector", image)

def detectObjectsVideo(source, model, classes, task):

        #Initializing variables
        fps_smooth = 0.0
        pipeline_fps = 0.0
        alpha = 0.9  #Smoothing factor for rolling average
        latency = 0.0
        total_pipeline_latency = 0.0
        frame_count = 0
        detected_frames = 0
        skip_stride = 3 #Detect every 3rd frame
    
        #Set up video capture
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
    
        while True:
    
            ret, frame = cap.read()
            if not ret:
                break

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
            
            if frame_count % skip_stride != 0:
                continue
    
            tr_start = time()

            if task == 'detect' or task == 'segment':

                results = model(source=frame, classes = classes, conf = 0.25, stream = True)

            elif task == 'track':

                results = model.track(
                            source=frame, 
                            persist=True, 
                            tracker="bytetrack.yaml", 
                            classes=classes,  
                            conf=0.40,               
                            verbose=False,
                            stream = True,
                            task = 'detect',
                
                        )
    
            #if results is not None: 
            for result in results:
        
                annotated_frame = result.plot()

                if task == 'track':
                    #Extract specific tracking data for your filtered items
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                        class_ids = result.boxes.cls.int().cpu().tolist()
                        
                        #Map index integers back to friendly text names
                        for class_id, track_id in zip(class_ids, track_ids):
                            item_name = "bird" if class_id == 14 else "No detection."
                            print(f"Active Track -> {item_name} (ID: {track_id})")
    
                #Extract the inference time
                inference_time = result.speed["inference"]
                print(f"Inference Time: {inference_time:.1f} ms")
    
            tr_end = time()
                                            
            # Calculations for latency and fps 
            latency = (tr_end - tr_start) * 1000.0
            fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
            fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * fps)
            fps_smooth = round(fps_smooth)
       
            total_pipeline_latency += latency
            pipeline_fps+= fps_smooth
            detected_frames+=1
            print(detected_frames)
                    
            #Display pipeline output

            if task == 'track':
                cv2.imshow("YOLOv26 Tracker", annotated_frame)

            else:
                cv2.imshow("YOLOv26 Detector", annotated_frame)
            
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()
    
        #Calculate Final Global Averages
        if detected_frames > 0:
            avg_latency = total_pipeline_latency / detected_frames
            avg_fps = pipeline_fps/detected_frames
    
        print(f"Average Latency: {avg_latency:.1f} ms")
        print(f"Average FPS: {avg_fps} ")

