from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo26n.pt")

#Export model in following formats
model.export(format = "openvino", imgsz = 320, nms= False)
#model.export(format="onnx", opset =12,imgsz = 320, dynamic = False, simplify = False, nms = False)