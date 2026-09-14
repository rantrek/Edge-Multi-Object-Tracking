from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

#Export model in following formats
model.export(format = "openvino", quantize = 8, imgsz = 320)
#model.export(format="onnx", opset =12,quantize =8)