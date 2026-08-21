from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

#Export model in following formats
model.export(format = "ncnn", quantize = 16)
#model.export(format="onnx", quantize=8, data="coco8.yaml")