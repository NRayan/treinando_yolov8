from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

model.export(format="tflite", imgsz=640)
model.export(format="onnx",opset=12)