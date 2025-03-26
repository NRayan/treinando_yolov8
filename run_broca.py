from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# model.export(format="onnx")
model.export(format="tflite")

# Escolha uma imagem para testar
img_path = "datasets/broca/train/1.jpg"
# img_path = "imagem_testes/1.jpg"

# Fazer a previsão
results = model(img_path)  

# Exibir a imagem com as detecções
for r in results:
    r.show()  # Abre a imagem em uma janela