import onnx
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# Carregar o modelo ONNX
model_path = 'best.onnx'  # Caminho para o arquivo .onnx
session = ort.InferenceSession(model_path)

# Função para carregar e preprocessar a imagem
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))  # Tamanho esperado pelo modelo
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalização
    img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) → (C, H, W)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch
    return img_array, img


# Caminho para a imagem de teste
image_path = 'datasets/broca/train/24.jpg'

# Preprocessar a imagem
input_data, img = preprocess_image(image_path)

# Executar a inferência
inputs = {session.get_inputs()[0].name: input_data}
outputs = session.run(None, inputs)

# Extrair dados de saída (5, 8400)
output_array = outputs[0].squeeze()  # Remove dimensões extras → (5, 8400)

# Salvar o array completo de saída em um arquivo .txt
output_txt_path = "raw_output_array.txt"
with open(output_txt_path, "w") as file:
    # Salvar o array de saída completo
    for i in range(output_array.shape[0]):
        file.write(f"Row {i+1}: {output_array[i].tolist()}\n")

print(f"Dados brutos do output_array salvos em: {output_txt_path}")

# Separar as caixas e as confidências
boxes = output_array[:4, :].transpose()  # (4, 8400) → (8400, 4)
confidences = output_array[4, :]  # (8400,)

# Aplicar Sigmoid na confiança
# confidences = 1 / (1 + np.exp(-confidences))

# Filtrar caixas com confiança acima de um limiar
confidence_threshold = 0.5
indices = np.where(confidences > confidence_threshold)[0]

# Aplicar filtro
boxes = boxes[indices]
confidences = confidences[indices]

print(f"Detecções: {len(boxes)}")

# Função para desenhar caixas na imagem
def draw_boxes(img, boxes, confidences):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        confidence = confidences[i]
        label = f"Conf: {confidence:.2f}"
        
        # Desenhar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

# Desenhar as caixas na imagem
img_with_boxes = draw_boxes(np.array(img), boxes, confidences)

# Exibir a imagem com as caixas
cv2.imshow('Predictions', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()



