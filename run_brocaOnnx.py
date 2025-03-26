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

# Salvar o array completo de saída em um arquivo .txt (50 primeiros valores por linha)
output_txt_path = "raw_output_array50.txt"
with open(output_txt_path, "w") as file:
    for i in range(output_array.shape[0]):  # 5 linhas
        first_50 = output_array[i][:50].tolist()
        file.write(f"Row {i+1}: {first_50}\n")

print(f"Dados brutos (50 primeiros valores) salvos em: {output_txt_path}")

# Separar as caixas e aplicar sigmoid nas confidências
detections = output_array.T  # Transpor de (5, 8400) para (8400, 5)

# Aplicar sigmoid na 5ª coluna (confiança)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

confidences = sigmoid(detections[:, 4])  # Aplicar sigmoid aos logits
boxes = detections[:, :4]  # [x_center, y_center, width, height]

# Converter para [x_min, y_min, x_max, y_max]
boxes_converted = np.zeros_like(boxes)
boxes_converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
boxes_converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
boxes_converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x_max
boxes_converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y_max

# Filtrar caixas com confiança acima de um limiar
confidence_threshold = 0.55
indices = np.where(confidences > confidence_threshold)[0]

# Aplicar filtro
filtered_boxes = boxes_converted[indices]
filtered_confidences = confidences[indices]

print(f"Detecções após filtro de confiança: {len(filtered_boxes)}")

# Função para desenhar caixas na imagem
def draw_boxes(img, boxes, confidences):
    img_array = np.array(img)  # Converter PIL Image para numpy array
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = map(int, boxes[i])  # Converter para inteiros
        confidence = confidences[i]
        label = f"Conf: {confidence:.2f}"
        
        # Garantir que as coordenadas estejam dentro da imagem 640x640
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(640, x_max)
        y_max = min(640, y_max)
        
        # Desenhar bounding box
        cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_array

# Desenhar as caixas na imagem
img_with_boxes = draw_boxes(img, filtered_boxes, filtered_confidences)

# Exibir a imagem com as caixas
cv2.imshow('Predictions', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Informações adicionais para depuração
print(f"Confiança máxima após sigmoid: {confidences.max()}")
print(f"Confiança mínima após sigmoid: {confidences.min()}")