import onnxruntime as ort
import numpy as np
import cv2
import torch
from torchvision.ops import nms

# Carregar o modelo ONNX
model_path = 'best.onnx'  # Caminho para o arquivo .onnx
session = ort.InferenceSession(model_path)

# =======================
# 🔹 Função de Letterbox
# =======================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # Altura, largura
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Escala
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)

# =======================
# 🔹 Função de Pré-processamento
# =======================
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Lê a imagem (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB

    # Obtém as dimensões originais
    h, w, _ = img.shape
    new_size = 640
    ratio = min(new_size / w, new_size / h)  # Mantém proporção
    new_w, new_h = int(w * ratio), int(h * ratio)

    # Redimensiona mantendo proporção
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Cria um fundo preto e coloca a imagem redimensionada no centro (letterbox)
    img_padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)  # Fundo cinza (114, 114, 114)
    dw, dh = (new_size - new_w) // 2, (new_size - new_h) // 2  # Padding

    img_padded[dh:dh + new_h, dw:dw + new_w, :] = img_resized  # Insere imagem no fundo

    # Normalização
    img_padded = img_padded.astype(np.float32) / 255.0  # [0,1]
    img_padded = np.transpose(img_padded, (2, 0, 1))  # (H, W, C) → (C, H, W)
    img_padded = np.expand_dims(img_padded, axis=0)  # Adiciona batch

    return img_padded, ratio, dw, dh

# =======================
# 🔹 Carregar Imagem e Processar
# =======================
image_path = 'datasets/broca/train/2.jpg'
input_data, ratio, dw, dh = preprocess_image(image_path)

# Executar a inferência
inputs = {session.get_inputs()[0].name: input_data}
outputs = session.run(None, inputs)

# Extrair dados de saída (5, 8400)
output_array = outputs[0].squeeze()  # Remove dimensões extras → (5, 8400)

# Aplicar sigmoid na 5ª coluna (confiança)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

detections = output_array.T  # Transpor de (5, 8400) para (8400, 5)
confidences = sigmoid(detections[:, 4])  # Aplicar sigmoid nas confidências
boxes = detections[:, :4]  # [x_center, y_center, width, height]

# Converter para [x_min, y_min, x_max, y_max]
boxes_converted = np.zeros_like(boxes)
boxes_converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
boxes_converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
boxes_converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x_max
boxes_converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y_max

# =======================
# 🔹 Ajustar caixas para a escala original
# =======================
boxes_converted[:, [0, 2]] -= dw  # Ajustar x_min e x_max
boxes_converted[:, [1, 3]] -= dh  # Ajustar y_min e y_max
boxes_converted[:, :4] /= ratio  # Reescalar para imagem original

# Filtrar caixas com confiança acima de um limiar
confidence_threshold = 0.55
indices = confidences > confidence_threshold
filtered_boxes = boxes_converted[indices]
filtered_confidences = confidences[indices]

print(f"Detecções após filtro de confiança: {len(filtered_boxes)}")

# =======================
# 🔹 Aplicar Non-Maximum Suppression (NMS)
# =======================
if len(filtered_boxes) > 0:
    boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(filtered_confidences, dtype=torch.float32)
    iou_threshold = 0.5
    keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    final_boxes = filtered_boxes[keep.numpy()]
    final_confidences = filtered_confidences[keep.numpy()]
    print(f"Detecções após NMS: {len(final_boxes)}")
else:
    final_boxes = filtered_boxes
    final_confidences = filtered_confidences
    print("Nenhuma detecção após filtro de confiança, NMS não aplicado.")

# =======================
# 🔹 Função para Desenhar Caixas
# =======================
def draw_boxes(image_path, boxes, confidences):
    img = cv2.imread(image_path)  # Recarregar imagem original
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = map(int, boxes[i])  # Converter para inteiros
        confidence = confidences[i]
        label = f"Conf: {confidence:.2f}"

        # Desenhar bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

# Desenhar as caixas na imagem
img_with_boxes = draw_boxes(image_path, final_boxes, final_confidences)

# Exibir a imagem com as caixas
cv2.imshow('Predictions', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =======================
# 🔹 Informações para Debug
# =======================
print(f"Confiança máxima após sigmoid: {confidences.max()}")
print(f"Confiança mínima após sigmoid: {confidences.min()}")
