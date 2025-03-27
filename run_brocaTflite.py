import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Carregar o modelo TFLite
model_path = "runs/detect/train/weights/best_saved_model/best_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obter detalhes dos tensores de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Função para carregar e preprocessar a imagem
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))  # Ajuste de tamanho para o modelo
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalização
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch (1, 640, 640, 3)
    return img_array, img

# Função para desenhar as caixas de detecção na imagem
def draw_boxes(image, boxes, confidences):
    image_array = np.array(image)  # Convertendo imagem para array numpy
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i]
        confidence = confidences[i]
        label = f"Conf: {confidence:.2f}"

        # Garantir que as coordenadas estejam dentro da imagem
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(640, int(x_max)), min(640, int(y_max))

        # Desenhar a caixa de detecção
        cv2.rectangle(image_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image_array, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(image_array)  # Convertendo de volta para PIL Image

# Função para processar as saídas do modelo
def process_model_output(output_data):
    detections = output_data[0]  # Supondo que as detecções estejam em output_data[0]

    boxes = []
    confidences = []
    
    # Iterar sobre as detecções (cada detecção tem 6 valores)
    for detection in detections:
        x_center, y_center, width, height, confidence = detection[:5]  # Pega as 5 primeiras colunas
        
        # Aplicar sigmoid na confiança
        confidence = sigmoid(confidence)
        
        # Ajustar as coordenadas para a escala de 640x640
        x_center *= 640
        y_center *= 640
        width *= 640
        height *= 640
        
        # Converter para x_min, y_min, x_max, y_max
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        boxes.append([x_min, y_min, x_max, y_max])
        confidences.append(confidence)
    
    return np.array(boxes), np.array(confidences)

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Caminho da imagem
image_path = 'datasets/broca/train/2.jpg'

# Preprocessar a imagem
input_data, img = preprocess_image(image_path)

# Atribuir os dados de entrada ao tensor de entrada
interpreter.set_tensor(input_details[0]['index'], input_data)

# Rodar a inferência
interpreter.invoke()

# Obter os resultados da saída
output_data = interpreter.get_tensor(output_details[0]['index'])

# Filtrar caixas com confiança acima de um limiar
confidence_threshold = 0.5
boxes, confidences = process_model_output(output_data)
filtered_boxes = boxes[confidences > confidence_threshold]
filtered_confidences = confidences[confidences > confidence_threshold]

print(f"Detecções após filtro de confiança: {len(filtered_boxes)}")

# Desenhar as caixas na imagem original
output_img = draw_boxes(img, filtered_boxes, filtered_confidences)

# Caminho para salvar a imagem com as caixas
output_image_path = 'output_with_boxes.jpg'

# Salvar a imagem com as caixas
output_img.save(output_image_path)

# Exibir a imagem com as caixas de detecção
cv2.imshow('Output with Boxes', np.array(output_img))
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Imagem com caixas salva em: {output_image_path}")
