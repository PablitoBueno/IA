import cv2
import numpy as np

# Carregar os nomes das classes (coco.names)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Carregar o modelo YOLOv4
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Função para realizar a detecção de objetos
def detect_objects(image_path):
    # Carregar a imagem
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Redimensionar a imagem para o tamanho adequado para o modelo (416x416)
    img_resized = cv2.resize(img, (416, 416))

    # Preparar a imagem para o modelo
    blob = cv2.dnn.blobFromImage(img_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Obter as saídas da rede
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Processar os resultados da detecção
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Limite de confiança
                # Coordenadas da caixa delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Retângulo da caixa delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para remover caixas redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Exibir resultados na imagem
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Exibir no console os objetos detectados
            print(f"Objeto detectado: {classes[class_ids[i]]}, Confiança: {confidences[i]:.2f}")

    # Mostrar a imagem com os resultados
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemplo de uso
image_path = 'image..jpg'
detect_objects(image_path)
