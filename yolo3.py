import cv2
import numpy as np

# Carregar a rede YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar nomes das classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar imagem
img = cv2.imread("image.jpg")  # Corrigir o nome da imagem, sem espaços!
if img is None:
    print("Erro ao carregar a imagem. Verifique o caminho e o nome do arquivo.")
    exit()

height, width, channels = img.shape

# Pre-processar a imagem
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Inicializar lista para armazenar dados das detecções
detections = []

# Processar saídas
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Apenas exibe objetos com confiança > 50%
            # Coordenadas da caixa delimitadora
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Adiciona à lista de detecções
            detections.append((classes[class_id], confidence, x, y, w, h))

            # Desenhar a caixa delimitadora na imagem
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{classes[class_id]} {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar dados das detecções no terminal
print("Resultados da Detecção:")
for det in detections:
    print(f"Classe: {det[0]}, Confiança: {det[1]:.2f}, Coordenadas: (x: {det[2]}, y: {det[3]}, w: {det[4]}, h: {det[5]})")

# Salvar imagem com resultados
cv2.imwrite("output.jpg", img)
print("\nImagem salva como 'output.jpg'")
