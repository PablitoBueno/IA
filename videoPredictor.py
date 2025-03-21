# Instala as bibliotecas necessárias (se ainda não estiverem instaladas)
!pip install ultralytics opencv-python-headless matplotlib

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import ceil, sqrt
from ultralytics import YOLO

# Carrega o modelo YOLOv8x (um dos mais precisos da família)
model = YOLO('yolov8x.pt')

# Define o dispositivo: usa GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Defina o caminho do vídeo (faça upload do arquivo no Colab ou informe o caminho correto)
video_path = 'videoteste.mp4'  # substitua pelo nome ou caminho do seu vídeo
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir o vídeo!")
except Exception as e:
    print(f"Erro: {e}")
    exit()

print("Processando vídeo...")

# Define parâmetros de processamento
frame_skip = 5          # Processa 1 a cada 5 frames para otimizar o tempo
conf_threshold = 0.3    # Limiar mínimo de confiança para considerar uma detecção

frame_count = 0
detection_crops = []    # Lista para armazenar os recortes, rótulo e confiança de cada detecção

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processa apenas 1 frame a cada 'frame_skip'
    if frame_count % frame_skip == 0:
        try:
            results = model(frame)
        except Exception as e:
            print(f"Erro durante a detecção no frame {frame_count}: {e}")
            continue

        # Itera sobre as detecções encontradas
        for box in results[0].boxes:
            # Extrai as coordenadas da caixa
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            # Garante que as coordenadas estejam dentro dos limites do frame
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            # Verifica se a caixa possui dimensões válidas
            if x2 <= x1 or y2 <= y1:
                continue

            # Extrai o rótulo e a confiança
            cls_id = int(box.cls)
            label = results[0].names[cls_id]
            conf = float(box.conf)

            # Aplica o filtro de confiança
            if conf < conf_threshold:
                continue

            # Recorta a região detectada e adiciona à lista
            crop = frame[y1:y2, x1:x2]
            detection_crops.append((crop, label, conf))
    frame_count += 1

cap.release()

# Exibe os recortes organizados em uma grade
if detection_crops:
    print("Exibindo os recortes dos objetos detectados:")
    num_detections = len(detection_crops)
    cols = int(sqrt(num_detections))
    rows = ceil(num_detections / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)  # "Achata" o array para facilitar a iteração

    for i, (crop, label, conf) in enumerate(detection_crops):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axes[i].imshow(crop_rgb)
        axes[i].set_title(f"{label}: {conf:.2f}")
        axes[i].axis('off')

    # Desativa os eixos dos subplots não utilizados, se houver
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Nenhum objeto detectado.")
