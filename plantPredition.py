# Instalar as bibliotecas necessárias (caso não estejam instaladas)
!pip install -q kaggle torch torchvision torchaudio albumentations opencv-python matplotlib numpy pandas

import os
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image

# Desabilitar warnings
warnings.filterwarnings('ignore')

# Configurar dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

#########################################
# Montar Google Drive e extrair o dataset
#########################################
from google.colab import drive
drive.mount('/content/drive')

# Caminho do arquivo ZIP no Google Drive
zip_file = "/content/drive/MyDrive/new-plant-diseases-dataset.zip"

# Diretório para extração
extract_path = "/content/dataset"
os.makedirs(extract_path, exist_ok=True)

# Extrair o ZIP (se já não tiver sido extraído)
if not os.path.exists(os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)")):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset extraído com sucesso!")
else:
    print("Dataset já extraído.")

# Definir diretórios de treino e validação
# Ajuste o caminho de acordo com a estrutura do ZIP
train_dir = os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", "train")
valid_dir = os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", "valid")

# Listar as classes presentes no diretório de treino
Diseases_classes = sorted(os.listdir(train_dir))
print(f"Total de classes: {len(Diseases_classes)}")
print(f"Classes encontradas: {Diseases_classes}")

#########################################
# Definir as transformações e carregar os dados
#########################################
# Transformações com Data Augmentation para treino e normalização para validação
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Carregar datasets usando ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)

print(f"Total de imagens no treino: {len(train_dataset)}")
print(f"Total de imagens na validação: {len(valid_dataset)}")

# Criar DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#########################################
# Preparar o modelo (ResNet50 com transferência de aprendizado)
#########################################
# Carregar modelo pré-treinado e ajustar a camada final
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(Diseases_classes))
model = model.to(device)

# Habilitar mixed precision
scaler = torch.cuda.amp.GradScaler()

# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduzir LR a cada 5 épocas

#########################################
# Treinamento com Early Stopping
#########################################
num_epochs = 20
best_acc = 0.0
patience = 5  # Early stopping
wait = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss = running_loss / total
    train_acc = 100. * correct / total
    
    # Validação
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= total
    val_acc = 100. * correct / total
    scheduler.step()
    
    elapsed_time = time.time() - start_time
    print(f"Época {epoch+1}/{num_epochs} | Tempo: {elapsed_time:.2f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Early stopping e salvamento do melhor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping ativado!")
            break

print("Treinamento finalizado. Modelo salvo como 'best_model.pth'")

#########################################
# Funções para carregar o modelo e realizar inferência
#########################################
def load_trained_model():
    # Recriar o modelo e carregar os pesos salvos
    model_trained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model_trained.fc.in_features
    model_trained.fc = nn.Linear(num_ftrs, len(Diseases_classes))
    model_trained.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model_trained.eval()
    return model_trained

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = valid_transform(image).unsqueeze(0)  # Adiciona dimensão de batch
    trained_model = load_trained_model()
    
    with torch.no_grad():
        outputs = trained_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    result = Diseases_classes[predicted.item()]
    print(f"A planta está com: {result}")
#########################################
# Upload de imagem e inferência (Google Colab)
#########################################
from google.colab import files

uploaded = files.upload()  # Selecione a imagem para inferência

# Se mais de uma imagem for enviada, usamos a primeira
for file_name in uploaded.keys():
    print(f"Arquivo carregado: {file_name}")
    predict_image(file_name)
    break  # Remova o break se desejar iterar por todas as imagens
