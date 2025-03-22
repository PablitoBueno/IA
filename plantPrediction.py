# Install required libraries (if not installed)
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

# Disable warnings
warnings.filterwarnings('ignore')

# Configure device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#########################################
# Mount Google Drive and extract dataset
#########################################
from google.colab import drive
drive.mount('/content/drive')

# Path to the ZIP file on Google Drive
zip_file = "/content/drive/MyDrive/new-plant-diseases-dataset.zip"

# Extraction directory
extract_path = "/content/dataset"
os.makedirs(extract_path, exist_ok=True)

# Extract ZIP (if not already extracted)
if not os.path.exists(os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)")):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset successfully extracted!")
else:
    print("Dataset already extracted.")

# Define training and validation directories
# Adjust path according to ZIP structure
train_dir = os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", "train")
valid_dir = os.path.join(extract_path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)", "valid")

# List classes present in the training directory
Diseases_classes = sorted(os.listdir(train_dir))
print(f"Total classes: {len(Diseases_classes)}")
print(f"Classes found: {Diseases_classes}")

#########################################
# Define transformations and load data
#########################################
# Data Augmentation transformations for training and normalization for validation
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

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)

print(f"Total training images: {len(train_dataset)}")
print(f"Total validation images: {len(valid_dataset)}")

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#########################################
# Prepare model (ResNet50 with transfer learning)
#########################################
# Load pre-trained model and adjust final layer
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(Diseases_classes))
model = model.to(device)

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

#########################################
# Training with Early Stopping
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
    
    # Validation
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
    print(f"Epoch {epoch+1}/{num_epochs} | Time: {elapsed_time:.2f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Early stopping and saving best model
    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping activated!")
            break

print("Training completed. Model saved as 'best_model.pth'")

#########################################
# Functions to load trained model and perform inference
#########################################
def load_trained_model():
    model_trained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model_trained.fc.in_features
    model_trained.fc = nn.Linear(num_ftrs, len(Diseases_classes))
    model_trained.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model_trained.eval()
    return model_trained

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = valid_transform(image).unsqueeze(0)
    trained_model = load_trained_model()
    
    with torch.no_grad():
        outputs = trained_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    result = Diseases_classes[predicted.item()]
    print(f"The plant is diagnosed with: {result}")
