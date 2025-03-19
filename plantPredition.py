{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install -q kaggle torch torchvision torchaudio albumentations opencv-python matplotlib numpy pandas"
      ],
      "metadata": {
        "id": "HygfVj6GqKfT"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "0CE4VC7tiHSv",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c43dd004-fa14-4b6f-98a1-c98d909d445e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Usando dispositivo: cuda\n"
          ]
        }
      ],
      "source": [
        "# Desabilitar warnings e importar bibliotecas necessárias\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "from PIL import Image\n",
        "\n",
        "import torchvision\n",
        "import torchvision.transforms as transforms\n",
        "from torchvision.datasets import ImageFolder\n",
        "\n",
        "# Se estiver usando GPU, exibe a disponibilidade\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Usando dispositivo: {device}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 🔽 Montar Google Drive e configurar dataset\n",
        "from google.colab import drive\n",
        "import os\n",
        "import zipfile\n",
        "import torch\n",
        "from torchvision import datasets, transforms\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "# 🔹 Montar o Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# 🔹 Caminho do ZIP no Google Drive\n",
        "zip_file = \"/content/drive/MyDrive/new-plant-diseases-dataset.zip\"\n",
        "\n",
        "# 🔹 Criar diretório de extração\n",
        "extract_path = \"/content/dataset\"\n",
        "os.makedirs(extract_path, exist_ok=True)\n",
        "\n",
        "# 🔹 Extrair o ZIP\n",
        "with zipfile.ZipFile(zip_file, 'r') as zip_ref:\n",
        "    zip_ref.extractall(extract_path)\n",
        "\n",
        "# 🔹 Definir caminhos do dataset\n",
        "train_dir = os.path.join(extract_path, \"New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train\")\n",
        "valid_dir = os.path.join(extract_path, \"New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid\")\n",
        "\n",
        "# 🔹 Listar classes do dataset\n",
        "Diseases_classes = os.listdir(train_dir)\n",
        "print(f\"Total de classes: {len(Diseases_classes)}\")\n",
        "print(f\"Classes encontradas: {Diseases_classes}\")\n",
        "\n",
        "# 🔹 Transformações das imagens\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# 🔹 Criar datasets\n",
        "train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)\n",
        "valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)\n",
        "\n",
        "# 🔹 Criar DataLoaders\n",
        "batch_size = 32\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "# 🔹 Exibir informações\n",
        "print(f\"Total de imagens no treino: {len(train_dataset)}\")\n",
        "print(f\"Total de imagens na validação: {len(valid_dataset)}\")\n"
      ],
      "metadata": {
        "id": "hftgBJiIgjM0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "288a867e-bc58-4200-8d57-355f798b0c15"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "Total de classes: 38\n",
            "Classes encontradas: ['Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Corn_(maize)___healthy', 'Peach___healthy', 'Apple___Cedar_apple_rust', 'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Soybean___healthy', 'Peach___Bacterial_spot', 'Corn_(maize)___Common_rust_', 'Raspberry___healthy', 'Cherry_(including_sour)___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Cherry_(including_sour)___Powdery_mildew', 'Apple___Black_rot', 'Apple___Apple_scab', 'Tomato___Bacterial_spot', 'Pepper,_bell___healthy', 'Strawberry___healthy', 'Grape___Esca_(Black_Measles)', 'Tomato___Early_blight', 'Orange___Haunglongbing_(Citrus_greening)', 'Grape___Black_rot', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Potato___Late_blight', 'Blueberry___healthy', 'Pepper,_bell___Bacterial_spot', 'Grape___healthy', 'Apple___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Septoria_leaf_spot', 'Tomato___Leaf_Mold', 'Tomato___Target_Spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Squash___Powdery_mildew', 'Potato___Early_blight', 'Tomato___healthy']\n",
            "Total de imagens no treino: 70295\n",
            "Total de imagens na validação: 17572\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Definir transformações para as imagens (resize, conversão para tensor e normalização)\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# Carregar os datasets utilizando ImageFolder\n",
        "train_data = ImageFolder(train_dir, transform=transform)\n",
        "valid_data = ImageFolder(valid_dir, transform=transform)\n",
        "\n",
        "# Criar DataLoaders para facilitar o treinamento em lotes\n",
        "train_loader = DataLoader(train_data, batch_size=32, shuffle=True)\n",
        "valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)\n",
        "\n",
        "print(f\"Total de imagens de treino: {len(train_data)}\")\n",
        "print(f\"Total de imagens de validação: {len(valid_data)}\")\n"
      ],
      "metadata": {
        "id": "XSFtoo--glLH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b145ba72-bc13-4991-a69e-a2a9e81a9d3f"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total de imagens de treino: 70295\n",
            "Total de imagens de validação: 17572\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Utilizando o modelo ResNet18 pré-treinado e ajustando a camada final para o número de classes do dataset\n",
        "model = torchvision.models.resnet18(pretrained=True)\n",
        "model.fc = nn.Linear(model.fc.in_features, len(Diseases_classes))\n",
        "model = model.to(device)\n",
        "\n",
        "# Definir a função de perda e o otimizador\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
        "\n",
        "print(model)\n"
      ],
      "metadata": {
        "id": "TmvWGjZygodU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "edef6ad3-8f22-47b3-9665-e57481cf1f47"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ResNet(\n",
            "  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)\n",
            "  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "  (relu): ReLU(inplace=True)\n",
            "  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)\n",
            "  (layer1): Sequential(\n",
            "    (0): BasicBlock(\n",
            "      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "    )\n",
            "    (1): BasicBlock(\n",
            "      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "    )\n",
            "  )\n",
            "  (layer2): Sequential(\n",
            "    (0): BasicBlock(\n",
            "      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (downsample): Sequential(\n",
            "        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
            "        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      )\n",
            "    )\n",
            "    (1): BasicBlock(\n",
            "      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "    )\n",
            "  )\n",
            "  (layer3): Sequential(\n",
            "    (0): BasicBlock(\n",
            "      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (downsample): Sequential(\n",
            "        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
            "        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      )\n",
            "    )\n",
            "    (1): BasicBlock(\n",
            "      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "    )\n",
            "  )\n",
            "  (layer4): Sequential(\n",
            "    (0): BasicBlock(\n",
            "      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (downsample): Sequential(\n",
            "        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
            "        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      )\n",
            "    )\n",
            "    (1): BasicBlock(\n",
            "      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "      (relu): ReLU(inplace=True)\n",
            "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
            "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "    )\n",
            "  )\n",
            "  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))\n",
            "  (fc): Linear(in_features=512, out_features=38, bias=True)\n",
            ")\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs = 10\n",
        "\n",
        "def train_model():\n",
        "    model.train()\n",
        "    for epoch in range(num_epochs):\n",
        "        running_loss = 0.0\n",
        "        for images, labels in train_loader:\n",
        "            # Mover dados para o dispositivo (CPU ou GPU)\n",
        "            images, labels = images.to(device), labels.to(device)\n",
        "\n",
        "            optimizer.zero_grad()       # Zerar gradientes\n",
        "            outputs = model(images)     # Forward pass\n",
        "            loss = criterion(outputs, labels)  # Calcular a perda\n",
        "            loss.backward()             # Backpropagation\n",
        "            optimizer.step()            # Atualizar os pesos\n",
        "\n",
        "            running_loss += loss.item()\n",
        "\n",
        "        epoch_loss = running_loss / len(train_loader)\n",
        "        print(f\"Época {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}\")\n",
        "\n",
        "# Executar o treinamento\n",
        "train_model()\n"
      ],
      "metadata": {
        "id": "gtFIu5TqgtTz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5900edb8-e9f0-4b03-8f5d-b6e5ed90fcf9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Época 1/10 - Loss: 0.2474\n",
            "Época 2/10 - Loss: 0.1089\n",
            "Época 3/10 - Loss: 0.0788\n",
            "Época 4/10 - Loss: 0.0626\n",
            "Época 5/10 - Loss: 0.0543\n",
            "Época 6/10 - Loss: 0.0443\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "MODEL_PATH = \"plant_disease_model.pth\"\n",
        "\n",
        "def save_model():\n",
        "    torch.save(model.state_dict(), MODEL_PATH)\n",
        "    print(\"Modelo salvo com sucesso!\")\n",
        "\n",
        "save_model()\n"
      ],
      "metadata": {
        "id": "QVfAIWmXgufy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Função para carregar o modelo salvo para inferência\n",
        "def load_trained_model():\n",
        "    # Recria o modelo e carrega os pesos\n",
        "    model_trained = torchvision.models.resnet18(pretrained=False)\n",
        "    model_trained.fc = nn.Linear(model_trained.fc.in_features, len(Diseases_classes))\n",
        "    model_trained.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))\n",
        "    model_trained.eval()\n",
        "    return model_trained\n",
        "\n",
        "# Função para fazer a previsão em uma nova imagem\n",
        "def predict_image(image_path):\n",
        "    image = Image.open(image_path).convert('RGB')\n",
        "    image = transform(image).unsqueeze(0)  # Adiciona dimensão do batch\n",
        "    trained_model = load_trained_model()\n",
        "\n",
        "    with torch.no_grad():\n",
        "        outputs = trained_model(image)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "\n",
        "    result = Diseases_classes[predicted.item()]\n",
        "    print(f\"A planta está com: {result}\")\n",
        "\n",
        "# Código para upload da imagem no Google Colab\n",
        "from google.colab import files\n",
        "\n",
        "uploaded = files.upload()  # Escolha a imagem para fazer a inferência\n",
        "\n",
        "# Se mais de uma imagem for enviada, usaremos a primeira\n",
        "for file_name in uploaded.keys():\n",
        "    print(f\"Arquivo carregado: {file_name}\")\n",
        "    predict_image(file_name)\n",
        "    break  # Remove o break se quiser iterar em todas as imagens\n"
      ],
      "metadata": {
        "id": "BejXWmgVgzOb"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}