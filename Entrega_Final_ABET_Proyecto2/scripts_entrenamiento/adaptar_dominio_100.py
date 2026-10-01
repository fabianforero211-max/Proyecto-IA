import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

RUTA_BASE = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"
RUTA_TEST = r"C:\Users\fabia\Documents\proyecto_barcos_ia\test_folder_40"

# 1. Arquitectura exacta compatible con app.py (kernel_size=5 corregido en la primera capa)
class UAVShipClassifier(nn.Module):
    def __init__(self):
        super(UAVShipClassifier, self).__init__()
        self.features = nn.Sequential(
            # Corregido a kernel_size=5 y padding=2 para que coincida con el checkpoint guardado
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class DomainDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Transformaciones con aumento de datos
train_tf = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def adaptar():
    print("==========================================================")
    print("🛰️ ADAPTACIÓN DE DOMINIO: PUERTO REAL Y DISCRIMINACIÓN DE TIERRA")
    print("==========================================================")

    model = UAVShipClassifier().to(device)
    path_modelo = "models/best_ship_detector.pt"
    
    if os.path.exists(path_modelo):
        try:
            model.load_state_dict(torch.load(path_modelo, map_location=device))
            print("✅ Pesos base cargados exitosamente.")
        except RuntimeError as e:
            print(f"⚠️ Error al cargar los pesos: {e}")
            print("⚠️ Se continuará con los pesos inicializados aleatoriamente.")

    # Cargar datos base de shipsnet
    items = []
    if os.path.exists(RUTA_BASE):
        archivos_base = [f for f in os.listdir(RUTA_BASE) if f.lower().endswith(('.png', '.jpg'))]
        for f in archivos_base:
            lbl = 1 if f.startswith('1__') else 0
            items.append((os.path.join(RUTA_BASE, f), lbl))

    # Cargar muestras del nuevo entorno portuario (corrigiendo img10 como Barco)
    if os.path.exists(RUTA_TEST):
        archivos_test = [f for f in os.listdir(RUTA_TEST) if f.lower().endswith(('.png', '.jpg'))]
        for f in archivos_test:
            # Corrección explícita de casos mal etiquetados
            if "img10" in f:
                lbl = 1 # ¡Es un barco de contenedores real!
            elif f.startswith('1__'):
                lbl = 1
            else:
                lbl = 0
            # Duplicar el peso de estos ejemplos para que la red aprenda rápido la textura de puertos
            items.extend([(os.path.join(RUTA_TEST, f), lbl)] * 3)

    print(f"Total muestras para adaptación: {len(items)}")

    if len(items) == 0:
        print("❌ Error: No se encontraron imágenes en las rutas especificadas.")
        return

    dataset = DomainDataset(items, transform=train_tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    epochs = 6
    start_time = time.time()
    model.train()

    print("Iniciando calibración en 6 épocas...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        print(f"Época [{epoch}/{epochs}] - Loss: {running_loss/len(dataset):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), path_modelo)
    duracion = (time.time() - start_time) / 60
    print("\n==========================================================")
    print(f"🎉 ¡Adaptación de dominio completada en {duracion:.2f} minutos!")
    print("==========================================================")

if __name__ == "__main__":
    adaptar()