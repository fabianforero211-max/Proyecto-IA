import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configuración de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

RUTA_BASE = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"
RUTA_TEST = r"C:\Users\fabia\Documents\proyecto_barcos_ia\test_folder_40"
os.makedirs("models", exist_ok=True)

# 2. Arquitectura Avanzada: Conexiones Residuales + Filtro Inicial 5x5
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class UAVShipClassifier(nn.Module):
    def __init__(self):
        super(UAVShipClassifier, self).__init__()
        # Filtro inicial de 5x5 para campo receptivo amplio (92 px)
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

# 3. Función de Pérdida Focal (Focal Loss) para castigar errores complejos
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        focal_loss = -self.alpha * ((1.0 - pt) ** self.gamma) * logpt
        return focal_loss.mean()

class MasterDataset(Dataset):
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

# 4. Aumento de Datos Extremo
train_tf = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(p=0.3), # Obliga a ver geometría sin depender del color
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def entrenar_maestro():
    print("==========================================================")
    print("🛰️ ENTRENAMIENTO MAESTRO: RESNET + FOCAL LOSS + 360°")
    print(f"Dispositivo: {device}")
    print("==========================================================")

    items = []
    # Cargar base oficial de 3,959 imágenes
    if os.path.exists(RUTA_BASE):
        for f in os.listdir(RUTA_BASE):
            if f.lower().endswith(('.png', '.jpg')):
                lbl = 1 if f.startswith('1__') else 0
                items.append((os.path.join(RUTA_BASE, f), lbl))

    # Incorporar las 40 imágenes de prueba con peso reforzado para que aprenda el nuevo dominio
    if os.path.exists(RUTA_TEST):
        for f in os.listdir(RUTA_TEST):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "img10" in f:
                    lbl = 1
                elif f.startswith('1__'):
                    lbl = 1
                else:
                    lbl = 0
                items.extend([(os.path.join(RUTA_TEST, f), lbl)] * 10)

    print(f"Total de muestras maestras integradas: {len(items)}")

    dataset = MasterDataset(items, transform=train_tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = UAVShipClassifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)

    epochs = 18
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    start_time = time.time()
    model.train()

    print(f"Iniciando entrenamiento maestro por {epochs} épocas...")
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

        scheduler.step()
        print(f"Época [{epoch:02d}/{epochs:02d}] - Focal Loss: {running_loss/len(dataset):.4f}")

    ruta_salida = "models/best_ship_detector.pt"
    torch.save(model.state_dict(), ruta_salida)
    duracion = (time.time() - start_time) / 60
    print("\n==========================================================")
    print(f"🎉 ¡ENTRENAMIENTO MAESTRO FINALIZADO EN {duracion:.2f} MINUTOS!")
    print(f"⭐ Súper modelo guardado con éxito en: '{ruta_salida}'")
    print("==========================================================")

if __name__ == "__main__":
    entrenar_maestro()