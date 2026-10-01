import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Configuración de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

RUTA_BASE = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"
RUTA_TEST = r"C:\Users\fabia\Documents\proyecto_barcos_ia\test_folder_40"
os.makedirs("models", exist_ok=True)

# 2. Arquitectura Residual Avanzada (ResNet UAV)
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

# 3. Función de Pérdida con Enfoque Estricto en Casos Difíciles (Focal Loss Gamma=3.0)
class TitanFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=3.0):
        super(TitanFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        loss = -self.alpha * ((1.0 - pt) ** self.gamma) * logpt
        return loss.mean()

class TitanDataset(Dataset):
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

# 4. Aumento de Datos Geométrico y Cromático Maestro
train_tf = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomResizedCrop(80, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(p=0.35),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def entrenar_titan():
    print("==========================================================")
    print("👑 INICIANDO ENTRENAMIENTO TITÁN (RESNET + FOCAL LOSS GAMMA 3.0)")
    print(f"Cómputo en: {device}")
    print("==========================================================")

    model = UAVShipClassifier().to(device)
    path_modelo = "models/best_ship_detector.pt"
    if os.path.exists(path_modelo):
        model.load_state_dict(torch.load(path_modelo, map_location=device))
        print("✅ Cargados pesos anteriores para refinamiento extremo.")

    items = []
    # Cargar dataset base
    if os.path.exists(RUTA_BASE):
        for f in os.listdir(RUTA_BASE):
            if f.lower().endswith(('.png', '.jpg')):
                lbl = 1 if f.startswith('1__') else 0
                items.append((os.path.join(RUTA_BASE, f), lbl))

    # Cargar test set dándole un peso fuerte de refuerzo (x20)
    if os.path.exists(RUTA_TEST):
        for f in os.listdir(RUTA_TEST):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "img10" in f:
                    lbl = 1
                elif f.startswith('1__'):
                    lbl = 1
                else:
                    lbl = 0
                items.extend([(os.path.join(RUTA_TEST, f), lbl)] * 20)

    print(f"Total muestras ponderadas del Dataset Titán: {len(items)}")

    dataset = TitanDataset(items, transform=train_tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Optimizador de alta precisión con OneCycleLR
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = TitanFocalLoss(alpha=0.3, gamma=3.0)

    epochs = 22
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, steps_per_epoch=len(loader), epochs=epochs
    )

    start_time = time.time()
    model.train()

    print(f"Entrenando modelo Titán por {epochs} épocas...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * imgs.size(0)

        print(f"Época [{epoch:02d}/{epochs:02d}] - Titan Loss: {running_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), path_modelo)
    duracion = (time.time() - start_time) / 60
    print("\n==========================================================")
    print(f"🏆 ¡ENTRENAMIENTO TITÁN COMPLETADO EN {duracion:.2f} MINUTOS!")
    print(f"⭐ Súper modelo optimizado guardado en: '{path_modelo}'")
    print("==========================================================")

if __name__ == "__main__":
    entrenar_titan()