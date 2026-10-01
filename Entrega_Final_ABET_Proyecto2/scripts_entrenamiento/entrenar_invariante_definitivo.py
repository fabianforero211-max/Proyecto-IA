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
np.random.seed(42)

RUTA_IMAGENES = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"
os.makedirs("models", exist_ok=True)

# Arquitectura idéntica para máxima compatibilidad con app.py
class UAVShipClassifier(nn.Module):
    def __init__(self):
        super(UAVShipClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 0
            nn.BatchNorm2d(32),                           # 1
            nn.ReLU(),                                    # 2
            nn.MaxPool2d(2, 2),                           # 3

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4
            nn.BatchNorm2d(64),                           # 5
            nn.ReLU(),                                    # 6
            nn.MaxPool2d(2, 2),                           # 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8
            nn.BatchNorm2d(128),                          # 9
            nn.ReLU(),                                    # 10
            nn.MaxPool2d(2, 2),                           # 11

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 12
            nn.BatchNorm2d(256),                          # 13
            nn.ReLU(),                                    # 14
            nn.AdaptiveAvgPool2d((1, 1))                  # 15
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

class ShipsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.labels = [1 if f.startswith('1__') else 0 for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# AUMENTO CROMÁTICO Y GEOMÉTRICO EXTREMO (INVARIANZA TOTAL DE AGUA Y ÁNGULO)
train_tf = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.RandomRotation(degrees=(0, 360)),            # Todos los ángulos
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(p=0.40),                     # 40% en escala de grises para forzar aprendizaje de FORMA
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.25), # Agua negra, café y turquesa
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def entrenar_invariante():
    print("==========================================================")
    print("🛰️ ENTRENAMIENTO CON INVARIANZA TOTAL DE SENSOR Y AGUA")
    print(f"Dispositivo: {device}")
    print("==========================================================")

    model = UAVShipClassifier().to(device)
    path_modelo = "models/best_ship_detector.pt"
    if os.path.exists(path_modelo):
        model.load_state_dict(torch.load(path_modelo, map_location=device))
        print("✅ Pesos base cargados. Iniciando adaptación profunda...")

    dataset = ShipsDataset(RUTA_IMAGENES, transform=train_tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Tasa de aprendizaje óptima para que la red aprenda los nuevos dominios
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    epochs = 18
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    start_time = time.time()

    print(f"Entrenando durante {epochs} épocas (Rotaciones 360° + Grayscale 40% + Jitter)...")

    for epoch in range(1, epochs + 1):
        model.train()
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
        print(f"Época [{epoch:02d}/{epochs:02d}] - Loss: {running_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), "models/best_ship_detector.pt")
    duracion = (time.time() - start_time) / 60
    print("\n==========================================================")
    print(f"🎉 ¡Modelo Invariante entrenado exitosamente en {duracion:.2f} minutos!")
    print(f"⭐ Guardado en: '{path_modelo}'")
    print("==========================================================")

if __name__ == "__main__":
    entrenar_invariante()