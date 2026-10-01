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

RUTA_IMAGENES = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"

# 1. Arquitectura exacta de tu red
class UAVShipClassifier(nn.Module):
    def __init__(self):
        super(UAVShipClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

# 2. Focal Loss: Castiga los errores en fotos difíciles (barcos oscuros y olas)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1.0 - pt) ** self.gamma) * logpt
        return focal_loss.mean()

# 3. Dataset
class ShipsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.labels = [1 if f.startswith('1__') else 0 for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transformaciones con variaciones de iluminación marina
tf_dificil = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def afinar():
    print("==========================================================")
    print("🎯 AFINAMIENTO FOCALIZADO (HARD EXAMPLE MINING - FROM SCRATCH)")
    print("==========================================================")

    model = UAVShipClassifier().to(device)
    path_modelo = "models/best_ship_detector.pt"
    model.load_state_dict(torch.load(path_modelo, map_location=device))
    print("✅ Modelo actual cargado exitosamente.")

    dataset = ShipsDataset(RUTA_IMAGENES, transform=tf_dificil)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Tasa de aprendizaje muy fina para no desarmar lo que ya aprendió
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = FocalLoss(gamma=2.5)

    print("Iniciando 5 épocas de micro-calibración sobre casos ambiguos...")
    start_time = time.time()

    model.train()
    for epoch in range(1, 6):
        loss_total = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * imgs.size(0)

        print(f"  -> Época [{epoch}/5] - Focal Loss: {loss_total/len(dataset):.4f}")

    # Sobrescribir con el modelo hiper-calibrado
    torch.save(model.state_dict(), "models/best_ship_detector.pt")
    print("\n==========================================================")
    print(f"🎉 ¡Modelo afinado y guardado en solo {(time.time()-start_time)/60:.2f} minutos!")
    print("==========================================================")

if __name__ == "__main__":
    afinar()