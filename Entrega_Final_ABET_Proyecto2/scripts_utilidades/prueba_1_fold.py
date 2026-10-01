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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Configuración de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

RUTA_IMAGENES = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"

# 2. Definición del Dataset directo con nombres originales
class ShipsDatasetOriginal(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. Arquitectura Convolucional Optimizada
class UAVShipClassifier(nn.Module):
    def __init__(self):
        super(UAVShipClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x40

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x20

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 10x10

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

# 4. Aumento de datos y Normalización
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Ejecución de la prueba de 1 solo Fold
def ejecutar_prueba():
    print("==========================================================")
    print("🔍 REVISANDO IMÁGENES ORIGINALES DEL DATASET")
    print("==========================================================")
    
    if not os.path.exists(RUTA_IMAGENES):
        print(f"Error: No existe la ruta {RUTA_IMAGENES}")
        return

    archivos = [f for f in sorted(os.listdir(RUTA_IMAGENES)) if f.lower().endswith(('.png', '.jpg'))]
    
    # Extraer la etiqueta original del prefijo del nombre
    paths = [os.path.join(RUTA_IMAGENES, f) for f in archivos]
    labels = [1 if f.startswith('1__') else 0 for f in archivos]

    n_barcos = sum(1 for l in labels if l == 1)
    n_nobarcos = sum(1 for l in labels if l == 0)

    print(f"Total imágenes encontradas: {len(archivos)}")
    print(f"🟢 Barcos reales (prefijo '1__'): {n_barcos}")
    print(f"🔴 No-Barcos reales (prefijo '0__'): {n_nobarcos}")
    print(f"Cómputo en: {device}")
    print("==========================================================")

    # Partición exacta 85% Entrenamiento / 15% Validación
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"Distribución de la prueba:")
    print(f"• Set de Entrenamiento (85%): {len(train_paths)} imágenes")
    print(f"• Set de Validación    (15%): {len(val_paths)} imágenes")
    print("==========================================================\n")

    train_ds = ShipsDatasetOriginal(train_paths, train_labels, transform=train_transform)
    val_ds = ShipsDatasetOriginal(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = UAVShipClassifier().to(device)
    
    # Manejo de balance automático según conteo real
    weight_val = n_nobarcos / max(n_barcos, 1)
    class_weights = torch.tensor([1.0, weight_val]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("🚀 Iniciando Entrenamiento de 1 Fold (10 Épocas)...")
    start_time = time.time()

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Evaluación en validación
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(targets.numpy())

        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100

        print(f"Época [{epoch:02d}/10] | Loss: {running_loss/len(train_paths):.4f} | Val Accuracy: {acc:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}% | F1: {f1:.2f}%")

    duracion = (time.time() - start_time) / 60
    print("\n" + "="*50)
    print("📊 RESULTADO FINAL DEL TEST (1 FOLD):")
    print("="*50)
    print(f"• Accuracy Final:  {acc:.2f}%")
    print(f"• Precision Final: {prec:.2f}%")
    print(f"• Recall Final:    {rec:.2f}%")
    print(f"• F1-Score Final:  {f1:.2f}%")
    print(f"⏱️ Tiempo transcurrido: {duracion:.2f} minutos")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nMatriz de Confusión del 15% de Validación:")
    print(f"                Predicho 0   Predicho 1")
    print(f"Real No-Barco (0):   {cm[0][0]:<10}   {cm[0][1]}")
    print(f"Real Barco    (1):   {cm[1][0]:<10}   {cm[1][1]}")
    print("="*50)

if __name__ == "__main__":
    ejecutar_prueba()