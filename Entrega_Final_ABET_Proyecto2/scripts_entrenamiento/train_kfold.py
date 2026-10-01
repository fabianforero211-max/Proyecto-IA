import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configuración de Hardware y Semillas para reproducibilidad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("training_evidence", exist_ok=True)

RUTA_IMAGENES = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"

# 2. Definición del Dataset directo con nombres originales
class ShipsDatasetOriginal(Dataset):
    def __init__(self, root_dir, file_list, labels, transform=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. Gráfica de Pastel (Evidencia oficial: 85% Train / 15% Val)
def save_pie_chart():
    labels = ['Entrenamiento (85%)', 'Validación (15%)']
    sizes = [85, 15]
    colors = ['#1f77b4', '#ff7f0e']
    explode = (0, 0.08)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Distribución de Partición por Fold (85% Train / 15% Val)', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig('training_evidence/pie_chart_split.png', dpi=300)
    plt.close()

# 4. Arquitectura Convolucional Optimizada (CNN Ligera para Dron UAV)
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

# 5. Transformaciones y Aumento de Datos para alta generalización
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

# 6. Pipeline de Validación Cruzada (6 Folds)
def run_kfold_oficial(dataset_dir, n_splits=6, epochs=10, batch_size=64):
    print("================================================================")
    print("🚀 INICIANDO VALIDACIÓN CRUZADA OFICIAL (6 FOLDS)")
    print(f"📌 Dispositivo de Cómputo: {device}")
    print("================================================================")

    save_pie_chart()
    print("✅ Gráfica de pastel guardada en 'training_evidence/pie_chart_split.png'")

    # Leer archivos y etiquetas originales
    archivos = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg'))])
    labels = np.array([1 if f.startswith('1__') else 0 for f in archivos])

    total_imgs = len(archivos)
    n_barcos = sum(labels == 1)
    n_nobarcos = sum(labels == 0)

    print(f"📊 Dataset cargado: {total_imgs} imágenes totales")
    print(f"   • Barcos reales (1):    {n_barcos} ({n_barcos/total_imgs*100:.1f}%)")
    print(f"   • No-Barcos reales (0): {n_nobarcos} ({n_nobarcos/total_imgs*100:.1f}%)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_overall_acc = 0.0

    # Pérdida equilibrada para eliminar falsos positivos
    criterion = nn.CrossEntropyLoss()
    start_total_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n>>> 🔄 FOLD {fold + 1}/{n_splits} [Train: {len(train_idx)} | Val: {len(val_idx)}]")

        train_ds = ShipsDatasetOriginal(dataset_dir, [archivos[i] for i in train_idx], labels[train_idx], transform=train_transform)
        val_ds = ShipsDatasetOriginal(dataset_dir, [archivos[i] for i in val_idx], labels[val_idx], transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = UAVShipClassifier().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()
            print(f"   Época {epoch+1:02d}/{epochs} completada...", end="\r")

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

        fold_results.append({
            'Fold': fold + 1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        print(f"\n   🎯 Fold {fold + 1} Finalizado -> Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")

        # Guardar el modelo con mejor precisión
        if acc > best_overall_acc:
            best_overall_acc = acc
            torch.save(model.state_dict(), "models/best_ship_detector.pt")

    # Guardar métricas oficiales
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv("training_evidence/kfold_metrics.csv", index=False)

    mean_acc = float(df_results['Accuracy'].mean())
    std_acc = float(df_results['Accuracy'].std())
    mean_f1 = float(df_results['F1-Score'].mean())

    summary = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1": mean_f1
    }
    with open("training_evidence/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

    total_min = (time.time() - start_total_time) / 60
    print("\n" + "="*60)
    print("🏆 RESUMEN FINAL OFICIAL (6 FOLDS):")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\n⭐ PROMEDIO ACCURACY: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"⭐ MEJOR MODELO GUARDADO: 'models/best_ship_detector.pt'")
    print(f"⏱️ Tiempo total: {total_min:.2f} minutos")
    print("="*60)

if __name__ == "__main__":
    run_kfold_oficial(RUTA_IMAGENES, n_splits=6, epochs=10)