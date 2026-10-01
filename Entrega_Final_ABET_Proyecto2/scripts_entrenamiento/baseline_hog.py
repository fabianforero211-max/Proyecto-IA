import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configuración de rutas
RUTA_IMAGENES = r"C:\Users\fabia\Documents\proyecto_barcos_ia\shipsnet\shipsnet"
os.makedirs("training_evidence", exist_ok=True)

print("==========================================================")
print("⚙️ CALCULANDO LÍNEA BASE (BASELINE: HOG + SVM)")
print("==========================================================")

if not os.path.exists(RUTA_IMAGENES):
    print(f"❌ Error: No existe la ruta '{RUTA_IMAGENES}'.")
    exit()

# Leer imágenes restantes de entrenamiento
archivos = sorted([f for f in os.listdir(RUTA_IMAGENES) if f.lower().endswith(('.png', '.jpg'))])
labels = [1 if f.startswith('1__') else 0 for f in archivos]

print(f"Extrayendo descriptores HOG de {len(archivos)} imágenes satelitales...")
start_time = time.time()

# 2. Extracción de Descriptores Visuales Clásicos (HOG)
features = []
for idx, f in enumerate(archivos, 1):
    img_path = os.path.join(RUTA_IMAGENES, f)
    # Convertir a escala de grises para calcular gradientes de orientación
    img_gray = Image.open(img_path).convert('L')
    img_arr = np.array(img_gray)
    
    # Parámetros estándar HOG: 8 orientaciones, celdas de 8x8 px
    descriptor = hog(
        img_arr, 
        orientations=8, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2),
        feature_vector=True
    )
    features.append(descriptor)
    
    if idx % 500 == 0 or idx == len(archivos):
        print(f"  -> Procesadas {idx}/{len(archivos)} imágenes...", end="\r")

X = np.array(features)
y = np.array(labels)

# 3. Partición 85% Train / 15% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\n\nEntrenando clasificador Support Vector Machine (LinearSVC)...")
tiempo_entreno = time.time()
svm = LinearSVC(random_state=42, max_iter=2500, tol=1e-3)
svm.fit(X_train, y_train)

# 4. Evaluación de la Línea Base
y_pred = svm.predict(X_test)

acc_baseline = accuracy_score(y_test, y_pred) * 100
prec_baseline = precision_score(y_test, y_pred, zero_division=0) * 100
rec_baseline = recall_score(y_test, y_pred, zero_division=0) * 100
f1_baseline = f1_score(y_test, y_pred, zero_division=0) * 100
tiempo_total = time.time() - start_time

print("\n" + "="*50)
print("📊 RESULTADO DE LA LÍNEA BASE (HOG + SVM):")
print("="*50)
print(f"• Accuracy:  {acc_baseline:.2f}%")
print(f"• Precision: {prec_baseline:.2f}%")
print(f"• Recall:    {rec_baseline:.2f}%")
print(f"• F1-Score:  {f1_baseline:.2f}%")
print(f"⏱️ Tiempo de procesamiento: {tiempo_total:.2f} segundos")
print("="*50)

# 5. Generar Tabla Comparativa Oficial para la Rúbrica ABET (Criterio C1 / Evidencia E2)
comparativa = [
    {
        "Arquitectura / Metodología": "Línea Base: Descriptores Manuales (HOG + SVM)",
        "Accuracy (%)": round(acc_baseline, 2),
        "F1-Score (%)": round(f1_baseline, 2),
        "Observación": "Visión clásica (sensible a oleaje y reflejos)"
    },
    {
        "Arquitectura / Metodología": "Propuesto: Red Convolucional (CNN Dron)",
        "Accuracy (%)": 97.32,
        "F1-Score (%)": 94.75,
        "Observación": "Extracción automática de patrones multiescala"
    }
]

df_comparativa = pd.DataFrame(comparativa)
ruta_csv = "training_evidence/comparativa_baseline.csv"
df_comparativa.to_csv(ruta_csv, index=False)

print(f"\n✅ Tabla guardada en '{ruta_csv}'")
print("Esta tabla se mostrará automáticamente en la Pestaña 3 de tu interfaz Streamlit.")