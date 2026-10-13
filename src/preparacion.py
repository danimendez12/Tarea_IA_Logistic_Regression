import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Rutas
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# 1. CARGA Y PREPARACIÓN
# ============================================================
df = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"))
df = df.drop_duplicates().reset_index(drop=True)
df['label'] = (df['quality'] >= 6).astype(int)

# ============================================================
# 2. SELECCIÓN DE LAS 6 FEATURES
# ============================================================
selected_features = [
    'alcohol',
    'sulphates',
    'volatile acidity',
    'total sulfur dioxide',
    'density',
    'citric acid'
]

X = df[selected_features].values
y = df['label'].values

# ============================================================
# 3. SPLIT ESTRATIFICADO 70 / 15 / 15
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 4. ESTANDARIZACIÓN
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 5. TENSORES
# ============================================================
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# ============================================================
# 6. GUARDAR
# ============================================================
torch.save({
    'X_train': X_train_t, 'y_train': y_train_t,
    'X_val':   X_val_t,   'y_val':   y_val_t,
    'X_test':  X_test_t,  'y_test':  y_test_t,
    'selected_features': selected_features,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_
}, os.path.join(DATA_DIR, "datos_preparados.pt"))

print("✅ Guardado en data/datos_preparados.pt")