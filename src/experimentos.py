import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt   # ← sin Agg, ahora abre ventanas

# ============================================================
# Rutas
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
ckpt = torch.load(os.path.join(DATA_DIR, "datos_preparados.pt"),
                  weights_only=False)

X_train, y_train = ckpt['X_train'], ckpt['y_train']
X_val,   y_val   = ckpt['X_val'],   ckpt['y_val']
X_test,  y_test  = ckpt['X_test'],  ckpt['y_test']
features = ckpt['selected_features']

INPUT_DIM = X_train.shape[1]
print(f"Input dim: {INPUT_DIM} | Features: {features}")

n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
POS_WEIGHT = torch.tensor([n_neg / n_pos], dtype=torch.float32)
print(f"pos_weight: {POS_WEIGHT.item():.4f}\n")

# ============================================================
# 2. MODELO
# ============================================================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# ============================================================
# 3. HELPERS
# ============================================================
def make_loaders(batch_size):
    tr = DataLoader(TensorDataset(X_train, y_train),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(X_val, y_val),
                    batch_size=batch_size, shuffle=False)
    te = DataLoader(TensorDataset(X_test, y_test),
                    batch_size=batch_size, shuffle=False)
    return tr, va, te

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        train_losses.append(tl / len(train_loader))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                vl += loss.item()
        val_losses.append(vl / len(val_loader))
    return train_losses, val_losses

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze(1)
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs > 0.5).astype(int)
    y_np = y.squeeze(1).numpy().astype(int)
    return {
        'accuracy':  accuracy_score(y_np, preds),
        'precision': precision_score(y_np, preds, zero_division=0),
        'recall':    recall_score(y_np, preds, zero_division=0),
        'f1':        f1_score(y_np, preds, zero_division=0),
        'auc':       roc_auc_score(y_np, probs),
    }, probs, preds

# ============================================================
# 4. EXPERIMENTOS
# ============================================================
experimentos = [
    (1,  'SGD',  0.01,  32, 100),
    (2,  'SGD',  0.001, 32, 100),
    (3,  'SGD',  0.1,   32, 100),
    (4,  'SGD',  0.01,  16, 100),
    (5,  'SGD',  0.01,  64, 100),
    (6,  'SGD',  0.01,  32, 200),
    (7,  'Adam', 0.001, 32, 100),
    (8,  'Adam', 0.01,  32, 100),
    (9,  'Adam', 0.001, 16, 150),
    (10, 'Adam', 0.001, 64, 150),
]

# ============================================================
# 5. BUCLE PRINCIPAL
# ============================================================
resultados   = []
mejor_f1     = -1.0
mejor_auc    = -1.0
mejor_modelo = None
mejor_config = None

for exp_id, opt_name, lr, batch, epochs in experimentos:
    print(f"\n{'='*60}")
    print(f"Experimento {exp_id}: {opt_name} | lr={lr} | batch={batch} | epochs={epochs}")
    print(f"{'='*60}")

    torch.manual_seed(42 + exp_id)
    train_loader, val_loader, test_loader = make_loaders(batch)

    model = LogisticRegressionModel(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)

    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tr_losses, vl_losses = train_model(model, train_loader, val_loader,
                                       criterion, optimizer, epochs)

    metrics, _, _ = evaluate(model, X_val, y_val)
    print(f"Val → Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} "
          f"| Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

    resultados.append({
        'Exp': exp_id,
        'Optimizer': opt_name,
        'LR': lr,
        'Batch': batch,
        'Epochs': epochs,
        'Train Loss (final)': tr_losses[-1],
        'Val Loss (final)': vl_losses[-1],
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'AUC': metrics['auc'],
    })

    # Desempate por AUC
    if (metrics['f1'] > mejor_f1) or \
       (abs(metrics['f1'] - mejor_f1) < 1e-6 and metrics['auc'] > mejor_auc):
        mejor_f1   = metrics['f1']
        mejor_auc  = metrics['auc']
        mejor_modelo = {k: v.clone() for k, v in model.state_dict().items()}
        mejor_config = (exp_id, opt_name, lr, batch, epochs)

    # Curva: guardar + mostrar ventana
    plt.figure(figsize=(8, 4))
    plt.plot(tr_losses, label='Train', color='tab:blue')
    plt.plot(vl_losses, label='Val',   color='tab:red')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Exp {exp_id}: {opt_name}, lr={lr}, batch={batch}, epochs={epochs}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"loss_exp{exp_id:02d}.png"), dpi=120)
    plt.show()   # ← abre la ventana emergente
    plt.close()

# ============================================================
# 6. TABLA COMPARATIVA
# ============================================================
df_results = pd.DataFrame(resultados).sort_values('F1', ascending=False).reset_index(drop=True)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

print("\n" + "="*100)
print("TABLA COMPARATIVA (ordenada por F1 descendente)")
print("="*100)
print(df_results.round(4).to_string(index=False))

csv_path = os.path.join(DATA_DIR, "resultados_experimentos.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n✅ Tabla guardada en {csv_path}")

# ============================================================
# 7. GUARDAR MEJOR MODELO
# ============================================================
torch.save({
    'model_state': mejor_modelo,
    'config': mejor_config,
    'f1_val': mejor_f1,
    'auc_val': mejor_auc,
    'features': features,
}, os.path.join(DATA_DIR, "mejor_modelo.pt"))

print(f"\n Mejor modelo: Exp {mejor_config[0]} "
      f"({mejor_config[1]}, lr={mejor_config[2]}, batch={mejor_config[3]}, epochs={mejor_config[4]})")
print(f"   F1 en validación:  {mejor_f1:.4f}")
print(f"   AUC en validación: {mejor_auc:.4f}")
print(f"   Guardado en data/mejor_modelo.pt")