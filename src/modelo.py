import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# Rutas
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. CARGAR DATOS PREPARADOS
# ============================================================
checkpoint = torch.load(
    os.path.join(DATA_DIR, "datos_preparados.pt"),
    weights_only=False
)

X_train = checkpoint['X_train']; y_train = checkpoint['y_train']
X_val   = checkpoint['X_val'];   y_val   = checkpoint['y_val']
X_test  = checkpoint['X_test'];  y_test  = checkpoint['y_test']
features = checkpoint['selected_features']

print("Features usadas:", features)
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ============================================================
# 2. DATALOADERS
# ============================================================
BATCH_SIZE = 32

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test),
                          batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 3. MODELO DE REGRESIÓN LOGÍSTICA
# ============================================================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # w^T x + b

    def forward(self, x):
        # Devolvemos LOGITS (sin sigmoide) porque BCEWithLogitsLoss lo aplica
        return self.linear(x)

# ============================================================
# 4. FUNCIÓN DE ENTRENAMIENTO
# ============================================================
def train_model(model, train_loader, val_loader, criterion,
                optimizer, epochs, verbose=True):
    """
    Entrena el modelo y devuelve las listas de pérdida de train y val
    por época.
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---------- Entrenamiento ----------
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------- Validación ----------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# ============================================================
# 5. CONFIGURACIÓN DEL EXPERIMENTO PILOTO
# ============================================================
torch.manual_seed(42)          # reproducibilidad

INPUT_DIM = X_train.shape[1]   # 6 features
LR        = 0.01
EPOCHS    = 100

model = LogisticRegressionModel(INPUT_DIM)

# Pérdida con pos_weight por si acaso (balance ~47/53, leve)
n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
print(f"\npos_weight = {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# ============================================================
# 6. ENTRENAR
# ============================================================
print("\n=== Entrenando modelo piloto ===")
train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS
)

# ============================================================
# 7. GRAFICAR CURVAS DE PÉRDIDA
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training', color='tab:blue', linewidth=2)
plt.plot(val_losses,   label='Validation', color='tab:red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "loss_piloto.png"), dpi=150)
plt.show()
plt.close()

# ============================================================
# 8. INSPECCIÓN DE COEFICIENTES
# ============================================================
weights = model.linear.weight.detach().numpy().flatten()
bias    = model.linear.bias.item()

print("\n=== Coeficientes aprendidos ===")
for feat, w in zip(features, weights):
    print(f"  {feat:25s}: {w:+.4f}")
print(f"  {'bias':25s}: {bias:+.4f}")

# ============================================================
# 9. GUARDAR MODELO PILOTO
# ============================================================
torch.save({
    'model_state': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'lr': LR,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'optimizer': 'SGD'
}, os.path.join(DATA_DIR, "modelo_piloto.pt"))

model.eval()
with torch.no_grad():
    train_preds = (torch.sigmoid(model(X_train)) > 0.5).float()
    val_preds   = (torch.sigmoid(model(X_val)) > 0.5).float()
    train_acc = (train_preds == y_train).float().mean().item()
    val_acc   = (val_preds == y_val).float().mean().item()

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")

print("\n✅ Modelo piloto guardado en data/modelo_piloto.pt")