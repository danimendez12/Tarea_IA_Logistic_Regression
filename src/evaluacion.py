import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import matplotlib.pyplot as plt

# ============================================================
# Rutas
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. CARGAR DATOS Y MEJOR MODELO
# ============================================================
ckpt = torch.load(os.path.join(DATA_DIR, "datos_preparados.pt"),
                  weights_only=False)
X_train, y_train = ckpt['X_train'], ckpt['y_train']
X_val,   y_val   = ckpt['X_val'],   ckpt['y_val']
X_test,  y_test  = ckpt['X_test'],  ckpt['y_test']
features = ckpt['selected_features']

best = torch.load(os.path.join(DATA_DIR, "mejor_modelo.pt"),
                  weights_only=False)
config = best['config']
print(f"Mejor modelo según validación: Exp {config[0]} "
      f"({config[1]}, lr={config[2]}, batch={config[3]}, epochs={config[4]})")
print(f"F1 val:  {best['f1_val']:.4f}")
print(f"AUC val: {best['auc_val']:.4f}")

# ============================================================
# 2. RECONSTRUIR MODELO Y CARGAR PESOS
# ============================================================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

model = LogisticRegressionModel(X_train.shape[1])
model.load_state_dict(best['model_state'])
model.eval()

# ============================================================
# 3. FUNCIÓN DE EVALUACIÓN
# ============================================================
def evaluate_split(X, y, split_name):
    with torch.no_grad():
        logits = model(X).squeeze(1)
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs > 0.5).astype(int)
    y_np = y.squeeze(1).numpy().astype(int)

    metrics = {
        'Split':     split_name,
        'N':         len(y_np),
        'Accuracy':  accuracy_score(y_np, preds),
        'Precision': precision_score(y_np, preds, zero_division=0),
        'Recall':    recall_score(y_np, preds, zero_division=0),
        'F1':        f1_score(y_np, preds, zero_division=0),
        'AUC':       roc_auc_score(y_np, probs),
    }
    return metrics, probs, preds, y_np

# ============================================================
# 4. EVALUAR EN TRAIN, VAL Y TEST
# ============================================================
m_train, p_train, pred_train, y_train_np = evaluate_split(X_train, y_train, 'Train')
m_val,   p_val,   pred_val,   y_val_np   = evaluate_split(X_val,   y_val,   'Val')
m_test,  p_test,  pred_test,  y_test_np  = evaluate_split(X_test,  y_test,  'Test')

df_final = pd.DataFrame([m_train, m_val, m_test])
print("\n" + "="*90)
print("MÉTRICAS FINALES POR SPLIT")
print("="*90)
print(df_final.round(4).to_string(index=False))

df_final.to_csv(os.path.join(DATA_DIR, "metricas_finales.csv"), index=False)
print(f"\n✅ Guardado en data/metricas_finales.csv")

# ============================================================
# 5. MATRIZ DE CONFUSIÓN EN TEST
# ============================================================
cm = confusion_matrix(y_test_np, pred_test)
print("\n=== Matriz de confusión (test) ===")
print(f"                    Pred MALA    Pred BUENA")
print(f"Real MALA   (0)      {cm[0,0]:5d}        {cm[0,1]:5d}")
print(f"Real BUENA  (1)      {cm[1,0]:5d}        {cm[1,1]:5d}")

# Matriz de confusión como figura
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_xticklabels(['MALA', 'BUENA'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['MALA', 'BUENA'])
ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
ax.set_title('Matriz de Confusión — Test')

for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color=color, fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_test.png"), dpi=150)
plt.show()
plt.close()

# ============================================================
# 6. CURVA ROC
# ============================================================
fpr, tpr, _ = roc_curve(y_test_np, p_test)
auc_test = roc_auc_score(y_test_np, p_test)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='tab:blue', lw=2,
         label=f'ROC curve (AUC = {auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC — Test')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve_test.png"), dpi=150)
plt.show()
plt.close()

# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================
print("\n=== Classification Report (test) ===")
print(classification_report(y_test_np, pred_test,
                             target_names=['MALA', 'BUENA'], digits=4))

# ============================================================
# 8. COEFICIENTES FINALES
# ============================================================
weights = model.linear.weight.detach().numpy().flatten()
bias    = model.linear.bias.item()
print("=== Coeficientes del modelo final ===")
for feat, w in zip(features, weights):
    print(f"  {feat:25s}: {w:+.4f}")
print(f"  {'bias':25s}: {bias:+.4f}")

print("\n Evaluación final completa.")