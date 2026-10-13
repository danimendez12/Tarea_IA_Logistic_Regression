import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Detectar raíz del proyecto (una carpeta arriba de src/)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figuras")

# Crear carpetas si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. CARGA Y PREPARACIÓN
# ============================================================
df = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"))

df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape tras eliminar duplicados: {df.shape}")

df['label'] = (df['quality'] >= 6).astype(int)
print("\nDistribución binaria:")
print(df['label'].value_counts(normalize=True).round(3))

# ============================================================
# 2. INFORMACIÓN MUTUA MANUAL
# ============================================================
def mutual_info_manual(x, y, n_bins=10):
    x_disc = pd.qcut(x, q=n_bins, labels=False, duplicates='drop')
    joint = pd.crosstab(x_disc, y, normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            pxy = joint.loc[i, j]
            if pxy > 0:
                mi += pxy * np.log(pxy / (px[i] * py[j]))
    return mi

features = [c for c in df.columns if c not in ['quality', 'label']]
mi_results = {f: mutual_info_manual(df[f], df['label']) for f in features}
mi_series = pd.Series(mi_results).sort_values(ascending=False)

print("\n=== INFORMACIÓN MUTUA con 'label' ===")
print(mi_series.round(4))

# ============================================================
# 3. HEATMAP DE CORRELACIÓN  ← aquí va el savefig
# ============================================================
corr_features = df[features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_features, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlación entre features fisicoquímicas")
plt.tight_layout()

# 👇 ESTA es la línea que preguntabas. Va aquí, justo antes de mostrar/cerrar.
plt.savefig(os.path.join(FIG_DIR, "heatmap_correlacion.png"), dpi=150)
plt.show()
plt.close()

# ============================================================
# 4. PARES CON |r| > 0.7
# ============================================================
upper = corr_features.where(
    np.triu(np.ones(corr_features.shape), k=1).astype(bool)
)
pairs = upper.stack().reset_index()
pairs.columns = ['Feature 1', 'Feature 2', 'r']
pairs['abs_r'] = pairs['r'].abs()
pairs = pairs.sort_values('abs_r', ascending=False)

print("\n=== PARES con |r| > 0.7 ===")
print(pairs[pairs['abs_r'] > 0.7].to_string(index=False))