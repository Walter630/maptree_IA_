"""
MapTree - Treinamento dos Modelos de IA
Treina 4 modelos:
  1. Regressão: estimar altura atual da árvore
  2. Regressão: estimar crescimento anual
  3. Regressão: prever dias até atingir o fio
  4. Classificação: classificar risco

Fluxo:
  python3 generate_data.py
  python3 train_models.py

Sempre retreine depois de mudar generate_data.py, BASE_FEATURES ou as fontes
reais de especie/solo/clima.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from generate_data import generate_dataset

# ─── Preparação ────────────────────────────────────────────────────────────────

print("🌳 MapTree - Treinamento de modelos\n")

# Gera ou carrega dataset
if os.path.exists("tree_dataset.csv"):
    print("📂 Carregando dataset existente...")
    df = pd.read_csv("tree_dataset.csv")
else:
    print("⚙️  Gerando dataset sintético...")
    df = generate_dataset(n_samples=3000)
    df.to_csv("tree_dataset.csv", index=False)

print(f"✅ Dataset: {len(df)} amostras\n")

# ─── Feature Engineering ──────────────────────────────────────────────────────

required_generated_columns = {"wire_height", "annual_growth_m", "height_next_year"}
if not required_generated_columns.issubset(df.columns):
    print("⚙️  Dataset antigo detectado. Regenerando com features de fio e crescimento anual...")
    df = generate_dataset(n_samples=3000)
    df.to_csv("tree_dataset.csv", index=False)

# Encoding de variáveis categóricas
soil_quality_map = {"GOOD": 2, "REGULAR": 1, "BAD": 0}
df["soil_quality_enc"] = df["soil_quality"].map(soil_quality_map)

# Features base: devem bater com api.build_features().
# Ordem importa porque os modelos sao treinados com esses nomes/colunas.
BASE_FEATURES = [
    "age_years",
    "species_height_max",
    "species_k",
    "pruning_count",
    "soil_depth",
    "soil_inclination",
    "soil_quality_enc",
    "soil_coverage",
    "annual_rainfall",
    "altitude",
    "avg_temperature",
    "has_fertilization",
    "has_irrigation",
    "nearby_trees_count",
    "avg_neighbor_distance",
    "wire_height",
    "fibonacci_modifier",    # calculado antes de chamar o modelo
    "canopy_ratio",          # calculado antes de chamar o modelo
]

# ─── MODELO 1: Estimativa de Altura ──────────────────────────────────────────
# Aprende a aproximar a formula botanica de altura atual a partir de especie,
# idade, solo, clima, manejo, competicao e altura do fio/contexto.

print("=" * 50)
print("📏 MODELO 1: Estimativa de Altura")
print("=" * 50)

X_height = df[BASE_FEATURES]
y_height = df["estimated_height"]

X_train, X_test, y_train, y_test = train_test_split(
    X_height, y_height, test_size=0.2, random_state=42
)

model_height = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_height.fit(X_train, y_train)

y_pred = model_height.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (erro médio): {mae:.3f} metros")
print(f"R²: {r2:.4f}")

# Feature importance
importance = pd.Series(
    model_height.feature_importances_,
    index=BASE_FEATURES
).sort_values(ascending=False)
print("\nTop 5 features mais importantes:")
print(importance.head(5))

# ─── MODELO 1B: Crescimento Anual ────────────────────────────────────────────
# Estima quantos metros a arvore tende a crescer nos proximos 12 meses.

print("\n" + "=" * 50)
print("🌱 MODELO 1B: Crescimento anual estimado")
print("=" * 50)

X_growth = df[BASE_FEATURES]
y_growth = df["annual_growth_m"]

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_growth, y_growth, test_size=0.2, random_state=42
)

model_growth = GradientBoostingRegressor(
    n_estimators=180,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)
model_growth.fit(X_train_g, y_train_g)

y_pred_g = model_growth.predict(X_test_g)
mae_g = mean_absolute_error(y_test_g, y_pred_g)
r2_g = r2_score(y_test_g, y_pred_g)

print(f"MAE (erro médio): {mae_g:.3f} m/ano")
print(f"R²: {r2_g:.4f}")

# ─── MODELO 2: Dias até o Fio ────────────────────────────────────────────────
# Treina apenas em arvores que biologicamente podem atingir o fio.

print("\n" + "=" * 50)
print("⚡ MODELO 2: Dias até atingir o fio")
print("=" * 50)

# Só treinar com árvores que VÃO atingir o fio (will_reach_wire = 1)
df_wire = df[df["will_reach_wire"] == 1].copy()
print(f"Amostras que atingirão o fio: {len(df_wire)}")

X_wire = df_wire[BASE_FEATURES]
y_wire = df_wire["days_to_wire"]

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wire, y_wire, test_size=0.2, random_state=42
)

model_wire = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    random_state=42
)
model_wire.fit(X_train_w, y_train_w)

y_pred_w = model_wire.predict(X_test_w)
mae_w = mean_absolute_error(y_test_w, y_pred_w)
r2_w = r2_score(y_test_w, y_pred_w)

print(f"MAE (erro médio): {mae_w:.1f} dias (~{mae_w/30:.1f} meses)")
print(f"R²: {r2_w:.4f}")

# ─── MODELO 3: Classificação de Risco ───────────────────────────────────────
# Classificador auxiliar. A API ainda aplica regra deterministica por cima para
# nao reduzir risco em casos de seguranca operacional.

print("\n" + "=" * 50)
print("🚦 MODELO 3: Classificação de Risco")
print("=" * 50)

# Adiciona altura estimada como feature (ela é output do modelo 1)
FEATURES_RISK = BASE_FEATURES + ["estimated_height"]

X_risk = df[FEATURES_RISK]
y_risk = df["risk_status"]

le = LabelEncoder()
y_risk_enc = le.fit_transform(y_risk)

# Em bases pequenas/curadas, pode haver classe com 1 amostra.
# Nesse caso, train_test_split com stratify quebra.
class_counts = pd.Series(y_risk_enc).value_counts()
can_stratify = (class_counts.min() >= 2)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_risk,
    y_risk_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_risk_enc if can_stratify else None,
)
if not can_stratify:
    print("⚠️  Classe rara detectada no risco (menos de 2 amostras). Treino sem stratify.")

model_risk = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
model_risk.fit(X_train_r, y_train_r)

y_pred_r = model_risk.predict(X_test_r)
present_labels = sorted(set(y_test_r) | set(y_pred_r))
present_names = [le.classes_[i] for i in present_labels]
print(classification_report(y_test_r, y_pred_r, labels=present_labels, target_names=present_names, zero_division=0))

# ─── Salvando os modelos ──────────────────────────────────────────────────────

print("=" * 50)
print("💾 Salvando modelos...")

os.makedirs("models", exist_ok=True)

joblib.dump(model_height, "models/model_height.pkl")
joblib.dump(model_growth, "models/model_annual_growth.pkl")
joblib.dump(model_wire, "models/model_wire_days.pkl")
joblib.dump(model_risk, "models/model_risk.pkl")
joblib.dump(le, "models/risk_label_encoder.pkl")
joblib.dump(BASE_FEATURES, "models/base_features.pkl")
joblib.dump(FEATURES_RISK, "models/risk_features.pkl")

print("✅ Modelos salvos em ./models/")
print("\nArquivos:")
for f in os.listdir("models"):
    print(f"  - {f}")
