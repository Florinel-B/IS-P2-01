"""
Script completo para entrenar el ensemble y guardar el modelo completo.
"""

import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from ensemble_model import EnsembleAnomalyDetector

print("="*70)
print("ENTRENAMIENTO COMPLETO DEL ENSEMBLE")
print("="*70)

# 1. Cargar datos
print("\n1️⃣  Cargando datos...")
with open("datos_procesados.pkl", "rb") as f:
    datos_lista = pickle.load(f)

df = pd.DataFrame(datos_lista)
df["tiempo"] = pd.to_datetime(df["tiempo"])
df = df.sort_values("tiempo").reset_index(drop=True)
print(f"   ✓ {len(df)} muestras cargadas")

# 2. Dividir train/test
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

print(f"   ✓ Train: {len(df_train)} muestras")
print(f"   ✓ Test: {len(df_test)} muestras")

# 3. Crear detector
print("\n2️⃣  Inicializando detector ensemble...")
detector = EnsembleAnomalyDetector(
    lstm_model_path="modelo_anomalias_finetuned.pth",
    require_rf=False
)

# 4. Extraer features LSTM para train
print("\n3️⃣  Extrayendo features LSTM (train)...")
lstm_probs_train, lstm_preds_train = detector.extract_lstm_features(df_train)
print(f"   ✓ {lstm_preds_train.sum():.0f} anomalías detectadas")

# 5. Detectar cuelgues
print("\n4️⃣  Detectando cuelgues...")
hang_train = detector.detect_hangs(df_train, hang_duration_minutes=2)
print(f"   ✓ {hang_train.sum():.0f} periodos de cuelgue detectados")

# 6. Crear labels de entrenamiento (ajustando tamaños: LSTM produce seq_len-1 muestras menos)
print("\n5️⃣  Creando labels de verdad fundamental...")
n_df = len(df_train)
n_lstm = len(lstm_probs_train)
offset = n_df - n_lstm

# Padding: repetir primer valor
if offset > 0:
    lstm_probs_padded = np.concatenate([np.zeros(offset), lstm_probs_train])
    lstm_preds_padded = np.concatenate([np.zeros(offset), lstm_preds_train])
else:
    lstm_probs_padded = lstm_probs_train[:n_df]
    lstm_preds_padded = lstm_preds_train[:n_df]

y_train = np.where(hang_train == 1, 2, lstm_preds_padded.astype(int))
print(f"   Normal (0): {np.sum(y_train == 0)}")
print(f"   Anomalía Voltaje (1): {np.sum(y_train == 1)}")
print(f"   Cuelgue Sistema (2): {np.sum(y_train == 2)}")

# 7. Entrenar Random Forest
print("\n6️⃣  Entrenando Random Forest...")
detector.train_random_forest(
    df=df_train,
    lstm_probs=lstm_probs_train,
    hang_labels=hang_train,
    target_labels=y_train,
    n_estimators=150,
    max_depth=20
)

# 8. Evaluación en test
print("\n7️⃣  Evaluación en test set...")
lstm_probs_test, lstm_preds_test = detector.extract_lstm_features(df_test)
hang_test = detector.detect_hangs(df_test, hang_duration_minutes=2)

# Ajustar tamaños para test también
n_df_test = len(df_test)
n_lstm_test = len(lstm_probs_test)
offset_test = n_df_test - n_lstm_test

if offset_test > 0:
    lstm_probs_test_padded = np.concatenate([np.zeros(offset_test), lstm_probs_test])
    lstm_preds_test_padded = np.concatenate([np.zeros(offset_test), lstm_preds_test])
else:
    lstm_probs_test_padded = lstm_probs_test[:n_df_test]
    lstm_preds_test_padded = lstm_preds_test[:n_df_test]

y_test = np.where(hang_test == 1, 2, lstm_preds_test_padded.astype(int))

predictions_test = detector.predict(df_test)
y_pred = predictions_test["predictions"]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 9. Guardar modelo completo
print("\n8️⃣  Guardando modelo completo...")
detector.save_complete_model("modelo_ensemble_completo.pkl")

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)
print(f"\nArchivos generados:")
print(f"   📦 modelo_ensemble_completo.pkl (USAR ESTE EN predict_realtime.py)")
print(f"   📊 modelo_ensemble_rf.pkl (componente RF)")
print(f"   📊 modelo_ensemble_rf_scaler.pkl (componente Scaler)")
