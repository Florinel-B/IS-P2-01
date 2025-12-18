"""
Script para entrenar el ensemble y guardar el modelo completo en un archivo.
Ejecutar esto una sola vez después de entrenar el Random Forest.
"""

import pickle
import numpy as np
import pandas as pd
from ensemble_model import EnsembleAnomalyDetector

print("="*70)
print("GUARDANDO MODELO COMPLETO DEL ENSEMBLE")
print("="*70)

# Cargar datos
print("\n1️⃣  Cargando datos...")
with open("datos_procesados.pkl", "rb") as f:
    datos_lista = pickle.load(f)

df = pd.DataFrame(datos_lista)
df["tiempo"] = pd.to_datetime(df["tiempo"])
df = df.sort_values("tiempo").reset_index(drop=True)
print(f"   ✓ {len(df)} muestras")

# Crear detector ensemble
print("\n2️⃣  Inicializando detector ensemble...")
detector = EnsembleAnomalyDetector(
    lstm_model_path="modelo_anomalias_finetuned.pth",
    require_rf=True  # Requiere que el RF esté entrenado
)

# Guardar modelo completo
print("\n3️⃣  Guardando modelo completo...")
detector.save_complete_model("modelo_ensemble_completo.pkl")

print("\n" + "="*70)
print("✅ MODELO COMPLETO GUARDADO")
print("="*70)
print("\nArchivo generado:")
print("   📦 modelo_ensemble_completo.pkl")
print("\nUso en predict_realtime.py:")
print("   detector = EnsembleAnomalyDetector.load_complete_model('modelo_ensemble_completo.pkl')")
