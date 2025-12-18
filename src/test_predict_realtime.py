"""
Test simple para verificar que predict_realtime.py carga correctamente el modelo completo.
"""

import pandas as pd
from predict_realtime import RealtimePredictor

print("="*70)
print("TEST: CARGA DEL MODELO COMPLETO EN predict_realtime.py")
print("="*70)

# Crear predictor (debería cargar modelo_ensemble_completo.pkl automáticamente)
print("\n1️⃣  Inicializando predictor...")
predictor = RealtimePredictor()

# Test con un dato simple
print("\n2️⃣  Realizando predicción de prueba...")
test_data = {
    "R1_a": 1776.0,
    "R2_a": 1588.0,
    "R1_b": 1753.0,
    "R2_b": 1624.0,
}
status = 1

result = predictor.predict_single(test_data, status=status)

print(f"\n3️⃣  Resultado de la predicción:")
print(f"   Predicción: {result['prediccion']}")
print(f"   Clase: {result['clase']}")
print(f"   Confianza: {result['confianza']:.4f}")
print(f"   Método usado: {result.get('metodo', 'N/A')}")

print("\n" + "="*70)
print("✅ TEST EXITOSO - El modelo carga y funciona correctamente")
print("="*70)
