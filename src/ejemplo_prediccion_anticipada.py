"""
Ejemplo de uso de predicción anticipada en tiempo real.
Muestra cómo obtener y usar el siguiente estado para alertas preventivas.
"""

import pandas as pd
from predict_realtime import RealtimePredictor

print("="*70)
print("EJEMPLO: PREDICCIÓN ANTICIPADA DEL SIGUIENTE ESTADO")
print("="*70)

# Crear predictor (carga automáticamente modelo_ensemble_completo.pkl)
print("\n1️⃣  Inicializando predictor...")
predictor = RealtimePredictor()

# Simular datos en tiempo real
print("\n2️⃣  Datos de ejemplo (voltajes):")
datos_ejemplo = [
    {
        "timestamp": "2025-04-23 15:00:00",
        "voltages": {
            "R1_a": 1776.0,
            "R2_a": 1588.0,
            "R1_b": 1753.0,
            "R2_b": 1624.0,
        },
        "status": 1
    },
    {
        "timestamp": "2025-04-23 15:01:00",
        "voltages": {
            "R1_a": 1775.5,
            "R2_a": 1587.8,
            "R1_b": 1752.9,
            "R2_b": 1623.9,
        },
        "status": 1
    },
    {
        "timestamp": "2025-04-23 15:02:00",
        "voltages": {
            "R1_a": 1730.0,  # Caída de voltaje
            "R2_a": 1540.0,  # Caída de voltaje
            "R1_b": 1700.0,  # Caída de voltaje
            "R2_b": 1570.0,  # Caída de voltaje
        },
        "status": 1
    },
]

print("\nProcesando datos en tiempo real...")
for dato in datos_ejemplo:
    print(f"\n⏱️  {dato['timestamp']}")
    print(f"   Voltajes: {dato['voltages']}")
    print(f"   Status: {dato['status']}")
    
    # Predicción para este timestamp
    resultado = predictor.predict_single(dato['voltages'], status=dato['status'])
    
    print(f"\n   📊 ESTADO ACTUAL (t):")
    print(f"      Predicción: {resultado['prediccion_actual']}")
    print(f"      Clase: {resultado['clase_actual']}")
    print(f"      Confianza: {resultado['confianza_actual']:.4f}")
    
    print(f"\n   🔮 ESTADO SIGUIENTE (t+1) ⭐:")
    print(f"      Predicción: {resultado['prediccion_siguiente']}")
    print(f"      Clase: {resultado['clase_siguiente']}")
    print(f"      Confianza: {resultado['confianza_siguiente']:.4f}")
    
    # Alerta preventiva
    if resultado.get('alerta_preventiva', False):
        print(f"\n   ⚠️  ALERTA PREVENTIVA DETECTADA!")
        print(f"      Cambio de '{resultado['clase_actual']}' → '{resultado['clase_siguiente']}'")
        print(f"      → REACCIONAR AHORA (antes de que ocurra)")
    else:
        print(f"\n   ✓ Sin cambios anticipados")
    
    # Probabilidades
    print(f"\n   Probabilidades ACTUALES:")
    print(f"      Normal: {resultado['prob_normal_actual']:.4f}")
    print(f"      Anomalía: {resultado['prob_anomalia_voltaje_actual']:.4f}")
    print(f"      Cuelgue: {resultado['prob_cuelgue_actual']:.4f}")
    
    print(f"\n   Probabilidades FUTURAS:")
    print(f"      Normal: {resultado['prob_normal_siguiente']:.4f}")
    print(f"      Anomalía: {resultado['prob_anomalia_voltaje_siguiente']:.4f}")
    print(f"      Cuelgue: {resultado['prob_cuelgue_siguiente']:.4f}")

print("\n" + "="*70)
print("✅ EJEMPLO COMPLETADO")
print("="*70)
print("""
CONCLUSIONES:
  1. El sistema detecta el ESTADO ACTUAL (lo que está pasando ahora)
  2. Pero lo MÁS IMPORTANTE es el ESTADO SIGUIENTE (lo que pasará después)
  3. Las ALERTAS PREVENTIVAS permiten reaccionar ANTES de problemas
  4. Esto es MANTENIMIENTO PREDICTIVO en acción

BENEFICIOS:
  ✓ Más tiempo para reaccionar
  ✓ Prevención de fallas
  ✓ Reducción de downtime
  ✓ Optimización de mantenimiento
""")
