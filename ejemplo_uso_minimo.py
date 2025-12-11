"""
Ejemplo mínimo de cómo usar el modelo en otro proyecto.

Para usar en tu proyecto:
1. Copia este archivo
2. Copia training_template.py (o todo el directorio src/)
3. Copia modelo_anomalias.pth
4. Instala: pip install torch pandas numpy scikit-learn
"""

import sys
import os

# Agregar src al path si es necesario
if os.path.exists('src'):
    sys.path.insert(0, 'src')

import torch
from training_template import importar_modelo_portable, OnlinePredictor


def ejemplo_basico():
    """Ejemplo más simple posible."""
    print("=== EJEMPLO BÁSICO ===\n")
    
    # 1. Cargar modelo
    modelo = importar_modelo_portable('modelo_anomalias.pth')
    
    # 2. Crear predictor
    predictor = OnlinePredictor(
        modelo['model'],
        modelo['scaler'],
        device=modelo['device']
    )
    
    # 3. Predecir
    muestra = {
        'status': 1.0,
        'R1_a': 2500.0,
        'R2_a': 2480.0,
        'R1_b': 2510.0,
        'R2_b': 2490.0
    }
    
    resultado = predictor.predict_single(muestra, threshold=modelo['threshold'])
    
    # 4. Mostrar resultado
    if resultado['is_anomaly']:
        print(f"⚠️  ANOMALÍA DETECTADA")
    else:
        print(f"✅ NORMAL")
    
    print(f"Probabilidad: {resultado['probability']:.4f}")


def ejemplo_stream():
    """Ejemplo procesando datos en tiempo real."""
    print("\n=== EJEMPLO STREAMING ===\n")
    
    # Cargar modelo
    modelo = importar_modelo_portable('modelo_anomalias.pth')
    predictor = OnlinePredictor(modelo['model'], modelo['scaler'])
    
    # Simular stream de datos
    datos_stream = [
        {'status': 1.0, 'R1_a': 2500.0, 'R2_a': 2480.0, 'R1_b': 2510.0, 'R2_b': 2490.0},
        {'status': 1.0, 'R1_a': 2505.0, 'R2_a': 2485.0, 'R1_b': 2515.0, 'R2_b': 2495.0},
        {'status': 0.0, 'R1_a': 2510.0, 'R2_a': 2490.0, 'R1_b': 2520.0, 'R2_b': 2500.0},
        # Anomalía: salto brusco
        {'status': 1.0, 'R1_a': 3200.0, 'R2_a': 3180.0, 'R1_b': 3210.0, 'R2_b': 3190.0},
        {'status': 1.0, 'R1_a': 3205.0, 'R2_a': 3185.0, 'R1_b': 3215.0, 'R2_b': 3195.0},
    ]
    
    for i, muestra in enumerate(datos_stream, 1):
        resultado = predictor.predict_single(muestra, threshold=modelo['threshold'])
        
        status = "⚠️  ANOMALÍA" if resultado['is_anomaly'] else "✅ Normal"
        print(f"Muestra {i}: {status} (prob: {resultado['probability']:.4f})")


def ejemplo_batch():
    """Ejemplo procesando un lote de datos."""
    print("\n=== EJEMPLO BATCH ===\n")
    
    import pandas as pd
    from training_template import VoltageDropDataset, evaluar_modelo
    from torch.utils.data import DataLoader
    
    # Cargar modelo
    modelo = importar_modelo_portable('modelo_anomalias.pth')
    
    # Cargar datos (simulados)
    datos_ejemplo = {
        'tiempo': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'id': ['device1'] * 100,
        'status': [1.0] * 100,
        'R1_a': [2500.0 + i for i in range(100)],
        'R2_a': [2480.0 + i for i in range(100)],
        'R1_b': [2510.0 + i for i in range(100)],
        'R2_b': [2490.0 + i for i in range(100)],
    }
    df = pd.DataFrame(datos_ejemplo)
    
    # Crear dataset
    dataset = VoltageDropDataset(
        df,
        seq_len=modelo['seq_len'],
        scaler=modelo['scaler']
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Evaluar
    metricas = evaluar_modelo(
        modelo['model'],
        loader,
        threshold=modelo['threshold'],
        device=modelo['device']
    )
    
    print(f"F1-Score: {metricas['f1_score']:.4f}")


def ejemplo_ajustar_umbral():
    """Ejemplo ajustando el umbral de detección."""
    print("\n=== EJEMPLO AJUSTE DE UMBRAL ===\n")
    
    modelo = importar_modelo_portable('modelo_anomalias.pth')
    predictor = OnlinePredictor(modelo['model'], modelo['scaler'])
    
    muestra = {
        'status': 1.0,
        'R1_a': 2500.0,
        'R2_a': 2480.0,
        'R1_b': 2510.0,
        'R2_b': 2490.0
    }
    
    # Probar diferentes umbrales
    umbrales = [0.3, 0.5, 0.7, 0.9]
    
    for umbral in umbrales:
        resultado = predictor.predict_single(muestra, threshold=umbral)
        status = "⚠️  ANOMALÍA" if resultado['is_anomaly'] else "✅ Normal"
        print(f"Umbral {umbral:.1f}: {status} (prob: {resultado['probability']:.4f})")


if __name__ == "__main__":
    # Ejecutar todos los ejemplos
    try:
        ejemplo_basico()
        ejemplo_stream()
        ejemplo_ajustar_umbral()
        # ejemplo_batch()  # Requiere datos reales
        
        print("\n✅ Todos los ejemplos ejecutados correctamente")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Asegúrate de tener:")
        print("   1. modelo_anomalias.pth")
        print("   2. training_template.py")
        print("   3. Ejecuta: pip install torch pandas numpy scikit-learn")
