# 📦 Guía de Uso del Modelo Ensemble Completo

## ¿Qué es el Modelo Ensemble Completo?

El archivo `modelo_ensemble_completo.pkl` contiene todo lo necesario para hacer predicciones:
- **LSTM**: Modelo entrenado para detectar anomalías de voltaje
- **Random Forest**: Clasificador para fusionar LSTM + detección de cuelgues
- **Scalers**: Normalizadores necesarios para preprocesar datos

## 📊 Cómo Fue Generado

El modelo se genera ejecutando:
```bash
cd /home/florin/Documentos/GitHub/IS-P2-01
.venv/bin/python src/entrenar_ensemble_completo.py
```

Esto:
1. Divide los datos en train (80%) y test (20%)
2. Extrae features LSTM
3. Detecta cuelgues del sistema (status != 1 por 2+ minutos)
4. Entrena un Random Forest para fusionar las predicciones
5. Guarda el modelo completo en `modelo_ensemble_completo.pkl`

## 🚀 Cómo Usar en predict_realtime.py

```python
from predict_realtime import RealtimePredictor

# Carga automáticamente modelo_ensemble_completo.pkl
predictor = RealtimePredictor()

# Hacer predicción con nuevos datos
voltage_data = {
    "R1_a": 1776.0,  # Voltaje receptor 1 fase A
    "R2_a": 1588.0,  # Voltaje receptor 2 fase A
    "R1_b": 1753.0,  # Voltaje receptor 1 fase B
    "R2_b": 1624.0,  # Voltaje receptor 2 fase B
}

resultado = predictor.predict_single(voltage_data, status=1)

print(f"Clase: {resultado['clase']}")  # "Normal", "Anomalía Voltaje", "Cuelgue Sistema"
print(f"Confianza: {resultado['confianza']:.4f}")
print(f"Probabilidades:")
print(f"  - Normal: {resultado['prob_normal']:.4f}")
print(f"  - Anomalía: {resultado['prob_anomalia_voltaje']:.4f}")
print(f"  - Cuelgue: {resultado['prob_cuelgue']:.4f}")
```

## 🔄 Carga Manual del Modelo

Si prefieres usar el modelo directamente sin `RealtimePredictor`:

```python
from ensemble_model import EnsembleAnomalyDetector

# Cargar modelo completo
detector = EnsembleAnomalyDetector.load_complete_model("modelo_ensemble_completo.pkl")

# Hacer predicciones
resultado = detector.predict(df_datos)
```

## 📋 Componentes Guardados Junto al Modelo Completo

- `modelo_ensemble_rf.pkl`: Random Forest entrenado
- `modelo_ensemble_rf_scaler.pkl`: Normalización para el RF
- `modelo_anomalias_finetuned.pth`: Modelo LSTM base

⚠️ **NOTA**: El archivo `modelo_ensemble_completo.pkl` contiene todo esto, así que solo necesitas este archivo para deploy.

## ✅ Verificar que el Modelo Funciona

Ejecuta el test:
```bash
.venv/bin/python src/test_predict_realtime.py
```

Debería ver:
```
✓ Modelo Random Forest restaurado desde modelo_ensemble_completo.pkl
✓ Modelo ensemble: ✓ (Completo con RF)
✅ TEST EXITOSO - El modelo carga y funciona correctamente
```

## 📊 Estructura de Datos Esperados

**Entrada (voltage_data):**
```python
{
    "R1_a": float,  # Voltaje receptor 1, fase A
    "R2_a": float,  # Voltaje receptor 2, fase A
    "R1_b": float,  # Voltaje receptor 1, fase B
    "R2_b": float,  # Voltaje receptor 2, fase B
}
```

**Salida:**
```python
{
    "prediccion": int,  # 0=Normal, 1=Anomalía Voltaje, 2=Cuelgue
    "clase": str,  # Nombre legible
    "prob_normal": float,
    "prob_anomalia_voltaje": float,
    "prob_cuelgue": float,
    "confianza": float,
    "metodo": str  # "random_forest" o "heuristic"
}
```

## 🔧 Regenerar el Modelo

Si necesitas reentrenar el modelo con nuevos datos:

```bash
# Asegúrate de que los datos procesados están en datos_procesados.pkl
.venv/bin/python src/entrenar_ensemble_completo.py
```

Esto sobrescribirá:
- `modelo_ensemble_completo.pkl`
- `modelo_ensemble_rf.pkl`
- `modelo_ensemble_rf_scaler.pkl`
