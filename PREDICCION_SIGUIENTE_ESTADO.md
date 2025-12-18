# 🚀 Predicción Anticipada del Siguiente Estado

## ¿Qué cambió?

El ensemble ahora **predice el siguiente estado (t+1)** en lugar de solo clasificar el estado actual (t). Esto es **mucho más valioso** para sistemas de alerta y mantenimiento predictivo.

## 📊 Comparación: Estado Actual vs. Siguiente Estado

### Antes (Estado Actual):
```
[2025-04-23 00:25:00] → Estado: NORMAL
```
**Problema**: Indica lo que YA PASÓ. No hay tiempo para reaccionar.

### Ahora (Estado Siguiente):
```
[2025-04-23 00:25:00] → Estado Actual: NORMAL
                     → Estado Siguiente: CUELGUE ⚠️
```
**Beneficio**: Detecta cambios ANTES de que ocurran. Tiempo para actuar.

## 🎯 Características Principales

### 1. **Predicción Dual**
- `prediccion_actual`: Lo que está pasando ahora (t)
- `prediccion_siguiente`: Lo que pasará después (t+1)

### 2. **Alertas Preventivas**
```
alerta_preventiva = True si:
  - Estado actual es Normal (0)
  - Estado siguiente es Anomalía (1) o Cuelgue (2)
```

Esto permite reaccionar ANTES de que ocurra el problema.

### 3. **Probabilidades Dobles**
- Probabilidades actuales (para entender el presente)
- Probabilidades futuras (para prepararse)

## 📋 Salida del CSV

Cada fila ahora contiene:

```
tiempo,
prediccion_actual,
clase_actual,
prob_normal_actual,
prob_anomalia_voltaje_actual,
prob_cuelgue_actual,
confianza_actual,
prediccion_siguiente,                ← NUEVO: Predicción futura
clase_siguiente,                      ← NUEVO: Nombre de la predicción futura
prob_normal_siguiente,                ← NUEVO: Probabilidades futuras
prob_anomalia_voltaje_siguiente,      ← NUEVO: Probabilidades futuras
prob_cuelgue_siguiente,               ← NUEVO: Probabilidades futuras
confianza_siguiente,                  ← NUEVO: Confianza en la predicción futura
alerta_preventiva,                    ← NUEVO: Hay cambio de estado
status
```

## 🔧 Cómo Usar en Código

### Script de Predicción (usar_ensemble.py)
```python
results = detector.predict_next_state(df, forecast_minutes=1)

# Acceso a predicciones
predictions_current = results["predictions_current"]     # Estado actual
predictions_future = results["predictions_future"]       # Estado siguiente
alerta_preventiva = results["alerta_preventiva"]         # ¿Hay cambio?
n_alertas = results["n_alertas"]                         # Cuántas alertas

# En predicción en vivo
predicciones_actuales = results["predictions_current"]
predicciones_futuras = results["predictions_future"]     # Lo más importante
alertas_preventivas = results["alerta_preventiva"]
```

### En predict_realtime.py
```python
# El predictor ahora retorna ambos estados
resultado = predictor.predict_single(voltage_data)

print(f"Ahora:    {resultado['clase_actual']}")
print(f"Después:  {resultado['clase_siguiente']}")  # ← IMPORTANTE
print(f"Alerta:   {resultado['alerta_preventiva']}")
```

## 📈 Ejemplo de Salida Real

```
4️⃣  Estadísticas de predicciones FUTURAS (siguiente estado):
   Normal (0):            21960 (79.0%)
   Anomalía Voltaje (1):   5765 (20.7%)
   Cuelgue Sistema (2):      59 (0.2%)

5️⃣  Alertas Preventivas: 59 (0.2%)
   ⚠️  Cambios detectados de estado normal → anomalía/cuelgue

7️⃣  Top 10 alertas preventivas (cambios anticipados):
   1. [2025-04-23 00:25:00]
      Estado actual: Normal (conf: 1.0000)
      → Siguiente: Cuelgue Sistema (conf: 1.0000)  ← ALERTA CRÍTICA
```

## 🚨 Casos de Uso

### 1. **Mantenimiento Predictivo**
- Predecir cuelgues antes de que ocurran
- Programar mantenimiento con anticipación

### 2. **Sistemas de Alerta Temprana**
- Alertar cuando se detecte cambio hacia anomalía
- Permitir intervención preventiva

### 3. **Análisis de Tendencias**
- Ver si normal → anomalía → cuelgue
- Entender patrones de degradación

## ⚙️ Parámetro: forecast_minutes

```python
# Predecir qué pasará en los próximos N minutos
results = detector.predict_next_state(df, forecast_minutes=1)   # +1 minuto
results = detector.predict_next_state(df, forecast_minutes=5)   # +5 minutos
results = detector.predict_next_state(df, forecast_minutes=10)  # +10 minutos
```

Valores mayores permiten más tiempo para reaccionar pero pueden ser menos precisos.

## 🔄 Métodos Disponibles

### En EnsembleAnomalyDetector

```python
# Predicción del estado ACTUAL
results = detector.predict(df)
# Retorna: predictions, probabilities, lstm_probs, hang_labels, method

# Predicción del SIGUIENTE estado (NUEVO)
results = detector.predict_next_state(df, forecast_minutes=1)
# Retorna: predictions_current, predictions_future, probabilities_current,
#          probabilities_future, alerta_preventiva, n_alertas, forecast_minutes
```

## 📊 Métricas Importantes

**Antes**:
- Precisión: ¿Acertamos en la clasificación actual?

**Ahora**:
- Precisión actual: ¿Acertamos en el estado actual?
- Precisión anticipada: ¿Acertamos en predecir el siguiente?
- Lead time: ¿Cuánto tiempo antes detectamos el problema?
- True positive rate: ¿Cuántas anomalías futuras detectamos?

## 🎯 Próximos Pasos Recomendados

1. **Validar precisión de predicción futura** con datos históricos
2. **Ajustar forecast_minutes** según el sistema (1, 2, 5, 10 minutos)
3. **Integrar alertas preventivas** en el dashboard
4. **Medir impacto** del tiempo de reacción ganado
