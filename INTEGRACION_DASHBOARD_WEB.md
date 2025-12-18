# 🚀 Integración de Predicción Anticipada en Página Web

## ¿Qué se implementó?

Se integró exitosamente la **predicción anticipada del siguiente estado** en la página web del dashboard.

### ✨ Nuevas Características en el Dashboard

1. **Panel de Predicciones Dual**
   - Estado Actual (t): Lo que está pasando ahora
   - Estado Siguiente (t+1): Lo que pasará después ⭐
   - Confianzas de ambas predicciones

2. **Alerta Preventiva Visual**
   - Parpadea en rojo cuando detecta cambios anticipados
   - Muestra: "Cambio anticipado: Estado Actual → Estado Siguiente"
   - Animación de pulso para máxima visibilidad

3. **Gráfica en Tiempo Real**
   - Predic del siguiente estado en naranja/dorado
   - Status del sistema en azul
   - Última versión del estado anticipado
   - Actualización automática cada nuevo dato

4. **Panel de Voltajes**
   - Muestra R1_a, R2_a, R1_b, R2_b en tiempo real
   - Badge de status (Normal/Desconectado)
   - Timestamp del último update

5. **Estadísticas en Vivo**
   - Contador de predicciones normales
   - Contador de anomalías detectadas
   - Contador de cuelgues predichos
   - **Contador de alertas preventivas ⭐**

6. **Centro de Notificaciones**
   - Notificaciones de incidencias
   - Alertas preventivas en naranja
   - Alertas críticas en rojo parpadeante
   - Historial scrolleable de últimas 50 eventos

---

## 🎨 Cambios en los Estilos

### Colores Nuevos:
- **Naranja/Dorado (#ff9800)**: Estado Siguiente, Alertas Preventivas
- **Gradiente Morado**: Header mejorado
- **Tema Oscuro (#1a1a2e)**: Fondo para mejor legibilidad

### Animaciones:
- **Pulse**: Alerta preventiva parpadeante
- **Alert-blink**: Notificaciones críticas
- **Gráfica animada**: Actualización suave

---

## 📱 Cambios en las Rutas

### Rutas Disponibles:

```
GET /                    → Dashboard Nuevo (con predicción anticipada)
GET /dashboard_legacy    → Dashboard Antiguo (para compatibilidad)
POST /api/predict        → Predicción con estado actual + siguiente
```

---

## 🔄 Cambios en API Routes (api_routes.py)

### Antes:
```python
predictor.predict_single()  → prediccion_actual (solo)
incidencia_tipo 1: Anomalía
incidencia_tipo 2: Cuelgue
```

### Ahora:
```python
predictor.predict_single()  → prediccion_actual + prediccion_siguiente
incidencia_tipo 1: Anomalía Predicha
incidencia_tipo 2: Cuelgue Predicho
incidencia_tipo 3: Alerta Preventiva (NUEVA) ⭐
```

### Estructura del Payload Emitido:

```python
{
    'tiempo': '2025-04-23T15:00:00',
    'status': 1,
    'R1_a': 1776.0,
    'R2_a': 1588.0,
    'R1_b': 1753.0,
    'R2_b': 1624.0,
    
    # NUEVO: Predicción Anticipada
    'prediccion_actual': 0,
    'prediccion_siguiente': 0,
    'clase_actual': 'Normal',
    'clase_siguiente': 'Normal',
    'confianza_actual': 1.0,
    'confianza_siguiente': 1.0,
    'alerta_preventiva': False,
    
    'pred': 1,  # Para gráfica
    'incidencia': None  # 1, 2, 3 o None
}
```

### Evento WebSocket para Notificaciones:

```python
socket.emit('notificacion_incidencia', {
    'tipo': 3,  # 1=Anomalía, 2=Cuelgue, 3=Preventiva
    'mensaje': '⚠️  ALERTA PREVENTIVA: Normal → Anomalía',
    'confianza': 0.95,
    'alerta_preventiva': True,  # NUEVO
    'tiempo': '2025-04-23T15:00:00'
})
```

---

## 🚀 Cómo Ejecutar

### 1. Asegurar que el modelo está entrenado:

```bash
cd /home/florin/Documentos/GitHub/IS-P2-01
.venv/bin/python src/entrenar_ensemble_completo.py
```

Debería crear:
- `modelo_ensemble_completo.pkl` ✓
- `modelo_ensemble_rf.pkl` ✓

### 2. Iniciar la aplicación Flask:

```bash
.venv/bin/python src/app.py
```

Verás:
```
 * Running on http://0.0.0.0:5000
 * Restarting with reloader
 ✓ Predictor de anomalías cargado
```

### 3. Abrir en navegador:

```
http://localhost:5000
```

Debería verse el dashboard con:
- ✅ Panel de predicciones actual + siguiente
- ✅ Gráfica en tiempo real
- ✅ Voltajes actuales
- ✅ Estadísticas
- ✅ Notificaciones

---

## 🧪 Testing

### Simular Simulación de Datos:

Abrir terminal adicional:

```bash
cd /home/florin/Documentos/GitHub/IS-P2-01
curl -X POST http://localhost:5000/api/iniciar_simulacion/DEVICE_ID \
  -H "Content-Type: application/json" \
  -d '{"speed": 10, "device_id": 7}'
```

Debería ver en el dashboard:
1. Predicciones actuales y futuras actualizándose
2. Gráfica en tiempo real
3. Voltajes cambiando
4. Alertas cuando hay cambios anticipados

---

## 📊 Ejemplo de Flujo de Alerta

### Escenario: Normal → Anomalía

**Minuto 1:**
- Estado Actual: ✅ Normal (100% confianza)
- Estado Siguiente: ⚠️ Anomalía (95% confianza)
- **ALERTA PREVENTIVA**: Activada ⚠️
- Usuario recibe notificación anticipada

**Minuto 2:**
- Estado Actual: ⚠️ Anomalía (95% confianza)
- Estado Siguiente: ⚠️ Anomalía (90% confianza)
- Alerta preventiva ya consumida, usuario ya preparado

---

## 💡 Mejoras Respecto a Versión Anterior

| Aspecto | Antes | Ahora |
|--------|-------|-------|
| **Predicción** | Solo estado actual | Estado actual + siguiente |
| **Anticipación** | Ninguna | Alertas preventivas |
| **Tiempo de reacción** | 0 minutos | 1-2 minutos (configurable) |
| **Visualización** | 1 estado | 2 estados lado a lado |
| **Alertas críticas** | 2 tipos | 3 tipos (+preventiva) |
| **Tipo de mantenimiento** | Correctivo | **Predictivo** ⭐ |

---

## 🔌 Integración con APIs Externas

Si necesitas integrar con otros sistemas:

### Endpoint JSON:

```bash
POST http://localhost:5000/api/predict
Content-Type: application/json

{
  "R1_a": 1776.0,
  "R2_a": 1588.0,
  "R1_b": 1753.0,
  "R2_b": 1624.0,
  "status": 1
}
```

### Respuesta:

```json
{
  "prediccion_actual": 0,
  "prediccion_siguiente": 1,
  "clase_actual": "Normal",
  "clase_siguiente": "Anomalía Voltaje",
  "confianza_actual": 1.0,
  "confianza_siguiente": 0.95,
  "alerta_preventiva": true
}
```

---

## 📁 Archivos Modificados/Creados

### Modificados:
- ✏️ `routes/api_routes.py` - Lógica de predicción anticipada
- ✏️ `routes/web_routes.py` - Nuevas rutas

### Creados:
- ✨ `templates/dashboard_anticipado.html` - Dashboard nuevo
- 📖 Documentación de integración

---

## ✅ Checklist de Validación

- [x] Predicción anticipada funcionando
- [x] Alertas preventivas visibles
- [x] Gráfica actualizando en tiempo real
- [x] WebSocket emitiendo datos correctos
- [x] Notificaciones mostrándose
- [x] Contador de alertas preventivas
- [x] Animaciones de pulso funcionando
- [x] Respons ivo en diferentes tamaños
- [x] Scrollbar personalizado
- [x] Timestamp en notificaciones

---

## 🎯 Próximos Pasos Opcionales

1. **Sonido de Alerta**: Agregar `new Audio('/sounds/alert.mp3').play()`
2. **Exportar Datos**: Botón para descargar CSV de predicciones
3. **Configurar Sensibilidad**: Slider para ajustar `forecast_minutes`
4. **Historial**: Tab para ver predicciones pasadas
5. **Comparación**: Ver predicción actual vs realidad

---

## 🚨 Troubleshooting

**P: La predicción siguiente no cambia**
R: Ejecutar `entrenar_ensemble_completo.py` para reentrenar el modelo

**P: No veo alertas preventivas**
R: Los datos deben cambiar significativamente. Ejecutar simulación con `speed: 100` para forzar cambios

**P: Gráfica no se actualiza**
R: Abrir consola (F12) y verificar que Socket.IO está conectado: `socket.connected === true`

**P: Las confianzas muestran NaN**
R: Recargar página (Ctrl+F5) para limpiar caché

---

## ¡Listo! 🎉

El dashboard está integrado y funcionando con predicción anticipada. 

Para iniciar: `python src/app.py` y acceder a `http://localhost:5000`
