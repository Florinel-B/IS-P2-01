"""
Ejemplo de integración del predictor en tiempo real con Flask
Para tu aplicación web existente
"""

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from src.predict_realtime import RealtimePredictor

# Crear blueprint para las rutas de predicción
prediction_routes = Blueprint('predictions', __name__, url_prefix='/api/predictions')

# Inicializar predictor global
_realtime_predictor = None

def init_predictor(lstm_model_path: str = "modelo_anomalias_finetuned.pth"):
    """Inicializa el predictor global."""
    global _realtime_predictor
    _realtime_predictor = RealtimePredictor(lstm_model_path)
    print("✓ Predictor en tiempo real inicializado")


# ========================
# ENDPOINTS DE PREDICCIÓN
# ========================

@prediction_routes.route('/single', methods=['POST'])
def predict_single():
    """
    Predice un único elemento.
    
    POST /api/predictions/single
    {
        "voltageReceiver1": 1765,
        "voltageReceiver2": 1588,
        "voltageReceiver3": 1750,
        "voltageReceiver4": 1620,
        "status": 1
    }
    
    Response:
    {
        "prediccion": 0,
        "clase": "Normal",
        "confianza": 0.9999,
        "prob_normal": 1.0,
        "prob_anomalia_voltaje": 0.0,
        "prob_cuelgue": 0.0,
        "estado": "completado"
    }
    """
    try:
        data = request.json
        status = int(data.pop("status", 1))
        
        # Extraer voltajes
        voltage_data = data
        
        # Predecir
        result = _realtime_predictor.predict_single(voltage_data, status)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@prediction_routes.route('/batch', methods=['POST'])
def predict_batch():
    """
    Predice un lote de elementos.
    
    POST /api/predictions/batch
    {
        "data": [
            {
                "voltageReceiver1": 1765,
                "voltageReceiver2": 1588,
                "voltageReceiver3": 1750,
                "voltageReceiver4": 1620,
                "status": 1
            },
            {...},
            ...
        ]
    }
    
    Response:
    [
        {
            "prediccion": 0,
            "clase": "Normal",
            "confianza": 0.9999,
            "prob_normal": 1.0,
            "prob_anomalia_voltaje": 0.0,
            "prob_cuelgue": 0.0,
            "estado": "completado"
        },
        ...
    ]
    """
    try:
        data = request.json
        measurements = data.get("data", [])
        
        results = []
        for measurement in measurements:
            status = int(measurement.pop("status", 1))
            voltage_data = measurement
            
            result = _realtime_predictor.predict_single(voltage_data, status)
            results.append(result)
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@prediction_routes.route('/reset', methods=['POST'])
def reset_buffer():
    """
    Limpia el buffer de secuencias.
    Usar cuando comience una nueva sesión de monitoreo.
    
    POST /api/predictions/reset
    
    Response:
    {
        "status": "ok",
        "message": "Buffer limpiado"
    }
    """
    try:
        _realtime_predictor.reset_buffer()
        return jsonify({
            "status": "ok",
            "message": "Buffer limpiado"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@prediction_routes.route('/status', methods=['GET'])
def get_status():
    """
    Obtiene estado del predictor.
    
    GET /api/predictions/status
    
    Response:
    {
        "initialized": true,
        "buffer_size": 42,
        "seq_len": 60,
        "use_ensemble": true,
        "lstm_threshold": 0.5
    }
    """
    return jsonify({
        "initialized": _realtime_predictor is not None,
        "buffer_size": len(_realtime_predictor.buffer) if _realtime_predictor else 0,
        "seq_len": _realtime_predictor.seq_len if _realtime_predictor else 0,
        "use_ensemble": _realtime_predictor.use_ensemble if _realtime_predictor else False,
        "lstm_threshold": _realtime_predictor.detector.lstm_threshold if _realtime_predictor else 0.0
    }), 200


@prediction_routes.route('/health', methods=['GET'])
def health_check():
    """
    Health check del servicio de predicción.
    
    GET /api/predictions/health
    
    Response:
    {
        "status": "healthy",
        "predictor": "ready"
    }
    """
    predictor_status = "ready" if _realtime_predictor else "not_initialized"
    
    return jsonify({
        "status": "healthy",
        "predictor": predictor_status
    }), 200


# ========================
# FUNCIONES AUXILIARES
# ========================

def get_prediction_html(result: dict) -> str:
    """
    Genera HTML para mostrar predicción en página web.
    
    Args:
        result: Dict con predicción
    
    Returns:
        HTML string con la predicción formateada
    """
    clase = result.get("clase", "Unknown")
    prediccion = result.get("prediccion", -1)
    confianza = result.get("confianza", 0) * 100
    
    # Colores según clase
    color_map = {
        0: "#28a745",  # Verde - Normal
        1: "#ffc107",  # Amarillo - Anomalía
        2: "#dc3545"   # Rojo - Cuelgue
    }
    
    color = color_map.get(prediccion, "#6c757d")
    
    html = f"""
    <div style="border: 2px solid {color}; border-radius: 8px; padding: 15px; background-color: {color}15;">
        <h3 style="color: {color}; margin: 0 0 10px 0;">{clase}</h3>
        <p style="margin: 5px 0;"><strong>Predicción:</strong> Clase {prediccion}</p>
        <p style="margin: 5px 0;"><strong>Confianza:</strong> {confianza:.1f}%</p>
        <div style="margin-top: 10px;">
            <strong>Probabilidades:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Normal: {result.get("prob_normal", 0)*100:.1f}%</li>
                <li>Anomalía: {result.get("prob_anomalia_voltaje", 0)*100:.1f}%</li>
                <li>Cuelgue: {result.get("prob_cuelgue", 0)*100:.1f}%</li>
            </ul>
        </div>
    </div>
    """
    
    return html


def get_prediction_json_for_dashboard(result: dict) -> dict:
    """
    Convierte predicción a formato para dashboard.
    
    Args:
        result: Dict con predicción
    
    Returns:
        Dict formateado para dashboard
    """
    clase_map = {
        0: "normal",
        1: "anomalia",
        2: "cuelgue"
    }
    
    return {
        "type": clase_map.get(result.get("prediccion"), "unknown"),
        "confidence": round(result.get("confianza", 0) * 100, 2),
        "class_name": result.get("clase", "Unknown"),
        "timestamp": pd.Timestamp.now().isoformat(),
        "details": {
            "lstm_prob": round(result.get("lstm_probabilidad", 0), 4),
            "status": result.get("status", 1),
            "buffer_size": result.get("buffer_size", 0)
        }
    }


# ========================
# EJEMPLO DE USO EN app.py
# ========================

"""
# En tu archivo principal (app.py):

from flask import Flask
from src.api_routes_predictions import prediction_routes, init_predictor

app = Flask(__name__)

# Registrar blueprint
app.register_blueprint(prediction_routes)

# Inicializar predictor cuando arranca la app
@app.before_request
def initialize():
    global predictor_initialized
    if not hasattr(initialize, 'predictor_initialized'):
        init_predictor("modelo_anomalias_finetuned.pth")
        initialize.predictor_initialized = True

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# ENDPOINTS DISPONIBLES:
# POST /api/predictions/single          - Predecir un elemento
# POST /api/predictions/batch           - Predecir lote
# POST /api/predictions/reset           - Limpiar buffer
# GET  /api/predictions/status          - Estado del predictor
# GET  /api/predictions/health          - Health check
"""
