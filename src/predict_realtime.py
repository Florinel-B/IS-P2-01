"""
Script de predicción en tiempo real/elemento a elemento
Para integración con página web y sistemas de monitoreo
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from ensemble_model import EnsembleAnomalyDetector


class RealtimePredictor:
    """
    Predictor en tiempo real que procesa datos elemento a elemento.
    Diseñado para integración con aplicaciones web y sistemas de monitoreo.
    """
    
    def __init__(self, lstm_model_path: str = "modelo_anomalias_finetuned.pth", 
                 model_completo_path: Optional[str] = "modelo_ensemble_completo.pkl"):
        """
        Inicializa el predictor.
        
        Args:
            lstm_model_path: Path al modelo LSTM
            model_completo_path: Path al modelo ensemble completo (si existe, tiene prioridad)
        """
        # Intentar cargar modelo completo primero
        if model_completo_path and os.path.exists(model_completo_path):
            print(f"📦 Cargando modelo completo desde {model_completo_path}...")
            self.detector = EnsembleAnomalyDetector.load_complete_model(model_completo_path)
            rf_status = "✓ (Completo con RF)"
        else:
            # Fallback: cargar solo LSTM con heurística
            print(f"📊 Cargando modelo LSTM desde {lstm_model_path}...")
            self.detector = EnsembleAnomalyDetector(lstm_model_path, require_rf=False)
            rf_status = "⚠️  (Heurística LSTM)"
        
        # Buffer para secuencias (necesario para LSTM)
        self.seq_len = self.detector.seq_len
        self.buffer = []
        self.scaler = self.detector.scaler
        
        print(f"✓ Predictor inicializado")
        print(f"   - LSTM threshold: {self.detector.lstm_threshold}")
        print(f"   - Sequence length: {self.seq_len}")
        print(f"   - Modelo ensemble: {rf_status}")
    
    def predict_single(
        self,
        voltage_data: Dict[str, float],
        status: int = 1,
        tiempo: Optional[str] = None
    ) -> Dict:
        """
        Predice para un único elemento/timestamp.
        SIEMPRE usa el modelo, rellenando con padding si buffer < seq_len.
        
        Args:
            voltage_data: Dict con las medidas de voltaje
                Ejemplo: {"voltageReceiver1": 1765, "voltageReceiver2": 1588, ...}
            status: Status del sistema (1=normal, 0=desconectado)
        
        Returns:
            Dict con predicción, clase, probabilidades y confianza
        """
        # Agregar al buffer
        row = {**voltage_data, "status": status}

        if tiempo is not None:
            try:
                row["tiempo"] = pd.to_datetime(tiempo)
            except Exception:
                row["tiempo"] = pd.Timestamp.utcnow()
        else:
            row["tiempo"] = pd.Timestamp.utcnow()
        self.buffer.append(row)
        
        # Mantener buffer de tamaño seq_len
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        
        # SIEMPRE predecir: si buffer < seq_len, rellenar con el primer elemento (padding)
        df_buffer = pd.DataFrame(self.buffer).copy()
        
        # Rellenar con padding si es necesario
        if len(df_buffer) < self.seq_len:
            first_row = df_buffer.iloc[0].to_dict() if len(df_buffer) > 0 else {col: 0 for col in ["R1_a", "R2_a", "R1_b", "R2_b", "status"]}
            padding_rows = [first_row] * (self.seq_len - len(df_buffer))
            df_padding = pd.DataFrame(padding_rows)
            df_buffer = pd.concat([df_padding, df_buffer], ignore_index=True)

        # Usar predicción anticipada
        result = self.detector.predict_next_state(df_buffer, forecast_minutes=1)

        # Tomar la predicción del último elemento (el que acabamos de agregar)
        idx = len(df_buffer) - 1
        
        # Predicción actual y futura
        prediction_actual = int(result["predictions_current"][idx])
        prediction_siguiente = int(result["predictions_future"][idx])
        
        probs_actual = result["probabilities_current"][idx]
        probs_siguiente = result["probabilities_future"][idx]
        probs_actual_list = probs_actual.tolist() if hasattr(probs_actual, "tolist") else list(probs_actual)
        probs_siguiente_list = probs_siguiente.tolist() if hasattr(probs_siguiente, "tolist") else list(probs_siguiente)
        
        alerta_preventiva = bool(result["alerta_preventiva"][idx])
        
        confianza_actual = float(max(probs_actual_list)) if len(probs_actual_list) else 0.0
        confianza_siguiente = float(max(probs_siguiente_list)) if len(probs_siguiente_list) else 0.0
        
        # Mapear a nombres
        class_names = {
            0: "Normal",
            1: "Anomalía Voltaje (+0.5V)",
            2: "Cuelgue Sistema"
        }
        
        return {
            # ACTUAL
            "prediccion_actual": prediction_actual,
            "clase_actual": class_names.get(prediction_actual, "Unknown"),
            "confianza_actual": confianza_actual,
            "prob_normal_actual": probs_actual_list[0],
            "prob_anomalia_voltaje_actual": probs_actual_list[1],
            "prob_cuelgue_actual": probs_actual_list[2],
            
            # SIGUIENTE (MÁS IMPORTANTE)
            "prediccion_siguiente": prediction_siguiente,
            "clase_siguiente": class_names.get(prediction_siguiente, "Unknown"),
            "confianza_siguiente": confianza_siguiente,
            "prob_normal_siguiente": probs_siguiente_list[0],
            "prob_anomalia_voltaje_siguiente": probs_siguiente_list[1],
            "prob_cuelgue_siguiente": probs_siguiente_list[2],
            
            # ALERTA
            "alerta_preventiva": alerta_preventiva,
            "status": status,
            "buffer_size": len(self.buffer)
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice un lote de datos.
        
        Args:
            df: DataFrame con columnas de voltaje y status
        
        Returns:
            DataFrame con predicciones
        """
        results = []
        
        for idx, row in df.iterrows():
            voltage_data = {col: row[col] for col in df.columns 
                           if "voltage" in col.lower()}
            status = int(row.get("status", 1))
            
            tiempo_val = row.get("tiempo")
            pred = self.predict_single(voltage_data, status, tiempo_val)
            results.append(pred)
        
        return pd.DataFrame(results)
    
    def reset_buffer(self):
        """Limpia el buffer (para iniciar nueva sesión)."""
        self.buffer = []
        print("✓ Buffer limpiado")


def predict_from_csv(
    csv_path: str,
    output_path: str = "predicciones_realtime.csv"
) -> None:
    """
    Realiza predicciones elemento a elemento desde un CSV.
    
    Args:
        csv_path: Path al CSV con datos
        output_path: Path donde guardar predicciones
    """
    print("="*70)
    print("PREDICCIÓN EN TIEMPO REAL (CSV)")
    print("="*70)
    
    # Cargar datos
    print(f"\n1️⃣  Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   ✓ {len(df)} registros")
    
    # Inicializar predictor
    print("\n2️⃣  Inicializando predictor...")
    predictor = RealtimePredictor()
    
    # Predicción elemento a elemento
    print(f"\n3️⃣  Prediciendo elemento a elemento...")
    predictions = []
    
    for idx_int, (_, row) in enumerate(df.iterrows()):
        if (idx_int + 1) % 100 == 0:
            print(f"   Procesados: {idx_int + 1}/{len(df)}")
        
        # Extraer voltajes
        voltage_data = {col: row[col] for col in df.columns 
                       if "voltage" in col.lower()}
        status = int(row.get("status", 1))
        tiempo = row.get("tiempo", f"record_{idx_int}")
        
        # Predecir
        tiempo_val = row.get("tiempo")
        pred = predictor.predict_single(voltage_data, status, tiempo_val)
        pred["tiempo"] = tiempo
        pred["indice"] = idx_int
        predictions.append(pred)
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(predictions)
    
    # Guardar
    results_df.to_csv(output_path, index=False)
    print(f"\n   ✓ Predicciones guardadas en {output_path}")
    
    # Estadísticas
    print("\n4️⃣  Estadísticas:")
    for pred_class in [0, 1, 2]:
        count = (results_df["prediccion"] == pred_class).sum()
        pct = count / len(results_df) * 100
        class_name = {0: "Normal", 1: "Anomalía", 2: "Cuelgue"}.get(pred_class)
        print(f"   {class_name:20s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n   Confianza promedio: {results_df['confianza'].mean():.4f}")


def predict_api_endpoint(
    new_measurements: List[Dict],
    lstm_model_path: str = "modelo_anomalias_finetuned.pth"
) -> List[Dict]:
    """
    Función para usar como endpoint de API.
    Predice para una lista de nuevas medidas.
    
    Args:
        new_measurements: Lista de dicts con medidas
            Ejemplo: [
                {"voltageReceiver1": 1765, "voltageReceiver2": 1588, "status": 1},
                {"voltageReceiver1": 1770, "voltageReceiver2": 1590, "status": 1},
                ...
            ]
        lstm_model_path: Path al modelo LSTM
    
    Returns:
        Lista de predicciones (una por medida)
    """
    predictor = RealtimePredictor(lstm_model_path)
    predictions = []
    
    for measurement in new_measurements:
        status = measurement.pop("status", 1)
        pred = predictor.predict_single(measurement, status)
        predictions.append(pred)
    
    return predictions


# Para usar en Flask
def setup_flask_predictor(app, lstm_model_path: str = "modelo_anomalias_finetuned.pth"):
    """
    Configura predictor para usar en Flask.
    
    Uso en app.py:
        from src.predict_realtime import setup_flask_predictor
        
        app = Flask(__name__)
        setup_flask_predictor(app)
        
        @app.route('/api/predict', methods=['POST'])
        def predict():
            data = request.json
            results = app.realtime_predictor.predict_batch(pd.DataFrame(data))
            return results.to_dict(orient='records')
    """
    app.realtime_predictor = RealtimePredictor(lstm_model_path)
    print("✓ Predictor Flask configurado")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "predicciones_realtime.csv"
        predict_from_csv(csv_path, output_path)
    else:
        print("Uso: python predict_realtime.py <csv_path> [output_path]")
        print("\nEjemplo:")
        print("  python predict_realtime.py datos.csv predicciones.csv")
