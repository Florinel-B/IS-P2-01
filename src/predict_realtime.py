"""
Script de predicción en tiempo real/elemento a elemento
Para integración con página web y sistemas de monitoreo
"""

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
    
    def __init__(self, lstm_model_path: str = "modelo_anomalias_finetuned.pth"):
        """
        Inicializa el predictor.
        
        Args:
            lstm_model_path: Path al modelo LSTM
        """
        self.detector = EnsembleAnomalyDetector(lstm_model_path)
        
        # Buffer para secuencias (necesario para LSTM)
        self.seq_len = self.detector.seq_len
        self.buffer = []
        self.scaler = self.detector.scaler
        
        print(f"✓ Predictor inicializado")
        print(f"   - LSTM threshold: {self.detector.lstm_threshold}")
        print(f"   - Sequence length: {self.seq_len}")
        print(f"   - Modelo completo (RF + LSTM): ✓")
    
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
            first_row = df_buffer.iloc[0].to_dict() if len(df_buffer) > 0 else {col: 0 for col in ["voltageReceiver1", "voltageReceiver2", "status"]}
            padding_rows = [first_row] * (self.seq_len - len(df_buffer))
            df_padding = pd.DataFrame(padding_rows)
            df_buffer = pd.concat([df_padding, df_buffer], ignore_index=True)

        result = self.detector.predict(df_buffer)

        # Tomar la predicción del último elemento (el que acabamos de agregar)
        idx = len(df_buffer) - 1
        prediction = int(result["predictions"][idx])

        probs_row = result["probabilities"][idx]
        probs = probs_row.tolist() if hasattr(probs_row, "tolist") else list(probs_row)

        hang_label = int(result["hang_labels"][idx]) if len(result["hang_labels"]) > idx else 0
        lstm_prob = float(result["lstm_probs"][idx]) if len(result["lstm_probs"]) > idx else 0.0
        lstm_pred = 1 if lstm_prob >= self.detector.lstm_threshold else 0
        confianza = float(max(probs)) if len(probs) else 0.0
        
        # Mapear a nombres
        class_names = {
            0: "Normal",
            1: "Anomalía Voltaje (+0.5V)",
            2: "Cuelgue Sistema"
        }
        
        return {
            "prediccion": prediction,
            "clase": class_names.get(prediction, "Unknown"),
            "confianza": confianza,
            "prob_normal": probs[0],
            "prob_anomalia_voltaje": probs[1],
            "prob_cuelgue": probs[2],
            "lstm_probabilidad": lstm_prob,
            "lstm_prediccion": lstm_pred,
            "cuelgue_detectado": int(hang_label),
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
