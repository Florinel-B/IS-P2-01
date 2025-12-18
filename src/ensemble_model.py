"""
Ensemble Model: LSTM Anomalías + Random Forest Cuelgues
Predice 3 clases: 0=Normal, 1=Anomalía Voltaje (+0.5V), 2=Cuelgue Sistema (2 min)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
from training_template import (
    VoltageDropDataset,
    importar_modelo_portable,
    DEVICE
)


class EnsembleAnomalyDetector:
    """
    Detector híbrido que combina:
    - LSTM para anomalías de voltaje
    - Detección de cuelgues basada en status
    - Random Forest para fusión y predicción multiclase
    """

    def __init__(self, lstm_model_path: str, rf_model_path: Optional[str] = None):
        """
        Args:
            lstm_model_path: Path al modelo LSTM (modelo_anomalias.pth o finetuned)
            rf_model_path: Path al Random Forest (opcional, se entrena si no existe)
        """
        self.lstm_model_path = lstm_model_path
        self.rf_model_path = rf_model_path or "modelo_ensemble_rf.pkl"

        # Cargar modelo LSTM
        print(f"📊 Cargando modelo LSTM desde {lstm_model_path}...")
        self.lstm_dict = importar_modelo_portable(lstm_model_path, device=DEVICE)
        self.lstm_model = self.lstm_dict["model"].to(DEVICE)
        self.lstm_threshold = self.lstm_dict.get("threshold", 0.5)
        self.scaler = self.lstm_dict["scaler"]
        self.seq_len = self.lstm_dict.get("seq_len", 60)

        # Random Forest (será cargado - requerido)
        self.rf_model: RandomForestClassifier = None  # type: ignore
        self.rf_scaler: StandardScaler = None  # type: ignore

        print(f"   ✓ LSTM threshold: {self.lstm_threshold:.2f}")
        print(f"   ✓ Sequence length: {self.seq_len}")
        
        # Intentar cargar RF (requerido para usar el modelo completo)
        if not self.load_random_forest():
            raise RuntimeError(f"❌ CRÍTICO: No se puede cargar el Random Forest desde {self.rf_model_path}. "
                             f"Se requiere el modelo RF entrenado para usar solo el modelo completo (sin heurística).")

    def detect_hangs(self, df: pd.DataFrame, hang_duration_minutes: int = 2) -> np.ndarray:
        """
        Detecta cuelgues del sistema (status=0 durante N minutos consecutivos).

        Args:
            df: DataFrame con columnas ['tiempo', 'status', ...]
            hang_duration_minutes: Duración mínima para considerar cuelgue

        Returns:
            Array binario [0=normal, 1=cuelgue] para cada timestamp
        """
        hang_labels = np.zeros(len(df), dtype=int)

        # Resetear index para garantizar alineación
        df = df.reset_index(drop=True).copy()

        # Asegurar columna tiempo para cálculo de duración
        if "tiempo" not in df.columns:
            df["tiempo"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="T")

        # Detectar periodos de cuelgue
        current_hang_start_idx: Optional[int] = None
        current_hang_start_time: Optional[pd.Timestamp] = None

        for idx, (_, row) in enumerate(df.iterrows()):
            status = row["status"]

            if status != 1:  # Sistema no está "1" (normal)
                if current_hang_start_idx is None:
                    current_hang_start_idx = idx
                    current_hang_start_time = row["tiempo"]
            else:
                if current_hang_start_idx is not None:
                    duration = (row["tiempo"] - current_hang_start_time).total_seconds() / 60
                    if duration >= hang_duration_minutes:
                        # Marcar los timestamps del cuelgue (entre current_hang_start_idx e idx)
                        for j in range(current_hang_start_idx, idx):
                            if df.iloc[j]["status"] != 1:
                                hang_labels[j] = 1
                    current_hang_start_idx = None
                    current_hang_start_time = None

        # Verificar si el último periodo es un cuelgue
        if current_hang_start_idx is not None:
            duration = (df.iloc[-1]["tiempo"] - current_hang_start_time).total_seconds() / 60
            if duration >= hang_duration_minutes:
                for j in range(current_hang_start_idx, len(df)):
                    if df.iloc[j]["status"] != 1:
                        hang_labels[j] = 1

        return hang_labels

    def extract_lstm_features(self, df: pd.DataFrame, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae predicciones del LSTM como features.

        Returns:
            (probabilities, predictions): Arrays de probabilidades y predicciones binarias
        """
        subset = df if limit is None else df.iloc[-limit:]
        dataset = VoltageDropDataset(subset, seq_len=self.seq_len, scaler=self.scaler)

        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=0)

        self.lstm_model.eval()
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                probs, _ = self.lstm_model.predict(X_batch)
                preds = (probs >= self.lstm_threshold).float()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())

        return np.array(all_probs), np.array(all_preds)

    def create_ensemble_features(
        self,
        df: pd.DataFrame,
        lstm_probs: np.ndarray,
        hang_labels: np.ndarray
    ) -> np.ndarray:
        """
        Crea matrix de features para el Random Forest combinando:
        - Predicciones LSTM (probabilidad, predicción)
        - Labels de cuelgues
        - Features estadísticas de voltaje
        - Status
        """
        n_samples = len(df)
        
        # Verificar que todos tienen el mismo tamaño
        assert len(lstm_probs) == n_samples, f"lstm_probs size mismatch: {len(lstm_probs)} vs {n_samples}"
        assert len(hang_labels) == n_samples, f"hang_labels size mismatch: {len(hang_labels)} vs {n_samples}"
        
        features_list = []

        # 1. LSTM probability and prediction
        features_list.append(lstm_probs.reshape(-1, 1))
        features_list.append((lstm_probs >= self.lstm_threshold).astype(float).reshape(-1, 1))

        # 2. Hang detection
        features_list.append(hang_labels.astype(float).reshape(-1, 1))

        # 3. Status column (si existe)
        if "status" in df.columns:
            features_list.append(df["status"].values.astype(float).reshape(-1, 1))

        X = np.hstack(features_list)
        return X

    def train_random_forest(
        self,
        df: pd.DataFrame,
        lstm_probs: np.ndarray,
        hang_labels: np.ndarray,
        target_labels: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 15
    ) -> None:
        """
        Entrena el Random Forest con los features del ensemble.

        Args:
            target_labels: Array con valores 0 (normal), 1 (anomalía voltaje), 2 (cuelgue)
        """
        print("\n🌳 Entrenando Random Forest Ensemble...")

        # Ajustar tamaños: LSTM produce seq_len - 1 muestras menos
        n_df = len(df)
        n_lstm = len(lstm_probs)
        offset = n_df - n_lstm

        if offset > 0:
            # Padding al inicio para alinear
            lstm_probs = np.concatenate([np.zeros(offset), lstm_probs])
            # hang_labels ya tiene el tamaño correcto, solo reordenar
            hang_labels = hang_labels[-n_df:]  # Tomar últimas n_df

        # Crear features
        X = self.create_ensemble_features(df, lstm_probs, hang_labels)

        # Normalizar features
        self.rf_scaler = StandardScaler()
        X_scaled = self.rf_scaler.fit_transform(X)

        # Entrenar RF
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"  # Manejar desbalance de clases
        )

        self.rf_model.fit(X_scaled, target_labels)

        # Guardar modelo
        joblib.dump(self.rf_model, self.rf_model_path)
        joblib.dump(self.rf_scaler, self.rf_model_path.replace(".pkl", "_scaler.pkl"))

        print(f"   ✓ Modelo guardado: {self.rf_model_path}")

        # Importancia de features
        feature_names = [
            "LSTM_prob",
            "LSTM_pred",
            "Hang_label",
            "Status"
        ]
        print("\n   Feature importance:")
        for name, importance in zip(feature_names, self.rf_model.feature_importances_):
            print(f"      {name:20s}: {importance:.4f}")

    def load_random_forest(self) -> bool:
        """Carga el Random Forest entrenado."""
        try:
            self.rf_model = joblib.load(self.rf_model_path)
            scaler_path = self.rf_model_path.replace(".pkl", "_scaler.pkl")
            self.rf_scaler = joblib.load(scaler_path)
            print(f"✓ Random Forest cargado desde {self.rf_model_path}")
            return True
        except FileNotFoundError:
            print(f"⚠️  No se encontró Random Forest. Será necesario entrenarlo primero.")
            return False

    def predict(
        self,
        df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Realiza predicción multiclase (0/1/2) usando el modelo completo (LSTM + RF).
        Funciona incluso con DataFrames pequeños (< seq_len).

        Args:
            df: DataFrame con datos

        Returns:
            Dict con:
                - 'predictions': Array de predicciones (0/1/2)
                - 'probabilities': Matriz de probabilidades
                - 'lstm_probs': Probabilidades del LSTM
                - 'hang_labels': Labels de cuelgues
        """
        # Extraer features LSTM
        lstm_probs, lstm_preds = self.extract_lstm_features(df)
        print(f"   ✓ LSTM: {lstm_preds.sum():.0f} anomalías detectadas")

        # Detectar cuelgues
        hang_labels = self.detect_hangs(df)
        print(f"   ✓ Cuelgues: {hang_labels.sum():.0f} periodos detectados")

        # Ajustar tamaños: LSTM produce seq_len - 1 muestras menos
        # Padding: repetir último valor para alinear
        n_df = len(df)
        n_lstm = len(lstm_probs)
        offset = n_df - n_lstm

        if offset > 0:
            # Padding al inicio para alinear
            lstm_probs_padded = np.concatenate([np.zeros(offset), lstm_probs])
            lstm_preds_padded = np.concatenate([np.zeros(offset), lstm_preds])
            lstm_probs = lstm_probs_padded
            lstm_preds = lstm_preds_padded

        # Usar Random Forest (modelo completo - sin heurística)
        print("   ✓ Usando Random Forest Ensemble")
        X = self.create_ensemble_features(df, lstm_probs, hang_labels)
        X_scaled = self.rf_scaler.transform(X)

        predictions = self.rf_model.predict(X_scaled)
        probabilities_rf = self.rf_model.predict_proba(X_scaled)

        # Asegurar que probabilities tenga 3 columnas (0, 1, 2)
        # El RF puede predecir solo 2 clases si no hay ejemplos de la 3ª
        if probabilities_rf.shape[1] < 3:
            probabilities = np.zeros((len(df), 3))
            for i, class_id in enumerate(self.rf_model.classes_):
                probabilities[:, class_id] = probabilities_rf[:, i]
        else:
            probabilities = probabilities_rf

        print(f"   Resultados: {np.sum(predictions == 0)} normal, "
              f"{np.sum(predictions == 1)} anomalía voltaje, "
              f"{np.sum(predictions == 2)} cuelgues")

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "lstm_probs": lstm_probs,
            "hang_labels": hang_labels
        }

    def get_classification_names(self, predictions: np.ndarray) -> List[str]:
        """Convierte predicciones numéricas a nombres legibles."""
        class_names = {
            0: "Normal",
            1: "Anomalía Voltaje (+0.5V)",
            2: "Cuelgue Sistema"
        }
        return [class_names.get(int(p), "Unknown") for p in predictions]


def create_target_labels(df: pd.DataFrame, hang_duration_minutes: int = 2) -> np.ndarray:
    """
    Crea labels de verdad fundamental para entrenamiento del RF.
    Combina:
    - Anomalía voltaje: aumentos >0.5V
    - Cuelgues: status != 1 durante N minutos
    """
    target = np.zeros(len(df), dtype=int)

    # Detectar cuelgues
    detector = EnsembleAnomalyDetector("modelo_anomalias.pth")  # Dummy instance
    hang_labels = detector.detect_hangs(df, hang_duration_minutes)

    # Detectar anomalías de voltaje (cambios grandes)
    voltage_cols = [col for col in df.columns if "voltage" in col.lower()]
    if voltage_cols:
        for i in range(1, len(df)):
            prev_voltages = df.iloc[i - 1][voltage_cols].values
            curr_voltages = df.iloc[i][voltage_cols].values

            voltage_diffs = np.abs(curr_voltages - prev_voltages)
            if np.any(voltage_diffs > 0.5):
                target[i] = 1  # Anomalía voltaje

    # Cuelgues sobrescriben (prioridad)
    target[hang_labels == 1] = 2

    return target
