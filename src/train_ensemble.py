"""
Script de entrenamiento del Ensemble (LSTM + RF) para predicción multiclase.
Genera labels de entrenamiento desde los datos históricos.
"""

import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from ensemble_model import (
    EnsembleAnomalyDetector,
    create_target_labels
)


def train_ensemble(
    data_pkl_path: str = "datos_procesados.pkl",
    lstm_model_path: str = "modelo_anomalias_finetuned.pth",
    hang_duration: int = 2,
    train_split: float = 0.8
) -> None:
    """
    Entrena el Random Forest del ensemble.

    Args:
        data_pkl_path: Path a los datos procesados
        lstm_model_path: Path al modelo LSTM
        hang_duration: Minutos para considerar cuelgue
        train_split: Proporción train/test
    """
    print("="*70)
    print("ENTRENAMIENTO DEL ENSEMBLE: LSTM + Random Forest")
    print("="*70)

    # Cargar datos
    print("\n1️⃣  Cargando datos...")
    with open(data_pkl_path, "rb") as f:
        datos_lista = pickle.load(f)

    df = pd.DataFrame(datos_lista)
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df = df.sort_values("tiempo").reset_index(drop=True)
    print(f"   ✓ {len(df)} muestras cargadas")

    # Dividir train/test
    split_idx = int(len(df) * train_split)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"   ✓ Train: {len(df_train)} muestras")
    print(f"   ✓ Test: {len(df_test)} muestras")

    # Crear labels de verdad fundamental
    print("\n2️⃣  Creando labels de verdad fundamental...")

    def create_targets(data_df, hang_dur, lstm_probs=None, lstm_threshold=0.5):
        """
        Crea labels combinando:
        - Cuelgues: status != 1 por hang_dur minutos
        - Anomalías: predicciones del LSTM (si disponibles) o cambios > 0.5V
        """
        targets = np.zeros(len(data_df), dtype=int)
        
        # Resetear index para garantizar alineación
        data_df = data_df.reset_index(drop=True).copy()

        # Detectar cuelgues (status != 1 por hang_dur minutos)
        if "status" in data_df.columns:
            hang_periods = []
            hang_start = None
            hang_start_idx = None

            for i, row in data_df.iterrows():
                if row["status"] != 1:
                    if hang_start is None:
                        hang_start = row["tiempo"]
                        hang_start_idx = i
                else:
                    if hang_start is not None:
                        duration_min = (row["tiempo"] - hang_start).total_seconds() / 60
                        if duration_min >= hang_dur:
                            hang_periods.append((hang_start_idx, i))
                        hang_start = None
                        hang_start_idx = None

            # Marcar periodos de cuelgue
            for start_idx, end_idx in hang_periods:
                # Marcar solo los índices donde status != 1
                for i in range(start_idx, end_idx):
                    if data_df.iloc[i]["status"] != 1:
                        targets[i] = 2

        # Detectar anomalías: usar LSTM si disponible, sino cambios > 0.5V
        if lstm_probs is not None and len(lstm_probs) > 0:
            # Usar predicciones del LSTM como verdad fundamental
            # Ajustar tamaño si es necesario
            offset = len(data_df) - len(lstm_probs)
            if offset > 0:
                lstm_probs_padded = np.concatenate([np.zeros(offset), lstm_probs])
            else:
                lstm_probs_padded = lstm_probs[:len(data_df)]
            
            for i in range(len(data_df)):
                if lstm_probs_padded[i] >= lstm_threshold and targets[i] != 2:
                    targets[i] = 1
        else:
            # Fallback: detectar cambios > 0.5V
            voltage_cols = [col for col in data_df.columns if "voltage" in col.lower()]
            if voltage_cols:
                for i in range(1, len(data_df)):
                    prev_vals = data_df.iloc[i - 1][voltage_cols].values
                    curr_vals = data_df.iloc[i][voltage_cols].values

                    if np.any(np.abs(curr_vals - prev_vals) > 0.5):
                        if targets[i] != 2:  # No sobrescribir cuelgues
                            targets[i] = 1

        return targets

    # Inicializar ensemble PRIMERO para extraer LSTM probs
    print("\n3️⃣  Inicializando detector ensemble...")
    detector = EnsembleAnomalyDetector(lstm_model_path=lstm_model_path)

    # Extraer features LSTM ANTES de crear targets
    print("\n4️⃣  Extrayendo features LSTM para crear labels...")
    lstm_probs_train, _ = detector.extract_lstm_features(df_train)
    lstm_probs_test, _ = detector.extract_lstm_features(df_test)

    print(f"   ✓ LSTM train anomalías: {np.sum(lstm_probs_train >= detector.lstm_threshold)}")
    print(f"   ✓ LSTM test anomalías: {np.sum(lstm_probs_test >= detector.lstm_threshold)}")

    # AHORA crear targets usando LSTM probs
    y_train = create_targets(df_train, hang_duration, lstm_probs=lstm_probs_train, 
                            lstm_threshold=detector.lstm_threshold)
    y_test = create_targets(df_test, hang_duration, lstm_probs=lstm_probs_test,
                           lstm_threshold=detector.lstm_threshold)

    print(f"\n   Train distribution: 0={np.sum(y_train==0)}, "
          f"1={np.sum(y_train==1)}, 2={np.sum(y_train==2)}")
    print(f"   Test distribution: 0={np.sum(y_test==0)}, "
          f"1={np.sum(y_test==1)}, 2={np.sum(y_test==2)}")

    # Detectar cuelgues
    hang_train = detector.detect_hangs(df_train, hang_duration_minutes=hang_duration)
    print(f"\n   ✓ Cuelgues detectados (train): {np.sum(hang_train)}")

    # Entrenar Random Forest
    print("\n5️⃣  Entrenando Random Forest...")
    detector.train_random_forest(
        df=df_train,
        lstm_probs=lstm_probs_train,
        hang_labels=hang_train,
        target_labels=y_train,
        n_estimators=150,
        max_depth=20
    )

    # Evaluación en test
    print("\n6️⃣  Evaluación en test set...")
    predictions_test = detector.predict(df_test)

    y_pred = predictions_test["predictions"]
    y_proba = predictions_test["probabilities"]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n   Reporte de clasificación:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomalía Voltaje", "Cuelgue Sistema"],
        labels=[0, 1, 2],
        zero_division=0,
        digits=4
    ))

    # Matriz de confusión
    print("\n   Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Métricas por clase
    print("\n   Métricas detalladas:")
    for class_id, class_name in enumerate(["Normal", "Anomalía Voltaje", "Cuelgue Sistema"]):
        mask = y_test == class_id
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"      {class_name:20s}: {acc:.4f} (n={np.sum(mask)})")

    # Guardar resultados
    print("\n7️⃣  Guardando resultados...")
    results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "class_names": ["Normal", "Anomalía Voltaje", "Cuelgue Sistema"],
        "confusion_matrix": cm
    }

    import joblib
    joblib.dump(results, "ensemble_test_results.pkl")
    print(f"   ✓ Resultados guardados en ensemble_test_results.pkl")

    print("\n" + "="*70)
    print("✅ ENSEMBLE ENTRENADO Y EVALUADO")
    print("="*70)
    print(f"\nModelos guardados:")
    print(f"   - Random Forest: modelo_ensemble_rf.pkl")
    print(f"   - Scaler: modelo_ensemble_rf_scaler.pkl")
    print(f"   - LSTM (original): {lstm_model_path}")


if __name__ == "__main__":
    train_ensemble(
        data_pkl_path="datos_procesados.pkl",
        lstm_model_path="modelo_anomalias_finetuned.pth",
        hang_duration=2,
        train_split=0.8
    )
