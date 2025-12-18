"""
Script de uso del Ensemble para predicciones en tiempo real/batch.
Predice 0=Normal, 1=Anomalía Voltaje, 2=Cuelgue Sistema
"""

import torch
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List
from ensemble_model import EnsembleAnomalyDetector


def predict_multiclass(
    data_path: str = "datos_procesados.pkl",
    lstm_model_path: str = "modelo_anomalias_finetuned.pth",
    use_ensemble: bool = True,
    output_csv: str = "predicciones_ensemble.csv"
) -> Dict:
    """
    Realiza predicción multiclase sobre los datos.

    Args:
        data_path: Path a datos procesados (pickle)
        lstm_model_path: Path al modelo LSTM
        use_ensemble: Si True, usa RF; si False, usa heurística
        output_csv: Archivo de salida CSV

    Returns:
        Dict con predicciones y análisis
    """
    print("="*70)
    print("PREDICCIÓN CON ENSEMBLE MULTICLASE")
    print("="*70)

    # Cargar datos
    print("\n1️⃣  Cargando datos...")
    with open(data_path, "rb") as f:
        datos_lista = pickle.load(f)

    df = pd.DataFrame(datos_lista)
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df = df.sort_values("tiempo").reset_index(drop=True)
    print(f"   ✓ {len(df)} muestras")

    # Inicializar detector
    print("\n2️⃣  Inicializando detector ensemble...")
    detector = EnsembleAnomalyDetector(lstm_model_path=lstm_model_path)

    # Cargar Random Forest si existe
    if use_ensemble:
        if not detector.load_random_forest():
            print("   ⚠️  RF no encontrado, usando heurística")
            use_ensemble = False

    # Realizar predicciones
    print("\n3️⃣  Realizando predicciones...")
    results = detector.predict(df, use_lstm_only=not use_ensemble)

    predictions = results["predictions"]
    probabilities = results["probabilities"]
    lstm_probs = results["lstm_probs"]
    hang_labels = results["hang_labels"]

    # Crear DataFrame de resultados
    output_df = df[["tiempo"]].copy()
    output_df["prediccion"] = predictions
    output_df["clase"] = detector.get_classification_names(predictions)

    # Agregar probabilidades por clase
    # Asegurar que probabilities tiene shape (n, 3)
    assert probabilities.shape[1] == 3, f"Probabilities debe tener 3 columnas, tiene {probabilities.shape[1]}"
    
    output_df["prob_normal"] = probabilities[:, 0]
    output_df["prob_anomalia_voltaje"] = probabilities[:, 1]
    output_df["prob_cuelgue"] = probabilities[:, 2]

    # Confianza (máx probabilidad)
    output_df["confianza"] = np.max(probabilities, axis=1)

    # Features adicionales
    output_df["lstm_probabilidad"] = lstm_probs
    output_df["cuelgue_detectado"] = hang_labels

    if "status" in df.columns:
        output_df["status"] = df["status"].values

    # Guardar CSV
    output_df.to_csv(output_csv, index=False)
    print(f"\n   ✓ Predicciones guardadas en {output_csv}")

    # Estadísticas
    print("\n4️⃣  Estadísticas de predicciones:")
    print(f"   Normal (0):            {np.sum(predictions == 0):5d} ({np.sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"   Anomalía Voltaje (1):  {np.sum(predictions == 1):5d} ({np.sum(predictions == 1)/len(predictions)*100:.1f}%)")
    print(f"   Cuelgue Sistema (2):   {np.sum(predictions == 2):5d} ({np.sum(predictions == 2)/len(predictions)*100:.1f}%)")

    # Confianza promedio por clase
    print("\n5️⃣  Confianza promedio por clase:")
    for class_id, class_name in enumerate(["Normal", "Anomalía Voltaje", "Cuelgue Sistema"]):
        mask = predictions == class_id
        if np.sum(mask) > 0:
            avg_conf = np.mean(probabilities[mask, class_id])
            print(f"   {class_name:20s}: {avg_conf:.4f}")

    # Top alertas de confianza
    print("\n6️⃣  Top 10 predicciones con más confianza:")
    top_indices = np.argsort(output_df["confianza"].values)[-10:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        row = output_df.iloc[idx]
        print(f"   {rank}. [{row['tiempo']}] {row['clase']:20s} "
              f"(confianza: {row['confianza']:.4f})")

    # Alertas críticas (cuelgues)
    cuelgue_mask = predictions == 2
    if np.sum(cuelgue_mask) > 0:
        print(f"\n⚠️  ALERTAS DE CUELGUE ({np.sum(cuelgue_mask)} registros):")
        cuelgue_times = output_df[cuelgue_mask]["tiempo"].values
        print(f"   Primero: {cuelgue_times[0]}")
        print(f"   Último: {cuelgue_times[-1]}")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "output_df": output_df,
        "method": results["method"]
    }


def predict_live(
    new_data: pd.DataFrame,
    lstm_model_path: str = "modelo_anomalias_finetuned.pth",
    use_ensemble: bool = True
) -> Dict:
    """
    Predicción en vivo para nuevos datos.

    Args:
        new_data: DataFrame con nuevas medidas
        lstm_model_path: Path al modelo LSTM
        use_ensemble: Si True, usa RF

    Returns:
        Predicciones para los nuevos datos
    """
    detector = EnsembleAnomalyDetector(lstm_model_path=lstm_model_path)

    if use_ensemble:
        detector.load_random_forest()

    results = detector.predict(new_data, use_lstm_only=not use_ensemble)

    output = {
        "predicciones": detector.get_classification_names(results["predictions"]),
        "clases_num": results["predictions"].tolist(),
        "confianzas": np.max(results["probabilities"], axis=1).tolist(),
        "probabilidades": {
            "normal": results["probabilities"][:, 0].tolist(),
            "anomalia_voltaje": results["probabilities"][:, 1].tolist(),
            "cuelgue": results["probabilities"][:, 2].tolist()
        }
    }

    return output


if __name__ == "__main__":
    # Predicción completa
    results = predict_multiclass(
        data_path="datos_procesados.pkl",
        lstm_model_path="modelo_anomalias_finetuned.pth",
        use_ensemble=True,
        output_csv="predicciones_ensemble.csv"
    )

    print("\n" + "="*70)
    print("✅ PREDICCIÓN COMPLETADA")
    print("="*70)
