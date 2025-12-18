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
    Realiza predicción multiclase sobre los datos usando el modelo completo.

    Args:
        data_path: Path a datos procesados (pickle)
        lstm_model_path: Path al modelo LSTM
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
    detector = EnsembleAnomalyDetector(lstm_model_path=lstm_model_path, require_rf=False)

    # Realizar predicciones ANTICIPADAS (siguiente estado)
    print("\n3️⃣  Realizando predicciones anticipadas...")
    results = detector.predict_next_state(df, forecast_minutes=1)

    predictions_current = results["predictions_current"]
    predictions_future = results["predictions_future"]
    probabilities_current = results["probabilities_current"]
    probabilities_future = results["probabilities_future"]
    alerta_preventiva = results["alerta_preventiva"]

    # Crear DataFrame de resultados
    output_df = df[["tiempo"]].copy()
    
    # Predicción ACTUAL
    output_df["prediccion_actual"] = predictions_current
    output_df["clase_actual"] = detector.get_classification_names(predictions_current)
    output_df["prob_normal_actual"] = probabilities_current[:, 0]
    output_df["prob_anomalia_voltaje_actual"] = probabilities_current[:, 1]
    output_df["prob_cuelgue_actual"] = probabilities_current[:, 2]
    output_df["confianza_actual"] = np.max(probabilities_current, axis=1)
    
    # Predicción FUTURA (siguiente estado - MÁS IMPORTANTE)
    output_df["prediccion_siguiente"] = predictions_future
    output_df["clase_siguiente"] = detector.get_classification_names(predictions_future)
    output_df["prob_normal_siguiente"] = probabilities_future[:, 0]
    output_df["prob_anomalia_voltaje_siguiente"] = probabilities_future[:, 1]
    output_df["prob_cuelgue_siguiente"] = probabilities_future[:, 2]
    output_df["confianza_siguiente"] = np.max(probabilities_future, axis=1)
    
    # Alerta preventiva (cambio anticipado)
    output_df["alerta_preventiva"] = alerta_preventiva.astype(int)

    if "status" in df.columns:
        output_df["status"] = df["status"].values

    # Guardar CSV
    output_df.to_csv(output_csv, index=False)
    print(f"\n   ✓ Predicciones guardadas en {output_csv}")

    # Estadísticas - Predicción Futura (MÁS IMPORTANTE)
    print("\n4️⃣  Estadísticas de predicciones FUTURAS (siguiente estado):")
    print(f"   Normal (0):            {np.sum(predictions_future == 0):5d} ({np.sum(predictions_future == 0)/len(predictions_future)*100:.1f}%)")
    print(f"   Anomalía Voltaje (1):  {np.sum(predictions_future == 1):5d} ({np.sum(predictions_future == 1)/len(predictions_future)*100:.1f}%)")
    print(f"   Cuelgue Sistema (2):   {np.sum(predictions_future == 2):5d} ({np.sum(predictions_future == 2)/len(predictions_future)*100:.1f}%)")

    # Alertas preventivas
    n_alertas = np.sum(alerta_preventiva)
    print(f"\n5️⃣  Alertas Preventivas: {n_alertas} ({n_alertas/len(alerta_preventiva)*100:.1f}%)")
    print(f"   ⚠️  Cambios detectados de estado normal → anomalía/cuelgue")

    # Confianza promedio por clase (futuro)
    print("\n6️⃣  Confianza promedio por clase (predicción futura):")
    for class_id, class_name in enumerate(["Normal", "Anomalía Voltaje", "Cuelgue Sistema"]):
        mask = predictions_future == class_id
        if np.sum(mask) > 0:
            avg_conf = np.mean(probabilities_future[mask, class_id])
            print(f"   {class_name:20s}: {avg_conf:.4f}")

    # Top alertas preventivas (cambios de estado)
    print("\n7️⃣  Top 10 alertas preventivas (cambios anticipados):")
    alert_indices = np.where(alerta_preventiva)[0]
    if len(alert_indices) > 0:
        # Ordenar por confianza en la predicción futura
        confianza_alertas = probabilities_future[alert_indices, predictions_future[alert_indices]]
        top_alert_indices = alert_indices[np.argsort(confianza_alertas)[-10:][::-1]]
        
        for rank, idx in enumerate(top_alert_indices, 1):
            row = output_df.iloc[idx]
            print(f"   {rank}. [{row['tiempo']}]")
            print(f"      Estado actual: {row['clase_actual']} (conf: {row['confianza_actual']:.4f})")
            print(f"      → Siguiente: {row['clase_siguiente']} (conf: {row['confianza_siguiente']:.4f})")

    # Alertas críticas (cuelgues predichos)
    cuelgue_mask_futuro = predictions_future == 2
    if np.sum(cuelgue_mask_futuro) > 0:
        print(f"\n🚨 CUELGUES PREDICHOS ({np.sum(cuelgue_mask_futuro)} registros):")
        cuelgue_times = output_df[cuelgue_mask_futuro]["tiempo"].values
        print(f"   Primero: {cuelgue_times[0]}")
        print(f"   Último: {cuelgue_times[-1]}")

    return {
        "predictions_current": predictions_current,
        "predictions_future": predictions_future,
        "probabilities_current": probabilities_current,
        "probabilities_future": probabilities_future,
        "alerta_preventiva": alerta_preventiva,
        "output_df": output_df
    }


def predict_live(
    new_data: pd.DataFrame,
    lstm_model_path: str = "modelo_anomalias_finetuned.pth",
    use_ensemble: bool = True
) -> Dict:
    """
    Predicción en vivo ANTICIPADA para nuevos datos.
    Predice el siguiente estado, no el actual.

    Args:
        new_data: DataFrame con nuevas medidas
        lstm_model_path: Path al modelo LSTM
        use_ensemble: Si True, usa RF

    Returns:
        Predicciones anticipadas para los nuevos datos
    """
    detector = EnsembleAnomalyDetector(lstm_model_path=lstm_model_path, require_rf=False)

    # Usar predicción anticipada
    results = detector.predict_next_state(new_data, forecast_minutes=1)

    output = {
        "predicciones_actuales": detector.get_classification_names(results["predictions_current"]),
        "predicciones_futuras": detector.get_classification_names(results["predictions_future"]),
        "clases_actuales": results["predictions_current"].tolist(),
        "clases_futuras": results["predictions_future"].tolist(),
        "confianzas_actuales": np.max(results["probabilities_current"], axis=1).tolist(),
        "confianzas_futuras": np.max(results["probabilities_future"], axis=1).tolist(),
        "alertas_preventivas": results["alerta_preventiva"].tolist(),
        "n_alertas": results["n_alertas"],
        "probabilidades_actuales": {
            "normal": results["probabilities_current"][:, 0].tolist(),
            "anomalia_voltaje": results["probabilities_current"][:, 1].tolist(),
            "cuelgue": results["probabilities_current"][:, 2].tolist()
        },
        "probabilidades_futuras": {
            "normal": results["probabilities_future"][:, 0].tolist(),
            "anomalia_voltaje": results["probabilities_future"][:, 1].tolist(),
            "cuelgue": results["probabilities_future"][:, 2].tolist()
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
