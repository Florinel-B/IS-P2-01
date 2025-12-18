"""Cargar el modelo exportado y ejecutar una predicción batch directa."""

import os
import pickle

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from training_template import VoltageDropDataset, importar_modelo_portable


MODEL_PATH = "modelo_anomalias_finetuned.pth"
DATA_PATH = "datos_procesados.pkl"
BATCH_SIZE = 32


def ensure_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encuentra {path}. Ejecuta el entrenamiento y el procesamiento necesarios."
        )


def load_dataframe(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    df = pd.DataFrame(raw)
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    return df.sort_values("tiempo").reset_index(drop=True)


def predict_batch(modelo_dict: dict, df: pd.DataFrame, limit: Optional[int] = None) -> dict:
    seq_len = modelo_dict["seq_len"]
    scaler = modelo_dict["scaler"]
    device = modelo_dict["device"]
    threshold = modelo_dict["threshold"]
    model = modelo_dict["model"].to(device)
    model.eval()

    subset = df if limit is None else df.iloc[-limit:]
    dataset = VoltageDropDataset(subset, seq_len=seq_len, scaler=scaler)

    loader_kwargs = {"pin_memory": True, "num_workers": 4} if torch.cuda.is_available() else {}
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    predictions = []
    probabilities = []
    labels = []
    preview = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            probs, _ = model(X_batch)
            preds = (probs >= threshold).long()

            predictions.extend(preds.cpu().flatten().tolist())
            probabilities.extend(probs.cpu().flatten().tolist())
            labels.extend(y_batch.cpu().flatten().tolist())

            if len(preview) < 5:
                for prob, pred, real in zip(probabilities[-len(preds):], predictions[-len(preds):], labels[-len(preds):]):
                    if len(preview) >= 5:
                        break
                    preview.append({"prob": prob, "pred": int(pred), "label": int(real)})

    total = len(predictions)
    anomalies = sum(predictions)
    normal = total - anomalies

    # Calcular métricas de clasificación
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # True Positives
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # False Positives
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # True Negatives
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "anomalies": anomalies,
        "normal": normal,
        "ratio": anomalies / total if total else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "preview": preview,
    }


def main() -> None:
    ensure_file(MODEL_PATH)
    ensure_file(DATA_PATH)

    modelo_dict = importar_modelo_portable(MODEL_PATH)

    df = load_dataframe(DATA_PATH)
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:].copy()

    print(f"Cargando {len(test_df)} muestras de test (últimos 15%).")

    resumen = predict_batch(modelo_dict, test_df)

    print("\nResumen de predicciones:")
    print(f"  Total: {resumen['total']}")
    print(f"  Anomalías detectadas: {resumen['anomalies']}")
    print(f"  De las cuales son reales: {resumen['true_positives']}")
    print(f"  Falsos positivos: {resumen['false_positives']}")
    print(f"  Ratio de detección: {resumen['ratio']:.2%}")
    print(f"  Precisión: {resumen['precision']:.2%}")
    print(f"  Recall: {resumen['recall']:.2%}")
    print(f"  F1-Score: {resumen['f1_score']:.2%}")

    if resumen["preview"]:
        print("\nPrimeras predicciones:")
        for idx, row in enumerate(resumen["preview"], 1):
            label = "anomalía" if row["label"] == 1 else "normal"
            pred = "anomalía" if row["pred"] == 1 else "normal"
            print(f"  {idx}. prob={row['prob']:.4f}, pred={pred}, etiqueta={label}")


if __name__ == "__main__":
    main()
