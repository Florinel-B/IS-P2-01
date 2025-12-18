import numpy as np
import pickle
import os
import torch
from incidence_detector import detectar_incidencias, entrenar_modelo
from visualization import plot_incidencias
import pandas as pd

def cargar_y_agrupar_dataset(ruta_csv: str):
    """
    Carga el CSV y devuelve un DataFrame con columnas:
    tiempo, id, R1_a, R1_b, R2_a, R2_b, status
    """
    df = pd.read_csv(ruta_csv, sep=';')

    # Convertir timestamp y id
    df['tiempo'] = pd.to_datetime(df['tiempo'], format="%d/%m/%Y %H:%M")
    df['id'] = df['id'].astype(int)

    # Pivotar tabla para que cada canal sea una columna
    df_pivot = df.pivot_table(
        index=['tiempo','id'], 
        columns=['medida','canal'], 
        values='valor', 
        aggfunc='mean'
    ).reset_index()

    # Renombrar columnas a formato plano (ajusta según tus nombres de CSV)
    df_pivot.columns = ['tiempo','id','R1_a','R1_b','R2_a','R2_b','status']

    # Ordenar por dispositivo y tiempo
    df_pivot = df_pivot.sort_values(['id','tiempo']).reset_index(drop=True)
    
    return df_pivot


if __name__ == "__main__":

    # 1. Cargar CSV
    df = cargar_y_agrupar_dataset("Dataset-CV.csv")

    # 2. Rellenar valores faltantes
    df.fillna(method='ffill', inplace=True)  # rellena con último valor conocido
    df.fillna(0, inplace=True)               # si queda NaN, poner 0

    # 3. Detectar incidencias
    df = detectar_incidencias(df)

    # 4. Entrenar modelo
    modelo, acc = entrenar_modelo(df)
    print(f"\n▶ Precisión del modelo predictivo: {acc:.2f}")

    # 5. Guardar CSV con incidencias
    df.to_csv("incidencias_detectadas.csv", index=False)
    print("\n📁 Archivo generado: incidencias_detectadas.csv")

    # 6. Mostrar gráfica
    plot_incidencias(df)

    # 7. Guardar/recuperar datos en pickle
    archivo_pickle = "datos_procesados.pkl"
    if os.path.exists(archivo_pickle):
        print(f"Cargando datos desde {archivo_pickle}...")
        with open(archivo_pickle, "rb") as f:
            datos = pickle.load(f)
    else:
        print("Procesando CSV original...")
        datos = cargar_y_agrupar_dataset("Dataset-CV.csv")
        print(f"Guardando datos en {archivo_pickle}...")
        with open(archivo_pickle, "wb") as f:
            pickle.dump(datos, f)

    print(datos[0])
