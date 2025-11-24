import pandas as pd
import numpy as np
import pickle
import os
import torch



def cargar_y_agrupar_dataset(ruta_csv: str):
    df = pd.read_csv(ruta_csv, sep=';')

    # Convertir timestamp
    df['tiempo'] = pd.to_datetime(df['tiempo'], format="%d/%m/%Y %H:%M")

    # Convertir id a int estándar
    df['id'] = df['id'].astype(int)

    # Agrupar por instante de tiempo + id de dispositivo
    grupos = df.groupby(['tiempo', 'id'])

    datos = []

    for (t, device_id), g in grupos:
        # Tomar valores de forma segura
        def v(medida, canal=None):
            if canal:
                sub = g[(g['medida'] == medida) & (g['canal'] == canal)]
            else:
                sub = g[g['medida'] == medida]

            return float(sub['valor'].iloc[0]) if len(sub) else np.nan

        datos.append({
            "tiempo": t,
            "id": int(device_id),        # aquí hacemos la conversión a int puro
            "status": v('status'),
            "R1_a": v('voltageReceiver1', 'a'),
            "R2_a": v('voltageReceiver2', 'a'),
            "R1_b": v('voltageReceiver1', 'b'),
            "R2_b": v('voltageReceiver2', 'b'),
        })

    return datos





if __name__ == "__main__":
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


