"""
Módulo para cargar, procesar y leer datasets de voltaje.

Funciones principales:
1. cargar_y_agrupar_dataset(ruta_csv):
   - Lee un CSV con datos de dispositivos y voltajes.
   - Agrupa por tiempo y ID, calcula medias de medidas por canal.
   - Devuelve un DataFrame ordenado por ID y tiempo.

2. leer_datos_procesados(ruta_pkl):
   - Carga datos previamente procesados desde un archivo pickle.
   - Convierte timestamps y devuelve un DataFrame ordenado por tiempo.
"""

import pandas as pd
import numpy as np
import pickle


def cargar_y_agrupar_dataset(ruta_csv: str):
    df = pd.read_csv(ruta_csv, sep=';')

    # Convertir timestamp
    df['tiempo'] = pd.to_datetime(df['tiempo'], format="%d/%m/%Y %H:%M")
    df['id'] = df['id'].astype(int)

    grupos = df.groupby(['tiempo', 'id'])
    datos = []

    for (t, device_id), g in grupos:
        def v(medida, canal=None):
            if canal:
                sub = g[(g['medida'] == medida) & (g['canal'] == canal)]
            else:
                sub = g[g['medida'] == medida]

            # Usar media en vez del primer valor
            return float(sub['valor'].astype(float).mean()) if len(sub) else np.nan


        datos.append({
            "tiempo": t,
            "id": int(device_id),
            "status": v('status'),
            "R1_a": v('voltageReceiver1', 'a'),
            "R2_a": v('voltageReceiver2', 'a'),
            "R1_b": v('voltageReceiver1', 'b'),
            "R2_b": v('voltageReceiver2', 'b'),
        })

    df_final = pd.DataFrame(datos)
    df_final = df_final.sort_values(by=["id", "tiempo"]).reset_index(drop=True)
    return df_final

def leer_datos_procesados(ruta_pkl: str):
    
    with open(ruta_pkl, 'rb') as f:
        datos = pickle.load(f)

    df = pd.DataFrame(datos)
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    print (df.sort_values('tiempo').reset_index(drop=True))
    return df.sort_values('tiempo').reset_index(drop=True)

if __name__ == "__main__":
    ruta_pkl = 'datos_procesados.pkl'

    leer_datos_procesados(ruta_pkl)