import pandas as pd
import numpy as np


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
            return float(sub['valor'].iloc[0]) if len(sub) else np.nan

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
