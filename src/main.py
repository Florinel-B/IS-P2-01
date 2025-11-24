import pandas as pd
import numpy as np

def cargar_y_agrupar_dataset(ruta_csv: str):
    """
    Carga el CSV, agrupa cada 6 filas y devuelve cada grupo como un diccionario
    sin provocar errores aunque falten valores.
    """

    df = pd.read_csv(ruta_csv, sep=';')
    df['tiempo'] = pd.to_datetime(df['tiempo'], format="%d/%m/%Y %H:%M")
    df = df.sort_values(by='tiempo').reset_index(drop=True)

    # Agrupar cada 6 filas
    grupos = [df.iloc[i:i+6] for i in range(0, len(df), 6)]

    def extraer_valor(g, medida, canal):
        """Extrae un valor de forma segura: si no existe, devuelve NaN."""
        sub = g[(g['medida'] == medida) & (g['canal'] == canal)]
        if len(sub) == 0:
            return np.nan
        return float(sub['valor'].iloc[0])

    def procesar_grupo(g):
        # Extraer status de forma segura
        sub_status = g[g['medida'] == 'status']
        status = int(sub_status['valor'].iloc[0]) if len(sub_status) > 0 else None

        return {
            "tiempo": g['tiempo'].iloc[0],
            "id": int(g['id'].iloc[0]),
            "status": status,
            "R1_a": extraer_valor(g, 'voltageReceiver1', 'a'),
            "R2_a": extraer_valor(g, 'voltageReceiver2', 'a'),
            "R1_b": extraer_valor(g, 'voltageReceiver1', 'b'),
            "R2_b": extraer_valor(g, 'voltageReceiver2', 'b')
        }

    datos = [procesar_grupo(g) for g in grupos]
    return datos




if __name__ == "__main__":

    datos = cargar_y_agrupar_dataset("Dataset-CV.csv")

    print(datos[0])

