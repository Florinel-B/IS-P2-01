import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def detectar_incidencias(df):
    df['incidencia'] = 0

    # (i) Bloqueo: diferencia > 2 min → ausencia de datos
    df['tiempo_shift'] = df.groupby('id')['tiempo'].shift(1)
    delta_t = (df['tiempo'] - df['tiempo_shift']).dt.total_seconds()
    df.loc[delta_t > 120, 'incidencia'] = 1

    # (ii) Saltos >= 0.5 voltios
    for col in ["R1_a", "R2_a", "R1_b", "R2_b"]:
        diff = df.groupby('id')[col].diff().abs()
        df.loc[diff >= 500, 'incidencia'] = 2

    df = df.drop(columns=["tiempo_shift"])
    return df


def entrenar_modelo(df):
    # Features numéricas
    X = df[["R1_a", "R2_a", "R1_b", "R2_b"]].fillna(0)
    y = df["incidencia"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    return modelo, accuracy
