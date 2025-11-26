import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os

# --- 1. Análisis de Datos y Correlaciones ---
def analizar_datos(df: pd.DataFrame):
    """
    Realiza un estudio estadístico y de correlación exhaustivo sobre el 100% de los datos.
    """
    print("="*70)
    print("--- ANÁLISIS EXPLORATORIO DE DATOS (100%) ---")
    print("="*70)
    
    # 1. Limpieza básica para análisis (rellenar huecos si existen)
    df_clean = df.copy().ffill().bfill()

    # === ESTADÍSTICAS DESCRIPTIVAS BÁSICAS ===
    print("\n1. ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 70)
    cols_numericas = ['status', 'R1_a', 'R2_a', 'R1_b', 'R2_b']
    print(df_clean[cols_numericas].describe())
    
    # Información sobre valores nulos en datos originales
    print("\n2. VALORES NULOS (antes de limpieza):")
    print(df[cols_numericas].isnull().sum())
    
    # Distribución temporal
    print("\n3. INFORMACIÓN TEMPORAL:")
    print(f"   Fecha inicio: {df_clean['tiempo'].min()}")
    print(f"   Fecha fin: {df_clean['tiempo'].max()}")
    print(f"   Duración total: {df_clean['tiempo'].max() - df_clean['tiempo'].min()}")
    print(f"   Número de dispositivos únicos: {df_clean['id'].nunique()}")
    
    # === MATRIZ DE CORRELACIÓN ESTÁNDAR ===
    print("\n4. MATRIZ DE CORRELACIÓN (Pearson):")
    print("-" * 70)
    cols_interes = ['status', 'R1_a', 'R2_a', 'R1_b', 'R2_b']
    corr_matrix = df_clean[cols_interes].corr()
    print(corr_matrix.round(4))

    # === CORRELACIÓN AUMENTADA: R1*status vs R2*status ===
    print("\n5. CORRELACIÓN AUMENTADA (Interacción con Status):")
    print("-" * 70)
    # Crear features aumentadas
    df_clean['R1_a_status'] = df_clean['R1_a'] * df_clean['status']
    df_clean['R2_a_status'] = df_clean['R2_a'] * df_clean['status']
    df_clean['R1_b_status'] = df_clean['R1_b'] * df_clean['status']
    df_clean['R2_b_status'] = df_clean['R2_b'] * df_clean['status']
    
    # Correlación entre las interacciones
    cols_aumentadas = ['R1_a_status', 'R2_a_status', 'R1_b_status', 'R2_b_status']
    corr_aumentada = df_clean[cols_aumentadas].corr()
    print(corr_aumentada.round(4))
    
    print("\n   Interpretación:")
    print(f"   Correlación R1_a*status <-> R2_a*status: {corr_aumentada.loc['R1_a_status', 'R2_a_status']:.4f}")
    print(f"   Correlación R1_b*status <-> R2_b*status: {corr_aumentada.loc['R1_b_status', 'R2_b_status']:.4f}")

    # === ANÁLISIS DEL IMPACTO DE TRENES ===
    print("\n6. IMPACTO DE LA PRESENCIA DE TRENES (status):")
    print("-" * 70)
    promedios = df_clean.groupby('status')[['R1_a', 'R2_a', 'R1_b', 'R2_b']].agg(['mean', 'std', 'min', 'max'])
    print(promedios)
    
    # Diferencia de voltaje entre con/sin tren
    print("\n   Diferencia promedio (con tren - sin tren):")
    if 1.0 in df_clean['status'].values and 0.0 in df_clean['status'].values:
        with_train = df_clean[df_clean['status'] == 1.0][['R1_a', 'R2_a']].mean()
        without_train = df_clean[df_clean['status'] == 0.0][['R1_a', 'R2_a']].mean()
        diff = with_train - without_train
        print(f"   R1_a: {diff['R1_a']:.2f} mV")
        print(f"   R2_a: {diff['R2_a']:.2f} mV")

    # === CONTEO DE ANOMALÍAS ===
    print("\n7. DETECCIÓN DE ANOMALÍAS (Saltos > 500mV):")
    print("-" * 70)
    # Calcular diferencias por dispositivo para evitar mezclar dispositivos
    deltas_list = []
    for device_id in df_clean['id'].unique():
        df_device = df_clean[df_clean['id'] == device_id].sort_values('tiempo')
        delta = df_device[['R1_a', 'R2_a', 'R1_b', 'R2_b']].diff().abs()
        deltas_list.append(delta)
    
    deltas = pd.concat(deltas_list, ignore_index=True)
    
    # Definir anomalías
    anomalies_r1a = deltas[deltas['R1_a'] > 500]
    anomalies_r2a = deltas[deltas['R2_a'] > 500]
    anomalies_r1b = deltas[deltas['R1_b'] > 500]
    anomalies_r2b = deltas[deltas['R2_b'] > 500]
    
    total_anomalies = len(anomalies_r1a) + len(anomalies_r2a) + len(anomalies_r1b) + len(anomalies_r2b)
    total_samples = len(df_clean)
    
    print(f"   Total de muestras: {total_samples}")
    print(f"   Anomalías en R1_a: {len(anomalies_r1a)}")
    print(f"   Anomalías en R2_a: {len(anomalies_r2a)}")
    print(f"   Anomalías en R1_b: {len(anomalies_r1b)}")
    print(f"   Anomalías en R2_b: {len(anomalies_r2b)}")
    print(f"   Total anomalías: {total_anomalies}")
    print(f"   Ratio de anomalías: {total_anomalies/total_samples:.4%}")
    
    # === ANÁLISIS DE VARIABILIDAD ===
    print("\n8. ANÁLISIS DE VARIABILIDAD:")
    print("-" * 70)
    print("   Coeficiente de Variación (CV = std/mean):")
    for col in ['R1_a', 'R2_a', 'R1_b', 'R2_b']:
        cv = df_clean[col].std() / df_clean[col].mean()
        print(f"   {col}: {cv:.4f}")
    
    # === RELACIÓN ENTRE R1 Y R2 ===
    print("\n9. RELACIÓN ENTRE R1 Y R2 (por canal):")
    print("-" * 70)
    print(f"   Correlación R1_a <-> R2_a: {df_clean['R1_a'].corr(df_clean['R2_a']):.4f}")
    print(f"   Correlación R1_b <-> R2_b: {df_clean['R1_b'].corr(df_clean['R2_b']):.4f}")
    
    # Diferencia promedio entre receptores
    df_clean['diff_a'] = (df_clean['R1_a'] - df_clean['R2_a']).abs()
    df_clean['diff_b'] = (df_clean['R1_b'] - df_clean['R2_b']).abs()
    print(f"   Diferencia media |R1_a - R2_a|: {df_clean['diff_a'].mean():.2f} mV")
    print(f"   Diferencia media |R1_b - R2_b|: {df_clean['diff_b'].mean():.2f} mV")

    print("\n" + "="*70)
    return total_anomalies

# --- 2. Datasets (Adaptados para DataFrame) ---

class VoltageDropDataset(Dataset):
    def __init__(self, df, seq_len=10):
        """
        Dataset para predecir caídas de voltaje.
        Input: Ventana de 'seq_len' minutos de [status, R1, R2]
        Target: 1 si en el siguiente minuto hay un salto > 500mV, 0 si no.
        """
        # Preprocesamiento: Normalizar y limpiar
        self.df = df.copy().ffill().fillna(0)
        
        # Features: status, R1_a, R2_a (simplificado para el ejemplo)
        # Nota: Un profesional debería normalizar estos valores (ej. MinMaxScaler)
        self.features = self.df[['status', 'R1_a', 'R2_a']].values.astype(np.float32)
        
        # Calcular etiquetas (Targets)
        # Shift(-1) para ver el futuro inmediato
        diffs = self.df[['R1_a', 'R2_a']].diff().shift(-1).abs()
        # Si el salto en R1 o R2 es > 500, es una anomalía (Clase 1)
        self.labels = ((diffs['R1_a'] > 500) | (diffs['R2_a'] > 500)).astype(int).values
        
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Ventana de tiempo t-seq_len hasta t
        x = self.features[idx : idx + self.seq_len]
        # Etiqueta en t (predicción para t+1 basada en diff shift)
        y = self.labels[idx + self.seq_len - 1] 
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)  # [y] para mantener dimensión

# ... (BlockageDataset se mantendría similar, adaptado a DataFrame) ...

# --- 3. Data Augmentation ---
def aumentar_datos_con_ruido(dataset, factor=5):
    """
    Genera datos sintéticos solo para la clase minoritaria (anomalías).
    """
    features_list = []
    labels_list = []
    
    print("\n--- Iniciando Data Augmentation ---")
    # Extraemos todos los datos originales
    for i in range(len(dataset)):
        x, y = dataset[i]
        features_list.append(x)
        labels_list.append(y)
        
        # Si es una anomalía (y == 1), generamos copias con ruido
        if y == 1.0:
            for _ in range(factor):
                noise = torch.randn_like(x) * 0.05  # Ruido gaussiano suave
                features_list.append(x + noise)
                labels_list.append(y) # La etiqueta sigue siendo 1
    
    print(f"Datos originales: {len(dataset)}")
    print(f"Datos tras aumentación: {len(features_list)}")
    
    # Reempaquetar en un TensorDataset simple para el DataLoader
    x_tensor = torch.stack(features_list)
    y_tensor = torch.stack(labels_list)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)

# --- 4. Modelos (Sin cambios mayores) ---
class VoltageAnomalyModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super(VoltageAnomalyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

# --- 5. Loop de Entrenamiento ---
def train_template(train_loader, model, epochs=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    archivo_pickle = "datos_procesados.pkl"
    
    if not os.path.exists(archivo_pickle):
        print(f"Error: No se encuentra {archivo_pickle}. Ejecuta main.py primero.")
    else:
        print(f"Cargando datos desde {archivo_pickle}...")
        with open(archivo_pickle, "rb") as f:
            datos_lista = pickle.load(f)
        
        # Convertir a DataFrame para facilitar manipulación
        df = pd.DataFrame(datos_lista)
        
        # Asegurar orden temporal
        df['tiempo'] = pd.to_datetime(df['tiempo'])
        df = df.sort_values('tiempo').reset_index(drop=True)

        # 1. Análisis de Correlación con el 100% de los datos
        num_anomalies = analizar_datos(df)

        # 2. División Train/Test (90% Train, 10% Test)
        # En series temporales NO hacemos shuffle, cortamos por tiempo.
        split_idx = int(len(df) * 0.9)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\nDatos de Entrenamiento: {len(train_df)} muestras")
        print(f"Datos de Validación (Test): {len(test_df)} muestras")

        # 3. Preparar Dataset de Entrenamiento
        # Usamos una ventana de 10 minutos (seq_len=10)
        train_dataset = VoltageDropDataset(train_df, seq_len=10)
        
        # 4. Aplicar Data Augmentation si hay pocas anomalías
        if num_anomalies < 1000: # Umbral arbitrario
            print("Detectadas pocas anomalías. Aplicando Data Augmentation...")
            train_dataset = aumentar_datos_con_ruido(train_dataset, factor=10)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 5. Instanciar y Entrenar Modelo
        print("\nEntrenando Modelo de Anomalías de Voltaje...")
        model = VoltageAnomalyModel(input_size=3) # status, R1, R2
        train_template(train_loader, model, epochs=5)
        
        print("\nEntrenamiento finalizado. El modelo está listo para validación con test_df.")