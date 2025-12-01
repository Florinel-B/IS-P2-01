import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, df, seq_len=10, scaler=None, fit_scaler=False):
        """
        Dataset para predecir caídas de voltaje.
        Input: Ventana de 'seq_len' minutos de [status, R1, R2]
        Target: 1 si en el siguiente minuto hay un salto > 500mV, 0 si no.
        """
        self.df = df.copy().ffill().fillna(0)
        
        # Features: status, R1_a, R2_a
        raw_features = self.df[['status', 'R1_a', 'R2_a']].values.astype(np.float32)
        
        # === NORMALIZACIÓN ===
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(raw_features).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(raw_features).astype(np.float32)
        else:
            self.scaler = None
            self.features = raw_features
        
        # Calcular etiquetas con umbral más bajo para capturar más anomalías
        diffs = self.df[['R1_a', 'R2_a']].diff().shift(-1).abs()
        # Umbral ajustable - 300mV puede ser más sensible
        self.labels = ((diffs['R1_a'] > 300) | (diffs['R2_a'] > 300)).astype(int).values
        
        self.seq_len = seq_len
        
        # Contar anomalías para información
        self.num_anomalies = self.labels.sum()
        self.anomaly_ratio = self.num_anomalies / len(self.labels)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1] 
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)
    
    def get_class_weights(self):
        """Calcula pesos para balancear clases."""
        n_samples = len(self.labels)
        n_pos = self.labels.sum()
        n_neg = n_samples - n_pos
        
        if n_pos == 0:
            return 1.0
        
        # Peso inversamente proporcional a la frecuencia
        weight_pos = n_samples / (2.0 * n_pos)
        return weight_pos

# --- 3. Data Augmentation Mejorado ---
def aumentar_datos_con_ruido(dataset, factor=5):
    """
    Genera datos sintéticos solo para la clase minoritaria (anomalías).
    """
    features_list = []
    labels_list = []
    
    print("\n--- Iniciando Data Augmentation ---")
    anomaly_count = 0
    normal_count = 0
    
    for i in range(len(dataset)):
        x, y = dataset[i]
        features_list.append(x)
        labels_list.append(y)
        
        if y == 1.0:
            anomaly_count += 1
            for _ in range(factor):
                # Ruido más variado
                noise = torch.randn_like(x) * 0.1
                features_list.append(x + noise)
                labels_list.append(y)
                
                # También añadir versión escalada
                scale = 0.95 + torch.rand(1).item() * 0.1
                features_list.append(x * scale)
                labels_list.append(y)
        else:
            normal_count += 1
    
    print(f"Anomalías originales: {anomaly_count}")
    print(f"Normales originales: {normal_count}")
    print(f"Datos tras aumentación: {len(features_list)}")
    
    x_tensor = torch.stack(features_list)
    y_tensor = torch.stack(labels_list)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)

# --- 4. Modelo Mejorado ---
class VoltageAnomalyModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super(VoltageAnomalyModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.bn(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)
    
    def predict(self, x):
        return self.forward(x)

# --- 5. Loop de Entrenamiento Mejorado ---
def train_template(train_loader, model, epochs=20, pos_weight=1.0):
    """Entrenamiento con pérdida ponderada para desbalance."""
    # BCEWithLogitsLoss con pos_weight para manejar desbalance
    # Nota: Requiere quitar sigmoid del modelo para logits
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Crear pesos para muestras
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # Pérdida ponderada manual
            weight = torch.where(y_batch == 1, pos_weight, 1.0)
            loss = nn.functional.binary_cross_entropy(y_pred, y_batch, weight=weight)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

# --- 6. Búsqueda de Umbral Óptimo ---
def encontrar_umbral_optimo(model, val_loader):
    """Encuentra el umbral que maximiza F1-score."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            probs = model.predict(X_batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Probar diferentes umbrales
    best_f1 = 0
    best_threshold = 0.5
    
    print("\n--- Búsqueda de Umbral Óptimo ---")
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        precision = precision_score(all_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            print(f"Threshold: {threshold:.2f} -> F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f} *")
    
    print(f"\nMejor umbral: {best_threshold:.2f} con F1: {best_f1:.4f}")
    return best_threshold

# --- 7. Evaluación del Modelo ---
def evaluar_modelo(model, test_loader, threshold=0.5):
    """
    Evalúa el modelo con métricas de clasificación.
    
    Args:
        model: Modelo entrenado
        test_loader: DataLoader con datos de test
        threshold: Umbral para clasificación binaria
    
    Returns:
        dict con métricas de evaluación
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n" + "="*70)
    print("--- EVALUACIÓN DEL MODELO ---")
    print("="*70)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model.predict(X_batch)
            preds = (probs >= threshold).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    
    # Convertir a arrays numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Mostrar resultados
    print(f"\n1. MÉTRICAS DE CLASIFICACIÓN (threshold={threshold}):")
    print("-" * 70)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n2. MATRIZ DE CONFUSIÓN:")
    print("-" * 70)
    print(f"   {'':>15} Predicho 0  Predicho 1")
    print(f"   {'Real 0':>15} {conf_matrix[0][0]:>10} {conf_matrix[0][1]:>10}")
    if len(conf_matrix) > 1:
        print(f"   {'Real 1':>15} {conf_matrix[1][0]:>10} {conf_matrix[1][1]:>10}")
    
    print(f"\n3. REPORTE DETALLADO:")
    print("-" * 70)
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal', 'Anomalía'],
                                zero_division=0))
    
    print(f"\n4. DISTRIBUCIÓN DE PREDICCIONES:")
    print("-" * 70)
    print(f"   Total muestras evaluadas: {len(all_labels)}")
    print(f"   Anomalías reales: {int(all_labels.sum())} ({all_labels.mean()*100:.2f}%)")
    print(f"   Anomalías predichas: {int(all_preds.sum())} ({all_preds.mean()*100:.2f}%)")
    
    print("\n" + "="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

if __name__ == "__main__":
    archivo_pickle = "datos_procesados.pkl"
    
    if not os.path.exists(archivo_pickle):
        print(f"Error: No se encuentra {archivo_pickle}. Ejecuta main.py primero.")
    else:
        print(f"Cargando datos desde {archivo_pickle}...")
        with open(archivo_pickle, "rb") as f:
            datos_lista = pickle.load(f)
        
        df = pd.DataFrame(datos_lista)
        df['tiempo'] = pd.to_datetime(df['tiempo'])
        df = df.sort_values('tiempo').reset_index(drop=True)

        num_anomalies = analizar_datos(df)

        # División 80% Train, 10% Val, 10% Test
        train_idx = int(len(df) * 0.8)
        val_idx = int(len(df) * 0.9)
        
        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()
        
        print(f"\nDatos de Entrenamiento: {len(train_df)} muestras")
        print(f"Datos de Validación: {len(val_df)} muestras")
        print(f"Datos de Test: {len(test_df)} muestras")

        # Dataset con normalización
        train_dataset = VoltageDropDataset(train_df, seq_len=10, fit_scaler=True)
        scaler = train_dataset.scaler
        
        val_dataset = VoltageDropDataset(val_df, seq_len=10, scaler=scaler)
        test_dataset = VoltageDropDataset(test_df, seq_len=10, scaler=scaler)
        
        print(f"\nRatio de anomalías en train: {train_dataset.anomaly_ratio:.4%}")
        print(f"Anomalías en train: {train_dataset.num_anomalies}")
        
        # Calcular peso para clase positiva
        pos_weight = train_dataset.get_class_weights()
        print(f"Peso para clase positiva: {pos_weight:.2f}")
        
        # Data Augmentation
        if train_dataset.num_anomalies < 500:
            print("Aplicando Data Augmentation...")
            augmented_dataset = aumentar_datos_con_ruido(train_dataset, factor=15)
            train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Modelo mejorado
        print("\nEntrenando Modelo de Anomalías de Voltaje...")
        model = VoltageAnomalyModel(input_size=3, hidden_size=64, num_layers=2, dropout=0.3)
        train_template(train_loader, model, epochs=30, pos_weight=pos_weight)
        
        # Encontrar umbral óptimo con validación
        optimal_threshold = encontrar_umbral_optimo(model, val_loader)
        
        # Evaluar con umbral óptimo
        print("\nEvaluando modelo con datos de test...")
        metricas = evaluar_modelo(model, test_loader, threshold=optimal_threshold)
        
        print(f"\nResumen Final:")
        print(f"  - Umbral óptimo: {optimal_threshold:.2f}")
        print(f"  - F1-Score: {metricas['f1_score']:.4f}")
        print(f"  - Recall: {metricas['recall']:.4f}")
        print(f"  - Precision: {metricas['precision']:.4f}")
        
        print("\nEntrenamiento y evaluación finalizados.")