import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
import math
import random
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# --- Configuración de dispositivo CUDA ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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

# --- 2. Dataset con Historial de Anomalías ---

class VoltageDropDataset(Dataset):
    def __init__(self, df, seq_len=60, scaler=None, fit_scaler=False, anomaly_history_len=30):
        """
        Dataset para predecir caídas de voltaje con memoria temporal.
        
        Args:
            df: DataFrame con los datos
            seq_len: Longitud de la secuencia de entrada (ventana temporal)
            scaler: Scaler para normalización
            fit_scaler: Si debe ajustar el scaler
            anomaly_history_len: Cuántas anomalías pasadas recordar como feature
        """
        self.df = df.copy().ffill().fillna(0)
        self.seq_len = seq_len
        self.anomaly_history_len = anomaly_history_len
        
        # Features base: status, R1_a, R2_a, R1_b, R2_b
        raw_features = self.df[['status', 'R1_a', 'R2_a', 'R1_b', 'R2_b']].values.astype(np.float32)
        
        # Calcular diferencias (cambios de voltaje) como features adicionales
        diffs = self.df[['R1_a', 'R2_a', 'R1_b', 'R2_b']].diff().fillna(0).values.astype(np.float32)
        
        # Calcular etiquetas de anomalías (para todo el dataset)
        diff_labels = self.df[['R1_a', 'R2_a']].diff().abs()
        self.all_anomaly_labels = ((diff_labels['R1_a'] > 500) | (diff_labels['R2_a'] > 500)).astype(int).values
        
        # Crear features de historial de anomalías (rolling sum de anomalías pasadas)
        anomaly_history = self._create_anomaly_history_features()
        
        # Combinar todas las features: [status, R1_a, R2_a, R1_b, R2_b, diff_R1_a, diff_R2_a, diff_R1_b, diff_R2_b, anomaly_history...]
        combined_features = np.hstack([raw_features, diffs, anomaly_history])
        
        # Normalización
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(combined_features).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(combined_features).astype(np.float32)
        else:
            self.scaler = None
            self.features = combined_features
        
        # Labels: predecir anomalía en el siguiente paso
        self.labels = np.roll(self.all_anomaly_labels, -1)  # shift para predecir futuro
        self.labels[-1] = 0  # último no tiene futuro
        
        # Estadísticas
        self.num_anomalies = self.labels[:len(self.labels)-self.seq_len].sum()
        self.anomaly_ratio = self.num_anomalies / max(1, len(self.labels) - self.seq_len)
        
        # Número de features
        self.input_size = self.features.shape[1]

    def _create_anomaly_history_features(self):
        """Crea features basadas en historial de anomalías pasadas."""
        n_samples = len(self.all_anomaly_labels)
        history_features = np.zeros((n_samples, 5), dtype=np.float32)
        
        for i in range(n_samples):
            # Ventanas de diferentes tamaños para capturar patrones
            windows = [5, 10, 30, 60, 120]
            for j, w in enumerate(windows):
                start = max(0, i - w)
                if start < i:
                    # Proporción de anomalías en la ventana
                    history_features[i, j] = self.all_anomaly_labels[start:i].mean()
        
        return history_features

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        # Secuencia de entrada: ventana temporal completa
        x = self.features[idx : idx + self.seq_len]
        # Label: anomalía en el paso siguiente al final de la ventana
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)
    
    def get_class_weights(self):
        """Calcula pesos para balancear clases."""
        n_pos = self.num_anomalies
        n_neg = len(self) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

# --- 3. Modelo con Estado Oculto Persistente ---

class VoltageAnomalyModelStateful(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=3, dropout=0.3):
        """
        Modelo LSTM con capacidad de mantener estado entre predicciones.
        
        Args:
            input_size: Número de features de entrada
            hidden_size: Tamaño del estado oculto
            num_layers: Número de capas LSTM
            dropout: Dropout rate
        """
        super(VoltageAnomalyModelStateful, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM principal
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de atención temporal
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 por concatenar atención + último estado
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Estado oculto persistente (para inferencia online)
        self.hidden_state = None

    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            hidden: Estado oculto opcional (h_n, c_n)
        
        Returns:
            output: Predicción de anomalía
            hidden: Nuevo estado oculto
        """
        batch_size = x.size(0)
        
        # Inicializar estado si no se proporciona
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM
        lstm_out, hidden_new = self.lstm(x, hidden)  # lstm_out: [batch, seq, hidden]
        
        # Atención temporal
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden]
        
        # Último estado oculto
        last_hidden = hidden_new[0][-1]  # [batch, hidden]
        
        # Concatenar contexto de atención + último estado
        combined = torch.cat([context, last_hidden], dim=1)  # [batch, hidden*2]
        
        # Clasificación
        output = self.classifier(combined)
        
        return output, hidden_new

    def _init_hidden(self, batch_size, device):
        """Inicializa el estado oculto."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def predict(self, x, hidden=None):
        """Wrapper para predicción."""
        return self.forward(x, hidden)

    def reset_state(self):
        """Reinicia el estado oculto persistente."""
        self.hidden_state = None


# --- 4. Predictor Online con Memoria ---

class OnlinePredictor:
    """
    Predictor que procesa datos uno a uno manteniendo contexto histórico.
    """
    def __init__(self, model, scaler, max_history=120, device=None):
        """
        Args:
            model: Modelo entrenado
            scaler: Scaler usado en entrenamiento
            max_history: Máximo de muestras a recordar
            device: Dispositivo de cómputo (auto-detecta si None)
        """
        self.device = device if device is not None else DEVICE
        self.model = model.to(self.device)
        self.scaler = scaler
        self.max_history = max_history
        
        # Buffer de datos históricos
        self.data_buffer = deque(maxlen=max_history)
        
        # Historial de anomalías detectadas
        self.anomaly_history = deque(maxlen=max_history)
        
        # Estado oculto del LSTM
        self.hidden_state = None
        
        # Contador de muestras procesadas
        self.samples_processed = 0
        
        self.model.eval()

    def _compute_features(self, raw_data):
        """
        Calcula features para una muestra incluyendo historial.
        
        Args:
            raw_data: dict con keys: status, R1_a, R2_a, R1_b, R2_b
        """
        # Features base
        base = np.array([
            raw_data['status'],
            raw_data['R1_a'],
            raw_data['R2_a'],
            raw_data['R1_b'],
            raw_data['R2_b']
        ], dtype=np.float32)
        
        # Diferencias (si hay historial)
        if len(self.data_buffer) > 0:
            prev = self.data_buffer[-1][:5]  # Features base anteriores
            diffs = base[1:5] - prev[1:5]  # Diff de voltajes
        else:
            diffs = np.zeros(4, dtype=np.float32)
        
        # Features de historial de anomalías
        anomaly_hist = np.zeros(5, dtype=np.float32)
        if len(self.anomaly_history) > 0:
            hist_array = np.array(self.anomaly_history)
            windows = [5, 10, 30, 60, 120]
            for i, w in enumerate(windows):
                start = max(0, len(hist_array) - w)
                if start < len(hist_array):
                    anomaly_hist[i] = hist_array[start:].mean()
        
        # Combinar todas las features
        combined = np.concatenate([base, diffs, anomaly_hist])
        return combined

    def predict_single(self, raw_data, threshold=0.5):
        """
        Predice anomalía para una única muestra nueva.
        
        Args:
            raw_data: dict con keys: status, R1_a, R2_a, R1_b, R2_b
            threshold: Umbral de clasificación
        
        Returns:
            dict con: probability, is_anomaly, confidence
        """
        # Calcular features
        features = self._compute_features(raw_data)
        
        # Normalizar
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        # Añadir al buffer
        self.data_buffer.append(features)
        self.samples_processed += 1
        
        # Necesitamos mínimo algunas muestras para predecir
        min_samples = 10
        if len(self.data_buffer) < min_samples:
            # No hay suficiente contexto
            self.anomaly_history.append(0)
            return {
                'probability': 0.0,
                'is_anomaly': False,
                'confidence': 0.0,
                'message': f'Acumulando contexto ({len(self.data_buffer)}/{min_samples})'
            }
        
        # Preparar secuencia para el modelo
        seq = np.array(list(self.data_buffer), dtype=np.float32)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predicción
        with torch.no_grad():
            prob, self.hidden_state = self.model.predict(x, self.hidden_state)
            prob = prob.cpu().item()
        
        is_anomaly = prob >= threshold
        
        # Registrar en historial
        self.anomaly_history.append(1 if is_anomaly else 0)
        
        # Calcular confianza basada en cantidad de datos
        confidence = min(1.0, len(self.data_buffer) / self.max_history)
        
        return {
            'probability': prob,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'samples_in_memory': len(self.data_buffer),
            'total_processed': self.samples_processed
        }

    def predict_batch(self, data_list, threshold=0.5):
        """Predice para una lista de muestras secuencialmente."""
        results = []
        for data in data_list:
            result = self.predict_single(data, threshold)
            results.append(result)
        return results

    def reset(self):
        """Reinicia todo el estado del predictor."""
        self.data_buffer.clear()
        self.anomaly_history.clear()
        self.hidden_state = None
        self.samples_processed = 0


# --- 5. Data Augmentation ---

def aumentar_datos_con_ruido(dataset, factor=5):
    """Genera datos sintéticos solo para la clase minoritaria."""
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
                noise = torch.randn_like(x) * 0.05
                features_list.append(x + noise)
                labels_list.append(y)
        else:
            normal_count += 1
    
    print(f"Anomalías originales: {anomaly_count}")
    print(f"Normales originales: {normal_count}")
    print(f"Datos tras aumentación: {len(features_list)}")
    
    x_tensor = torch.stack(features_list)
    y_tensor = torch.stack(labels_list)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)


# --- 6. Cosine Annealing Learning Rate Scheduler con Warm Restarts ---

class CosineAnnealingLRWithRestarts:
    """
    Wrapper personalizado sobre CosineAnnealingWarmRestarts con tracking adicional.
    
    El LR sigue un patrón coseno que decae y se reinicia periódicamente,
    permitiendo escapar de mínimos locales.
    """
    def __init__(self, optimizer, T_0=20, T_mult=2, eta_min=1e-6, initial_lr=0.01):
        """
        Args:
            optimizer: Optimizador de PyTorch
            T_0: Número de epochs hasta el primer restart
            T_mult: Factor de multiplicación del período después de cada restart
            eta_min: Learning rate mínimo
            initial_lr: Learning rate inicial/máximo
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_mult = T_mult
        
        # Establecer LR inicial
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
        
        # Scheduler nativo de PyTorch
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        
        # Tracking
        self.epoch = 0
        self.lr_history = []
        self.restart_epochs = []
        self.current_T = T_0
        self.restarts = 0
        self.best_loss = float('inf')
        self.loss_history = []
    
    def step(self, current_loss=None):
        """
        Actualiza el learning rate.
        
        Args:
            current_loss: Loss actual (opcional, para tracking)
        """
        # Detectar restart (cuando el LR vuelve al máximo)
        old_lr = self.get_lr()
        
        self.scheduler.step()
        self.epoch += 1
        
        new_lr = self.get_lr()
        
        # Detectar si hubo un restart (LR subió significativamente)
        if new_lr > old_lr * 1.5 and self.epoch > 1:
            self.restarts += 1
            self.restart_epochs.append(self.epoch)
        
        # Tracking de loss
        if current_loss is not None:
            self.loss_history.append(current_loss)
            if current_loss < self.best_loss:
                self.best_loss = current_loss
        
        # Guardar historial
        self.lr_history.append(new_lr)
        
        return new_lr
    
    def get_lr(self):
        """Retorna el learning rate actual."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_stats(self):
        """Retorna estadísticas del scheduler."""
        return {
            'current_lr': self.get_lr(),
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'restarts': self.restarts,
            'restart_epochs': self.restart_epochs,
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'lr_history': self.lr_history
        }


# --- 7. Entrenamiento con Cosine Annealing ---

def train_template(train_loader, model, val_loader=None, epochs=200, pos_weight=1.0, patience=10, device=None):
    """Entrenamiento con Cosine Annealing LR scheduler y soporte CUDA."""
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    pos_weight_tensor = torch.tensor(pos_weight, device=device)
    
    criterion = nn.BCELoss()
    initial_lr = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Cosine Annealing con Warm Restarts
    # T_0=30: primer ciclo de 30 epochs
    # T_mult=2: cada ciclo siguiente dura el doble (30, 60, 120, ...)
    ca_scheduler = CosineAnnealingLRWithRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=1e-6,
        initial_lr=initial_lr
    )
    
    best_val_f1 = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    print(f"\n{'='*70}")
    print("Iniciando entrenamiento con Cosine Annealing Warm Restarts")
    print(f"   T_0={ca_scheduler.T_0}, T_mult={ca_scheduler.T_mult}, eta_min={ca_scheduler.eta_min}")
    print(f"{'='*70}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)
            
            weight = torch.where(y_batch == 1, pos_weight_tensor, torch.tensor(1.0, device=device))
            loss = nn.functional.binary_cross_entropy(y_pred, y_batch, weight=weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Actualizar scheduler
        current_lr = ca_scheduler.step(avg_loss)
        
        # Validación
        val_f1 = 0
        if val_loader is not None:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    probs, _ = model(X_batch)
                    preds = (probs >= 0.5).float()
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(y_batch.cpu().numpy().flatten())
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            model.train()
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        
        # Mostrar progreso
        if (epoch + 1) % 10 == 0:
            restart_info = f"Restarts: {ca_scheduler.restarts}"
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f} | {restart_info}")
        
        # Mostrar eventos de restart
        if ca_scheduler.restart_epochs and ca_scheduler.restart_epochs[-1] == epoch + 1:
            print(f"   🔄 WARM RESTART en epoch {epoch+1}: LR -> {current_lr:.6f}")
        
    
    # Estadísticas finales
    stats = ca_scheduler.get_stats()
    print(f"\n{'='*70}")
    print("Estadísticas de Cosine Annealing:")
    print(f"   Epochs completados: {stats['epoch']}")
    print(f"   Mejor loss: {stats['best_loss']:.4f}")
    print(f"   Warm restarts: {stats['restarts']}")
    print(f"   Epochs de restart: {stats['restart_epochs']}")
    print(f"   LR final: {stats['current_lr']:.6f}")
    print(f"{'='*70}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        print(f"Restaurado mejor modelo con Val F1: {best_val_f1:.4f}")
    
    return model, ca_scheduler


# --- 8. Búsqueda de Umbral Óptimo con CUDA ---

def encontrar_umbral_optimo(model, val_loader, device=None):
    """Encuentra el umbral que maximiza F1-score."""
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            
            probs, _ = model(X_batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_f1 = 0
    best_threshold = 0.5
    
    print("\n--- Búsqueda de Umbral Óptimo ---")
    for threshold in np.arange(0.05, 0.95, 0.05):
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


# --- 9. Evaluación del Modelo con CUDA ---

def evaluar_modelo(model, test_loader, threshold=0.5, device=None):
    """Evalúa el modelo con métricas de clasificación."""
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n" + "="*70)
    print("--- EVALUACIÓN DEL MODELO ---")
    print("="*70)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            probs, _ = model(X_batch)
            preds = (probs >= threshold).float()
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"\n1. MÉTRICAS DE CLASIFICACIÓN (threshold={threshold:.2f}):")
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
    
    print(f"\n4. DISTRIBUCIÓN:")
    print("-" * 70)
    print(f"   Total muestras: {len(all_labels)}")
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


# --- 10. Demostración de Predicción Online ---

def demo_prediccion_online(model, scaler, test_df, threshold=0.5, device=None):
    """Demuestra el uso del predictor online."""
    if device is None:
        device = DEVICE
    
    print("\n" + "="*70)
    print("--- DEMO: PREDICCIÓN ONLINE (dato a dato) ---")
    print("="*70)
    
    predictor = OnlinePredictor(model, scaler, max_history=120, device=device)
    
    # Simular llegada de datos uno a uno
    resultados = []
    anomalias_detectadas = 0
    
    for i, row in test_df.head(200).iterrows():
        data = {
            'status': row['status'],
            'R1_a': row['R1_a'],
            'R2_a': row['R2_a'],
            'R1_b': row['R1_b'],
            'R2_b': row['R2_b']
        }
        
        result = predictor.predict_single(data, threshold=threshold)
        resultados.append(result)
        
        if result['is_anomaly']:
            anomalias_detectadas += 1
            if anomalias_detectadas <= 5:
                print(f"\n⚠️  ANOMALÍA DETECTADA en muestra {i}")
                print(f"   Probabilidad: {result['probability']:.4f}")
                print(f"   Confianza: {result['confidence']:.2%}")
                print(f"   Muestras en memoria: {result['samples_in_memory']}")
    
    print(f"\n📊 Resumen de predicción online:")
    print(f"   Muestras procesadas: {len(resultados)}")
    print(f"   Anomalías detectadas: {anomalias_detectadas}")
    print(f"   Ratio: {anomalias_detectadas/len(resultados)*100:.2f}%")
    
    return predictor, resultados


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

        # División temporal
        train_idx = int(len(df) * 0.70)
        val_idx = int(len(df) * 0.85)
        
        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()
        
        print(f"\nDatos de Entrenamiento: {len(train_df)} muestras")
        print(f"Datos de Validación: {len(val_df)} muestras")
        print(f"Datos de Test: {len(test_df)} muestras")

        SEQ_LEN = 60
        
        train_dataset = VoltageDropDataset(train_df, seq_len=SEQ_LEN, fit_scaler=True)
        scaler = train_dataset.scaler
        input_size = train_dataset.input_size
        
        val_dataset = VoltageDropDataset(val_df, seq_len=SEQ_LEN, scaler=scaler)
        test_dataset = VoltageDropDataset(test_df, seq_len=SEQ_LEN, scaler=scaler)
        
        print(f"\nFeatures de entrada: {input_size}")
        print(f"Longitud de secuencia: {SEQ_LEN}")
        print(f"Ratio de anomalías en train: {train_dataset.anomaly_ratio:.4%}")
        print(f"Anomalías en train: {train_dataset.num_anomalies}")
        
        pos_weight = train_dataset.get_class_weights()
        print(f"Peso para clase positiva: {pos_weight:.2f}")
        
        loader_kwargs = {'pin_memory': True, 'num_workers': 4} if torch.cuda.is_available() else {}
        
        if train_dataset.num_anomalies < 200:
            print("Aplicando Data Augmentation...")
            augmented_dataset = aumentar_datos_con_ruido(train_dataset, factor=20)
            train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True, **loader_kwargs)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **loader_kwargs)
        
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **loader_kwargs)

        print("\n" + "="*70)
        print(f"Entrenando Modelo con Cosine Annealing en {DEVICE}...")
        print("="*70)
        
        model = VoltageAnomalyModelStateful(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        ).to(DEVICE)
        
        # Ahora train_template retorna también el scheduler
        model, ca_scheduler = train_template(
            train_loader, model, val_loader, 
            epochs=200, pos_weight=pos_weight, patience=20, device=DEVICE
        )
        
        optimal_threshold = encontrar_umbral_optimo(model, val_loader, device=DEVICE)
        
        print("\nEvaluando modelo con datos de test (modo batch)...")
        metricas = evaluar_modelo(model, test_loader, threshold=optimal_threshold, device=DEVICE)
        
        predictor, resultados = demo_prediccion_online(model, scaler, test_df, threshold=optimal_threshold, device=DEVICE)
        
        # Guardar modelo con info del scheduler
        print("\nGuardando modelo...")
        model_cpu = model.cpu()
        torch.save({
            'model_state_dict': model_cpu.state_dict(),
            'scaler': scaler,
            'threshold': optimal_threshold,
            'input_size': input_size,
            'seq_len': SEQ_LEN,
            'ca_stats': ca_scheduler.get_stats()
        }, 'modelo_anomalias.pth')
        
        print(f"\n{'='*70}")
        print("RESUMEN FINAL")
        print(f"{'='*70}")
        print(f"  Dispositivo usado: {DEVICE}")
        print(f"  Umbral óptimo: {optimal_threshold:.2f}")
        print(f"  F1-Score: {metricas['f1_score']:.4f}")
        print(f"  Recall: {metricas['recall']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Warm Restarts: {ca_scheduler.restarts}")
        print(f"\nModelo guardado en: modelo_anomalias.pth")