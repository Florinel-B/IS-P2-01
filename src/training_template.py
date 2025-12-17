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

# --- 3. Variational Dropout para Secuencias Temporales ---

class VariationalDropout(nn.Module):
    """
    Dropout variacional que usa la misma máscara en todos los timesteps.
    Esto es más apropiado para secuencias temporales que el dropout estándar.
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
    def forward(self, x):
        """
        Args:
            x: tensor [batch, seq_len, features]
        Returns:
            tensor con dropout aplicado con la misma máscara en todos los timesteps
        """
        if not self.training or self.dropout == 0:
            return x
        
        # Crear máscara que se repite en el tiempo: [batch, 1, features]
        # La máscara se expande automáticamente a [batch, seq_len, features]
        mask = torch.bernoulli(
            torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.dropout)
        )
        # Escalar por (1 - dropout) para mantener la expectativa
        return x * mask / (1 - self.dropout)


# --- 4. Modelo con Estado Oculto Persistente y Regularización Mejorada ---

class VoltageAnomalyModelStateful(nn.Module):
    def __init__(self, input_size=14, hidden_size=192, num_layers=4, dropout=0.3):
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
        
        # Dropout variacional para entrada (mismo dropout en todos los timesteps)
        self.variational_dropout = VariationalDropout(dropout=dropout * 0.5)  # Más suave en entrada
        
        # LSTM principal
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout variacional para salida LSTM
        self.variational_dropout_out = VariationalDropout(dropout=dropout)
        
        # Capa de atención temporal simplificada y corregida
        self.attention_weights = nn.Linear(hidden_size, 1)
        
        # Clasificador mejorado con LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Sin Sigmoid, usaremos BCEWithLogitsLoss
        )
        
        # Estado oculto persistente (para inferencia online)
        self.hidden_state = None

        self._init_weights()

    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            hidden: Estado oculto opcional (h_n, c_n)
        
        Returns:
            logits: Logits sin sigmoid [batch, 1]
            hidden: Nuevo estado oculto
        """
        batch_size = x.size(0)
        
        # Inicializar estado si no se proporciona
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # Aplicar dropout variacional a la entrada
        x_dropped = self.variational_dropout(x)
        
        # LSTM
        lstm_out, hidden_new = self.lstm(x_dropped, hidden)  # lstm_out: [batch, seq, hidden]
        
        # Aplicar dropout variacional a la salida del LSTM
        lstm_out = self.variational_dropout_out(lstm_out)
        
        # Atención temporal corregida
        attn_logits = self.attention_weights(lstm_out)  # [batch, seq, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)  # Normalizar sobre secuencia
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden]
        
        # Clasificación (retorna logits sin sigmoid)
        logits = self.classifier(context)
        
        return logits, hidden_new

    def _init_hidden(self, batch_size, device):
        """Inicializa el estado oculto."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                hidden = self.hidden_size
                param.data[hidden:2*hidden].fill_(1.0)

        nn.init.xavier_uniform_(self.attention_weights.weight)
        if self.attention_weights.bias is not None:
            nn.init.zeros_(self.attention_weights.bias)

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict(self, x, hidden=None):
        """Wrapper para predicción con sigmoid."""
        logits, hidden_new = self.forward(x, hidden)
        probs = torch.sigmoid(logits)
        return probs, hidden_new

    def reset_state(self):
        """Reinicia el estado oculto persistente."""
        self.hidden_state = None


# --- 5. Predictor Online con Memoria ---

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


# --- 6. Data Augmentation ---

def aumentar_datos_con_ruido(dataset, factor=5):
    """Genera datos sintéticos solo para la clase minoritaria."""
    features_list = []
    labels_list = []
    
    print("\n--- Iniciando Data Augmentation ---")
    anomaly_count = 0
    normal_count = 0
    
    # Primero agregar todos los datos originales
    for i in range(len(dataset)):
        x, y = dataset[i]
        features_list.append(x)
        labels_list.append(y)
        
        if y == 1.0:
            anomaly_count += 1
        else:
            normal_count += 1
    
    # Luego agregar augmentación solo de anomalías
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == 1.0:
            for _ in range(factor):
                noise = torch.randn_like(x) * 0.02  # Reducido de 0.05 a 0.02
                features_list.append(x + noise)
                labels_list.append(y)
    
    print(f"Anomalías originales: {anomaly_count}")
    print(f"Normales originales: {normal_count}")
    print(f"Datos tras aumentación: {len(features_list)}")
    print(f"Ratio final: {(anomaly_count * (factor + 1)) / len(features_list):.2%}")
    
    x_tensor = torch.stack(features_list)
    y_tensor = torch.stack(labels_list)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)


# --- 7. Cosine Annealing Learning Rate Scheduler con Warm Restarts ---

class CosineAnnealingLRWithRestarts:
    """
    Wrapper personalizado sobre CosineAnnealingWarmRestarts con tracking adicional.
    
    El LR sigue un patrón coseno que decae y se reinicia periódicamente,
    permitiendo escapar de mínimos locales.
    """
    def __init__(self, optimizer, T_0=50, T_mult=1, eta_min=1e-6, initial_lr=0.075):
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

        # Capar LR para evitar saltos por encima del inicial en los restarts
        if new_lr > self.initial_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr
            new_lr = self.initial_lr
        
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


# --- 8. Entrenamiento con Cosine Annealing y SWA ---

def train_template(train_loader, model, val_loader=None, epochs=50, pos_weight=1.0, patience=10, device=None, use_swa=True, swa_start=120):
    """
    Entrenamiento con Cosine Annealing LR scheduler, SWA y soporte CUDA.
    
    Args:
        use_swa: Si usar Stochastic Weight Averaging
        swa_start: Epoch en el que comenzar SWA
    """
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    
    # Usar BCEWithLogitsLoss (más estable, incluye sigmoid)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    initial_lr = 0.0025  # LR más bajo para estabilizar
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Cosine Annealing con Warm Restarts
    # T_0=20: primer ciclo de 20 epochs (más corto para ver progreso antes)
    # T_mult=2: cada ciclo siguiente dura el doble (20, 40, 80, ...)
    ca_scheduler = CosineAnnealingLRWithRestarts(
        optimizer,
        T_0=50,  # Efectivamente sin restarts tempranos
        T_mult=1,
        eta_min=1e-7,
        initial_lr=initial_lr
    )
    
    # Configurar SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=initial_lr * 0.1)  # LR más bajo para SWA
        print(f"   SWA habilitado: comenzará en epoch {swa_start}")
    
    best_val_f1 = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    print(f"\n{'='*70}")
    print("Iniciando entrenamiento con Cosine Annealing Warm Restarts")
    print(f"   T_0={ca_scheduler.T_0}, T_mult={ca_scheduler.T_mult}, eta_min={ca_scheduler.eta_min}")
    print(f"{'='*70}")
    
    model.train()
    warmup_epochs = 10

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(X_batch)  # Ahora retorna logits
            
            loss = criterion(logits, y_batch)  # BCEWithLogitsLoss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Warmup manual los primeros epochs
        if epoch < warmup_epochs:
            warmup_lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            # Actualizar scheduler (después de swa_start, usar swa_scheduler)
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                current_lr = swa_scheduler.get_last_lr()[0] if hasattr(swa_scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            else:
                current_lr = ca_scheduler.step(avg_loss)
        
        # Validación
        val_f1 = 0
        val_loss = 0
        val_precision = 0
        val_recall = 0
        if val_loader is not None:
            model.eval()
            all_preds, all_labels = [], []
            val_loss_sum = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    probs, _ = model.predict(X_batch) 
                    preds = (probs >= 0.5).float()
                    
                    # Loss de validación
                    logits, _ = model(X_batch)
                    val_loss_sum += criterion(logits, y_batch).item()
                    val_batch_count += 1
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(y_batch.cpu().numpy().flatten())
            
            val_loss = val_loss_sum / val_batch_count
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            val_recall = recall_score(all_labels, all_preds, zero_division=0)
            val_precision = precision_score(all_labels, all_preds, zero_division=0)
            model.train()
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        
        # Mostrar progreso cada 5 epochs
        if (epoch + 1) % 5 == 0:
            restart_info = f"Restarts: {ca_scheduler.restarts}"
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} (P:{val_precision:.3f} R:{val_recall:.3f}) | "
                  f"LR: {current_lr:.6f} | {restart_info}")
        
        # Mostrar eventos de restart
        if ca_scheduler.restart_epochs and ca_scheduler.restart_epochs[-1] == epoch + 1:
            print(f"   🔄 WARM RESTART en epoch {epoch+1}: LR -> {current_lr:.6f}")
        
        # Mostrar cuando comienza SWA
        if use_swa and epoch == swa_start:
            print(f"   📊 SWA ACTIVADO en epoch {epoch+1}: promediando pesos")
        
    
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
    
    # Si usamos SWA, actualizar batch normalization y usar el modelo promediado
    if use_swa and swa_model is not None:
        print("\n📊 Finalizando SWA...")
        # Actualizar estadísticas de batch normalization
        from torch.optim.swa_utils import update_bn
        update_bn(train_loader, swa_model, device=device)
        
        # Evaluar modelo SWA vs mejor modelo
        print("Evaluando modelo SWA...")
        swa_model.eval()
        swa_preds, swa_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                probs, _ = swa_model.module.predict(X_batch)
                preds = (probs >= 0.5).float()
                swa_preds.extend(preds.cpu().numpy().flatten())
                swa_labels.extend(y_batch.cpu().numpy().flatten())
        
        swa_f1 = f1_score(swa_labels, swa_preds, zero_division=0)
        print(f"   F1 con SWA: {swa_f1:.4f}")
        print(f"   Mejor F1 sin SWA: {best_val_f1:.4f}")
        
        # Usar el mejor entre SWA y el mejor modelo guardado
        if swa_f1 > best_val_f1:
            print(f"✅ Usando modelo SWA (mejor F1: {swa_f1:.4f})")
            model = swa_model.module
            best_val_f1 = swa_f1
        elif best_model_state is not None:
            print(f"✅ Usando mejor modelo guardado (mejor F1: {best_val_f1:.4f})")
            model.load_state_dict(best_model_state)
            model = model.to(device)
    elif best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        print(f"✅ Restaurado mejor modelo con Val F1: {best_val_f1:.4f}")
    
    return model, ca_scheduler


# --- 9. Búsqueda de Umbral Óptimo con CUDA ---

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


def encontrar_umbral_por_precision(model, val_loader, target_precision=0.70, device=None):
    """Busca el umbral mínimo que alcanza la precisión objetivo en validación.

    Si no se alcanza la precisión objetivo, retorna el umbral con mayor precisión.
    """
    if device is None:
        device = DEVICE

    model = model.to(device)
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs, _ = model.predict(X_batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n--- Búsqueda de Umbral por Precisión ---")
    best_prec = 0.0
    best_thr = 0.5
    best_metrics = (0.0, 0.0)  # (recall, f1)

    # Barrido más fino para priorizar precisión
    for threshold in np.arange(0.10, 0.95, 0.02):
        preds = (all_probs >= threshold).astype(int)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        # Guardar el mejor por precisión alcanzada
        if prec > best_prec or (prec == best_prec and f1 > best_metrics[1]):
            best_prec = prec
            best_thr = threshold
            best_metrics = (rec, f1)

        # Primer umbral que alcanza la precisión objetivo
        if prec >= target_precision:
            print(f"Objetivo alcanzado: Thr {threshold:.2f} -> Precision {prec:.4f}, Recall {rec:.4f}, F1 {f1:.4f} *")
            return threshold

    print(f"No se alcanzó precisión {target_precision:.2f}. Mejor: Thr {best_thr:.2f} -> Precision {best_prec:.4f}, Recall {best_metrics[0]:.4f}, F1 {best_metrics[1]:.4f}")
    return best_thr


# --- 10. Evaluación del Modelo con CUDA ---

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
            
            probs, _ = model.predict(X_batch)  # Usar predict() que aplica sigmoid
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


# --- 11. Demostración de Predicción Online ---

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


# --- 12. Funciones de Exportación/Importación ---

def exportar_modelo_portable(model, scaler, threshold, input_size, seq_len, save_path='modelo_anomalias.pth', ca_stats=None):
    """
    Exporta el modelo de manera portable para usarlo en otros proyectos.
    
    Args:
        model: Modelo entrenado
        scaler: StandardScaler ajustado
        threshold: Umbral óptimo de clasificación
        input_size: Número de features de entrada
        seq_len: Longitud de secuencia
        save_path: Ruta donde guardar el modelo
        ca_stats: Estadísticas del Cosine Annealing (opcional)
    """
    model_cpu = model.cpu()
    
    checkpoint = {
        'model_state_dict': model_cpu.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout': 0.3  # Valor por defecto usado
        },
        'scaler': scaler,
        'threshold': threshold,
        'seq_len': seq_len,
        'input_size': input_size,
        'ca_stats': ca_stats
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✅ Modelo exportado exitosamente a: {save_path}")
    print(f"   Input size: {input_size}")
    print(f"   Seq length: {seq_len}")
    print(f"   Threshold: {threshold:.4f}")
    return checkpoint


def importar_modelo_portable(checkpoint_path, device=None):
    """
    Importa un modelo exportado de manera portable.
    
    Args:
        checkpoint_path: Ruta al archivo .pth
        device: Dispositivo donde cargar el modelo (auto-detecta si es None)
    
    Returns:
        dict con: model, scaler, threshold, config
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📦 Cargando modelo desde: {checkpoint_path}")
    print(f"   Dispositivo: {device}")
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extraer configuración
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Compatibilidad con versiones antiguas
        config = {
            'input_size': checkpoint['input_size'],
            'hidden_size': 192,
            'num_layers': 3,
            'dropout': 0.3
        }
    
    # Reconstruir modelo
    model = VoltageAnomalyModelStateful(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extraer otros componentes
    scaler = checkpoint['scaler']
    threshold = checkpoint['threshold']
    seq_len = checkpoint.get('seq_len', 60)
    
    print(f"✅ Modelo cargado exitosamente")
    print(f"   Input size: {config['input_size']}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Num layers: {config['num_layers']}")
    print(f"   Seq length: {seq_len}")
    print(f"   Threshold: {threshold:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'config': config,
        'seq_len': seq_len,
        'device': device
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
        
        pos_weight_base = train_dataset.get_class_weights()
        pos_weight = pos_weight_base * 1.2  # Aumentar penalidad para falsos negativos (maximizar recall)
        print(f"Peso para clase positiva (base): {pos_weight_base:.2f}")
        print(f"Peso para clase positiva (ajustado x1.2 para recall): {pos_weight:.2f}")
        
        loader_kwargs = {'pin_memory': True, 'num_workers': 4} if torch.cuda.is_available() else {}
        
        if train_dataset.num_anomalies < 200:
            print("Aplicando Data Augmentation...")
            augmented_dataset = aumentar_datos_con_ruido(train_dataset, factor=6)  # Reducido a 6 para balance natural
            train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, **loader_kwargs)  # shuffle=True, batch_size=32
        else:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **loader_kwargs)  # shuffle=True, batch_size=32
        
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **loader_kwargs)

        print("\n" + "="*70)
        print(f"Entrenando Modelo con Cosine Annealing en {DEVICE}...")
        print("="*70)
        
        model = VoltageAnomalyModelStateful(
            input_size=input_size,
        ).to(DEVICE)
        
        # Ahora train_template retorna también el scheduler
        model, ca_scheduler = train_template(
            train_loader, model, val_loader, 
            epochs=400, pos_weight=pos_weight, patience=15, device=DEVICE,
            use_swa=True, swa_start=120  # SWA activado, comienza en epoch 100
        )
        
        # Elegir umbral priorizando RECALL (detección de anomalías) sobre precisión
        # Objetivo más bajo para maximizar true positives, aceptando más falsos positivos
        target_precision = 0.60
        optimal_threshold = encontrar_umbral_por_precision(model, val_loader, target_precision=target_precision, device=DEVICE)
        
        print("\nEvaluando modelo con datos de test (modo batch)...")
        metricas = evaluar_modelo(model, test_loader, threshold=optimal_threshold, device=DEVICE)
        
        predictor, resultados = demo_prediccion_online(model, scaler, test_df, threshold=optimal_threshold, device=DEVICE)
        
        # Guardar modelo de manera portable
        print("\nGuardando modelo...")
        exportar_modelo_portable(
            model, scaler, optimal_threshold, input_size, SEQ_LEN,
            save_path='modelo_anomalias.pth',
            ca_stats=ca_scheduler.get_stats()
        )
        
        print(f"\n{'='*70}")
        print("RESUMEN FINAL")
        print(f"{'='*70}")
        print(f"  Dispositivo usado: {DEVICE}")
        print(f"  Umbral óptimo (maximizando recall, precisión≥{target_precision:.2f}): {optimal_threshold:.2f}")
        print(f"  F1-Score: {metricas['f1_score']:.4f}")
        print(f"  Recall: {metricas['recall']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Warm Restarts: {ca_scheduler.restarts}")
        print(f"\nModelo guardado en: modelo_anomalias.pth")