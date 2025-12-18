"""
Script de Fine-tuning para el modelo de detección de anomalías.
Carga un modelo preentrenado y lo ajusta con datos específicos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from training_template import (
    VoltageDropDataset, 
    importar_modelo_portable,
    DEVICE
)

def fine_tune_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=0.0001,
    freeze_lstm_layers=2,
    pos_weight=1.0,
    device=None
):
    """
    Fine-tuning del modelo con estrategia de congelamiento parcial.
    
    Args:
        model: Modelo cargado
        train_loader: DataLoader de entrenamiento (puede ser val/test del entrenamiento original)
        val_loader: DataLoader de validación
        epochs: Número de epochs de fine-tuning
        lr: Learning rate bajo para fine-tuning
        freeze_lstm_layers: Cuántas capas LSTM congelar (desde el inicio)
        pos_weight: Peso para clase positiva
        device: Dispositivo de cómputo
    """
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    
    # Congelar capas LSTM iniciales (mantener conocimiento base)
    print(f"\n🔒 Congelando primeras {freeze_lstm_layers} capas LSTM...")
    for name, param in model.lstm.named_parameters():
        # Congelar pesos de las primeras N capas
        layer_match = False
        for i in range(freeze_lstm_layers):
            if f'_l{i}' in name or (i == 0 and '_l' not in name):
                layer_match = True
                break
        
        if layer_match:
            param.requires_grad = False
            print(f"   Congelado: {name}")
    
    # Asegurar que atención y clasificador están descongelados
    for param in model.attention_weights.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Contar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros entrenables: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Configurar entrenamiento
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    # Optimizador solo para parámetros no congelados
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5
    )
    
    # Scheduler suave
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1
    )
    
    best_f1 = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f"Iniciando Fine-tuning: {epochs} epochs, LR={lr}")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=0.5
            )
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validación
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits, _ = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                
                probs, _ = model.predict(X_batch)
                preds = (probs >= 0.5).float()
                
                val_probs.extend(probs.cpu().numpy().flatten())
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels.extend(y_batch.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Métricas
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Guardar mejor modelo
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "✓"
        else:
            patience_counter += 1
            status = ""
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"P: {val_precision:.3f} R: {val_recall:.3f} F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f} {status}")
        
        scheduler.step()
        
    
    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        print(f"\n✅ Restaurado mejor modelo con Val F1: {best_f1:.4f}")
    
    return model, best_f1


def calibrar_threshold_finetuned(model, val_loader, device=None):
    """
    Recalibra el threshold tras fine-tuning para maximizar F1.
    """
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    model.eval()
    
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs, _ = model.predict(X_batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_f1 = 0.0
    best_thr = 0.5
    
    print("\n--- Recalibración de Threshold Post Fine-tuning ---")
    for thr in np.arange(0.10, 0.95, 0.02):
        preds = (all_probs >= thr).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            prec = precision_score(all_labels, preds, zero_division=0)
            rec = recall_score(all_labels, preds, zero_division=0)
            print(f"Thr {thr:.2f} -> F1: {f1:.4f} (P: {prec:.3f}, R: {rec:.3f}) *")
    
    print(f"\n✅ Mejor threshold: {best_thr:.2f} con F1: {best_f1:.4f}")
    return best_thr


def evaluar_finetuned(model, test_loader, threshold, device=None):
    """Evalúa modelo tras fine-tuning."""
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs, _ = model.predict(X_batch)
            preds = (probs >= threshold).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf = confusion_matrix(all_labels, all_preds)
    
    tp = conf[1, 1] if len(conf) > 1 else 0
    fp = conf[0, 1] if len(conf) > 1 else 0
    fn = conf[1, 0] if len(conf) > 1 else 0
    
    print("\n" + "="*70)
    print("RESULTADOS POST FINE-TUNING")
    print("="*70)
    print(f"Threshold: {threshold:.2f}")
    print(f"Precisión: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nTrue Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("="*70)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


if __name__ == "__main__":
    print("="*70)
    print("FINE-TUNING DEL MODELO DE DETECCIÓN DE ANOMALÍAS")
    print("="*70)
    
    # Cargar datos
    print("\n1. Cargando datos...")
    with open("datos_procesados.pkl", "rb") as f:
        datos_lista = pickle.load(f)
    
    df = pd.DataFrame(datos_lista)
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    df = df.sort_values('tiempo').reset_index(drop=True)
    
    # División: usar últimos datos para fine-tuning
    # Opción 1: Usar test set original como fine-tuning set
    train_idx = int(len(df) * 0.80)
    val_idx = int(len(df) * 0.90)
    
    finetune_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    print(f"   Fine-tuning set: {len(finetune_df)} muestras")
    print(f"   Test set: {len(test_df)} muestras")
    
    # Cargar modelo preentrenado
    print("\n2. Cargando modelo preentrenado...")
    model_dict = importar_modelo_portable('modelo_anomalias.pth', device=DEVICE)
    model = model_dict['model']
    scaler = model_dict['scaler']
    old_threshold = model_dict['threshold']
    
    print(f"   Threshold original: {old_threshold:.2f}")
    
    # Crear datasets
    SEQ_LEN = 60
    finetune_dataset = VoltageDropDataset(finetune_df, seq_len=SEQ_LEN, scaler=scaler)
    test_dataset = VoltageDropDataset(test_df, seq_len=SEQ_LEN, scaler=scaler)
    
    # Split finetune en train/val para fine-tuning
    finetune_size = len(finetune_dataset)
    finetune_train_size = int(finetune_size * 0.7)
    finetune_val_size = finetune_size - finetune_train_size
    
    finetune_train, finetune_val = torch.utils.data.random_split(
        finetune_dataset,
        [finetune_train_size, finetune_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_kwargs = {'pin_memory': torch.cuda.is_available(), 'num_workers': 0}
    finetune_train_loader = DataLoader(finetune_train, batch_size=32, shuffle=True, **loader_kwargs)
    finetune_val_loader = DataLoader(finetune_val, batch_size=64, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **loader_kwargs)
    
    print(f"\n   Fine-tune train: {len(finetune_train)} muestras")
    print(f"   Fine-tune val: {len(finetune_val)} muestras")
    
    # Calcular pos_weight para fine-tuning
    anomaly_count = sum(1 for _, (_, y) in enumerate(finetune_train) if y.item() == 1.0)
    normal_count = len(finetune_train) - anomaly_count
    pos_weight = (normal_count / anomaly_count) if anomaly_count > 0 else 1.0
    pos_weight = pos_weight * 0.7  # Ajuste para balancear precision/recall
    
    print(f"   Pos weight para fine-tuning: {pos_weight:.2f}")
    
    # Fine-tuning
    print("\n3. Ejecutando fine-tuning...")
    finetuned_model, best_f1 = fine_tune_model(
        model=model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        epochs=60,
        lr=0.0001,  # LR bajo pero suficiente para ajuste fino
        freeze_lstm_layers=1,  # Congelar solo 1 capa para más flexibilidad
        pos_weight=pos_weight,
        device=DEVICE
    )
    
    # Recalibrar threshold
    print("\n4. Recalibrando threshold...")
    new_threshold = calibrar_threshold_finetuned(finetuned_model, finetune_val_loader, device=DEVICE)
    
    # Evaluación final en test
    print("\n5. Evaluación final en test set...")
    metricas = evaluar_finetuned(finetuned_model, test_loader, new_threshold, device=DEVICE)
    
    # Guardar modelo fine-tuned
    print("\n6. Guardando modelo fine-tuned...")
    from training_template import exportar_modelo_portable
    
    exportar_modelo_portable(
        finetuned_model,
        scaler,
        new_threshold,
        model_dict['config']['input_size'],
        SEQ_LEN,
        save_path='modelo_anomalias_finetuned.pth',
        ca_stats=None
    )
    
    print("\n" + "="*70)
    print("COMPARACIÓN PRE vs POST FINE-TUNING")
    print("="*70)
    print(f"Threshold: {old_threshold:.2f} → {new_threshold:.2f}")
    print(f"F1-Score: 71.46% → {metricas['f1']*100:.2f}%")
    print(f"Precisión: 65.89% → {metricas['precision']*100:.2f}%")
    print(f"Recall: 78.05% → {metricas['recall']*100:.2f}%")
    print("="*70)
    print("\n✅ Fine-tuning completado!")
    print(f"   Modelo guardado: modelo_anomalias_finetuned.pth")
