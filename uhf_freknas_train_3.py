# improved_uhf_multi_signal_model.py
# Üst üste binmiş UHF sinyalleri için geliştirilmiş CNN modeli

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# Hyperparameters ve Ayarlar
# =====================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32  # Multi-signal için daha küçük batch
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 8
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model parametreleri
INPUT_CHANNELS = 4
DROPOUT_RATE = 0.3
MAX_SIGNALS = 4  # Maksimum eşzamanlı sinyal sayısı

# Normalizasyon parametreleri
FREQ_NORM_FACTOR = 3e9  # 3 GHz
BW_NORM_FACTOR = 1e6    # 1 MHz

# Dinamik öğrenme oranı azaltma parametreleri
LR_PATIENCE = 4  # 4 epoch iyileşme yoksa öğrenme oranını azalt
LR_FACTOR = 0.6  # Öğrenme oranını yarıya düşür

print(f"🖥️ Device: {DEVICE}")
print(f"📊 Batch Size: {BATCH_SIZE}")

# =====================================================
# Geliştirilmiş Dataset Loader
# =====================================================
class MultiSignalUHFDataset(Dataset):
    def __init__(self, data_dir, subset_ratio=1.0):
        self.data_dir = data_dir
        self.samples = []
        self.max_signals = MAX_SIGNALS
        self._load_data(subset_ratio)
        
    def _load_data(self, subset_ratio):
        print(f"🔍 Loading multi-signal data from {self.data_dir}")
        
        with open(os.path.join(self.data_dir, 'manifest.json'), 'r') as f:
            self.manifest = json.load(f)
            
        shard_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('shard_')]
        shard_dirs.sort()
        
        total_loaded = 0
        for shard_dir in tqdm(shard_dirs, desc="Loading shards"):
            shard_path = os.path.join(self.data_dir, shard_dir)
            
            features = np.load(os.path.join(shard_path, 'features.npy'))
            with open(os.path.join(shard_path, 'labels.pkl'), 'rb') as f:
                labels = pickle.load(f)
                
            for i, label in enumerate(labels):
                if label['type'] != 'noise' and label['num_signals'] > 0:
                    feature = features[i]  # [4, F, T]
                    
                    # Multi-signal target hazırlama
                    signals_data = self._prepare_multi_signal_targets(label['signals'])
                    
                    if signals_data is not None:
                        self.samples.append({
                            'feature': feature.astype(np.float32),
                            'num_signals': len(label['signals']),
                            'signals_data': signals_data,
                            'snr_db': label['snr_db_total']
                        })
                        
            total_loaded += len(labels)
            
            if len(self.samples) >= int(subset_ratio * total_loaded):
                break
                
        print(f"✅ Loaded {len(self.samples)} multi-signal samples")
        
        # İstatistikler
        signal_counts = [s['num_signals'] for s in self.samples]
        print(f"📈 Signal count distribution:")
        for i in range(1, self.max_signals + 1):
            count = sum(1 for x in signal_counts if x == i)
            print(f"   {i} signals: {count} samples ({count/len(self.samples)*100:.1f}%)")
        
    def _prepare_multi_signal_targets(self, signals):
        """Multi-signal targets hazırla"""
        if len(signals) > self.max_signals:
            return None  # Çok fazla sinyal, atla
            
        # Fixed size arrays hazırla
        freq_targets = np.zeros(self.max_signals, dtype=np.float32)
        bw_targets = np.zeros(self.max_signals, dtype=np.float32)
        signal_mask = np.zeros(self.max_signals, dtype=np.float32)  # Hangi pozisyonlar valid
        
        for i, signal_info in enumerate(signals[:self.max_signals]):
            freq_center = signal_info.get('f_center_est_hz', 0.0)
            bandwidth = signal_info.get('bw_occ99_hz', 0.0)
            
            if abs(freq_center) < 1e6 and bandwidth > 0:
                # Normalize
                freq_targets[i] = freq_center / FREQ_NORM_FACTOR
                bw_targets[i] = bandwidth / BW_NORM_FACTOR
                signal_mask[i] = 1.0
            else:
                return None  # Geçersiz sinyal, atla
                
        return {
            'frequencies': freq_targets,
            'bandwidths': bw_targets,
            'mask': signal_mask,
            'num_valid': int(np.sum(signal_mask))
        }
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        feature = torch.from_numpy(sample['feature'])
        
        # Multi-signal targets
        signals_data = sample['signals_data']
        targets = {
            'frequencies': torch.from_numpy(signals_data['frequencies']),
            'bandwidths': torch.from_numpy(signals_data['bandwidths']),
            'mask': torch.from_numpy(signals_data['mask']),
            'num_signals': torch.tensor(signals_data['num_valid'], dtype=torch.long)
        }
        
        return feature, targets

# =====================================================
# Attention-Enhanced Multi-Signal CNN
# =====================================================
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class MultiSignalUHFCNN(nn.Module):
    def __init__(self, input_channels=4, max_signals=4, dropout_rate=0.3):
        super(MultiSignalUHFCNN, self).__init__()
        
        self.max_signals = max_signals
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionBlock(64),  # Attention eklendi
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.3),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.4),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AttentionBlock(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            AttentionBlock(512),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_rate * 0.6),
        )
        
        # Global feature pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Sinyal sayısı tahmin başı
        self.num_signals_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, max_signals + 1)  # 0 to max_signals
        )
        
        # Multi-signal detection heads
        self.freq_detector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(128, 1)
            ) for _ in range(max_signals)
        ])
        
        self.bw_detector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(128, 1)
            ) for _ in range(max_signals)
        ])
        
        # Signal confidence heads (hangi çıkışlar geçerli)
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(max_signals)
        ])
        
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)  # [batch, 512, 4, 4]
        
        # Global pooling
        global_features = self.global_pool(features)  # [batch, 512, 1, 1]
        global_features = global_features.view(global_features.size(0), -1)  # [batch, 512]
        
        # Sinyal sayısı tahmini
        num_signals_logits = self.num_signals_head(global_features)
        
        # Her sinyal pozisyonu için tahminler
        frequencies = []
        bandwidths = []
        confidences = []
        
        for i in range(self.max_signals):
            freq_out = self.freq_detector[i](global_features).squeeze(-1)
            bw_out = self.bw_detector[i](global_features).squeeze(-1)
            conf_out = self.confidence_heads[i](global_features).squeeze(-1)
            
            frequencies.append(freq_out)
            bandwidths.append(bw_out)
            confidences.append(conf_out)
        
        # Stack outputs
        frequencies = torch.stack(frequencies, dim=1)  # [batch, max_signals]
        bandwidths = torch.stack(bandwidths, dim=1)    # [batch, max_signals]
        confidences = torch.stack(confidences, dim=1)  # [batch, max_signals]
        
        return {
            'frequencies': frequencies,
            'bandwidths': bandwidths,
            'confidences': confidences,
            'num_signals_logits': num_signals_logits
        }

# =====================================================
# Multi-Signal Loss Function
# =====================================================
class MultiSignalLoss(nn.Module):
    def __init__(self, freq_weight=1.0, bw_weight=1.0, conf_weight=0.5, count_weight=0.3):
        super(MultiSignalLoss, self).__init__()
        self.freq_weight = freq_weight
        self.bw_weight = bw_weight
        self.conf_weight = conf_weight
        self.count_weight = count_weight
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        batch_size = predictions['frequencies'].size(0)
        
        # Mask: hangi pozisyonlar geçerli
        mask = targets['mask']  # [batch, max_signals]
        
        # Frequency loss (sadece geçerli pozisyonlarda)
        freq_loss = self.mse_loss(predictions['frequencies'], targets['frequencies'])
        freq_loss = (freq_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Bandwidth loss
        bw_loss = self.mse_loss(predictions['bandwidths'], targets['bandwidths'])
        bw_loss = (bw_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Confidence loss (signal detection)
        conf_loss = self.bce_loss(predictions['confidences'], mask)
        conf_loss = conf_loss.mean()
        
        # Signal count loss
        count_loss = self.ce_loss(predictions['num_signals_logits'], targets['num_signals'])
        
        # Total loss
        total_loss = (self.freq_weight * freq_loss + 
                     self.bw_weight * bw_loss + 
                     self.conf_weight * conf_loss + 
                     self.count_weight * count_loss)
        
        return {
            'total': total_loss,
            'freq': freq_loss,
            'bandwidth': bw_loss,
            'confidence': conf_loss,
            'count': count_loss
        }

# =====================================================
# Training Functions with Dynamic LR
# =====================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    losses_dict = {'freq': [], 'bandwidth': [], 'confidence': [], 'count': []}
    
    pbar = tqdm(dataloader, desc="Training")
    for features, targets in pbar:
        features = features.to(device)
        # Move all target tensors to device
        targets = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        outputs = model(features)
        losses = criterion(outputs, targets)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses['total'].item()
        for key in losses_dict:
            if key in losses:
                losses_dict[key].append(losses[key].item())
        
        pbar.set_postfix({
            'Loss': f"{losses['total'].item():.6f}",
            'Freq': f"{losses['freq'].item():.6f}",
            'Conf': f"{losses['confidence'].item():.6f}"
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        **{k: np.mean(v) for k, v in losses_dict.items()}
    }

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    losses_dict = {'freq': [], 'bandwidth': [], 'confidence': [], 'count': []}
    
    # Metrics collection
    all_freq_pred, all_freq_true = [], []
    all_bw_pred, all_bw_true = [], []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(features)
            losses = criterion(outputs, targets)
            
            total_loss += losses['total'].item()
            for key in losses_dict:
                if key in losses:
                    losses_dict[key].append(losses[key].item())
            
            # Collect valid predictions for metrics
            mask = targets['mask']  # [batch, max_signals]
            
            # Flatten and collect only valid predictions
            for b in range(mask.size(0)):
                for s in range(mask.size(1)):
                    if mask[b, s] > 0.5:  # Valid signal
                        all_freq_pred.append(outputs['frequencies'][b, s].cpu().item())
                        all_freq_true.append(targets['frequencies'][b, s].cpu().item())
                        all_bw_pred.append(outputs['bandwidths'][b, s].cpu().item())
                        all_bw_true.append(targets['bandwidths'][b, s].cpu().item())
    
    # Calculate metrics
    if len(all_freq_true) > 0:
        freq_mae = mean_absolute_error(all_freq_true, all_freq_pred)
        bw_mae = mean_absolute_error(all_bw_true, all_bw_pred)
        freq_r2 = r2_score(all_freq_true, all_freq_pred) if len(all_freq_true) > 1 else 0.0
        bw_r2 = r2_score(all_bw_true, all_bw_pred) if len(all_bw_true) > 1 else 0.0
    else:
        freq_mae = bw_mae = freq_r2 = bw_r2 = 0.0
    
    return {
        'total_loss': total_loss / len(dataloader),
        'freq_mae': freq_mae,
        'bw_mae': bw_mae,
        'freq_r2': freq_r2,
        'bw_r2': bw_r2,
        **{k: np.mean(v) for k, v in losses_dict.items()}
    }

# =====================================================
# Enhanced Training Function
# =====================================================
def train_multi_signal_model(data_dir, model_save_path="multi_signal_uhf_model.pth"):
    print("🚀 Starting Multi-Signal UHF Prediction Model Training")
    print("=" * 70)
    
    # Load dataset
    dataset = MultiSignalUHFDataset(data_dir, subset_ratio=1.0)
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(TEST_SPLIT * total_size)
    val_size = int(VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📈 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    model = MultiSignalUHFCNN(input_channels=INPUT_CHANNELS, 
                             max_signals=MAX_SIGNALS, 
                             dropout_rate=DROPOUT_RATE)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = MultiSignalLoss(freq_weight=1.0, bw_weight=1.0, conf_weight=0.5, count_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Dinamik öğrenme oranı scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, 
        verbose=True, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_freq_loss': [], 'val_freq_loss': [],
        'train_bw_loss': [], 'val_bw_loss': [],
        'val_freq_mae': [], 'val_bw_mae': [],
        'val_freq_r2': [], 'val_bw_r2': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🎯 Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        print(f"📚 Learning Rate: {current_lr:.2e}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Update scheduler - 4 epoch iyileşme yoksa LR düşür
        scheduler.step(val_metrics['total_loss'])
        
        # Record history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_freq_loss'].append(train_metrics.get('freq', 0))
        history['val_freq_loss'].append(val_metrics.get('freq', 0))
        history['train_bw_loss'].append(train_metrics.get('bandwidth', 0))
        history['val_bw_loss'].append(val_metrics.get('bandwidth', 0))
        history['val_freq_mae'].append(val_metrics['freq_mae'])
        history['val_bw_mae'].append(val_metrics['bw_mae'])
        history['val_freq_r2'].append(val_metrics['freq_r2'])
        history['val_bw_r2'].append(val_metrics['bw_r2'])
        
        # Print epoch summary
        print(f"📊 Train Loss: {train_metrics['total_loss']:.6f}")
        print(f"📊 Val Loss: {val_metrics['total_loss']:.6f}")
        print(f"🎯 Val Freq MAE: {val_metrics['freq_mae']:.6f} | R²: {val_metrics['freq_r2']:.4f}")
        print(f"🎯 Val BW MAE: {val_metrics['bw_mae']:.6f} | R²: {val_metrics['bw_r2']:.4f}")
        
        # Early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'model_config': {
                    'input_channels': INPUT_CHANNELS,
                    'max_signals': MAX_SIGNALS,
                    'dropout_rate': DROPOUT_RATE
                }
            }, model_save_path)
            print(f"💾 Model saved! Best val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"⏱️ Early stopping triggered after {PATIENCE} epochs without improvement")
            break
    
    print("\n" + "=" * 70)
    print("✅ Training completed!")
    
    # Final test evaluation
    print("\n🧪 Final test evaluation...")
    test_metrics = validate_epoch(model, test_loader, criterion, DEVICE)
    
    print(f"🎯 Test Results:")
    print(f"   Total Loss: {test_metrics['total_loss']:.6f}")
    print(f"   Freq MAE: {test_metrics['freq_mae']:.6f} (norm) | R²: {test_metrics['freq_r2']:.4f}")
    print(f"   BW MAE: {test_metrics['bw_mae']:.6f} (norm) | R²: {test_metrics['bw_r2']:.4f}")
    
    # Convert to actual units
    freq_mae_hz = test_metrics['freq_mae'] * FREQ_NORM_FACTOR
    bw_mae_hz = test_metrics['bw_mae'] * BW_NORM_FACTOR
    
    print(f"\n🔍 Test Results (Actual Units):")
    print(f"   Freq MAE: {freq_mae_hz:.0f} Hz")
    print(f"   BW MAE: {bw_mae_hz:.0f} Hz")
    
    return model, history, test_metrics

# =====================================================
# Enhanced Inference Function
# =====================================================
def predict_multi_signals(model_path, features, confidence_threshold=0.5):
    """
    Trained model ile multiple signal detection ve parameter estimation
    
    Args:
        model_path: Eğitilmiş model dosya yolu
        features: Input features [4, F, T] veya [batch, 4, F, T]
        confidence_threshold: Sinyal detection threshold
    
    Returns:
        list: [{'freq_center_hz': float, 'bandwidth_hz': float, 'confidence': float}, ...]
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_config = checkpoint['model_config']
    
    model = MultiSignalUHFCNN(
        input_channels=model_config['input_channels'],
        max_signals=model_config['max_signals'],
        dropout_rate=model_config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Prepare input
    if features.ndim == 3:
        features = features.unsqueeze(0)
    
    features = torch.from_numpy(features).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(features)
        
        frequencies = outputs['frequencies'][0]  # [max_signals]
        bandwidths = outputs['bandwidths'][0]    # [max_signals]
        confidences = outputs['confidences'][0]  # [max_signals]
        num_signals_logits = outputs['num_signals_logits'][0]  # [max_signals+1]
    
    # Extract detected signals
    detected_signals = []
    
    for i in range(model_config['max_signals']):
        confidence = confidences[i].cpu().item()
        
        if confidence > confidence_threshold:
            freq_norm = frequencies[i].cpu().item()
            bw_norm = bandwidths[i].cpu().item()
            
            freq_hz = freq_norm * FREQ_NORM_FACTOR
            bw_hz = bw_norm * BW_NORM_FACTOR
            
            detected_signals.append({
                'freq_center_hz': freq_hz,
                'bandwidth_hz': bw_hz,
                'confidence': confidence,
                'freq_normalized': freq_norm,
                'bw_normalized': bw_norm
            })
    
    # Predicted signal count
    predicted_count = torch.argmax(num_signals_logits).cpu().item()
    
    # Sort by confidence (highest first)
    detected_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'detected_signals': detected_signals,
        'predicted_count': predicted_count,
        'actual_detections': len(detected_signals)
    }

# =====================================================
# Visualization Functions
# =====================================================
def plot_training_history(history, save_path=None):
    """Training history görselleştirme"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Signal UHF Model Training History', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency loss
    axes[0, 1].plot(history['train_freq_loss'], label='Train', alpha=0.8)
    axes[0, 1].plot(history['val_freq_loss'], label='Validation', alpha=0.8)
    axes[0, 1].set_title('Frequency Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bandwidth loss
    axes[0, 2].plot(history['train_bw_loss'], label='Train', alpha=0.8)
    axes[0, 2].plot(history['val_bw_loss'], label='Validation', alpha=0.8)
    axes[0, 2].set_title('Bandwidth Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # MAE curves
    axes[1, 0].plot(history['val_freq_mae'], label='Frequency MAE', alpha=0.8)
    axes[1, 0].plot(history['val_bw_mae'], label='Bandwidth MAE', alpha=0.8)
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (normalized)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # R² scores
    axes[1, 1].plot(history['val_freq_r2'], label='Frequency R²', alpha=0.8)
    axes[1, 1].plot(history['val_bw_r2'], label='Bandwidth R²', alpha=0.8)
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.1, 1.1)
    
    # Learning rate
    axes[1, 2].plot(history['learning_rates'], alpha=0.8, color='red')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {save_path}")
    
    plt.show()

def visualize_prediction(features, predictions, ground_truth=None, save_path=None):
    """Prediction sonuçlarını spektrogram üzerinde görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Signal Detection Results', fontsize=16)
    
    # Log power spectrum
    log_power = features[0]  # [F, T] - first channel
    axes[0, 0].imshow(log_power, aspect='auto', origin='lower', cmap='plasma')
    axes[0, 0].set_title('Log Power Spectrum')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Frequency Bin')
    
    # Phase
    phase = features[1]  # [F, T] - second channel
    axes[0, 1].imshow(phase, aspect='auto', origin='lower', cmap='hsv')
    axes[0, 1].set_title('Phase')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Frequency Bin')
    
    # Instantaneous frequency
    inst_freq = features[2]  # [F, T] - third channel
    axes[1, 0].imshow(inst_freq, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title('Instantaneous Frequency')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Frequency Bin')
    
    # Detection overlay
    axes[1, 1].imshow(log_power, aspect='auto', origin='lower', cmap='gray', alpha=0.7)
    
    # Predicted signals overlay (kırmızı)
    if 'detected_signals' in predictions:
        for i, signal in enumerate(predictions['detected_signals']):
            freq_norm = signal['freq_normalized']
            bw_norm = signal['bw_normalized'] 
            confidence = signal['confidence']
            
            # Convert to frequency bin (rough approximation)
            freq_bin = (freq_norm + 1) * log_power.shape[0] / 2  # Assuming freq range [-1, 1]
            bw_bins = bw_norm * log_power.shape[0] / 2
            
            # Draw detection box
            y_min = max(0, freq_bin - bw_bins/2)
            y_max = min(log_power.shape[0], freq_bin + bw_bins/2)
            
            axes[1, 1].axhspan(y_min, y_max, alpha=0.3, color='red', 
                              label=f'Pred {i+1} (conf={confidence:.2f})')
    
    # Ground truth overlay (yeşil) if provided
    if ground_truth:
        for i, signal in enumerate(ground_truth):
            freq_norm = signal.get('freq_normalized', 0)
            bw_norm = signal.get('bw_normalized', 0)
            
            freq_bin = (freq_norm + 1) * log_power.shape[0] / 2
            bw_bins = bw_norm * log_power.shape[0] / 2
            
            y_min = max(0, freq_bin - bw_bins/2)
            y_max = min(log_power.shape[0], freq_bin + bw_bins/2)
            
            axes[1, 1].axhspan(y_min, y_max, alpha=0.2, color='green',
                              label=f'GT {i+1}')
    
    axes[1, 1].set_title('Detection Overlay (Red=Pred, Green=GT)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Frequency Bin')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Prediction visualization saved to: {save_path}")
    
    plt.show()

# =====================================================
# Model Evaluation Functions
# =====================================================
def evaluate_multi_signal_detection(model_path, test_loader, device=DEVICE):
    """Multi-signal detection performansını değerlendir"""
    print("🧪 Evaluating Multi-Signal Detection Performance...")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = MultiSignalUHFCNN(
        input_channels=model_config['input_channels'],
        max_signals=model_config['max_signals'],
        dropout_rate=model_config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Metrics
    total_samples = 0
    correct_count_predictions = 0
    detection_scores = []  # Per-signal detection accuracy
    freq_errors = []
    bw_errors = []
    
    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(features)
            
            batch_size = features.size(0)
            total_samples += batch_size
            
            # Count prediction accuracy
            predicted_counts = torch.argmax(outputs['num_signals_logits'], dim=1)
            true_counts = targets['num_signals']
            correct_count_predictions += (predicted_counts == true_counts).sum().item()
            
            # Per-signal detection accuracy
            confidences = outputs['confidences']  # [batch, max_signals]
            masks = targets['mask']  # [batch, max_signals]
            
            # Detection: confidence > 0.5 and mask == 1
            detections = (confidences > 0.5).float()
            
            for b in range(batch_size):
                true_signals = masks[b].sum().int().item()
                detected_signals = detections[b].sum().int().item()
                
                # True positives
                tp = ((detections[b] == 1) & (masks[b] == 1)).sum().item()
                # False positives  
                fp = ((detections[b] == 1) & (masks[b] == 0)).sum().item()
                # False negatives
                fn = ((detections[b] == 0) & (masks[b] == 1)).sum().item()
                
                # Detection metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                detection_scores.append({
                    'true_count': true_signals,
                    'detected_count': detected_signals,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                # Parameter errors for correctly detected signals
                for s in range(model_config['max_signals']):
                    if masks[b, s] == 1 and detections[b, s] == 1:  # True positive
                        freq_error = abs(outputs['frequencies'][b, s] - targets['frequencies'][b, s]).item()
                        bw_error = abs(outputs['bandwidths'][b, s] - targets['bandwidths'][b, s]).item()
                        
                        freq_errors.append(freq_error)
                        bw_errors.append(bw_error)
    
    # Calculate overall metrics
    count_accuracy = correct_count_predictions / total_samples
    
    avg_precision = np.mean([s['precision'] for s in detection_scores])
    avg_recall = np.mean([s['recall'] for s in detection_scores])
    avg_f1 = np.mean([s['f1'] for s in detection_scores])
    
    avg_freq_error = np.mean(freq_errors) if freq_errors else 0.0
    avg_bw_error = np.mean(bw_errors) if bw_errors else 0.0
    
    # Convert to actual units
    freq_error_hz = avg_freq_error * FREQ_NORM_FACTOR
    bw_error_hz = avg_bw_error * BW_NORM_FACTOR
    
    print("\n📊 Multi-Signal Detection Results:")
    print(f"   Signal Count Accuracy: {count_accuracy:.3f}")
    print(f"   Average Precision: {avg_precision:.3f}")
    print(f"   Average Recall: {avg_recall:.3f}")
    print(f"   Average F1-Score: {avg_f1:.3f}")
    print(f"   Average Freq Error: {freq_error_hz:.0f} Hz")
    print(f"   Average BW Error: {bw_error_hz:.0f} Hz")
    
    return {
        'count_accuracy': count_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'freq_error_hz': freq_error_hz,
        'bw_error_hz': bw_error_hz,
        'detection_scores': detection_scores
    }

# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # Veri klasör yolu
    DATA_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_9_1"
    MODEL_SAVE_PATH = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_9_1\model_train\multi_signal_uhf_predictor.pth"
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please run your UHF dataset generation script first!")
        exit(1)
    
    # Eğitim başlat
    print("🚀 Starting Multi-Signal UHF Model Training with Dynamic LR...")
    print(f"📚 Learning rate will be reduced by {LR_FACTOR} after {LR_PATIENCE} epochs without improvement")
    
    model, history, test_metrics = train_multi_signal_model(DATA_DIR, MODEL_SAVE_PATH)
    
    # Training history görselleştir
    plot_training_history(history, "training_history.png")
    
    print(f"\n🎉 Multi-signal model saved to: {MODEL_SAVE_PATH}")
    print("🚀 You can now use predict_multi_signals() function for inference!")
    
    # Example usage
    print("\n📝 Example Usage:")
    print("```python")
    print("# Load a sample")
    print("features = your_feature_array  # [4, F, T]")
    print()
    print("# Predict multiple signals")
    print("results = predict_multi_signals('multi_signal_uhf_predictor.pth', features)")
    print()
    print("# Print results")
    print("for i, signal in enumerate(results['detected_signals']):")
    print("    print(f'Signal {i+1}: {signal['freq_center_hz']:.0f} Hz, {signal['bandwidth_hz']:.0f} Hz, conf={signal['confidence']:.3f}')")
    print("```")