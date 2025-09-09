#!/usr/bin/env python3
"""
Hibrit UHF Sinyal Tespiti - CNN + Transformer Yaklaşımı
Dataset üreticinizle tam uyumlu eğitim kodu
"""

import os
import json
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# Konfigürasyon
# =====================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {DEVICE}")

# Dataset parametreleri (generator kodunuzla uyumlu)
FS = 2_000_000  # 2 MHz örnekleme hızı
N_FFT = 256
N_OVERLAP = 128
MAX_SIGNALS = 3
UHF_MIN = 300e6
UHF_MAX = 3e9

# Eğitim konfigürasyonu
CONFIG = {
    'model': {
        'input_channels': 1,  # STFT magnitude
        'max_signals': MAX_SIGNALS,
        'dropout_rate': 0.3,
        'hidden_dim': 256,
        'transformer_layers': 4,
        'attention_heads': 8
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 50,
        'patience': 11,
        'lr_patience': 5,
        'lr_factor': 0.5,
        'validation_split': 0.2,
        'gradient_clip': 1.0
    },
    'loss': {
        'freq_weight': 2.0,
        'bw_weight': 1.5,
        'power_weight': 1.0,
        'count_weight': 1.0,
        'mod_weight': 0.8,
        'spectral_weight': 0.5
    }
}

# =====================================================
# Hibrit Model Mimarisi
# =====================================================
class PositionalEncoding(nn.Module):
    """Transformer için pozisyonel kodlama"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SelfAttention(nn.Module):
    """Self-attention mekanizması"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        # Multi-head attention
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return self.output(out)

class HybridUHFDetector(nn.Module):
    """Hibrit CNN + Transformer UHF Sinyal Tespit Modeli"""
    
    def __init__(self, input_channels=1, max_signals=3, dropout_rate=0.3, 
                 hidden_dim=256, transformer_layers=4, attention_heads=8):
        super().__init__()
        
        self.max_signals = max_signals
        self.hidden_dim = hidden_dim
        
        # CNN Feature Extractor (Spektrogram için)
        self.cnn_backbone = nn.Sequential(
            # İlk blok: Zaman-frekans özellik çıkarma
            nn.Conv2d(input_channels, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1),
            
            # İkinci blok: Daha karmaşık desenler
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.15),
            
            # Üçüncü blok: Yüksek seviye özellikler
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Dropout2d(0.2),
        )
        
        # CNN çıkışını dönüştürme
        cnn_output_dim = 128 * 16 * 16
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Transformer komponenti
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers
        )
        
        # Self-attention mekanizması
        self.self_attention = SelfAttention(hidden_dim, attention_heads, dropout_rate)
        
        # Global özellik füzyon
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # CNN + Transformer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        
        # Sinyal sayısı tahmini
        self.count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, max_signals + 1)  # 0 ile max_signals arası
        )
        
        # Her sinyal için parametreleri tahmin et
        # Frekans merkezi, bant genişliği, güç, modülasyon türü, spektral özellikler
        signal_param_dim = 9  # freq, bw, power, mod_type(4-class), spectral_entropy, spectral_kurtosis
        self.signal_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim // 2, signal_param_dim)
            ) for _ in range(max_signals)
        ])
        
        # Sinyal varlık tespiti (binary classification)
        self.presence_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(max_signals)
        ])
        
    def forward(self, spectrogram):
        batch_size = spectrogram.size(0)
        
        # CNN özellik çıkarma
        cnn_features = self.cnn_backbone(spectrogram)  # [B, 128, 16, 16]
        cnn_features = cnn_features.view(batch_size, -1)  # [B, 128*16*16]
        cnn_projected = self.cnn_projection(cnn_features)  # [B, hidden_dim]
        
        # Transformer için sekans oluştur (suni sekans)
        # Her batch öğesi için sabit bir sekans uzunluğu kullan
        seq_length = 32
        transformer_input = cnn_projected.unsqueeze(1).repeat(1, seq_length, 1)  # [B, seq_len, hidden_dim]
        
        # Pozisyonel kodlama ekle
        transformer_input = transformer_input.transpose(0, 1)  # [seq_len, B, hidden_dim]
        transformer_input = self.pos_encoding(transformer_input)
        
        # Transformer
        transformer_features = self.transformer(transformer_input)  # [seq_len, B, hidden_dim]
        transformer_features = transformer_features.mean(0)  # [B, hidden_dim] - global pooling
        
        # Self-attention
        attention_input = cnn_projected.unsqueeze(1)  # [B, 1, hidden_dim]
        attention_features = self.self_attention(attention_input)  # [B, 1, hidden_dim]
        attention_features = attention_features.squeeze(1)  # [B, hidden_dim]
        
        # Özellik füzyonu
        fused_features = torch.cat([cnn_projected, transformer_features], dim=1)  # [B, hidden_dim*2]
        global_features = self.feature_fusion(fused_features)  # [B, hidden_dim]
        
        # Sinyal sayısı tahmini
        count_logits = self.count_predictor(global_features)  # [B, max_signals+1]
        
        # Her sinyal için tahminler
        signal_params = []
        presence_logits = []
        
        for i in range(self.max_signals):
            # Sinyal parametreleri
            params = self.signal_predictors[i](global_features)  # [B, signal_param_dim]
            signal_params.append(params)
            
            # Sinyal varlığı
            presence = self.presence_predictors[i](global_features)  # [B, 1]
            presence_logits.append(presence)
        
        signal_params = torch.stack(signal_params, dim=1)  # [B, max_signals, signal_param_dim]
        presence_logits = torch.stack(presence_logits, dim=1).squeeze(-1)  # [B, max_signals]
        
        # Sinyal parametrelerini ayır
        frequencies = signal_params[:, :, 0]  # [B, max_signals]
        bandwidths = F.relu(signal_params[:, :, 1])  # [B, max_signals] - pozitif
        powers = signal_params[:, :, 2]  # [B, max_signals]
        mod_logits = signal_params[:, :, 3:7]  # [B, max_signals, 4] - FM, OFDM, GFSK, QPSK
        spectral_entropy = F.relu(signal_params[:, :, 7])  # [B, max_signals]
        spectral_kurtosis = signal_params[:, :, 8]  # [B, max_signals]
        
        # Sinyal varlık olasılıkları
        presence_probs = torch.sigmoid(presence_logits)  # [B, max_signals]
        
        return {
            'count_logits': count_logits,
            'frequencies': frequencies,
            'bandwidths': bandwidths,
            'powers': powers,
            'mod_logits': mod_logits,
            'spectral_entropy': spectral_entropy,
            'spectral_kurtosis': spectral_kurtosis,
            'presence_probs': presence_probs,
            'global_features': global_features
        }

# =====================================================
# Gelişmiş Loss Fonksiyonu
# =====================================================
class AdvancedUHFLoss(nn.Module):
    """Çok-görevli loss fonksiyonu"""
    
    def __init__(self, fs=FS, **loss_weights):
        super().__init__()
        self.fs = fs
        self.freq_weight = loss_weights.get('freq_weight', 2.0)
        self.bw_weight = loss_weights.get('bw_weight', 1.5)
        self.power_weight = loss_weights.get('power_weight', 1.0)
        self.count_weight = loss_weights.get('count_weight', 1.0)
        self.mod_weight = loss_weights.get('mod_weight', 0.8)
        self.spectral_weight = loss_weights.get('spectral_weight', 0.5)
        
        # Normalizasyon faktörleri
        self.freq_norm = fs / 2.0  # [-fs/2, fs/2] -> [-1, 1]
        self.bw_norm = fs / 4.0    # Makul BW ~ fs/4
        
    def forward(self, predictions, targets):
        batch_size = targets['num_signals'].size(0)
        
        # Sinyal maskesi
        mask = targets['presence_mask']  # [B, max_signals]
        valid_signals = mask.sum()
        
        losses = {}
        
        # 1. Sinyal sayısı loss'u
        count_loss = F.cross_entropy(predictions['count_logits'], targets['num_signals'])
        losses['count'] = count_loss
        
        # 2. Sinyal varlığı loss'u (binary classification)
        presence_loss = F.binary_cross_entropy(predictions['presence_probs'], mask.float())
        losses['presence'] = presence_loss
        
        if valid_signals > 0:
            # 3. Frekans loss'u (normalize edilmiş)
            freq_targets_norm = targets['frequencies'] / self.freq_norm
            freq_loss = F.mse_loss(predictions['frequencies'], freq_targets_norm, reduction='none')
            freq_loss = (freq_loss * mask).sum() / valid_signals
            losses['frequency'] = freq_loss
            
            # 4. Bant genişliği loss'u
            bw_targets_norm = targets['bandwidths'] / self.bw_norm
            bw_loss = F.mse_loss(predictions['bandwidths'], bw_targets_norm, reduction='none')
            bw_loss = (bw_loss * mask).sum() / valid_signals
            losses['bandwidth'] = bw_loss
            
            # 5. Güç loss'u
            power_loss = F.mse_loss(predictions['powers'], targets['powers'], reduction='none')
            power_loss = (power_loss * mask).sum() / valid_signals
            losses['power'] = power_loss
            
            # 6. Modülasyon sınıflandırma loss'u
            mod_targets = targets['modulation_types'].long()  # [B, max_signals]
            mod_logits = predictions['mod_logits']  # [B, max_signals, 4]
            
            # Sadece geçerli sinyaller için modülasyon loss'u hesapla
            mod_loss_total = 0
            for b in range(batch_size):
                for s in range(mask.size(1)):
                    if mask[b, s] > 0.5:  # Geçerli sinyal
                        mod_loss_total += F.cross_entropy(
                            mod_logits[b, s].unsqueeze(0), 
                            mod_targets[b, s].unsqueeze(0)
                        )
            
            if valid_signals > 0:
                mod_loss = mod_loss_total / valid_signals
                losses['modulation'] = mod_loss
            
            # 7. Spektral özellikler loss'u
            if 'spectral_entropy' in targets:
                entropy_loss = F.mse_loss(predictions['spectral_entropy'], 
                                        targets['spectral_entropy'], reduction='none')
                entropy_loss = (entropy_loss * mask).sum() / valid_signals
                losses['spectral_entropy'] = entropy_loss
                
            if 'spectral_kurtosis' in targets:
                kurtosis_loss = F.mse_loss(predictions['spectral_kurtosis'], 
                                         targets['spectral_kurtosis'], reduction='none')
                kurtosis_loss = (kurtosis_loss * mask).sum() / valid_signals
                losses['spectral_kurtosis'] = kurtosis_loss
        
        # Toplam loss
        total_loss = (
            self.count_weight * losses.get('count', 0) +
            self.count_weight * losses.get('presence', 0) +
            self.freq_weight * losses.get('frequency', 0) +
            self.bw_weight * losses.get('bandwidth', 0) +
            self.power_weight * losses.get('power', 0) +
            self.mod_weight * losses.get('modulation', 0) +
            self.spectral_weight * losses.get('spectral_entropy', 0) +
            self.spectral_weight * losses.get('spectral_kurtosis', 0)
        )
        
        losses['total'] = total_loss
        losses['freq_norm'] = self.freq_norm
        losses['bw_norm'] = self.bw_norm
        
        return losses

# =====================================================
# Dataset Sınıfı (Shard tabanlı)
# =====================================================
class UHFShardDataset(Dataset):
    """Dataset üreticinizle uyumlu shard tabanlı veri yükleyici"""
    
    def __init__(self, dataset_dir, max_signals=MAX_SIGNALS, subset_ratio=1.0):
        self.dataset_dir = dataset_dir
        self.max_signals = max_signals
        self.samples = []
        self.modulation_map = {'FM': 0, 'OFDM': 1, 'GFSK': 2, 'QPSK': 3, 'NOISE': -1}
        
        self._load_dataset(subset_ratio)
        
    def _load_dataset(self, subset_ratio):
        """Shard dosyalarını yükle"""
        print(f"Dataset yükleniyor: {self.dataset_dir}")
        
        # Dataset istatistiklerini oku
        stats_path = os.path.join(self.dataset_dir, 'dataset_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
                print(f"Toplam örnek: {self.stats['total_samples']:,}")
        
        # Shard dosyalarını bul
        shard_files = [f for f in os.listdir(self.dataset_dir) 
                      if f.startswith('shard_') and f.endswith('.pkl')]
        shard_files.sort()
        
        print(f"Bulunan shard sayısı: {len(shard_files)}")
        
        total_loaded = 0
        target_samples = int(self.stats.get('total_samples', 50000) * subset_ratio)
        
        for shard_file in tqdm(shard_files, desc="Shard'lar yükleniyor"):
            if len(self.samples) >= target_samples:
                break
                
            shard_path = os.path.join(self.dataset_dir, shard_file)
            
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                for sample in shard_data:
                    if len(self.samples) >= target_samples:
                        break
                        
                    # Sadece sinyal içeren örnekleri al (noise-only hariç)
                    if (sample['sample_type'] != 'noise' and 
                        sample['n_signals'] > 0 and 
                        sample['n_signals'] <= self.max_signals):
                        
                        processed_sample = self._process_sample(sample)
                        if processed_sample is not None:
                            self.samples.append(processed_sample)
                
                total_loaded += len(shard_data)
                
            except Exception as e:
                print(f"Shard yüklenirken hata: {shard_file} - {e}")
                continue
        
        print(f"Toplam yüklenen örnek: {len(self.samples)}")
        
        # İstatistikler
        signal_counts = [s['targets']['num_signals'] for s in self.samples]
        for i in range(1, self.max_signals + 1):
            count = sum(1 for x in signal_counts if x == i)
            if count > 0:
                print(f"   {i} sinyal: {count} örnek ({count/len(self.samples)*100:.1f}%)")
    
    def _process_sample(self, sample):
        """Tek bir sample'ı işle"""
        try:
            # Spektrogram (STFT magnitude)
            spectrogram = sample['spectrogram']  # [freq_bins, time_bins]
            
            # Tek kanal haline getir
            if spectrogram.ndim == 2:
                spectrogram = spectrogram[np.newaxis, ...]  # [1, freq_bins, time_bins]
            
            # Hedef değerleri hazırla
            targets = self._prepare_targets(sample['signals'])
            
            if targets is None:
                return None
            
            return {
                'spectrogram': spectrogram.astype(np.float32),
                'targets': targets,
                'metadata': {
                    'sample_id': sample['sample_id'],
                    'sample_type': sample['sample_type'],
                    'uhf_carrier_hz': sample['uhf_carrier_hz'],
                    'fs': sample['fs']
                }
            }
            
        except Exception as e:
            print(f"Sample işlenirken hata: {e}")
            return None
    
    def _prepare_targets(self, signals):
        """Hedef değerleri hazırla"""
        try:
            num_signals = min(len(signals), self.max_signals)
            
            # Sabit boyutlu arrays
            frequencies = np.zeros(self.max_signals, dtype=np.float32)
            bandwidths = np.zeros(self.max_signals, dtype=np.float32)
            powers = np.zeros(self.max_signals, dtype=np.float32)
            modulation_types = np.zeros(self.max_signals, dtype=np.int64)
            spectral_entropies = np.zeros(self.max_signals, dtype=np.float32)
            spectral_kurtosis = np.zeros(self.max_signals, dtype=np.float32)
            presence_mask = np.zeros(self.max_signals, dtype=np.float32)
            
            valid_count = 0
            for i, signal in enumerate(signals[:self.max_signals]):
                # Frekans merkezi (baseband)
                freq_center = signal.get('f_center_est_hz', signal.get('f_off_hz', 0.0))
                
                # Bant genişliği
                bandwidth = signal.get('bw_occ99_hz', signal.get('bw_rms_hz', 0.0))
                
                # Güç
                power = signal.get('power_db', signal.get('rel_power_db', 0.0))
                
                # Modülasyon türü
                mod_type = signal.get('mod', 'NOISE')
                mod_idx = self.modulation_map.get(mod_type, -1)
                
                # Spektral özellikler
                entropy = signal.get('spectral_entropy', 0.0)
                kurtosis = signal.get('spectral_kurtosis', 0.0)
                
                # Geçerlilik kontrolü
                if (abs(freq_center) <= FS/2 and bandwidth > 0 and 
                    bandwidth <= FS/2 and mod_idx >= 0):
                    
                    frequencies[i] = freq_center
                    bandwidths[i] = bandwidth
                    powers[i] = power
                    modulation_types[i] = mod_idx
                    spectral_entropies[i] = entropy
                    spectral_kurtosis[i] = kurtosis
                    presence_mask[i] = 1.0
                    valid_count += 1
            
            if valid_count == 0:
                return None
            
            return {
                'num_signals': valid_count,
                'frequencies': frequencies,
                'bandwidths': bandwidths,
                'powers': powers,
                'modulation_types': modulation_types,
                'spectral_entropy': spectral_entropies,
                'spectral_kurtosis': spectral_kurtosis,
                'presence_mask': presence_mask
            }
            
        except Exception as e:
            print(f"Hedef hazırlanırken hata: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        spectrogram = torch.from_numpy(sample['spectrogram'])
        
        targets = {}
        for key, value in sample['targets'].items():
            if isinstance(value, np.ndarray):
                targets[key] = torch.from_numpy(value)
            else:
                targets[key] = torch.tensor(value)
        
        return spectrogram, targets, sample['metadata']

# =====================================================
# Eğitim Fonksiyonları
# =====================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0.0
    losses_dict = {
        'count': [], 'presence': [], 'frequency': [], 'bandwidth': [],
        'power': [], 'modulation': [], 'spectral_entropy': [], 'spectral_kurtosis': []
    }
    
    pbar = tqdm(dataloader, desc="Eğitim", leave=False)
    for spectrograms, targets, metadata in pbar:
        spectrograms = spectrograms.to(device)
        targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        outputs = model(spectrograms)
        losses = criterion(outputs, targets)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
        
        optimizer.step()
        
        total_loss += losses['total'].item()
        for key in losses_dict:
            if key in losses and losses[key] != 0:
                losses_dict[key].append(losses[key].item())
        
        pbar.set_postfix({
            'Loss': f"{losses['total'].item():.4f}",
            'Freq': f"{losses.get('frequency', 0.0):.4f}",
            'Count': f"{losses.get('count', 0.0):.4f}",
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        **{k: np.mean(v) if v else 0.0 for k, v in losses_dict.items()}
    }

def validate_epoch(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0.0
    losses_dict = {
        'count': [], 'presence': [], 'frequency': [], 'bandwidth': [],
        'power': [], 'modulation': []
    }
    
    # Metrik koleksiyonu
    all_freq_pred, all_freq_true = [], []
    all_bw_pred, all_bw_true = [], []
    all_count_pred, all_count_true = [], []
    all_mod_pred, all_mod_true = [], []
    
    with torch.no_grad():
        for spectrograms, targets, metadata in dataloader:
            spectrograms = spectrograms.to(device)
            targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
            
            outputs = model(spectrograms)
            losses = criterion(outputs, targets)
            
            total_loss += losses['total'].item()
            for key in losses_dict:
                if key in losses and losses[key] != 0:
                    losses_dict[key].append(losses[key].item())
            
            # Metrikler için veri topla
            mask = targets['presence_mask']
            freq_norm_factor = losses['freq_norm']
            bw_norm_factor = losses['bw_norm']
            
            # Sinyal sayısı tahminleri
            count_pred = torch.argmax(outputs['count_logits'], dim=1)
            all_count_pred.extend(count_pred.cpu().numpy())
            all_count_true.extend(targets['num_signals'].cpu().numpy())
            
            # Her geçerli sinyal için
            for b in range(mask.size(0)):
                for s in range(mask.size(1)):
                    if mask[b, s] > 0.5:  # Geçerli sinyal
                        # Frekans (denormalize et)
                        freq_pred_hz = outputs['frequencies'][b, s].item() * freq_norm_factor
                        freq_true_hz = targets['frequencies'][b, s].item()
                        all_freq_pred.append(freq_pred_hz)
                        all_freq_true.append(freq_true_hz)
                        
                        # Bant genişliği (denormalize et)
                        bw_pred_hz = outputs['bandwidths'][b, s].item() * bw_norm_factor
                        bw_true_hz = targets['bandwidths'][b, s].item()
                        all_bw_pred.append(bw_pred_hz)
                        all_bw_true.append(bw_true_hz)
                        
                        # Modülasyon türü
                        mod_pred = torch.argmax(outputs['mod_logits'][b, s])
                        mod_true = targets['modulation_types'][b, s]
                        all_mod_pred.append(mod_pred.cpu().item())
                        all_mod_true.append(mod_true.cpu().item())
    
    # Metrikleri hesapla
    metrics = {}
    
    if len(all_freq_true) > 0:
        metrics['freq_mae'] = mean_absolute_error(all_freq_true, all_freq_pred)
        metrics['freq_r2'] = r2_score(all_freq_true, all_freq_pred) if len(all_freq_true) > 1 else 0.0
        
        metrics['bw_mae'] = mean_absolute_error(all_bw_true, all_bw_pred)
        metrics['bw_r2'] = r2_score(all_bw_true, all_bw_pred) if len(all_bw_true) > 1 else 0.0
        
        metrics['mod_accuracy'] = accuracy_score(all_mod_true, all_mod_pred)
    else:
        metrics['freq_mae'] = metrics['freq_r2'] = 0.0
        metrics['bw_mae'] = metrics['bw_r2'] = 0.0
        metrics['mod_accuracy'] = 0.0
    
    metrics['count_accuracy'] = accuracy_score(all_count_true, all_count_pred)
    
    return {
        'total_loss': total_loss / len(dataloader),
        **{k: np.mean(v) if v else 0.0 for k, v in losses_dict.items()},
        **metrics
    }

# =====================================================
# Ana Eğitim Fonksiyonu
# =====================================================
def train_uhf_hybrid_model(dataset_dir, model_save_path="uhf_hybrid_model.pth"):
    """Hibrit UHF modelini eğit"""
    print("Hibrit UHF Sinyal Tespiti Eğitimi Başlıyor")
    print("=" * 60)
    
    # Dataset yükle
    dataset = UHFShardDataset(dataset_dir, max_signals=MAX_SIGNALS, subset_ratio=1.0)
    
    # Dataset bölümlemesi
    total_size = len(dataset)
    val_size = int(CONFIG['training']['validation_split'] * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset bölümlemesi: Eğitim={len(train_dataset)}, Doğrulama={len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=4 if DEVICE.type == 'cuda' else 2,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # Model oluştur
    model = HybridUHFDetector(**CONFIG['model'])
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")
    
    # Loss ve optimizer
    criterion = AdvancedUHFLoss(fs=FS, **CONFIG['loss'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG['training']['lr_factor'],
        patience=CONFIG['training']['lr_patience'],
        verbose=True,
        min_lr=1e-7
    )
    
    # Eğitim geçmişi
    history = {
        'train_loss': [], 'val_loss': [],
        'val_freq_mae': [], 'val_bw_mae': [],
        'val_freq_r2': [], 'val_bw_r2': [],
        'val_mod_accuracy': [], 'val_count_accuracy': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Eğitim {CONFIG['training']['num_epochs']} epoch için başlıyor...")
    print("=" * 60)
    
    for epoch in range(CONFIG['training']['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['training']['num_epochs']} (LR: {current_lr:.2e})")
        
        # Eğitim
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Doğrulama
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Scheduler güncelleme
        scheduler.step(val_metrics['total_loss'])
        
        # Geçmişe kaydet
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['val_freq_mae'].append(val_metrics['freq_mae'])
        history['val_bw_mae'].append(val_metrics['bw_mae'])
        history['val_freq_r2'].append(val_metrics['freq_r2'])
        history['val_bw_r2'].append(val_metrics['bw_r2'])
        history['val_mod_accuracy'].append(val_metrics['mod_accuracy'])
        history['val_count_accuracy'].append(val_metrics['count_accuracy'])
        
        # Metrikleri yazdır
        print(f"Eğitim Loss: {train_metrics['total_loss']:.4f} | Doğrulama Loss: {val_metrics['total_loss']:.4f}")
        print(f"Frekans MAE: {val_metrics['freq_mae']:.0f} Hz | R²: {val_metrics['freq_r2']:.3f}")
        print(f"BW MAE: {val_metrics['bw_mae']:.0f} Hz | R²: {val_metrics['bw_r2']:.3f}")
        print(f"Mod Doğruluk: {val_metrics['mod_accuracy']:.3f} | Sayı Doğruluk: {val_metrics['count_accuracy']:.3f}")
        
        # Model kaydetme ve erken durdurma
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # Model checkpoint kaydet
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': CONFIG,
                'fs': FS,
                'model_class': 'HybridUHFDetector'
            }, model_save_path)
            print(f"Model kaydedildi! En iyi doğrulama loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['training']['patience']:
            print(f"{CONFIG['training']['patience']} epoch boyunca iyileşme olmadığı için eğitim durduruldu")
            break
    
    print("\nEğitim tamamlandı!")
    return model, history

# =====================================================
# Tahmin Fonksiyonu
# =====================================================
def predict_signals_hybrid(model_path, spectrogram, confidence_threshold=0.5):
    """Eğitilmiş hibrit modeli kullanarak sinyal tespiti yap"""
    
    # Model yükle
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']
    fs = checkpoint.get('fs', FS)
    
    model = HybridUHFDetector(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Giriş hazırla
    if isinstance(spectrogram, np.ndarray):
        spectrogram = torch.from_numpy(spectrogram)
    
    if spectrogram.ndim == 2:
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
    elif spectrogram.ndim == 3:
        spectrogram = spectrogram.unsqueeze(0)  # [1, C, F, T]
    
    spectrogram = spectrogram.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(spectrogram)
        
        frequencies = outputs['frequencies'][0]    # [max_signals]
        bandwidths = outputs['bandwidths'][0]      # [max_signals]
        powers = outputs['powers'][0]              # [max_signals]
        mod_logits = outputs['mod_logits'][0]      # [max_signals, 4]
        presence_probs = outputs['presence_probs'][0]  # [max_signals]
        count_logits = outputs['count_logits'][0]  # [max_signals+1]
        spectral_entropy = outputs['spectral_entropy'][0]  # [max_signals]
        spectral_kurtosis = outputs['spectral_kurtosis'][0]  # [max_signals]
    
    # Denormalizasyon faktörleri
    freq_norm_factor = fs / 2.0
    bw_norm_factor = fs / 4.0
    
    # Modülasyon etiketleri
    mod_labels = ['FM', 'OFDM', 'GFSK', 'QPSK']
    
    # Tespit edilen sinyaller
    detected_signals = []
    for i in range(len(frequencies)):
        presence_conf = presence_probs[i].item()
        
        if presence_conf > confidence_threshold:
            freq_hz = frequencies[i].item() * freq_norm_factor
            bw_hz = bandwidths[i].item() * bw_norm_factor
            power_db = powers[i].item()
            
            # Modülasyon tahmini
            mod_probs = F.softmax(mod_logits[i], dim=0)
            mod_idx = torch.argmax(mod_probs).item()
            mod_confidence = mod_probs[mod_idx].item()
            
            # Baseband aralığında kontrol
            if abs(freq_hz) <= fs/2 and bw_hz > 0:
                detected_signals.append({
                    'freq_center_hz': freq_hz,
                    'bandwidth_hz': bw_hz,
                    'power_db': power_db,
                    'modulation': mod_labels[mod_idx],
                    'mod_confidence': mod_confidence,
                    'presence_confidence': presence_conf,
                    'spectral_entropy': spectral_entropy[i].item(),
                    'spectral_kurtosis': spectral_kurtosis[i].item()
                })
    
    # Sinyal sayısı tahmini
    predicted_count = torch.argmax(count_logits).item()
    count_confidence = F.softmax(count_logits, dim=0).max().item()
    
    # Güven skoruna göre sırala
    detected_signals.sort(key=lambda x: x['presence_confidence'], reverse=True)
    
    return {
        'detected_signals': detected_signals,
        'predicted_count': predicted_count,
        'count_confidence': count_confidence,
        'actual_detections': len(detected_signals)
    }

# =====================================================
# Görselleştirme Fonksiyonları
# =====================================================
def plot_hybrid_training_history(history, save_path=None):
    """Hibrit model eğitim geçmişini çiz"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hibrit UHF Model Eğitim Geçmişi', fontsize=16)
    
    # Loss eğrileri
    axes[0, 0].plot(history['train_loss'], label='Eğitim', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Doğrulama', alpha=0.8)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE eğrileri
    axes[0, 1].plot(history['val_freq_mae'], label='Frekans MAE (Hz)', alpha=0.8)
    axes[0, 1].plot(history['val_bw_mae'], label='Bant Genişliği MAE (Hz)', alpha=0.8)
    axes[0, 1].set_title('Ortalama Mutlak Hata')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (Hz)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # R² skorları
    axes[0, 2].plot(history['val_freq_r2'], label='Frekans R²', alpha=0.8)
    axes[0, 2].plot(history['val_bw_r2'], label='BW R²', alpha=0.8)
    axes[0, 2].set_title('R² Skoru')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('R² Skoru')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-0.1, 1.1)
    
    # Doğruluk metrikleri
    axes[1, 0].plot(history['val_mod_accuracy'], label='Modülasyon Doğruluk', alpha=0.8)
    axes[1, 0].plot(history['val_count_accuracy'], label='Sinyal Sayısı Doğruluk', alpha=0.8)
    axes[1, 0].set_title('Sınıflandırma Doğruluğu')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Doğruluk')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'], alpha=0.8, color='red')
    axes[1, 1].set_title('Öğrenme Hızı Planlaması')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Öğrenme Hızı')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Loss bileşenleri (en son epoch)
    if len(history['train_loss']) > 0:
        loss_components = ['frequency', 'bandwidth', 'power', 'modulation', 'count']
        loss_values = [history.get(f'val_{comp}', [0])[-1] if history.get(f'val_{comp}') else 0 
                      for comp in loss_components]
        
        axes[1, 2].bar(loss_components, loss_values, alpha=0.7)
        axes[1, 2].set_title('Loss Bileşenleri (Son Epoch)')
        axes[1, 2].set_ylabel('Loss Değeri')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eğitim geçmişi kaydedildi: {save_path}")
    
    plt.show()

def visualize_prediction_results(spectrogram, predictions, fs=FS, save_path=None):
    """Tahmin sonuçlarını görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hibrit UHF Model Tahmin Sonuçları', fontsize=16)
    
    # Spektrogram
    axes[0, 0].imshow(spectrogram.squeeze(), aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Giriş Spektrogram')
    axes[0, 0].set_xlabel('Zaman')
    axes[0, 0].set_ylabel('Frekans')
    
    # Tespit edilen sinyaller - frekans dağılımı
    if predictions['detected_signals']:
        frequencies = [s['freq_center_hz'] for s in predictions['detected_signals']]
        bandwidths = [s['bandwidth_hz'] for s in predictions['detected_signals']]
        confidences = [s['presence_confidence'] for s in predictions['detected_signals']]
        
        scatter = axes[0, 1].scatter(frequencies, bandwidths, c=confidences, 
                                   cmap='coolwarm', s=100, alpha=0.8)
        axes[0, 1].set_title('Tespit Edilen Sinyaller')
        axes[0, 1].set_xlabel('Frekans Merkezi (Hz)')
        axes[0, 1].set_ylabel('Bant Genişliği (Hz)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Güven Skoru')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Modülasyon dağılımı
        modulations = [s['modulation'] for s in predictions['detected_signals']]
        mod_counts = {mod: modulations.count(mod) for mod in set(modulations)}
        
        axes[1, 0].bar(mod_counts.keys(), mod_counts.values(), alpha=0.7)
        axes[1, 0].set_title('Modülasyon Türü Dağılımı')
        axes[1, 0].set_ylabel('Sinyal Sayısı')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spektral özellikler
        entropies = [s['spectral_entropy'] for s in predictions['detected_signals']]
        kurtosis = [s['spectral_kurtosis'] for s in predictions['detected_signals']]
        
        axes[1, 1].scatter(entropies, kurtosis, c=confidences, 
                          cmap='coolwarm', s=100, alpha=0.8)
        axes[1, 1].set_title('Spektral Özellikler')
        axes[1, 1].set_xlabel('Spektral Entropi')
        axes[1, 1].set_ylabel('Spektral Kurtosis')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tahmin görselleştirmesi kaydedildi: {save_path}")
    
    plt.show()

# =====================================================
# Ana Çalıştırma
# =====================================================
if __name__ == "__main__":
    # Konfigürasyon
    DATASET_DIR = r"C:\Users\Osman\Desktop\BITES\sinyal_uhf\uhf_dataset_real_11"
    MODEL_SAVE_PATH = r"C:\Users\Osman\Desktop\BITES\sinyal_uhf\uhf_dataset_real_11\model_train\uhf_hybrid_detector22.pth"
    
    # Model kaydetme dizinini oluştur
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset dizini bulunamadı: {DATASET_DIR}")
        print("Lütfen önce UHF dataset üretim kodunuzu çalıştırın!")
        exit(1)
    
    print(f"Konfigürasyon:")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Model kaydedilecek: {MODEL_SAVE_PATH}")
    print(f"  Örnekleme hızı: {FS/1e6:.1f} MHz")
    print(f"  Maksimum sinyal: {MAX_SIGNALS}")
    print(f"  Cihaz: {DEVICE}")
    
    # Eğitimi başlat
    model, history = train_uhf_hybrid_model(DATASET_DIR, MODEL_SAVE_PATH)
    
    # Sonuçları görselleştir
    plot_hybrid_training_history(history, "uhf_hybrid_training_history.png")
    
    print(f"\nModel kaydedildi: {MODEL_SAVE_PATH}")
    print("\nTahmin için örnek kullanım:")
    print("```python")
    print("import numpy as np")
    print("# Spektrogram verilerinizi yükleyin [1, freq_bins, time_bins]")
    print("spectrogram = your_spectrogram_data")
    print()
    print("# Sinyal tespiti yapın")
    print(f"results = predict_signals_hybrid('{MODEL_SAVE_PATH}', spectrogram)")
    print()
    print("# Sonuçları görselleştirin")
    print("visualize_prediction_results(spectrogram, results)")
    print("```")