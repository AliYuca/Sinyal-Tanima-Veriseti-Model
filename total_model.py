#!/usr/bin/env python3
"""
 Bu eğitim kodunda daha önceden pek çok farklı deneme ile ulaşılıp karar verilen frekans
ve modülasyon özellikleri çıkartımı yapan 2 kod birleştirilmiştir. Veriseti girişi olarak
temel spektrum alınacaktır.

Görevler
1. Sinyal Tipi Sınıflandırma
2. Sinyal Sayısı Tahmini
3. Modülasyon Tanıma
4. Frekans Maskesi Segmentasyonu

Girdi: 2D Spektrogram
Çıktı: Sınıflandırma sonuçları ve 1D frekans maskesi
"""

import os
import json
import pickle
import numpy as np
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import wandb


# ==================================
# 1. BİRLEŞTİRİLMİŞ CONFIGURATION
# ==================================
class Config:
    """Eğitim ve model konfigürasyonlarını içeren sınıf."""
    
    # Dataset paths
    DATA_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_11_2"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 80
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 12
    
    DROPOUT_RATE = 0.4
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    
    # Output
    RESULTS_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_11_2\total_train_1\unified_model_results_v1"
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "best_unified_uhf_model.pth")
    
    # Task-specific parameters
    MAX_SIGNALS = 3
    
    # Input boyutları
    FREQ_BINS = 256
    TIME_BINS = None  # Otomatik hesaplanacak


# ==================================
# 2. SEGMENTASYON İÇİN YARDIMCI FONKSİYONLAR VE LOSS'LAR
# ==================================

class DiceLoss(nn.Module):
    """Dice Loss implementasyonu segmentasyon görevi için."""
    
    def __init__(self, smooth=1e-5):
        """
        Dice Loss'u başlatır.
        
        Args: smooth: Bölme hatalarını önlemek için kullanılan düzgünleştirme parametresi
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """Dice Loss hesaplar."""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


def create_label_mask(signals, freqs):
    """
    Verilen sinyaller için frekans maskesi oluşturur.
    
    Args:
        signals: Sinyal bilgilerini içeren liste
        freqs: Frekans vektörü
        
    Returns:
        numpy.ndarray: Frekans maskesi
    """
    label_mask = np.zeros_like(freqs, dtype=np.float32)
    
    for signal in signals:
        center_freq = signal['f_off_hz']
        bandwidth = signal['bw_occ99_hz']
        
        # Sinyal frekans aralığını hesapla
        start_freq = center_freq - bandwidth / 2
        end_freq = center_freq + bandwidth / 2
        
        # Frekans indekslerini bul
        start_idx = np.searchsorted(freqs, start_freq, side='left')
        end_idx = np.searchsorted(freqs, end_freq, side='right')
        
        # Sınırları kontrol et
        start_idx = max(0, start_idx)
        end_idx = min(len(freqs), end_idx)
        
        # Maskeyi güncelle
        if start_idx < end_idx:
            label_mask[start_idx:end_idx] = 1.0
            
    return label_mask


# ==================================
# 3. BİRLEŞTİRİLMİŞ DATASET CLASS
# ==================================
class UnifiedUhfDataset(Dataset):
    """Birleştirilmiş UHF sinyal dataset sınıfı."""
    
    def __init__(self, data_dir, split='train', train_ratio=0.75, val_ratio=0.15, seed=42):
        """
        Dataset'i başlatır ve verileri yükler.
        
        Args:
            data_dir: Veri dizini yolu
            split: 'train', 'val' veya 'test'
            train_ratio: Eğitim verisi oranı
            val_ratio: Doğrulama verisi oranı
            seed: Random seed
        """
        self.data_dir = data_dir
        self.split = split
        np.random.seed(seed)
        
        self._load_all_data()
        self._prepare_labels()
        self._split_data(train_ratio, val_ratio, seed)
        print(f"Unified {split.upper()} dataset: {len(self.indices)} samples")

    def _load_all_data(self):
        """Tüm shard dosyalarını yükler."""
        shard_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith('shard_') and f.endswith('.pkl')
        ])
        
        self.all_samples = []
        
        for shard_file in tqdm(shard_files, desc="Loading shards"):
            shard_path = os.path.join(self.data_dir, shard_file)
            try:
                with open(shard_path, 'rb') as f:
                    self.all_samples.extend(pickle.load(f))
            except Exception as e:
                print(f"Error loading {shard_file}: {e}")
        
        # TIME_BINS'i ilk örnekten otomatik ayarla
        if self.all_samples and Config.TIME_BINS is None:
            Config.TIME_BINS = self.all_samples[0]['spectrogram'].shape[-1]
            print(f"TIME_BINS automatically set to: {Config.TIME_BINS}")

    def _prepare_labels(self):
        """Label encoder'ları hazırlar ve etiketleri oluşturur."""
        # Sınıflandırma etiketleri için encoderlar
        self.signal_type_encoder = LabelEncoder()
        
        all_mods_in_data = set(
            mod for sample in self.all_samples 
            for signal in sample['signals'] 
            for mod in [signal.get('mod', 'NOISE')]
        )
        
        self.mlb_modulations = MultiLabelBinarizer(classes=sorted(list(all_mods_in_data)))

        # Geçici etiket listeleri oluştur
        temp_signal_types = [sample['sample_type'] for sample in self.all_samples]
        temp_modulations = [
            [signal.get('mod', 'NOISE') for signal in sample['signals']] 
            for sample in self.all_samples
        ]
        
        # Etiketleri encode et
        self.signal_types_encoded = self.signal_type_encoder.fit_transform(temp_signal_types)
        self.modulations_encoded = self.mlb_modulations.fit_transform(temp_modulations)
        self.signal_counts = [
            min(sample['n_signals'], Config.MAX_SIGNALS) for sample in self.all_samples
        ]
        
        print(f"Signal types found: {list(self.signal_type_encoder.classes_)}")
        print(f"Modulations found: {list(self.mlb_modulations.classes_)}")

    def _split_data(self, train_ratio, val_ratio, seed):
        """Veriyi train/val/test olarak böler."""
        n_total = len(self.all_samples)
        indices = np.arange(n_total)
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        else:  # test
            self.indices = indices[n_train + n_val:]

    def __len__(self):
        """Dataset boyutunu döndürür."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Belirtilen indeksteki veri örneğini döndürür."""
        actual_idx = self.indices[idx]
        sample = self.all_samples[actual_idx]
        
        # Spektrogramı tensor'a çevir
        spectrogram = torch.tensor(sample['spectrogram'], dtype=torch.float32).unsqueeze(0)
        
        # Sınıflandırma etiketleri
        signal_type = torch.tensor(self.signal_types_encoded[actual_idx], dtype=torch.long)
        signal_count = torch.tensor(self.signal_counts[actual_idx], dtype=torch.long)
        modulation = torch.tensor(self.modulations_encoded[actual_idx], dtype=torch.float32)

        # Segmentasyon maskesi (1D) - Anlık oluşturuluyor
        freqs = sample['freqs']  # Frekans vektörünü veriden al
        segmentation_mask = create_label_mask(sample['signals'], freqs)
        segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.float32)
        
        return spectrogram, (signal_type, signal_count, modulation, segmentation_mask)


# ==================================
# 4. BİRLEŞTİRİLMİŞ MODEL MİMARİSİ
# ==================================
class UnifiedUhfModel(nn.Module):
    """Birleştirilmiş UHF sinyal analiz modeli."""
    
    def __init__(self, n_signal_types, n_modulations, dropout_rate=0.4):
        """
        Modeli başlatır.
        
        Args:
            n_signal_types: Sinyal tipi sayısı
            n_modulations: Modülasyon sayısı
            dropout_rate: Dropout oranı
        """
        super(UnifiedUhfModel, self).__init__()
        
        # --- Ortak Gövde (2D CNN Backbone) ---
        self.backbone = nn.Sequential(
            #Blok 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout_rate * 0.3),

            # Blok 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout_rate * 0.4),
            
            # Blok 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout_rate * 0.5),

            #Blok4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
        )  # Çıktı: [B, 512, F/8, T/8]
        
        # --- Sınıflandırma Görevi için Ortak Katman ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classification_fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(True), 
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(True), 
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # --- Görev-Spesifik Başlıklar (Heads) ---
        # 1. Sınıflandırma Başlıkları
        self.signal_type_head = nn.Linear(512, n_signal_types)
        self.signal_count_head = nn.Linear(512, Config.MAX_SIGNALS + 1)
        self.modulation_head = nn.Linear(512, n_modulations)
        
        # 2. Segmentasyon Başlığı (1D Decoder)
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # F/4
            nn.Conv1d(256, 128, 3, padding=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # F/2
            nn.Conv1d(128, 64, 3, padding=1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # F/1
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, x):
        """Model forward pass'ını gerçekleştirir."""
        # --- Ortak Gövdeden Özellik Çıkarımı ---
        features_2d = self.backbone(x)  # Shape: [B, 512, F/8, T/8]
        
        # --- 1. Sınıflandırma Yolu ---
        pooled_features = self.adaptive_pool(features_2d)
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        shared_repr = self.classification_fc(flat_features)
        
        type_pred = self.signal_type_head(shared_repr)
        count_pred = self.signal_count_head(shared_repr)
        mod_pred = self.modulation_head(shared_repr)
        
        # --- 2. Segmentasyon Yolu ---
        # Zaman eksenini ortalama alarak 1D'ye indir
        features_1d = features_2d.mean(dim=3)  # Shape: [B, 512, F/8]
        seg_pred_logits = self.segmentation_head(features_1d).squeeze(1)  # Shape: [B, F]
        
        return type_pred, count_pred, mod_pred, seg_pred_logits


# ==================================
# 5. BİRLEŞTİRİLMİŞ LOSS FONKSİYONU
# ==================================
class UnifiedMultiTaskLoss(nn.Module):
    """Birden fazla görev için birleştirilmiş loss fonksiyonu."""
    
    def __init__(self, task_weights=None):
        """
        Multi-task loss'u başlatır.
        
        Args:
            task_weights: Her görev için ağırlık dict'i
        """
        super(UnifiedMultiTaskLoss, self).__init__()
        
        if task_weights is None:
            task_weights = {'type': 0.5, 'count': 1.5, 'mod': 2, 'seg': 1.5} # Özelliklerin değerine ve çıkartma kolaylığına göre farklı ağılıklar atandı
                                                                             # farklı ağılıklar atandı.
        self.task_weights = task_weights
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, predictions, targets):
        """Tüm görevler için loss hesaplar ve birleştirir."""
        type_pred, count_pred, mod_pred, seg_pred = predictions
        type_target, count_target, mod_target, seg_target = targets
        
        # Her bir görev için loss hesabı
        type_loss = self.ce_loss(type_pred, type_target)
        count_loss = self.ce_loss(count_pred, count_target)
        mod_loss = self.bce_loss(mod_pred, mod_target)
        seg_loss = self.dice_loss(seg_pred, seg_target)  # Sadece Dice Loss kullanıldı
        
        # Ağırlıklı toplam loss
        total_loss = (
            self.task_weights['type'] * type_loss +
            self.task_weights['count'] * count_loss +
            self.task_weights['mod'] * mod_loss +
            self.task_weights['seg'] * seg_loss
        )
                      
        return {
            'total_loss': total_loss,
            'type_loss': type_loss,
            'count_loss': count_loss,
            'modulation_loss': mod_loss,
            'segmentation_loss': seg_loss
        }


# ==================================
# 6. GÜNCELLENMİŞ EĞİTİM VE DOĞRULAMA DÖNGÜLERİ
# ==================================
def calculate_seg_metrics(pred, target, threshold=0.5):
    """
    Segmentasyon metrikleri hesaplar.
    
    Args:
        pred: Tahmin edilen maskeler
        target: Gerçek maskeler
        threshold: Binary threshold değeri
        
    Returns:
        dict: F1, precision, recall metrikleri
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # True/False positive/negative hesapla
    tp = ((pred_binary == 1) & (target == 1)).float().sum()
    fp = ((pred_binary == 1) & (target == 0)).float().sum()
    fn = ((pred_binary == 0) & (target == 1)).float().sum()
    
    # Metrikleri hesapla
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'f1': f1.item(), 
        'precision': precision.item(), 
        'recall': recall.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Bir epoch eğitim gerçekleştirir.
    
    Args:
        model: Eğitilecek model
        dataloader: Eğitim dataloader'ı
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        device: Hesaplama cihazı
        
    Returns:
        tuple: Loss'lar, doğruluk oranları ve segmentasyon metrikleri
    """
    model.train()
    
    total_losses = {
        k: 0.0 for k in ['total_loss', 'type_loss', 'count_loss', 
                         'modulation_loss', 'segmentation_loss']
    }
    
    total_samples, correct_type, correct_count = 0, 0, 0
    all_seg_preds, all_seg_targets = [], []

    pbar = tqdm(dataloader, desc="Training")
    
    for features, targets in pbar:
        features = features.to(device)
        targets = [t.to(device) for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        losses = criterion(predictions, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Loss'ları topla
        for key in total_losses:
            total_losses[key] += losses[key].item()
        
        # Sınıflandırma metrikleri hesabı
        _, pred_type = torch.max(predictions[0], 1)
        correct_type += (pred_type == targets[0]).sum().item()
        _, pred_count = torch.max(predictions[1], 1)
        correct_count += (pred_count == targets[1]).sum().item()
        total_samples += targets[0].size(0)

        # Segmentasyon metrikleri için biriktir
        all_seg_preds.append(predictions[3].detach())
        all_seg_targets.append(targets[3].detach())
        
        pbar.set_postfix({'Loss': f"{losses['total_loss'].item():.4f}"})

    # Ortalama loss'ları hesapla
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    # Doğruluk oranlarını hesapla
    type_acc = 100. * correct_type / total_samples
    count_acc = 100. * correct_count / total_samples
    
    # Segmentasyon metriklerini hesapla
    seg_metrics = calculate_seg_metrics(
        torch.cat(all_seg_preds), 
        torch.cat(all_seg_targets)
    )
    
    return total_losses, type_acc, count_acc, seg_metrics


def validate_epoch(model, dataloader, criterion, device):
    """
    Bir epoch doğrulama gerçekleştirir.
    
    Args:
        model: Doğrulanacak model
        dataloader: Doğrulama dataloader'ı
        criterion: Loss fonksiyonu
        device: Hesaplama cihazı
        
    Returns:
        tuple: Loss'lar, doğruluk oranları ve segmentasyon metrikleri
    """
    model.eval()
    
    total_losses = {
        k: 0.0 for k in ['total_loss', 'type_loss', 'count_loss', 
                         'modulation_loss', 'segmentation_loss']
    }
    
    total_samples, correct_type, correct_count = 0, 0, 0
    all_seg_preds, all_seg_targets = [], []

    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            predictions = model(features)
            losses = criterion(predictions, targets)

            # Loss'ları topla
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            # Sınıflandırma metrikleri hesapla
            _, pred_type = torch.max(predictions[0], 1)
            correct_type += (pred_type == targets[0]).sum().item()
            _, pred_count = torch.max(predictions[1], 1)
            correct_count += (pred_count == targets[1]).sum().item()
            total_samples += targets[0].size(0)

            # Segmentasyon için biriktir
            all_seg_preds.append(predictions[3])
            all_seg_targets.append(targets[3])

    # Ortalama loss'ları hesapla
    for key in total_losses:
        total_losses[key] /= len(dataloader)

    # Doğruluk oranlarını hesapla
    type_acc = 100. * correct_type / total_samples
    count_acc = 100. * correct_count / total_samples
    
    # Segmentasyon metriklerini hesapla
    seg_metrics = calculate_seg_metrics(
        torch.cat(all_seg_preds), 
        torch.cat(all_seg_targets)
    )
    
    return total_losses, type_acc, count_acc, seg_metrics


# ==================================
# 7. GÖRSELLEŞTİRME VE ANA EĞİTİM FONKSİYONU
# ==================================

def visualize_unified_prediction(model, dataset, device, epoch, num_samples=3):
    """
    Model tahminlerini görselleştirir.
    
    Args:
        model: Tahmin yapacak model
        dataset: Veri seti
        device: Hesaplama cihazı
        epoch: Mevcut epoch numarası
        num_samples: Görselleştirilecek örnek sayısı
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(15, 5 * len(indices)))
    if len(indices) == 1: 
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            spectrogram, targets = dataset[idx]
            spectrogram_gpu = spectrogram.unsqueeze(0).to(device)
            
            predictions = model(spectrogram_gpu)
            
            # Verileri CPU'ya çek ve numpy'ye çevir
            spec_np = spectrogram.squeeze(0).cpu().numpy()
            
            # Sınıflandırma sonuçları
            pred_type_idx = torch.argmax(predictions[0], 1).item()
            pred_count_idx = torch.argmax(predictions[1], 1).item()
            pred_type = dataset.signal_type_encoder.classes_[pred_type_idx]
            true_type = dataset.signal_type_encoder.classes_[targets[0].item()]
            true_count = targets[1].item()
            
            # Segmentasyon sonuçları
            pred_mask = torch.sigmoid(predictions[3]).squeeze(0).cpu().numpy()
            true_mask = targets[3].cpu().numpy()
            
            # Sol taraf: Spektrogram ve başlık
            ax_left = axes[i, 0]
            ax_left.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
            title = (
                f"Sample {dataset.indices[idx]} | True Type: {true_type} (Count: {true_count})\n"
                f"Pred Type: {pred_type} (Count: {pred_count_idx})"
            )
            ax_left.set_title(title)
            ax_left.set_xlabel("Time Bins")
            ax_left.set_ylabel("Frequency Bins")

            # Sağ taraf: Segmentasyon maskesi
            ax_right = axes[i, 1]
            ax_right.plot(true_mask, 'g-', label='True Mask', linewidth=2)
            ax_right.plot(pred_mask, 'r--', label='Predicted Mask', linewidth=2)
            ax_right.fill_between(range(len(true_mask)), true_mask, alpha=0.2, color='green')
            ax_right.set_title("Frequency Segmentation")
            ax_right.set_xlabel("Frequency Bins")
            ax_right.set_ylabel("Signal Presence")
            ax_right.legend()
            ax_right.grid(True)

    plt.tight_layout()
    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(plt)})
    plt.close()


def main():
    """Ana eğitim fonksiyonu."""
    # Sonuç dizinini oluştur
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # WandB başlat
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    run = wandb.init(project="Total_Model", config=config_dict)

    print("Unified Multi-Task & Segmentation Training Started")
    print(f"Device: {Config.DEVICE}")
    
    # Dataset ve DataLoader oluştur
    train_dataset = UnifiedUhfDataset(
        Config.DATA_DIR, 'train', Config.TRAIN_SPLIT, Config.VAL_SPLIT
    )
    val_dataset = UnifiedUhfDataset(
        Config.DATA_DIR, 'val', Config.TRAIN_SPLIT, Config.VAL_SPLIT
    )
    
    train_loader = DataLoader(
        train_dataset, Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # Model, Loss, Optimizer oluştur
    n_signal_types = len(train_dataset.signal_type_encoder.classes_)
    n_modulations = len(train_dataset.mlb_modulations.classes_)
    
    model = UnifiedUhfModel(n_signal_types, n_modulations, Config.DROPOUT_RATE).to(Config.DEVICE)
    criterion = UnifiedMultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5, verbose=True
    )

    # WandB model tracking başlat
    wandb.watch(model, log='all', log_freq=100)
    
    # Early stopping değişkenleri
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training loop...")
    
    # Ana eğitim döngüsü
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        
        # Eğitim ve doğrulama epoch'larını çalıştır
        train_losses, train_type_acc, train_count_acc, train_seg_metrics = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_losses, val_type_acc, val_count_acc, val_seg_metrics = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )

        # Sonuçları yazdır
        print(
            f"Train - Loss: {train_losses['total_loss']:.4f} | "
            f"Type Acc: {train_type_acc:.2f}% | "
            f"Count Acc: {train_count_acc:.2f}% | "
            f"Seg F1: {train_seg_metrics['f1']:.3f}"
        )
        print(
            f"Val   - Loss: {val_losses['total_loss']:.4f} | "
            f"Type Acc: {val_type_acc:.2f}% | "
            f"Count Acc: {val_count_acc:.2f}% | "
            f"Seg F1: {val_seg_metrics['f1']:.3f}"
        )

        # WandB Logging
        log_data = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            # Train metrikleri
            **{f'train/{k}': v for k, v in train_losses.items()},
            "train/type_accuracy": train_type_acc,
            "train/count_accuracy": train_count_acc,
            "train/segmentation_f1": train_seg_metrics['f1'],
            # Val metrikleri
            **{f'val/{k}': v for k, v in val_losses.items()},
            "val/type_accuracy": val_type_acc,
            "val/count_accuracy": val_count_acc,
            "val/segmentation_f1": val_seg_metrics['f1'],
        }
        wandb.log(log_data)

        # Her 10 epoch'ta görselleştirme yap
        if (epoch + 1) % 10 == 0:
            visualize_unified_prediction(model, val_dataset, Config.DEVICE, epoch + 1)

        # Model kaydetme ve Early Stopping
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Best model saved! (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Learning rate scheduler step
        scheduler.step(val_losses['total_loss'])

        # Early stopping kontrolü
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {patience_counter} epochs.")
            break
            
    print("\nTraining completed!")
    
    # En iyi modeli WandB'ye artifact olarak kaydet
    best_model_artifact = wandb.Artifact("unified-uhf-model", type="model")
    best_model_artifact.add_file(Config.MODEL_SAVE_PATH)
    run.log_artifact(best_model_artifact)
    wandb.finish()


if __name__ == "__main__":
    main()