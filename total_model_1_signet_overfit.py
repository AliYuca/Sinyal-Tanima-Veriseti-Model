#!/usr/bin/env python3
"""
Dual-Input (Multi-Modal) Multi-Task UHF Signal Analysis CNN
VERSION 3: Enhanced Visualization & Final Evaluation with Confusion Matrix

MİMARİ:
- 1D CNN Encoder (PSD) -> Segmentasyon
- 2D CNN Encoder (Spectrogram)
- Birleştirilmiş özellikler -> Sınıflandırma (Modülasyon, Tip, Sayı)
"""
# 1. Standart Kütüphaneler
import os
import pickle
import random
import warnings
from datetime import datetime

# 2. Üçüncü Parti Kütüphaneler (Bilimsel & ML)
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 3. Özel & Harici Kütüphaneler
import wandb
try:
    # Sadece Kaggle ortamında çalışacak bir kütüphane
    from kaggle_secrets import UserSecretsClient
except ImportError:

    UserSecretsClient = None

# ==================================
# 1. KONFİGÜRASYON
# ==================================
class Config:
    """Proje için tüm yapılandırma ve hiper-parametreleri barındıran sınıf."""
    # --- Veri Yolları ---
    SPECTROGRAM_DATA_DIR = "/kaggle/input/uhf-dataset-12/uhf_dataset_11"
    PSD_DATA_DIR = "/kaggle/input/uhf-dataset-12-psd/psd_dataset"
    
    # --- Çıktı Yolları ---
    RESULTS_DIR = "/kaggle/working/results"
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "best_dual_input_model_v3.pth")

    # --- Model ve Eğitim Parametreleri ---
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.5
    EARLY_STOPPING_PATIENCE = 15
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 16 
    NUM_EPOCHS = 80
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15  # Geriye kalan %10 test için kullanılır
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2  # Veri yükleme işlemini hızlandırmak için
    MAX_SIGNALS = 3
    FREQ_BINS = 256  # Spektrogramların yeniden boyutlandırılacağı hedef boyut
    POST_PROCESSING_THRESHOLD = 0.5

# ==================================
# 2. YARDIMCI SINIFLAR VE FONKSİYONLAR
# ==================================
class DiceLoss(nn.Module):
    """Segmentasyon görevi için Dice katsayısına dayalı bir kayıp fonksiyonu."""
    def __init__(self, smooth=1e-5):
        """DiceLoss sınıfını ilklendirir."""
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """Tahmin ve hedef arasındaki Dice kaybını hesaplar."""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice_score

def create_label_mask(signals, freqs, target_bins):
    """Sinyal meta verilerinden segmentasyon için bir hedef maske oluşturur."""
    label_mask = np.zeros(target_bins, dtype=np.float32)
    fs_total = freqs[-1] - freqs[0]
    f_min = freqs[0]

    # Frekans aralığının sıfır olma durumunu kontrol et
    if fs_total == 0:
        return label_mask
    
    for signal in signals:
        center_freq = signal.get('f_off_hz', 0)
        bandwidth = signal.get('bw_occ99_hz', 0)
        
        start_freq = center_freq - (bandwidth / 2)
        end_freq = center_freq + (bandwidth / 2)
        
        # Frekansları indekslere çevir
        start_idx = int(((start_freq - f_min) / fs_total) * target_bins)
        end_idx = int(((end_freq - f_min) / fs_total) * target_bins)
        
        # İndekslerin sınırlar içinde kalmasını sağla
        start_idx = max(0, start_idx)
        end_idx = min(target_bins, end_idx)
        
        if start_idx < end_idx:
            label_mask[start_idx:end_idx] = 1.0
            
    return label_mask

def calculate_signal_parameters(mask_pred, freqs, threshold=0.5):
    """Tahmin edilen segmentasyon maskesinden sinyal parametrelerini (merkez frekans, bant genişliği) çıkarır."""
    binary_mask = (mask_pred > threshold).astype(int)
    labeled_mask, num_features = ndi.label(binary_mask)
    detected_signals = []

    if num_features == 0:
        return detected_signals
        
    for i in range(1, num_features + 1):
        signal_indices = np.where(labeled_mask == i)[0]
        if len(signal_indices) == 0:
            continue
            
        start_idx = signal_indices.min()
        end_idx = signal_indices.max()
        
        f_lower = freqs[start_idx]
        f_upper = freqs[end_idx]
        
        center_freq = (f_lower + f_upper) / 2
        bandwidth = f_upper - f_lower
        
        if bandwidth > 0:
            detected_signals.append({
                'center_freq_mhz': center_freq / 1e6,
                'bandwidth_khz': bandwidth / 1e3
            })
            
    return detected_signals

# ==================================
# 3. VERİSETİ SINIFI (DATASET CLASS)
# ==================================
class DualInputDataset(Dataset):
    """Spektrogram ve PSD verilerini yükleyen, işleyen ve modele hazır hale getiren özel PyTorch veri kümesi."""
    def __init__(self, spec_dir, psd_dir, split='train', train_ratio=0.75, val_ratio=0.15, seed=42):
        """Veri kümesini ilklendirir, verileri yükler ve ayırır."""
        self.split = split
        np.random.seed(seed)
        
        if split == 'train':
            print("Tüm veri setleri (train, val, test) için veriler yükleniyor...")
            
        self.spectrogram_samples = self._load_data_from_dir(spec_dir, "shard_")
        self.psd_samples = self._load_data_from_dir(psd_dir, "psd_shard_")
        
        if len(self.spectrogram_samples) != len(self.psd_samples):
            raise ValueError("Spektrogram ve PSD örnek sayıları eşleşmiyor!")
            
        self._prepare_labels()
        self._split_data(train_ratio, val_ratio, seed)
        print(f"Dual-Input {split.upper()} veri kümesi {len(self.indices)} örnek ile oluşturuldu.")

    def _load_data_from_dir(self, data_dir, prefix):
        """Belirtilen bir dizindeki tüm .pkl parçalarını (shards) yükler."""
        all_samples = []
        try:
            shard_files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('.pkl')])
            for shard_file in tqdm(shard_files, desc=f"{os.path.basename(data_dir)}'dan '{prefix}' yükleniyor", leave=False):
                file_path = os.path.join(data_dir, shard_file)
                with open(file_path, 'rb') as f:
                    all_samples.extend(pickle.load(f))
        except FileNotFoundError:
            print(f"HATA: Dizin bulunamadı: {data_dir}")
        return all_samples

    def _prepare_labels(self):
        """Tüm veri kümesi için etiket kodlayıcıları (label encoders) hazırlar."""
        self.signal_type_encoder = LabelEncoder()
        all_mods_in_data = set(mod for s in self.spectrogram_samples for signal in s['signals'] for mod in [signal.get('mod', 'NOISE')])
        self.mlb_modulations = MultiLabelBinarizer(classes=sorted(list(all_mods_in_data)))
        
        temp_signal_types = [s['sample_type'] for s in self.spectrogram_samples]
        temp_modulations = [[signal.get('mod', 'NOISE') for signal in s['signals']] for s in self.spectrogram_samples]
        
        self.signal_types_encoded = self.signal_type_encoder.fit_transform(temp_signal_types)
        self.modulations_encoded = self.mlb_modulations.fit_transform(temp_modulations)
        self.signal_counts = [min(s['n_signals'], Config.MAX_SIGNALS) for s in self.spectrogram_samples]

    def _split_data(self, train_ratio, val_ratio, seed):
        """Veri kümesini eğitim, doğrulama ve test setlerine ayırır."""
        n_total = len(self.spectrogram_samples)
        indices = np.arange(n_total)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train : n_train + n_val]
        else:  # 'test'
            self.indices = indices[n_train + n_val :]

    def __len__(self):
        """Veri kümesindeki örnek sayısını döndürür."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Verilen indekse karşılık gelen bir veri örneğini döndürür."""
        actual_idx = self.indices[idx]
        spec_sample = self.spectrogram_samples[actual_idx]
        psd_sample = self.psd_samples[actual_idx]
        
        # Spektrogramı hedeflenen boyuta yeniden boyutlandır (örn: 256x256)
        spectrogram_np = spec_sample['spectrogram']
        zoom_factors = (Config.FREQ_BINS / spectrogram_np.shape[0], Config.FREQ_BINS / spectrogram_np.shape[1])
        resized_spectrogram = ndi.zoom(spectrogram_np, zoom_factors, order=1)  # order=1: bilinear interpolation
        
        # Verileri Torch tensörlerine dönüştür
        spectrogram = torch.tensor(resized_spectrogram, dtype=torch.float32).unsqueeze(0)
        psd = torch.tensor(psd_sample['psd'], dtype=torch.float32).unsqueeze(0)
        
        # Etiketleri ve maskeyi hazırla
        signal_type = torch.tensor(self.signal_types_encoded[actual_idx], dtype=torch.long)
        signal_count = torch.tensor(self.signal_counts[actual_idx], dtype=torch.long)
        modulation = torch.tensor(self.modulations_encoded[actual_idx], dtype=torch.float32)
        freqs = psd_sample['freqs']
        segmentation_mask = create_label_mask(spec_sample['signals'], freqs, len(freqs))
        segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.float32)
        
        inputs = (spectrogram, psd)
        targets = (signal_type, signal_count, modulation, segmentation_mask)
        
        return inputs, targets

# ==================================
# 4. MODEL MİMARİSİ
# ==================================
class DualInputUHFModel(nn.Module):
    """Spektrogram ve PSD'yi girdi olarak alan çift modlu, çok görevli bir CNN modeli."""
    def __init__(self, n_signal_types, n_modulations, dropout_rate=0.5):
        """Modelin katmanlarını tanımlar."""
        super(DualInputUHFModel, self).__init__()
        
        # --- Spektrogram Kodlayıcı (2D CNN) ---
        self.spec_encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spec_encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spec_encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spec_flatten = nn.AdaptiveAvgPool2d((1, 1))

        # --- PSD Kodlayıcı (1D CNN) ve Segmentasyon Kod Çözücü ---
        self.psd_encoder_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.psd_encoder_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.psd_encoder_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.segmentation_decoder_block2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.segmentation_decoder_block1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.segmentation_final_conv = nn.Conv1d(64, 1, kernel_size=1)
        self.psd_flatten = nn.AdaptiveAvgPool1d(1)

        # --- Birleşik Sınıflandırma Katmanı ---
        self.classification_head = nn.Sequential(
            nn.Linear(256 + 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7)
        )
        self.signal_type_predictor = nn.Linear(512, n_signal_types)
        self.signal_count_predictor = nn.Linear(512, Config.MAX_SIGNALS + 1)
        self.modulation_predictor = nn.Linear(512, n_modulations)

    def forward(self, x_spec, x_psd):
        """Modelin ileri geçiş (forward pass) işlemini gerçekleştirir."""
        # Spektrogram yolunu işle
        spec_features = self.spec_encoder_block1(x_spec)
        spec_features = self.spec_encoder_block2(spec_features)
        spec_features = self.spec_encoder_block3(spec_features)
        spec_features_flat = self.spec_flatten(spec_features).view(spec_features.size(0), -1)

        # PSD yolunu işle
        psd_features = self.psd_encoder_block1(x_psd)
        psd_features = self.psd_encoder_block2(psd_features)
        psd_features = self.psd_encoder_block3(psd_features)
        psd_features_flat = self.psd_flatten(psd_features).view(psd_features.size(0), -1)
        
        # Segmentasyon tahminini üret
        seg_output = self.segmentation_decoder_block2(psd_features)
        seg_output = self.segmentation_decoder_block1(seg_output)
        segmentation_logits = self.segmentation_final_conv(seg_output).squeeze(1)

        # İki yoldan gelen özellikleri birleştir ve sınıflandırma yap
        combined_features = torch.cat([spec_features_flat, psd_features_flat], dim=1)
        shared_representation = self.classification_head(combined_features)
        
        type_prediction = self.signal_type_predictor(shared_representation)
        count_prediction = self.signal_count_predictor(shared_representation)
        modulation_prediction = self.modulation_predictor(shared_representation)
        
        return type_prediction, count_prediction, modulation_prediction, segmentation_logits

class UnifiedMultiTaskLoss(nn.Module):
    """Dört farklı görev (tip, sayı, modülasyon, segmentasyon) için kayıpları ağırlıklı olarak birleştiren sınıf."""
    def __init__(self):
        """Kayıp fonksiyonlarını ve ağırlıklarını ilklendirir."""
        super(UnifiedMultiTaskLoss, self).__init__()
        self.task_weights = {'type': 0.1, 'count': 1.0, 'mod': 2.0, 'seg': 1.5}
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, predictions, targets):
        """Tüm görevler için toplam ağırlıklı kaybı hesaplar."""
        pred_type, pred_count, pred_mod, pred_seg = predictions
        target_type, target_count, target_mod, target_seg = targets
        
        loss_type = self.ce_loss(pred_type, target_type)
        loss_count = self.ce_loss(pred_count, target_count)
        loss_mod = self.bce_loss(pred_mod, target_mod)
        loss_seg = self.dice_loss(pred_seg, target_seg)
        
        total_loss = (self.task_weights['type'] * loss_type +
                      self.task_weights['count'] * loss_count +
                      self.task_weights['mod'] * loss_mod +
                      self.task_weights['seg'] * loss_seg)
                      
        return {
            'total_loss': total_loss,
            'type_loss': loss_type,
            'count_loss': loss_count,
            'modulation_loss': loss_mod,
            'segmentation_loss': loss_seg
        }

def calculate_seg_f1_score(pred, target, threshold=0.5):
    """Segmentasyon tahmini için F1 skorunu hesaplar."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    tp = ((pred_binary == 1) & (target == 1)).float().sum()  # True Positives
    fp = ((pred_binary == 1) & (target == 0)).float().sum()  # False Positives
    fn = ((pred_binary == 0) & (target == 1)).float().sum()  # False Negatives
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1_score.item()

# ==================================
# 5. EĞİTİM VE DOĞRULAMA DÖNGÜLERİ
# ==================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Modeli bir epoch boyunca eğitir ve metrikleri hesaplar."""
    model.train()
    total_losses = {k: 0.0 for k in ['total_loss', 'type_loss', 'count_loss', 'modulation_loss', 'segmentation_loss']}
    total_samples, correct_type, correct_count = 0, 0, 0
    all_seg_preds, all_seg_targets = [], []
    
    progress_bar = tqdm(dataloader, desc="Eğitim")
    for (spec, psd), targets in progress_bar:
        spec = spec.to(device)
        psd = psd.to(device)
        targets = [t.to(device) for t in targets]
        
        # İleri ve geri geçiş
        optimizer.zero_grad()
        predictions = model(spec, psd)
        losses = criterion(predictions, targets)
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradients exploding'i önlemek için
        optimizer.step()
        
        # Kayıpları ve metrikleri topla
        for key in total_losses:
            total_losses[key] += losses[key].item()
            
        _, pred_type = torch.max(predictions[0], 1)
        correct_type += (pred_type == targets[0]).sum().item()
        
        _, pred_count = torch.max(predictions[1], 1)
        correct_count += (pred_count == targets[1]).sum().item()
        
        total_samples += targets[0].size(0)
        all_seg_preds.append(predictions[3].detach())
        all_seg_targets.append(targets[3].detach())
        
        progress_bar.set_postfix({'Loss': f"{losses['total_loss']:.4f}"})
        
    # Epoch sonu metriklerini hesapla
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    type_accuracy = 100. * correct_type / total_samples
    count_accuracy = 100. * correct_count / total_samples
    segmentation_f1 = calculate_seg_f1_score(torch.cat(all_seg_preds), torch.cat(all_seg_targets))
    
    return total_losses, type_accuracy, count_accuracy, segmentation_f1

def validate_epoch(model, dataloader, criterion, device):
    """Modeli doğrulama veri kümesi üzerinde değerlendirir."""
    model.eval()
    total_losses = {k: 0.0 for k in ['total_loss', 'type_loss', 'count_loss', 'modulation_loss', 'segmentation_loss']}
    total_samples, correct_type, correct_count = 0, 0, 0
    all_seg_preds, all_seg_targets = [], []
    
    with torch.no_grad():
        for (spec, psd), targets in tqdm(dataloader, desc="Doğrulama"):
            spec = spec.to(device)
            psd = psd.to(device)
            targets = [t.to(device) for t in targets]
            
            predictions = model(spec, psd)
            losses = criterion(predictions, targets)
            
            for key in total_losses:
                total_losses[key] += losses[key].item()
                
            _, pred_type = torch.max(predictions[0], 1)
            correct_type += (pred_type == targets[0]).sum().item()
            
            _, pred_count = torch.max(predictions[1], 1)
            correct_count += (pred_count == targets[1]).sum().item()
            
            total_samples += targets[0].size(0)
            all_seg_preds.append(predictions[3].detach())
            all_seg_targets.append(targets[3].detach())
            
    for key in total_losses:
        total_losses[key] /= len(dataloader)
        
    type_accuracy = 100. * correct_type / total_samples
    count_accuracy = 100. * correct_count / total_samples
    segmentation_f1 = calculate_seg_f1_score(torch.cat(all_seg_preds), torch.cat(all_seg_targets))
    
    return total_losses, type_accuracy, count_accuracy, segmentation_f1

# ==================================
# 6. GÖRSELLEŞTİRME VE DEĞERLENDİRME
# ==================================
def visualize_prediction(model, dataset, device, epoch, num_samples=3):
    """Modelin tahminlerini rastgele seçilen birkaç örnek üzerinde görselleştirir."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(24, 8 * len(indices)), gridspec_kw={'width_ratios': [2, 3, 2]})
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            (spectrogram, psd), targets = dataset[idx]
            spec_gpu = spectrogram.unsqueeze(0).to(device)
            psd_gpu = psd.unsqueeze(0).to(device)
            
            predictions = model(spec_gpu, psd_gpu)
            
            # Verileri CPU'ya alıp NumPy'a çevir
            spec_np = spectrogram.squeeze(0).cpu().numpy()
            psd_np = psd.squeeze(0).cpu().numpy()
            pred_prob = torch.sigmoid(predictions[3]).squeeze(0).cpu().numpy()
            pred_binary = (pred_prob > Config.POST_PROCESSING_THRESHOLD).astype(int)
            true_mask = targets[3].cpu().numpy()
            
            # Diğer bilgileri al
            actual_idx = dataset.indices[idx]
            freqs_hz = dataset.psd_samples[actual_idx]['freqs']
            freqs_mhz = freqs_hz / 1e6
            detected_params = calculate_signal_parameters(pred_prob, freqs_hz, Config.POST_PROCESSING_THRESHOLD)
            
            # --- Sütun 1: Spektrogram ---
            ax1 = axes[i, 0]
            extent = [freqs_mhz.min(), freqs_mhz.max(), 0, spec_np.shape[0]]
            ax1.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis', extent=extent)
            ax1.set_title(f"Örnek {actual_idx} - Spektrogram")
            ax1.set_ylabel("Zaman Dilimleri")
            ax1.set_xlabel("Frekans (MHz)")

            # --- Sütun 2: Frekans Segmentasyonu ---
            ax2 = axes[i, 1]
            ax2.plot(freqs_mhz, psd_np, 'k-', label='Girdi PSD', alpha=0.5)
            ax2.plot(freqs_mhz, true_mask, 'g-', label='Gerçek Maske', linewidth=2)
            ax2.plot(freqs_mhz, pred_prob, 'r--', label='Tahmin Olasılığı')
            ax2.plot(freqs_mhz, pred_binary, 'b:', label='İkili Tahmin', alpha=0.7)
            ax2.set_title("Frekans Segmentasyonu")
            ax2.legend()
            ax2.grid(True)
            param_text = "Tespit Edilen Sinyaller:\n"
            if detected_params:
                param_text += '\n'.join([f" F: {s['center_freq_mhz']:.2f}MHz, BW: {s['bandwidth_khz']:.1f}kHz" for s in detected_params])
            else:
                param_text += "Yok"
            ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

            # --- Sütun 3: Modülasyon Tahmini ---
            ax3 = axes[i, 2]
            mod_classes = dataset.mlb_modulations.classes_
            y_pos = np.arange(len(mod_classes))
            pred_mod_probs = torch.sigmoid(predictions[2]).cpu().numpy()[0]
            true_mod_vec = targets[2].cpu().numpy()
            width = 0.35
            ax3.barh(y_pos - width/2, true_mod_vec, width, label='Gerçek', color='green')
            ax3.barh(y_pos + width/2, pred_mod_probs, width, label='Tahmini', color='red', alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(mod_classes)
            ax3.set_title("Modülasyon")
            ax3.legend()

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(Config.RESULTS_DIR, f"predictions_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200)
    
    # wandb'ye logla
    if wandb and wandb.run:
        wandb.log({f"predictions/epoch_{epoch}": wandb.Image(plt)})
    plt.close()

def generate_confusion_matrix(model, test_loader, device, class_names):
    """Test veri kümesi üzerinde çok etiketli karmaşıklık matrisleri oluşturur ve kaydeder."""
    model.eval()
    all_targets = []
    all_preds = []
    print("\nTest seti üzerinde karmaşıklık matrisi oluşturuluyor...")
    
    with torch.no_grad():
        for (spec_data, psd_data), targets in tqdm(test_loader, desc="Test Ediliyor"):
            spec_data = spec_data.to(device)
            psd_data = psd_data.to(device)
            mod_target = targets[2].to(device)
            
            # Modelden sadece modülasyon tahminini al
            mod_pred = model(spec_data, psd_data)[2]
            
            mod_pred_binary = (torch.sigmoid(mod_pred) > 0.5).cpu().numpy()
            all_preds.extend(mod_pred_binary)
            all_targets.extend(mod_target.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Her sınıf için ayrı bir 2x2 matris hesapla
    mcm = multilabel_confusion_matrix(all_targets, all_preds)
    
    # Matrisleri görselleştir
    num_classes = len(class_names)
    cols = 4
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for i, (matrix, name) in enumerate(zip(mcm, class_names)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Tahmin 0', 'Tahmin 1'],
                    yticklabels=['Gerçek 0', 'Gerçek 1'])
        axes[i].set_title(f'CM: {name}')
        axes[i].set_ylabel('Gerçek Etiket')
        axes[i].set_xlabel('Tahmin Edilen Etiket')

    # Kullanılmayan subplot'ları gizle
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(Config.RESULTS_DIR, "modulation_confusion_matrix.png")
    plt.savefig(save_path, dpi=200)
    print(f"\nKarmaşıklık matrisi şuraya kaydedildi: {save_path}")
    
    if wandb and wandb.run:
        wandb.log({"final_evaluation/confusion_matrix": wandb.Image(plt)})
    plt.close()

# ==================================
# 7. ANA EĞİTİM FONKSİYONU
# ==================================
def main():
    """Ana fonksiyon; veri yüklemeyi, model oluşturmayı, eğitimi ve değerlendirmeyi yönetir."""
    # Çıktı klasörünü oluştur
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # wandb yapılandırmasını oluştur
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and k != 'DEVICE'}
    config_dict['DEVICE'] = str(Config.DEVICE)
    
    run = None
    # wandb'yi ilklendirmeyi dene
    if wandb and UserSecretsClient:
        try:
            user_secrets = UserSecretsClient()
            wandb_api_key = user_secrets.get_secret("wandb_api_key")
            wandb.login(key=wandb_api_key)
            
            run_name = f"dual_input_refactored_{datetime.now().strftime('%Y%m%d_%H%M')}"
            run = wandb.init(project="Total_Model_UNet", config=config_dict, name=run_name)
            print("W&B loglaması başarıyla başlatıldı.")
        except Exception as e:
            print(f"W&B başlatılamadı. Loglama olmadan devam ediliyor. Hata: {e}")
            
    print(f"Kullanılacak Cihaz: {Config.DEVICE}")
    print("Çift Girdili Model Eğitimi Başlatıldı...")
    
    # Veri kümelerini ve yükleyicileri oluştur
    train_dataset = DualInputDataset(Config.SPECTROGRAM_DATA_DIR, Config.PSD_DATA_DIR, 'train', Config.TRAIN_SPLIT, Config.VAL_SPLIT)
    val_dataset = DualInputDataset(Config.SPECTROGRAM_DATA_DIR, Config.PSD_DATA_DIR, 'val', Config.TRAIN_SPLIT, Config.VAL_SPLIT)
    test_dataset = DualInputDataset(Config.SPECTROGRAM_DATA_DIR, Config.PSD_DATA_DIR, 'test', Config.TRAIN_SPLIT, Config.VAL_SPLIT)

    train_loader = DataLoader(train_dataset, Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Modeli, kayıp fonksiyonunu ve optimize ediciyi oluştur
    n_signal_types = len(train_dataset.signal_type_encoder.classes_)
    n_modulations = len(train_dataset.mlb_modulations.classes_)
    
    model = DualInputUHFModel(n_signal_types, n_modulations, Config.DROPOUT_RATE).to(Config.DEVICE)
    criterion = UnifiedMultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    if run:
        wandb.watch(model, log='all', log_freq=100)
        
    best_val_loss = float('inf')
    patience_counter = 0

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch+1}/{Config.NUM_EPOCHS}\n{'='*60}")
        
        train_losses, train_type_acc, train_count_acc, train_seg_f1 = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_losses, val_type_acc, val_count_acc, val_seg_f1 = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        print(f"\nEĞİTİM    - Loss: {train_losses['total_loss']:.4f} | Seg F1: {train_seg_f1:.4f} | Tip Acc: {train_type_acc:.2f}%")
        print(f"DOĞRULAMA - Loss: {val_losses['total_loss']:.4f} | Seg F1: {val_seg_f1:.4f} | Tip Acc: {val_type_acc:.2f}%")
        
        if run:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                **{f'train/{k}': v for k, v in train_losses.items()},
                "train/type_accuracy": train_type_acc,
                "train/count_accuracy": train_count_acc,
                "train/segmentation_f1": train_seg_f1,
                **{f'val/{k}': v for k, v in val_losses.items()},
                "val/type_accuracy": val_type_acc,
                "val/count_accuracy": val_count_acc,
                "val/segmentation_f1": val_seg_f1,
            })

        # Belirli epoch'larda görselleştirme yap
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_prediction(model, val_dataset, Config.DEVICE, epoch + 1)

        # En iyi modeli kaydet ve erken durdurmayı kontrol et
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"\n EN İYİ MODEL KAYDEDİLDİ! (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_losses['total_loss'])

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n Erken durdurma {patience_counter} epoch sonra tetiklendi.")
            break
            
    print(f"\n🎉 Eğitim tamamlandı! En iyi doğrulama kaybı: {best_val_loss:.4f}")
    
    # --- SON DEĞERLENDİRME ---
    print("\nEn iyi model test seti üzerinde son değerlendirme için yükleniyor...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    generate_confusion_matrix(model, test_loader, Config.DEVICE, train_dataset.mlb_modulations.classes_)
    
    if run:
        print("\nEn iyi model artifact'i WandB'ye yükleniyor...")
        best_model_artifact = wandb.Artifact("dual-input-uhf-model-v3", type="model")
        best_model_artifact.add_file(Config.MODEL_SAVE_PATH)
        run.log_artifact(best_model_artifact)
        wandb.finish()

if __name__ == "__main__":
    main()