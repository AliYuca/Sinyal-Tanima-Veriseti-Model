#!/usr/bin/env python3
"""
SigdetNet: 1D Frekans Spektrumunda Sinyal Tespiti için Derin Öğrenme Modeli
Bu betik, önceden işlenmiş PSD veri setini kullanarak sinyal tespit modeli eğitir.
WandB entegrasyonu ve görselleştirme ile geliştirildi.
"""

import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb


def set_seed(seed=42):
    """Reproducible sonuçlar için seed ayarlama"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceLoss(nn.Module):
    """Segmentasyon için Dice Loss implementasyonu"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Sigmoid aktivasyon
        pred = torch.sigmoid(pred)
        
        # Flatten tensörler
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Dice katsayısı hesaplama
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Dengesiz sınıflar için Focal Loss"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Dice ve Focal Loss kombinasyonu"""
    def __init__(self, dice_weight=0.7, focal_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


def create_label_mask(signals, freqs):
    """
    Sinyal listesi ve frekans vektöründen etiket maskesi oluşturur.
    
    Args:
        signals (list): Her elementi sinyal metadata'sını içeren liste
        freqs (np.ndarray): Frekans vektörü
    
    Returns:
        np.ndarray: Binary mask (0: arka plan, 1: sinyal)
    """
    label_mask = np.zeros_like(freqs, dtype=np.float32)
    
    for signal in signals:
        # Sinyal frekans aralığını hesapla
        center_freq = signal['f_off_hz']
        bandwidth = signal['bw_occ99_hz']
        
        start_freq = center_freq - bandwidth / 2
        end_freq = center_freq + bandwidth / 2
        
        # Frekans aralığına karşılık gelen indisleri bul
        start_idx = np.searchsorted(freqs, start_freq, side='left')
        end_idx = np.searchsorted(freqs, end_freq, side='right')
        
        # Sınırları kontrol et
        start_idx = max(0, start_idx)
        end_idx = min(len(freqs), end_idx)
        
        # Sinyal bölgesini 1 olarak işaretle
        if start_idx < end_idx:
            label_mask[start_idx:end_idx] = 1.0
    
    return label_mask


class SignalDataset(Dataset):
    """PSD veri seti için PyTorch Dataset sınıfı"""
    
    def __init__(self, data_files, augment=False):
        self.data_files = data_files
        self.augment = augment
        
        # Tüm veriyi belleğe yükle
        self.samples = []
        print(f"Veri dosyaları yükleniyor: {len(data_files)} dosya")
        
        for file_path in tqdm(data_files, desc="Veri yükleme"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    self.samples.extend(data)
                else:
                    self.samples.append(data)
        
        print(f"Toplam {len(self.samples)} örnek yüklendi")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # PSD vektörü ve frekans bilgisi
        psd = sample['psd'].astype(np.float32)
        freqs = sample['freqs']
        signals = sample['signals']
        
        # Etiket maskesi oluştur
        label_mask = create_label_mask(signals, freqs)
        
        # Veri augmentasyonu (eğitim sırasında)
        if self.augment:
            psd = self._augment_psd(psd)
        
        # Tensör formatına çevir
        psd_tensor = torch.from_numpy(psd).unsqueeze(0)  # (1, N)
        label_tensor = torch.from_numpy(label_mask)       # (N,)
        
        return psd_tensor, label_tensor
    
    def _augment_psd(self, psd):
        """PSD verisine augmentasyon uygula"""
        # Gaussian noise ekleme
        if random.random() < 0.3:
            noise_level = 0.01 * np.random.rand()
            psd += np.random.normal(0, noise_level, psd.shape)
        
        # Amplitude scaling
        if random.random() < 0.3:
            scale = 0.8 + 0.4 * np.random.rand()  # 0.8-1.2 arası
            psd *= scale
        
        return psd


class ResidualBlock1D(nn.Module):
    """1D Rezidüel Blok"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection için boyut uyumlama
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class PyramidPoolingModule(nn.Module):
    """Piramit Havuzlama Modülü"""
    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        
        self.pool_sizes = [1, 2, 4, 8]
        pool_out_channels = out_channels // len(self.pool_sizes)
        
        self.pools = nn.ModuleList()
        for pool_size in self.pool_sizes:
            self.pools.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(pool_size),
                    nn.Conv1d(in_channels, pool_out_channels, 1, bias=False),
                    nn.BatchNorm1d(pool_out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h = x.size(2)
        
        pool_features = []
        for pool in self.pools:
            pooled = pool(x)
            # Orijinal boyuta interpolasyon
            upsampled = F.interpolate(pooled, size=h, mode='linear', align_corners=False)
            pool_features.append(upsampled)
        
        # Tüm özellikleri birleştir
        out = torch.cat([x] + pool_features, dim=1)
        out = self.final_conv(out)
        
        return out


class AttentionGate(nn.Module):
    """Dikkat mekanizması geçidi"""
    def __init__(self, gate_channels, in_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv1d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm1d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(inter_channels, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Boyut uyumlama
        if g1.size(2) != x1.size(2):
            g1 = F.interpolate(g1, size=x1.size(2), mode='linear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class SigdetNet(nn.Module):
    """
    1D Frekans Spektrumunda Sinyal Tespiti için U-Net Benzeri Mimari
    Gelişmiş özellikler: Residual blocks, Pyramid Pooling, Attention Gates
    """
    
    def __init__(self, input_channels=1, num_classes=1, base_channels=64):
        super(SigdetNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(input_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck with Pyramid Pooling
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(base_channels * 8, base_channels * 16),
            PyramidPoolingModule(base_channels * 16, base_channels * 16),
            nn.Dropout1d(0.3)
        )
        
        # Attention Gates
        self.att4 = AttentionGate(base_channels * 16, base_channels * 8)
        self.att3 = AttentionGate(base_channels * 8, base_channels * 4)
        self.att2 = AttentionGate(base_channels * 4, base_channels * 2)
        self.att1 = AttentionGate(base_channels * 2, base_channels)
        
        # Decoder (concat sonrası kanal sayıları dikkate alınarak)
        self.dec4 = self._make_decoder_block(base_channels * 24, base_channels * 8)  # 16 + 8
        self.dec3 = self._make_decoder_block(base_channels * 12, base_channels * 4)  # 8 + 4  
        self.dec2 = self._make_decoder_block(base_channels * 6, base_channels * 2)   # 4 + 2
        self.dec1 = self._make_decoder_block(base_channels * 3, base_channels)       # 2 + 1
        
        # Final classifier
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels // 2, num_classes, 1),
        )
        
        # Maxpool
        self.pool = nn.MaxPool1d(2)
        
        self._initialize_weights()
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock1D(in_channels, out_channels),
            ResidualBlock1D(out_channels, out_channels),
            nn.Dropout1d(0.1)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock1D(in_channels, out_channels),
            ResidualBlock1D(out_channels, out_channels),
            nn.Dropout1d(0.1)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)        # N/1
        e2 = self.enc2(self.pool(e1))  # N/2
        e3 = self.enc3(self.pool(e2))  # N/4
        e4 = self.enc4(self.pool(e3))  # N/8
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # N/16
        
        # Decoder path with attention gates
        d4 = F.interpolate(b, size=e4.size(2), mode='linear', align_corners=False)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, size=e3.size(2), mode='linear', align_corners=False)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, size=e2.size(2), mode='linear', align_corners=False)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, size=e1.size(2), mode='linear', align_corners=False)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.final_conv(d1)
        
        return out.squeeze(1)  # (batch_size, N)


def calculate_metrics(pred, target, threshold=0.5):
    """Performans metriklerini hesapla"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # True/False Positives/Negatives
    tp = ((pred_binary == 1) & (target == 1)).float().sum()
    tn = ((pred_binary == 0) & (target == 0)).float().sum()
    fp = ((pred_binary == 1) & (target == 0)).float().sum()
    fn = ((pred_binary == 0) & (target == 1)).float().sum()
    
    # Metrikleri hesapla
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item()
    }


def safe_wandb_save(file_path, max_retries=3):
    """WandB'ye güvenli dosya kaydetme"""
    for retry in range(max_retries):
        try:
            # Dosyanın mevcut olduğunu kontrol et
            if not os.path.exists(file_path):
                print(f"Uyarı: Dosya bulunamadı: {file_path}")
                return False
            
            # WandB'ye kaydet
            wandb.save(file_path)
            print(f"✓ Dosya WandB'ye başarıyla kaydedildi: {file_path}")
            return True
            
        except Exception as e:
            print(f"WandB kaydetme hatası (deneme {retry + 1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                print("Yeniden deneniyor...")
                continue
            else:
                print("WandB kaydetme başarısız oldu, devam ediliyor...")
                return False


def visualize_prediction(model, dataset, device, save_dir, epoch, num_samples=3):
    """Model tahminlerini görselleştir"""
    model.eval()
    
    # Rastgele örnekler seç
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            psd, target = dataset[idx]
            psd = psd.unsqueeze(0).to(device)
            
            # Model tahmini
            pred = model(psd)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0]
            pred_binary = (pred_prob > 0.5).astype(int)
            
            # Veriyi numpy'ye çevir
            psd_np = psd.cpu().numpy()[0, 0]
            target_np = target.numpy()
            
            # PSD grafiği
            axes[i, 0].plot(psd_np)
            axes[i, 0].set_title(f'PSD Signal {idx}')
            axes[i, 0].set_ylabel('Power (dB)')
            axes[i, 0].grid(True)
            
            # Gerçek etiketler
            axes[i, 1].plot(target_np, 'g-', label='True', linewidth=2)
            axes[i, 1].fill_between(range(len(target_np)), target_np, alpha=0.3, color='green')
            axes[i, 1].set_title(f'True Labels')
            axes[i, 1].set_ylabel('Signal Presence')
            axes[i, 1].legend()
            axes[i, 1].grid(True)
            
            # Tahmin vs Gerçek
            axes[i, 2].plot(target_np, 'g-', label='True', linewidth=2)
            axes[i, 2].plot(pred_prob, 'r--', label='Predicted (prob)', linewidth=2)
            axes[i, 2].plot(pred_binary, 'b:', label='Predicted (binary)', linewidth=2)
            axes[i, 2].set_title(f'Prediction vs True')
            axes[i, 2].set_ylabel('Signal Presence')
            axes[i, 2].legend()
            axes[i, 2].grid(True)
    
    plt.tight_layout()
    
    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'predictions_epoch_{epoch}_{timestamp}.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Görselleştirme kaydedildi: {save_path}")
        
        # WandB'ye güvenli yükleme
        try:
            wandb.log({f"predictions/epoch_{epoch}": wandb.Image(save_path)})
        except Exception as e:
            print(f"WandB görsel yükleme hatası: {str(e)}")
        
    except Exception as e:
        print(f"Görsel kaydetme hatası: {str(e)}")
        save_path = None
    
    plt.close()
    return save_path


def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """Eğitim geçmişini görselleştir"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss grafiği
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score grafiği
    train_f1 = [m['f1'] for m in train_metrics]
    val_f1 = [m['f1'] for m in val_metrics]
    axes[0, 1].plot(epochs, train_f1, 'b-', label='Train F1')
    axes[0, 1].plot(epochs, val_f1, 'r-', label='Val F1')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy grafiği
    train_acc = [m['accuracy'] for m in train_metrics]
    val_acc = [m['accuracy'] for m in val_metrics]
    axes[1, 0].plot(epochs, train_acc, 'b-', label='Train Acc')
    axes[1, 0].plot(epochs, val_acc, 'r-', label='Val Acc')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision vs Recall
    train_prec = [m['precision'] for m in train_metrics]
    train_rec = [m['recall'] for m in train_metrics]
    val_prec = [m['precision'] for m in val_metrics]
    val_rec = [m['recall'] for m in val_metrics]
    
    axes[1, 1].plot(epochs, train_prec, 'b-', label='Train Precision')
    axes[1, 1].plot(epochs, train_rec, 'b--', label='Train Recall')
    axes[1, 1].plot(epochs, val_prec, 'r-', label='Val Precision')
    axes[1, 1].plot(epochs, val_rec, 'r--', label='Val Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'training_history_{timestamp}.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Eğitim geçmişi grafiği kaydedildi: {save_path}")
        
        # WandB'ye güvenli yükleme
        try:
            wandb.log({"training_history": wandb.Image(save_path)})
        except Exception as e:
            print(f"WandB eğitim geçmişi yükleme hatası: {str(e)}")
        
    except Exception as e:
        print(f"Eğitim geçmişi kaydetme hatası: {str(e)}")
        save_path = None
    
    plt.close()
    return save_path


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Eğitim")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(output.detach())
        all_targets.append(target.detach())
    
    # Metrikleri hesapla
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Bir epoch doğrulama"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Doğrulama"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.append(output)
            all_targets.append(target)
    
    # Metrikleri hesapla
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def main():
    # Sabit parametreler
    DATA_PATH = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_11_psd"
    MODEL_SAVE_DIR = os.path.join(DATA_PATH, "model_train")
    BATCH_SIZE = 32
    EPOCHS = 60
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    
    # Klasör oluştur
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Seed ayarlama
    set_seed(42)
    
    # WandB başlat
    try:
        wandb.init(
            project="psd_frekans",
            name=f"sigdetnet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "val_split": VAL_SPLIT,
                "architecture": "SigdetNet",
                "data_path": DATA_PATH
            }
        )
        print("✓ WandB başarıyla başlatıldı")
    except Exception as e:
        print(f"WandB başlatma hatası: {str(e)}")
        print("WandB olmadan devam ediliyor...")
    
    # Device ayarlama
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # WandB'ye device bilgisi ekle
    try:
        wandb.config.update({"device": str(device)})
    except:
        pass
    
    # Veri dosyalarını bul
    data_files = glob(os.path.join(DATA_PATH, "psd_shard_*.pkl"))
    if not data_files:
        raise ValueError(f"Veri dosyası bulunamadı: {DATA_PATH}")
    
    print(f"Bulunan veri dosyaları: {len(data_files)}")
    try:
        wandb.config.update({"num_data_files": len(data_files)})
    except:
        pass
    
    # Dataset oluştur
    full_dataset = SignalDataset(data_files, augment=True)
    
    # Train/Validation split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Validation dataset için augmentasyon kapat
    val_dataset.dataset = SignalDataset(data_files, augment=False)
    
    print(f"Eğitim örnekleri: {len(train_dataset)}")
    print(f"Doğrulama örnekleri: {len(val_dataset)}")
    
    try:
        wandb.config.update({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        })
    except:
        pass
    
    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Model oluştur
    model = SigdetNet(input_channels=1, num_classes=1, base_channels=64)
    model = model.to(device)
    
    # Model parametrelerini yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametreler: {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    
    try:
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
        # WandB model takibi
        wandb.watch(model, log="all", log_freq=100)
    except:
        pass
    
    # Loss ve optimizer
    criterion = CombinedLoss(dice_weight=0.7, focal_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Model kaydetme yolu
    model_save_path = os.path.join(MODEL_SAVE_DIR, 'best_sigdetnet_model.pth')
    
    # Eğitim geçmişi için listeler
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    # Eğitim döngüsü
    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    max_patience = 11
    
    print("\nEğitim başlıyor...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        # Eğitim
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Doğrulama
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Geçmişe ekle
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Scheduler güncelleme
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # WandB'ye metrikleri gönder
        try:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/f1": train_metrics['f1'],
                "train/accuracy": train_metrics['accuracy'],
                "train/precision": train_metrics['precision'],
                "train/recall": train_metrics['recall'],
                "val/loss": val_loss,
                "val/f1": val_metrics['f1'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "learning_rate": current_lr
            })
        except Exception as e:
            print(f"WandB log hatası: {str(e)}")
        
        # Sonuçları yazdır
        print(f"Eğitim - Loss: {train_loss:.6f} | F1: {train_metrics['f1']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")
        print(f"Doğrulama - Loss: {val_loss:.6f} | F1: {val_metrics['f1']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # En iyi modeli kaydet
        is_best = False
        if val_loss < best_val_loss or val_metrics['f1'] > best_f1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✓ En iyi doğrulama loss güncellendi: {val_loss:.6f}")
                is_best = True
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                print(f"✓ En iyi F1 skoru güncellendi: {best_f1:.4f}")
                is_best = True
            
            # Model kaydet
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_f1': best_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_metrics_history': train_metrics_history,
                    'val_metrics_history': val_metrics_history,
                }, model_save_path)
                print(f"✓ Model kaydedildi: {model_save_path}")
                
                # WandB'ye güvenli model kaydet
                if is_best:
                    safe_wandb_save(model_save_path)
                    try:
                        wandb.log({
                            "best_val_loss": best_val_loss,
                            "best_f1": best_f1
                        })
                    except:
                        pass
                        
            except Exception as e:
                print(f"Model kaydetme hatası: {str(e)}")
        else:
            patience_counter += 1
        
        # Her 10 epoch'ta bir görselleştirme yap
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Tahmin görselleştirmesi oluşturuluyor...")
            try:
                vis_path = visualize_prediction(model, val_dataset, device, MODEL_SAVE_DIR, epoch + 1)
                if vis_path:
                    print(f"✓ Görselleştirme kaydedildi: {vis_path}")
            except Exception as e:
                print(f"Görselleştirme hatası: {str(e)}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping! {max_patience} epoch boyunca iyileşme görülmedi.")
            break
    
    print("\n" + "=" * 80)
    print("Eğitim tamamlandı!")
    print(f"En iyi doğrulama loss: {best_val_loss:.6f}")
    print(f"En iyi F1 skoru: {best_f1:.4f}")
    print(f"Model kaydedildi: {model_save_path}")
    
    # Final görselleştirmeler
    print("\nFinal görselleştirmeler oluşturuluyor...")
    
    try:
        # Eğitim geçmişi grafiği
        history_path = plot_training_history(
            train_losses, val_losses, 
            train_metrics_history, val_metrics_history, 
            MODEL_SAVE_DIR
        )
        if history_path:
            print(f"✓ Eğitim geçmişi grafiği: {history_path}")
    except Exception as e:
        print(f"Eğitim geçmişi grafiği hatası: {str(e)}")
    
    try:
        # Final tahmin görselleştirmesi
        final_vis_path = visualize_prediction(model, val_dataset, device, MODEL_SAVE_DIR, "final", num_samples=5)
        if final_vis_path:
            print(f"✓ Final tahminler: {final_vis_path}")
    except Exception as e:
        print(f"Final görselleştirme hatası: {str(e)}")
    
    # WandB'ye final metrikleri gönder
    try:
        wandb.log({
            "final/best_val_loss": best_val_loss,
            "final/best_f1": best_f1,
            "final/total_epochs": len(train_losses)
        })
    except:
        pass
    
    # WandB oturumunu sonlandır
    try:
        wandb.finish()
        print("✓ WandB oturumu sonlandırıldı")
    except:
        pass
    
    print("\nTüm işlemler tamamlandı!")
    print(f"Çıktılar: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()