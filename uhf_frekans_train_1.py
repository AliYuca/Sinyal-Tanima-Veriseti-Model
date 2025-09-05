# UHF Frekans ve Bant Genişliği Tahmin Modeli

import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =========================
# DATASET
# =========================
class UHFDataset(Dataset):
    def __init__(self, dataset_dir, mode='multi_signal', normalize_freq=True, normalize_bw=True):
        """
        Args:
            dataset_dir: Veri setinin bulunduğu dizin
            mode: 'single' (tek sinyal), 'multi_signal' (çoklu sinyal), 'mixed' (karışık)
            normalize_freq: Frekans normalizasyonu yapılsın mı
            normalize_bw: Bant genişliği normalizasyonu yapılsın mı
        """
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.normalize_freq = normalize_freq
        self.normalize_bw = normalize_bw
        
        # Manifest bilgilerini yükle
        with open(os.path.join(dataset_dir, "manifest.json"), 'r') as f:
            self.manifest = json.load(f)
        
        self.fs = self.manifest['baseband_fs']
        
        # Tüm shard'ları yükle
        self.features = []
        self.targets = []
        self.signal_counts = []
        
        shard_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('shard_')]
        
        print(f"📊 {len(shard_dirs)} shard yükleniyor...")
        
        for shard_dir in tqdm(shard_dirs, desc="Shard'lar yükleniyor"):
            shard_path = os.path.join(dataset_dir, shard_dir)
            
            # Features yükle
            features = np.load(os.path.join(shard_path, "features.npy"))
            
            # Labels yükle
            with open(os.path.join(shard_path, "labels.pkl"), 'rb') as f:
                labels = pickle.load(f)
            
            for i, label in enumerate(labels):
                if label['type'] == 'noise':
                    continue  # Gürültü örneklerini atla
                
                feature = features[i]  
                signals = label['signals']
                
                if self.mode == 'single' and len(signals) != 1:
                    continue
                elif self.mode == 'multi_signal' and len(signals) < 2:
                    continue
                
                # Hedef değerleri hazırla
                if self.mode == 'single':
                    # Tek sinyal için
                    sig = signals[0]
                    target = {
                        'f_center': sig['f_center_est_hz'],
                        'bw_occ99': sig['bw_occ99_hz'],
                        'bw_rms': sig['bw_rms_hz'],
                        'bw_3db': sig['bw_3db_hz']
                    }
                else:
                    # Çoklu sinyal için - en güçlü sinyalin parametreleri
                    sig = max(signals, key=lambda s: s['rel_power_db'])
                    target = {
                        'f_center': sig['f_center_est_hz'],
                        'bw_occ99': sig['bw_occ99_hz'],
                        'bw_rms': sig['bw_rms_hz'],
                        'bw_3db': sig['bw_3db_hz'],
                        'num_signals': len(signals)
                    }
                
                self.features.append(feature)
                self.targets.append(target)
                self.signal_counts.append(len(signals))
        
        print(f"✅ {len(self.features)} örnek yüklendi")
        
        # Normalizasyon parametrelerini hesapla
        if self.normalize_freq or self.normalize_bw:
            self._compute_normalization_params()
    
    def _compute_normalization_params(self):
        """Normalizasyon parametrelerini hesapla"""
        f_centers = [t['f_center'] for t in self.targets]
        bw_occ99s = [t['bw_occ99'] for t in self.targets]
        bw_rmss = [t['bw_rms'] for t in self.targets]
        bw_3dbs = [t['bw_3db'] for t in self.targets]
        
        if self.normalize_freq:
            self.f_center_mean = np.mean(f_centers)
            self.f_center_std = np.std(f_centers)
            print(f"📈 Frekans normalizasyonu: μ={self.f_center_mean:.2e}, σ={self.f_center_std:.2e}")
        
        if self.normalize_bw:
            self.bw_occ99_mean = np.mean(bw_occ99s)
            self.bw_occ99_std = np.std(bw_occ99s)
            self.bw_rms_mean = np.mean(bw_rmss)
            self.bw_rms_std = np.std(bw_rmss)
            self.bw_3db_mean = np.mean(bw_3dbs)
            self.bw_3db_std = np.std(bw_3dbs)
            print(f"📊 BW normalizasyonu:")
            print(f"  OCC99: μ={self.bw_occ99_mean:.2e}, σ={self.bw_occ99_std:.2e}")
            print(f"  RMS: μ={self.bw_rms_mean:.2e}, σ={self.bw_rms_std:.2e}")
            print(f"  3dB: μ={self.bw_3db_mean:.2e}, σ={self.bw_3db_std:.2e}")
    
    def normalize_target(self, target):
        """Hedef değerleri normalize et"""
        norm_target = {}
        
        if self.normalize_freq:
            norm_target['f_center'] = (target['f_center'] - self.f_center_mean) / (self.f_center_std + 1e-8)
        else:
            norm_target['f_center'] = target['f_center'] / self.fs  # normalize
        
        if self.normalize_bw:
            norm_target['bw_occ99'] = (target['bw_occ99'] - self.bw_occ99_mean) / (self.bw_occ99_std + 1e-8)
            norm_target['bw_rms'] = (target['bw_rms'] - self.bw_rms_mean) / (self.bw_rms_std + 1e-8)
            norm_target['bw_3db'] = (target['bw_3db'] - self.bw_3db_mean) / (self.bw_3db_std + 1e-8)
        else:
            norm_target['bw_occ99'] = target['bw_occ99'] / self.fs
            norm_target['bw_rms'] = target['bw_rms'] / self.fs
            norm_target['bw_3db'] = target['bw_3db'] / self.fs
        
        if 'num_signals' in target:
            norm_target['num_signals'] = target['num_signals']
        
        return norm_target
    
    def denormalize_prediction(self, norm_pred):
        """Tahmin değerlerini denormalize et"""
        pred = {}
        
        if self.normalize_freq:
            pred['f_center'] = norm_pred['f_center'] * self.f_center_std + self.f_center_mean
        else:
            pred['f_center'] = norm_pred['f_center'] * self.fs
        
        if self.normalize_bw:
            pred['bw_occ99'] = norm_pred['bw_occ99'] * self.bw_occ99_std + self.bw_occ99_mean
            pred['bw_rms'] = norm_pred['bw_rms'] * self.bw_rms_std + self.bw_rms_mean
            pred['bw_3db'] = norm_pred['bw_3db'] * self.bw_3db_std + self.bw_3db_mean
        else:
            pred['bw_occ99'] = norm_pred['bw_occ99'] * self.fs
            pred['bw_rms'] = norm_pred['bw_rms'] * self.fs
            pred['bw_3db'] = norm_pred['bw_3db'] * self.fs
        
        if 'num_signals' in norm_pred:
            pred['num_signals'] = norm_pred['num_signals']
        
        return pred
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float() 
        target = self.normalize_target(self.targets[idx])
        
        # Hedef vektörünü oluştur
        if self.mode == 'single':
            target_vec = torch.tensor([
                target['f_center'],
                target['bw_occ99'],
                target['bw_rms'],
                target['bw_3db']
            ], dtype=torch.float32)
        else:
            target_vec = torch.tensor([
                target['f_center'],
                target['bw_occ99'],
                target['bw_rms'],
                target['bw_3db'],
                target['num_signals']
            ], dtype=torch.float32)
        
        return feature, target_vec

# =========================
# MODEL Mimarisi
# =========================
class SpectralCNN(nn.Module):
    def __init__(self, input_channels=4, output_dim=4, dropout=0.2):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regresyon başlığı
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Regresyon
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc_out(x)
        
        return x

# Daha basit ve güvenilir model
class SpectralCNNSimple(nn.Module):
    def __init__(self, input_channels=4, output_dim=4, dropout=0.2):
        super().__init__()
        
        # Basit konvolüsyon katmanları
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Regression head - 2x256 (avg + max pooling)
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction with adaptive pooling
        x = F.relu(self.bn1(self.conv1(x)))
        if min(x.size(2), x.size(3)) >= 4:
            x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        if min(x.size(2), x.size(3)) >= 4:
            x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Dual global pooling
        avg_pool = self.global_avg_pool(x).view(batch_size, -1)
        max_pool = self.global_max_pool(x).view(batch_size, -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Regression
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x

# =========================
# Eğitim Fonksiyonları
# =========================
class FrequencyBandwidthTrainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50):
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Model kaydı
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # En iyi modeli yükle
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()

# =========================
# EVALUATION FUNCTIONS
# =========================
def evaluate_model(model, test_loader, dataset, device):
    """Model performansını değerlendir"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Denormalizasyon
    results = []
    for i in range(len(predictions)):
        if dataset.mode == 'single':
            pred_dict = {
                'f_center': predictions[i, 0],
                'bw_occ99': predictions[i, 1],
                'bw_rms': predictions[i, 2],
                'bw_3db': predictions[i, 3]
            }
            target_dict = {
                'f_center': targets[i, 0],
                'bw_occ99': targets[i, 1],
                'bw_rms': targets[i, 2],
                'bw_3db': targets[i, 3]
            }
        else:
            pred_dict = {
                'f_center': predictions[i, 0],
                'bw_occ99': predictions[i, 1],
                'bw_rms': predictions[i, 2],
                'bw_3db': predictions[i, 3],
                'num_signals': predictions[i, 4]
            }
            target_dict = {
                'f_center': targets[i, 0],
                'bw_occ99': targets[i, 1],
                'bw_rms': targets[i, 2],
                'bw_3db': targets[i, 3],
                'num_signals': targets[i, 4]
            }
        
        pred_denorm = dataset.denormalize_prediction(pred_dict)
        target_denorm = dataset.denormalize_prediction(target_dict)
        
        results.append({
            'pred': pred_denorm,
            'target': target_denorm
        })
    
    # Metrikleri hesapla
    metrics = {}
    for param in ['f_center', 'bw_occ99', 'bw_rms', 'bw_3db']:
        pred_vals = [r['pred'][param] for r in results]
        target_vals = [r['target'][param] for r in results]
        
        mae = mean_absolute_error(target_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(target_vals, pred_vals))
        
        metrics[param] = {'mae': mae, 'rmse': rmse}
        
        print(f"\n📊 {param.upper()}:")
        print(f"  MAE: {mae:.2e} Hz")
        print(f"  RMSE: {rmse:.2e} Hz")
        
        if param == 'f_center':
            # Frekans için yüzdelik hata
            relative_error = np.mean(np.abs(np.array(pred_vals) - np.array(target_vals)) / (np.array(target_vals) + 1e-8))
            print(f"  Relative Error: {relative_error*100:.2f}%")
    
    return results, metrics

# =========================
# MAIN 
# =========================
def train_frequency_bandwidth_model(
    dataset_dir,
    mode='multi_signal',
    batch_size=32,
    epochs=100,
    learning_rate=1e-3
):
    """Ana eğitim fonksiyonu"""
    
    print("🚀 UHF Frekans ve Bant Genişliği Tahmin Modeli Eğitimi")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Dataset yükleme
    print("\n📊 Veri seti yükleniyor...")
    dataset = UHFDataset(dataset_dir, mode=mode, normalize_freq=True, normalize_bw=True)
    
    # Train/validation split
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"📈 Train samples: {len(train_dataset)}")
    print(f"📉 Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    output_dim = 4 if mode == 'single' else 5
    
    sample_feature, _ = dataset[0]
    print(f"🔍 Spektrogram boyutu: {sample_feature.shape}")  # [4, F, T]
    
    # Eğer spektrogram çok küçükse basit modeli kullan
    if sample_feature.shape[1] < 32 or sample_feature.shape[2] < 32:
        print("Küçük spektrogram boyutu tespit edildi, basit model kullanılıyor...")
        model = SpectralCNNSimple(input_channels=4, output_dim=output_dim, dropout=0.2)
    else:
        print("Normal spektrogram boyutu, standart model kullanılıyor...")
        model = SpectralCNN(input_channels=4, output_dim=output_dim, dropout=0.2)
    
    print(f"\n🧠 Model oluşturuldu - Output dim: {output_dim}")
    print(f"📊 Toplam parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    

    trainer = FrequencyBandwidthTrainer(model, device, learning_rate)
    
    # Training
    print("\n🎯 Eğitim başlıyor...")
    trainer.train(train_loader, val_loader, epochs)
    
    # Training history
    trainer.plot_training_history()
    
    # Evaluation
    print("\n📊 Model değerlendiriliyor...")
    results, metrics = evaluate_model(model, val_loader, dataset, device)
    
    return model, dataset, results, metrics

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_test_8"

    model, dataset, results, metrics = train_frequency_bandwidth_model(
        dataset_dir=DATASET_DIR,
        mode='multi_signal',  # 'single' veya 'multi_signal'
        batch_size=32,
        epochs=50,
        learning_rate=1e-3
    )
    
    print("\n✅ Eğitim tamamlandı!")
    print("💾 En iyi model 'best_model.pth' dosyasına kaydedildi.")