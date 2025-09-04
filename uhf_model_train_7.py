#!/usr/bin/env python3
"""
Enhanced Multi-Task UHF Signal Classification CNN
- Signal type classification (noise, single, mixed_close, mixed_far)
- Signal count estimation for mixed signals
- Multi-label modulation classification
- Multi-output center frequency estimation
- CUDA optimized with comprehensive evaluation
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =========================
# CONFIGURATION
# =========================
class Config:
    # Dataset paths
    DATA_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_7"
    
    # Training parameters
    BATCH_SIZE = 16  # Reduced for complex multi-task learning
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15
    
    # Model parameters
    DROPOUT_RATE = 0.4
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    
    # Output
    MODEL_SAVE_PATH = "best_multitask_uhf_model.pth"
    RESULTS_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_7\model_resuts_7_2"
    
    # Task-specific parameters
    MAX_SIGNALS = 4  # Maximum number of signals in mixed cases
    FREQ_TOLERANCE = 0.15  # 15% frequency estimation tolerance
    MODULATIONS = ['FM', 'OFDM', 'GFSK', 'QPSK']
    SIGNAL_TYPES = ['noise', 'single', 'mixed_close', 'mixed_far']

# =========================
# ENHANCED DATASET CLASS
# =========================
class MultiTaskUHFDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.7, val_ratio=0.15, seed=42):
        """Enhanced dataset for multi-task learning"""
        self.data_dir = data_dir
        self.split = split
        np.random.seed(seed)
        
        # Load manifest
        with open(os.path.join(data_dir, 'manifest.json'), 'r') as f:
            self.manifest = json.load(f)
        
        # Load all data
        self._load_all_data()
        
        # Prepare multi-task labels
        self._prepare_multitask_labels()
        
        # Split data
        self._split_data(train_ratio, val_ratio, seed)
        
        print(f"{split.upper()} split: {len(self.indices)} samples")
        self._print_dataset_stats()
    
    def _load_all_data(self):
        """Load data from all shards"""
        shard_dirs = sorted([d for d in os.listdir(self.data_dir) if d.startswith('shard_')])
        
        all_features, all_labels = [], []
        print(f"Loading data from {len(shard_dirs)} shards...")
        
        for shard_dir in tqdm(shard_dirs):
            shard_path = os.path.join(self.data_dir, shard_dir)
            features = np.load(os.path.join(shard_path, 'features.npy'))
            with open(os.path.join(shard_path, 'labels.pkl'), 'rb') as f:
                labels = pickle.load(f)
            
            all_features.append(features)
            all_labels.extend(labels)
        
        self.features = np.concatenate(all_features, axis=0)
        self.raw_labels = all_labels
        print(f"Loaded {len(self.raw_labels)} samples with feature shape {self.features.shape}")
    
    def _prepare_multitask_labels(self):
        """Prepare labels for all tasks"""
        self.signal_types = []           # noise, single, mixed_close, mixed_far
        self.signal_counts = []          # 0 for noise, 1 for single, 2-4 for mixed
        self.modulation_labels = []      # multi-hot encoding for modulations
        self.center_frequencies = []     # list of center frequencies per sample
        
        # Initialize encoders
        self.signal_type_encoder = LabelEncoder()
        self.mlb_modulations = MultiLabelBinarizer(classes=Config.MODULATIONS)
        
        temp_signal_types = []
        temp_modulations = []
        
        for label in self.raw_labels:
            signal_type = label['type']
            num_signals = label['num_signals']
            signals_info = label['signals']
            
            # Signal type
            temp_signal_types.append(signal_type)
            
            # Signal count
            self.signal_counts.append(num_signals)
            
            # Modulations and frequencies
            if num_signals == 0:  # noise
                temp_modulations.append([])
                self.center_frequencies.append([0.0])
            else:
                sample_mods = []
                sample_freqs = []
                for signal in signals_info:
                    sample_mods.append(signal['mod'])
                    # Use estimated center frequency
                    freq = signal.get('f_center_est_hz', 0.0)
                    sample_freqs.append(freq)
                
                temp_modulations.append(sample_mods)
                self.center_frequencies.append(sample_freqs)
        
        # Encode signal types
        self.signal_types = self.signal_type_encoder.fit_transform(temp_signal_types)
        
        # Encode modulations (multi-hot)
        self.modulation_labels = self.mlb_modulations.fit_transform(temp_modulations)
        
        # Normalize frequencies
        all_freqs = []
        for freqs in self.center_frequencies:
            all_freqs.extend(freqs)
        all_freqs = np.array([f for f in all_freqs if f != 0.0])
        
        if len(all_freqs) > 0:
            self.freq_mean = np.mean(all_freqs)
            self.freq_std = np.std(all_freqs) + 1e-8
        else:
            self.freq_mean = 0.0
            self.freq_std = 1.0
        
        print(f"Signal types: {list(self.signal_type_encoder.classes_)}")
        print(f"Modulations: {Config.MODULATIONS}")
        print(f"Frequency normalization: mean={self.freq_mean:.0f}, std={self.freq_std:.0f}")
    
    def _split_data(self, train_ratio, val_ratio, seed):
        """Split data maintaining class balance"""
        n_total = len(self.raw_labels)
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
    
    def _print_dataset_stats(self):
        """Print dataset statistics for current split"""
        split_signal_types = [self.signal_types[i] for i in self.indices]
        split_counts = [self.signal_counts[i] for i in self.indices]
        
        print(f"\nDataset statistics for {self.split}:")
        for i, stype in enumerate(self.signal_type_encoder.classes_):
            count = sum(1 for x in split_signal_types if x == i)
            print(f"  {stype}: {count}")
        
        print(f"\nSignal count distribution:")
        for count in range(5):
            n_samples = sum(1 for x in split_counts if x == count)
            print(f"  {count} signals: {n_samples}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Features
        feature = torch.tensor(self.features[actual_idx], dtype=torch.float32)
        
        # Signal type
        signal_type = torch.tensor(self.signal_types[actual_idx], dtype=torch.long)
        
        # Signal count
        signal_count = torch.tensor(self.signal_counts[actual_idx], dtype=torch.long)
        
        # Modulation labels (multi-hot)
        modulation = torch.tensor(self.modulation_labels[actual_idx], dtype=torch.float32)
        
        # Center frequencies (normalized, padded to MAX_SIGNALS)
        freqs = self.center_frequencies[actual_idx]
        freq_tensor = torch.zeros(Config.MAX_SIGNALS, dtype=torch.float32)
        
        for i, freq in enumerate(freqs[:Config.MAX_SIGNALS]):
            if freq != 0.0:
                freq_tensor[i] = (freq - self.freq_mean) / self.freq_std
            # else remains 0.0 (padding)
        
        # Frequency mask (1 for valid frequencies, 0 for padding)
        freq_mask = torch.zeros(Config.MAX_SIGNALS, dtype=torch.float32)
        for i in range(min(len(freqs), Config.MAX_SIGNALS)):
            if freqs[i] != 0.0:
                freq_mask[i] = 1.0
        
        return feature, signal_type, signal_count, modulation, freq_tensor, freq_mask

# =========================
# MULTI-TASK CNN MODEL
# =========================
class MultiTaskUHFCNN(nn.Module):
    def __init__(self, input_channels=4, dropout_rate=0.4):
        super(MultiTaskUHFCNN, self).__init__()
        
        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.3),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Task-specific heads
        # 1. Signal type classification (4 classes)
        self.signal_type_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, len(Config.SIGNAL_TYPES))
        )
        
        # 2. Signal count regression/classification (0-4)
        self.signal_count_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, Config.MAX_SIGNALS + 1)  # 0-4 signals
        )
        
        # 3. Modulation multi-label classification
        self.modulation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, len(Config.MODULATIONS))
        )
        
        # 4. Frequency estimation (multiple outputs)
        self.frequency_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, Config.MAX_SIGNALS)  # Up to 4 frequencies
        )
    
    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        shared_repr = self.shared_fc(features)
        
        # Task-specific predictions
        signal_type_logits = self.signal_type_head(shared_repr)
        signal_count_logits = self.signal_count_head(shared_repr)
        modulation_logits = self.modulation_head(shared_repr)
        frequency_preds = self.frequency_head(shared_repr)
        
        return signal_type_logits, signal_count_logits, modulation_logits, frequency_preds

# =========================
# MULTI-TASK LOSS FUNCTION
# =========================
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        
        # Default task weights
        if task_weights is None:
            task_weights = {
                'signal_type': 1.0,
                'signal_count': 1.0,
                'modulation': 1.0,
                'frequency': 0.5  # Lower weight for regression task
            }
        
        self.task_weights = task_weights
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss(reduction='none')  # For masked frequency loss
    
    def forward(self, predictions, targets):
        signal_type_logits, signal_count_logits, modulation_logits, frequency_preds = predictions
        signal_type_targets, signal_count_targets, modulation_targets, freq_targets, freq_mask = targets
        
        # 1. Signal type loss
        signal_type_loss = self.ce_loss(signal_type_logits, signal_type_targets)
        
        # 2. Signal count loss
        signal_count_loss = self.ce_loss(signal_count_logits, signal_count_targets)
        
        # 3. Modulation loss (multi-label)
        modulation_loss = self.bce_loss(modulation_logits, modulation_targets)
        
        # 4. Frequency loss (masked MSE)
        freq_loss_raw = self.mse_loss(frequency_preds, freq_targets)
        freq_loss = (freq_loss_raw * freq_mask).sum() / (freq_mask.sum() + 1e-8)
        
        # Combined loss
        total_loss = (
            self.task_weights['signal_type'] * signal_type_loss +
            self.task_weights['signal_count'] * signal_count_loss +
            self.task_weights['modulation'] * modulation_loss +
            self.task_weights['frequency'] * freq_loss
        )
        
        return {
            'total_loss': total_loss,
            'signal_type_loss': signal_type_loss,
            'signal_count_loss': signal_count_loss,
            'modulation_loss': modulation_loss,
            'frequency_loss': freq_loss
        }

# =========================
# TRAINING FUNCTIONS
# =========================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_losses = {
        'total_loss': 0.0,
        'signal_type_loss': 0.0,
        'signal_count_loss': 0.0,
        'modulation_loss': 0.0,
        'frequency_loss': 0.0
    }
    
    correct_signal_type = 0
    correct_signal_count = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_data in pbar:
        features, signal_type, signal_count, modulation, freq_tensor, freq_mask = batch_data
        
        # Move to device
        features = features.to(device)
        signal_type = signal_type.to(device)
        signal_count = signal_count.to(device)
        modulation = modulation.to(device)
        freq_tensor = freq_tensor.to(device)
        freq_mask = freq_mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features)
        targets = (signal_type, signal_count, modulation, freq_tensor, freq_mask)
        
        # Calculate losses
        losses = criterion(predictions, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()
        
        # Calculate accuracies
        signal_type_logits, signal_count_logits, _, _ = predictions
        
        _, pred_signal_type = torch.max(signal_type_logits, 1)
        correct_signal_type += (pred_signal_type == signal_type).sum().item()
        
        _, pred_signal_count = torch.max(signal_count_logits, 1)
        correct_signal_count += (pred_signal_count == signal_count).sum().item()
        
        total_samples += signal_type.size(0)
        
        # Update progress
        pbar.set_postfix({
            'Loss': f"{losses['total_loss'].item():.4f}",
            'Type Acc': f"{100.*correct_signal_type/total_samples:.1f}%",
            'Count Acc': f"{100.*correct_signal_count/total_samples:.1f}%"
        })
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    # Calculate accuracies
    signal_type_acc = 100. * correct_signal_type / total_samples
    signal_count_acc = 100. * correct_signal_count / total_samples
    
    return epoch_losses, signal_type_acc, signal_count_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_losses = {
        'total_loss': 0.0,
        'signal_type_loss': 0.0,
        'signal_count_loss': 0.0,
        'modulation_loss': 0.0,
        'frequency_loss': 0.0
    }
    
    correct_signal_type = 0
    correct_signal_count = 0
    total_samples = 0
    
    all_signal_type_preds = []
    all_signal_type_targets = []
    all_modulation_preds = []
    all_modulation_targets = []
    all_freq_errors = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", leave=False):
            features, signal_type, signal_count, modulation, freq_tensor, freq_mask = batch_data
            
            # Move to device
            features = features.to(device)
            signal_type = signal_type.to(device)
            signal_count = signal_count.to(device)
            modulation = modulation.to(device)
            freq_tensor = freq_tensor.to(device)
            freq_mask = freq_mask.to(device)
            
            # Forward pass
            predictions = model(features)
            targets = (signal_type, signal_count, modulation, freq_tensor, freq_mask)
            
            # Calculate losses
            losses = criterion(predictions, targets)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Extract predictions
            signal_type_logits, signal_count_logits, modulation_logits, frequency_preds = predictions
            
            _, pred_signal_type = torch.max(signal_type_logits, 1)
            correct_signal_type += (pred_signal_type == signal_type).sum().item()
            
            _, pred_signal_count = torch.max(signal_count_logits, 1)
            correct_signal_count += (pred_signal_count == signal_count).sum().item()
            
            total_samples += signal_type.size(0)
            
            # Store predictions for detailed analysis
            all_signal_type_preds.extend(pred_signal_type.cpu().numpy())
            all_signal_type_targets.extend(signal_type.cpu().numpy())
            
            # Modulation predictions (multi-label)
            mod_probs = torch.sigmoid(modulation_logits)
            mod_preds = (mod_probs > 0.5).float()
            all_modulation_preds.append(mod_preds.cpu().numpy())
            all_modulation_targets.append(modulation.cpu().numpy())
            
            # Frequency errors
            freq_errors = torch.abs(frequency_preds - freq_tensor) * freq_mask
            all_freq_errors.extend(freq_errors[freq_mask > 0].cpu().numpy())
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    # Calculate accuracies
    signal_type_acc = 100. * correct_signal_type / total_samples
    signal_count_acc = 100. * correct_signal_count / total_samples
    
    # Additional metrics
    all_modulation_preds = np.vstack(all_modulation_preds) if all_modulation_preds else np.array([])
    all_modulation_targets = np.vstack(all_modulation_targets) if all_modulation_targets else np.array([])
    
    return (epoch_losses, signal_type_acc, signal_count_acc, 
            all_signal_type_preds, all_signal_type_targets,
            all_modulation_preds, all_modulation_targets, all_freq_errors)

# =========================
# EVALUATION FUNCTIONS
# =========================
def evaluate_frequency_accuracy(freq_preds, freq_targets, freq_mask, dataset, tolerance=0.15):
    """Evaluate frequency prediction accuracy with tolerance"""
    correct_freqs = 0
    total_freqs = 0
    
    # Denormalize predictions
    freq_preds_denorm = freq_preds * dataset.freq_std + dataset.freq_mean
    freq_targets_denorm = freq_targets * dataset.freq_std + dataset.freq_mean
    
    for i in range(freq_mask.shape[0]):
        for j in range(freq_mask.shape[1]):
            if freq_mask[i, j] > 0:  # Valid frequency
                pred_freq = freq_preds_denorm[i, j].item()
                true_freq = freq_targets_denorm[i, j].item()
                
                if true_freq > 0:  # Non-zero frequency
                    error_pct = abs(pred_freq - true_freq) / true_freq
                    if error_pct <= tolerance:
                        correct_freqs += 1
                total_freqs += 1
    
    return correct_freqs / (total_freqs + 1e-8) * 100.0

def calculate_modulation_metrics(mod_preds, mod_targets):
    """Calculate multi-label modulation classification metrics"""
    if len(mod_preds) == 0:
        return {}
    
    # Calculate per-class metrics
    n_classes = mod_targets.shape[1]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = np.sum((mod_preds[:, i] == 1) & (mod_targets[:, i] == 1))
        fp = np.sum((mod_preds[:, i] == 1) & (mod_targets[:, i] == 0))
        fn = np.sum((mod_preds[:, i] == 0) & (mod_targets[:, i] == 1))
        
        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1': np.mean(f1)
    }

# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_training_progress(history, save_dir):
    """Plot comprehensive training progress"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['total_loss']['train']) + 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Multi-Task Training Progress', fontsize=16, fontweight='bold')
    
    # Total Loss
    axes[0, 0].plot(epochs, history['total_loss']['train'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['total_loss']['val'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Signal Type Accuracy
    axes[0, 1].plot(epochs, history['signal_type_acc']['train'], 'g-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['signal_type_acc']['val'], 'orange', label='Val', linewidth=2)
    axes[0, 1].set_title('Signal Type Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Signal Count Accuracy
    axes[1, 0].plot(epochs, history['signal_count_acc']['train'], 'purple', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['signal_count_acc']['val'], 'brown', label='Val', linewidth=2)
    axes[1, 0].set_title('Signal Count Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Task-specific losses
    task_losses = ['signal_type_loss', 'signal_count_loss', 'modulation_loss', 'frequency_loss']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (task, color) in enumerate(zip(task_losses, colors)):
        if i < 2:
            ax = axes[1, 1] if i == 0 else axes[2, 0]
            ax.plot(epochs, history[task]['train'], color=color, linestyle='-', 
                   label=f'Train {task}', linewidth=2)
            ax.plot(epochs, history[task]['val'], color=color, linestyle='--', 
                   label=f'Val {task}', linewidth=2)
            ax.set_title(f'{task.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Combined task losses
    axes[2, 1].plot(epochs, history['modulation_loss']['train'], 'r-', label='Modulation (train)', linewidth=2)
    axes[2, 1].plot(epochs, history['frequency_loss']['train'], 'purple', label='Frequency (train)', linewidth=2)
    axes[2, 1].set_title('Modulation & Frequency Losses')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close(fig) # Belleği boşaltmak için figürü kapat

def plot_confusion_matrices(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot confusion matrix for signal type classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Belleği boşaltmak için figürü kapat

def plot_modulation_metrics(mod_metrics, class_names, save_path):
    """Plot modulation classification metrics"""
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = mod_metrics[metric]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Modulation Types')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Label Modulation Classification Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Belleği boşaltmak için figürü kapat


# ===============================================================
# DÜZELTME: JSON SERILEŞTIRME IÇIN YARDIMCI FONKSIYON
# ===============================================================
def json_converter(o):
    """NumPy ve diğer JSON-uyumsuz tipleri dönüştüren yardımcı fonksiyon."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# =========================
# MAIN TRAINING FUNCTION
# =========================
def main():
    print("Starting Multi-Task UHF Signal CNN Training")
    print(f"Device: {Config.DEVICE}")
    print("="*60)
    
    # Create results directory
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MultiTaskUHFDataset(Config.DATA_DIR, split='train', 
                                       train_ratio=Config.TRAIN_SPLIT, val_ratio=Config.VAL_SPLIT)
    val_dataset = MultiTaskUHFDataset(Config.DATA_DIR, split='val', 
                                     train_ratio=Config.TRAIN_SPLIT, val_ratio=Config.VAL_SPLIT)
    test_dataset = MultiTaskUHFDataset(Config.DATA_DIR, split='test', 
                                      train_ratio=Config.TRAIN_SPLIT, val_ratio=Config.VAL_SPLIT)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    model = MultiTaskUHFCNN(input_channels=4, dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=8, verbose=True, min_lr=1e-7)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Signal types: {Config.SIGNAL_TYPES}")
    print(f"Modulations: {Config.MODULATIONS}")
    
    # Training history
    history = {
        'total_loss': {'train': [], 'val': []},
        'signal_type_loss': {'train': [], 'val': []},
        'signal_count_loss': {'train': [], 'val': []},
        'modulation_loss': {'train': [], 'val': []},
        'frequency_loss': {'train': [], 'val': []},
        'signal_type_acc': {'train': [], 'val': []},
        'signal_count_acc': {'train': [], 'val': []},
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Training
        train_losses, train_type_acc, train_count_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validation
        val_results = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        val_losses, val_type_acc, val_count_acc = val_results[:3]
        
        # Update history
        for loss_name in ['total_loss', 'signal_type_loss', 'signal_count_loss', 
                         'modulation_loss', 'frequency_loss']:
            history[loss_name]['train'].append(train_losses[loss_name])
            history[loss_name]['val'].append(val_losses[loss_name])
        
        history['signal_type_acc']['train'].append(train_type_acc)
        history['signal_type_acc']['val'].append(val_type_acc)
        history['signal_count_acc']['train'].append(train_count_acc)
        history['signal_count_acc']['val'].append(val_count_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"Train - Total: {train_losses['total_loss']:.4f}, "
              f"Type: {train_type_acc:.2f}%, Count: {train_count_acc:.2f}%")
        print(f"Val   - Total: {val_losses['total_loss']:.4f}, "
              f"Type: {val_type_acc:.2f}%, Count: {val_count_acc:.2f}%")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total_loss'],
                'val_type_acc': val_type_acc,
                'val_count_acc': val_count_acc,
                'signal_type_encoder': train_dataset.signal_type_encoder,
                'mlb_modulations': train_dataset.mlb_modulations,
                'freq_normalization': {
                    'mean': train_dataset.freq_mean,
                    'std': train_dataset.freq_std
                }
            }, Config.MODEL_SAVE_PATH)
            print(f"Best model saved! (Val Loss: {val_losses['total_loss']:.4f})")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_losses['total_loss'])
        
        # Early stopping
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Plot progress every 15 epochs
        if (epoch + 1) % 15 == 0:
            plot_training_progress(history, Config.RESULTS_DIR)
    
    print(f"\nTraining completed! Best epoch: {best_epoch+1} with val loss: {best_val_loss:.4f}")
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)['model_state_dict'])
    
    test_results = validate_epoch(model, test_loader, criterion, Config.DEVICE)
    test_losses, test_type_acc, test_count_acc = test_results[:3]
    test_type_preds, test_type_targets = test_results[3:5]
    test_mod_preds, test_mod_targets, test_freq_errors = test_results[5:8]
    
    # Calculate additional metrics
    mod_metrics = calculate_modulation_metrics(test_mod_preds, test_mod_targets)
    freq_mae = np.mean(test_freq_errors) if len(test_freq_errors) > 0 else 0
    
    print(f"\nTest Results:")
    print(f"  Total Loss: {test_losses['total_loss']:.4f}")
    print(f"  Signal Type Accuracy: {test_type_acc:.2f}%")
    print(f"  Signal Count Accuracy: {test_count_acc:.2f}%")
    print(f"  Modulation F1 (mean): {mod_metrics.get('mean_f1', 0):.3f}")
    print(f"  Frequency MAE: {freq_mae:.4f} (normalized)")
    
    # Classification reports
    print(f"\nSignal Type Classification Report:")
    print(classification_report(test_type_targets, test_type_preds, 
                               target_names=Config.SIGNAL_TYPES))
    
    # Final plots
    plot_training_progress(history, Config.RESULTS_DIR)
    
    # Confusion matrix for signal types
    cm_path = os.path.join(Config.RESULTS_DIR, 'signal_type_confusion_matrix.png')
    plot_confusion_matrices(test_type_targets, test_type_preds, 
                           Config.SIGNAL_TYPES, cm_path, 
                           "Signal Type Classification Confusion Matrix")
    
    # Modulation metrics plot
    if len(mod_metrics) > 0:
        mod_path = os.path.join(Config.RESULTS_DIR, 'modulation_metrics.png')
        plot_modulation_metrics(mod_metrics, Config.MODULATIONS, mod_path)
    
    # Save results
    final_results = {
        'training_history': history,
        'test_results': {
            'total_loss': test_losses['total_loss'],
            'signal_type_accuracy': test_type_acc,
            'signal_count_accuracy': test_count_acc,
            'modulation_metrics': mod_metrics,
            'frequency_mae': freq_mae
        },
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss
        }
    }
    
    # ===============================================================
    # DÜZELTME: GÜNCELLENEN JSON KAYDETME SATIRI
    # ===============================================================
    with open(os.path.join(Config.RESULTS_DIR, 'training_results.json'), 'w') as f:
        # Hatalı lambda yerine yeni dönüştürücü fonksiyonumuzu kullanıyoruz
        json.dump(final_results, f, indent=2, default=json_converter)
    
    print(f"\nResults saved to: {Config.RESULTS_DIR}")
    print("Multi-task training pipeline completed successfully!")
    
    return model, test_results

# =========================
# INFERENCE AND DEMO FUNCTIONS
# =========================
def predict_sample(model, dataset, sample_idx, device):
    """Make predictions on a single sample and analyze results"""
    model.eval()
    
    # Get sample
    features, signal_type, signal_count, modulation, freq_tensor, freq_mask = dataset[sample_idx]
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(features)
        signal_type_logits, signal_count_logits, modulation_logits, frequency_preds = predictions
        
        # Convert to predictions
        pred_signal_type = torch.argmax(signal_type_logits, dim=1).item()
        pred_signal_count = torch.argmax(signal_count_logits, dim=1).item()
        
        # Modulation predictions
        mod_probs = torch.sigmoid(modulation_logits)
        pred_modulations = (mod_probs > 0.5)[0].cpu().numpy()
        
        # Frequency predictions (denormalized)
        pred_freqs = frequency_preds[0].cpu().numpy()
        pred_freqs_denorm = pred_freqs * dataset.freq_std + dataset.freq_mean
        
        # True values
        true_signal_type = signal_type.item()
        true_signal_count = signal_count.item()
        true_modulations = modulation.numpy()
        true_freqs = freq_tensor.numpy()
        true_freqs_denorm = true_freqs * dataset.freq_std + dataset.freq_mean
        true_freq_mask = freq_mask.numpy()
    
    # Format results
    result = {
        'signal_type': {
            'predicted': dataset.signal_type_encoder.classes_[pred_signal_type],
            'true': dataset.signal_type_encoder.classes_[true_signal_type],
            'correct': pred_signal_type == true_signal_type
        },
        'signal_count': {
            'predicted': pred_signal_count,
            'true': true_signal_count,
            'correct': pred_signal_count == true_signal_count
        },
        'modulations': {
            'predicted': [Config.MODULATIONS[i] for i, x in enumerate(pred_modulations) if x],
            'true': [Config.MODULATIONS[i] for i, x in enumerate(true_modulations) if x > 0.5],
            'correct': np.array_equal(pred_modulations, (true_modulations > 0.5))
        },
        'frequencies': []
    }
    
    # Frequency analysis
    for i in range(Config.MAX_SIGNALS):
        if true_freq_mask[i] > 0:  # Valid frequency
            pred_freq = pred_freqs_denorm[i]
            true_freq = true_freqs_denorm[i]
            error_pct = abs(pred_freq - true_freq) / (true_freq + 1e-8) if true_freq > 0 else float('inf')
            
            result['frequencies'].append({
                'predicted': pred_freq,
                'true': true_freq,
                'error_percent': error_pct * 100,
                'within_tolerance': error_pct <= Config.FREQ_TOLERANCE
            })
    
    return result

def demonstrate_predictions(model, dataset, device, num_samples=10):
    """Demonstrate model predictions on multiple samples"""
    print(f"\nDemonstrating predictions on {num_samples} samples:")
    print("=" * 80)
    
    correct_counts = {
        'signal_type': 0,
        'signal_count': 0,
        'modulation': 0,
        'frequency': 0
    }
    
    for i in range(min(num_samples, len(dataset))):
        result = predict_sample(model, dataset, i, device)
        
        print(f"\nSample {i+1}:")
        print(f"  Signal Type: {result['signal_type']['predicted']} "
              f"(True: {result['signal_type']['true']}) "
              f"{'✓' if result['signal_type']['correct'] else '✗'}")
        
        print(f"  Signal Count: {result['signal_count']['predicted']} "
              f"(True: {result['signal_count']['true']}) "
              f"{'✓' if result['signal_count']['correct'] else '✗'}")
        
        print(f"  Modulations: {result['modulations']['predicted']} "
              f"(True: {result['modulations']['true']}) "
              f"{'✓' if result['modulations']['correct'] else '✗'}")
        
        if result['frequencies']:
            print("  Frequencies:")
            for j, freq_result in enumerate(result['frequencies']):
                print(f"    {j+1}: {freq_result['predicted']:.0f} Hz "
                      f"(True: {freq_result['true']:.0f} Hz, "
                      f"Error: {freq_result['error_percent']:.1f}%) "
                      f"{'✓' if freq_result['within_tolerance'] else '✗'}")
        
        # Count correct predictions
        if result['signal_type']['correct']:
            correct_counts['signal_type'] += 1
        if result['signal_count']['correct']:
            correct_counts['signal_count'] += 1
        if result['modulations']['correct']:
            correct_counts['modulation'] += 1
        if result['frequencies'] and all(f['within_tolerance'] for f in result['frequencies']):
            correct_counts['frequency'] += 1
    
    print("\n" + "=" * 80)
    print("Summary:")
    for task, count in correct_counts.items:
        accuracy = count / num_samples * 100
        print(f"  {task.replace('_', ' ').title()}: {count}/{num_samples} ({accuracy:.1f}%)")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available! GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")
        
        # Run training
        model, test_results = main()
        
        # Load test dataset for demonstrations
        test_dataset = MultiTaskUHFDataset(Config.DATA_DIR, split='test',
                                          train_ratio=Config.TRAIN_SPLIT, 
                                          val_ratio=Config.VAL_SPLIT)
        
        # Demonstrate predictions
        demonstrate_predictions(model, test_dataset, Config.DEVICE, num_samples=10)
        
        print(f"\nAll results saved to: {Config.RESULTS_DIR}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup completed")