import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import logging
import time
import json
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"🚀 GPU Available:")
        logger.info(f"  Device: {gpu_name}")
        logger.info(f"  Memory: {gpu_memory:.1f} GB")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device, 64  # GPU, batch_size
    else:
        raise RuntimeError("CUDA GPU not available! This script requires GPU acceleration.")

class DataCleaner:
    """Utility class for cleaning and validating dataset"""
    
    def __init__(self, data_root):
        self.data_root = data_root
        self.duplicate_map = {}
        self.problematic_images = []
        
    def find_duplicates(self, image_paths):
        """Find duplicate images using perceptual hashing"""
        logger.info("🔍 Finding duplicate images...")
        
        image_hashes = {}
        duplicates = []
        
        for img_path in tqdm(image_paths, desc="Hashing images"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    self.problematic_images.append(img_path)
                    continue
                    
                # Create perceptual hash
                small = cv2.resize(image, (8, 8))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                img_hash = hashlib.md5(gray.tobytes()).hexdigest()
                
                if img_hash in image_hashes:
                    duplicates.append((img_path, image_hashes[img_hash]))
                else:
                    image_hashes[img_hash] = img_path
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                self.problematic_images.append(img_path)
                
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def remove_duplicates(self, image_paths, labels, duplicates):
        """Remove duplicate images, keeping the first occurrence"""
        logger.info("🧹 Removing duplicate images...")
        
        to_remove = set()
        for dup_path, original_path in duplicates:
            to_remove.add(dup_path)
            
        cleaned_paths = []
        cleaned_labels = []
        
        for path, label in zip(image_paths, labels):
            if path not in to_remove:
                cleaned_paths.append(path)
                cleaned_labels.append(label)
                
        logger.info(f"Removed {len(image_paths) - len(cleaned_paths)} duplicate images")
        logger.info(f"Dataset size: {len(image_paths)} → {len(cleaned_paths)}")
        
        return cleaned_paths, cleaned_labels
    
    def validate_images(self, image_paths):
        """Validate image quality and fix brightness issues"""
        logger.info("🔍 Validating image quality...")
        
        valid_paths = []
        brightness_issues = []
        
        for img_path in tqdm(image_paths, desc="Validating images"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Cannot read image: {img_path}")
                    continue
                    
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                if brightness > 235:
                    brightness_issues.append((img_path, brightness))
                    
                valid_paths.append(img_path)
                
            except Exception as e:
                logger.warning(f"Error validating {img_path}: {e}")
                
        if brightness_issues:
            logger.warning(f"Found {len(brightness_issues)} images with brightness issues")
                
        return valid_paths, brightness_issues

class EmotionEyeDataset(Dataset):
    """Dataset class with error handling and brightness correction"""
    
    def __init__(self, image_paths, labels, transform=None, brightness_issues=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.brightness_issues = set(brightness_issues) if brightness_issues else set()
        
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        logger.info(f"Dataset initialized with {len(image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if img_path in self.brightness_issues:
                image = self._correct_brightness(image)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            if self.transform:
                black_image = np.zeros((75, 75, 3), dtype=np.uint8)
                transformed = self.transform(image=black_image)
                return transformed['image'], self.labels[idx]
            else:
                return torch.zeros(3, 224, 224), self.labels[idx]
    
    def _correct_brightness(self, image):
        """Correct overly bright images"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l * 0.8, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return corrected

def get_transforms(split, image_size=224):
    """Get transforms optimized for GPU training"""
    
    if split == 'train':
        # Aggressive augmentation for GPU
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def load_and_clean_dataset_split(split_dir, cleaner=None):
    """Load dataset split with cleaning and validation"""
    logger.info(f"📂 Loading {split_dir}...")
    
    image_paths = []
    labels = []
    
    if not os.path.exists(split_dir):
        logger.error(f"Directory not found: {split_dir}")
        return [], [], []
    
    for emotion in sorted(os.listdir(split_dir)):
        emotion_dir = os.path.join(split_dir, emotion)
        if os.path.isdir(emotion_dir):
            emotion_count = 0
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(emotion)
                    emotion_count += 1
            logger.info(f"  {emotion}: {emotion_count} images")
    
    logger.info(f"Total images loaded: {len(image_paths)}")
    
    if cleaner:
        duplicates = cleaner.find_duplicates(image_paths)
        if duplicates:
            image_paths, labels = cleaner.remove_duplicates(image_paths, labels, duplicates)
        
        image_paths, brightness_issues = cleaner.validate_images(image_paths)
        
        cleaned_labels = []
        for path in image_paths:
            emotion = os.path.basename(os.path.dirname(path))
            cleaned_labels.append(emotion)
        labels = cleaned_labels
        
        return image_paths, labels, brightness_issues
    
    return image_paths, labels, []

def prepare_datasets(data_root='model_eye', clean_data=True):
    """Prepare datasets with cleaning and validation"""
    logger.info("🚀 Starting dataset preparation...")
    
    cleaner = DataCleaner(data_root) if clean_data else None
    
    train_imgs, train_labels, train_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'train'), cleaner
    )
    val_imgs, val_labels, val_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'val'), cleaner
    )
    test_imgs, test_labels, test_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'test'), cleaner
    )
    
    logger.info("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    all_labels = train_labels + val_labels + test_labels
    label_encoder.fit(all_labels)
    
    train_labels_enc = label_encoder.transform(train_labels)
    val_labels_enc = label_encoder.transform(val_labels)
    test_labels_enc = label_encoder.transform(test_labels)
    
    logger.info(f"Label classes: {list(label_encoder.classes_)}")
    
    train_set = EmotionEyeDataset(
        train_imgs, train_labels_enc, 
        get_transforms('train'),
        brightness_issues=[item[0] for item in train_brightness]
    )
    val_set = EmotionEyeDataset(
        val_imgs, val_labels_enc, 
        get_transforms('val'),
        brightness_issues=[item[0] for item in val_brightness]
    )
    test_set = EmotionEyeDataset(
        test_imgs, test_labels_enc, 
        get_transforms('test'),
        brightness_issues=[item[0] for item in test_brightness]
    )
    
    logger.info("✅ Dataset preparation completed!")
    logger.info(f"📊 Final dataset sizes:")
    logger.info(f"  Train: {len(train_set)} samples")
    logger.info(f"  Validation: {len(val_set)} samples") 
    logger.info(f"  Test: {len(test_set)} samples")
    
    return train_set, val_set, test_set, label_encoder

def calculate_class_weights(labels, label_encoder):
    """Calculate class weights for handling imbalance"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info("⚖️ Class weights calculated:")
    for i, (class_name, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
        logger.info(f"  {class_name}: {weight:.3f}")
    
    return class_weights_tensor

class ResNet18EmotionModel(nn.Module):
    """ResNet18-based emotion recognition model for GPU training"""
    
    def __init__(self, num_classes=8, pretrained=True):
        super(ResNet18EmotionModel, self).__init__()
        
        # Use ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        
        # Replace final classifier with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch with mixed precision"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training function with GPU acceleration"""
    # Check GPU availability
    device, optimal_batch_size = check_gpu()
    
    # Configuration for GPU training
    CONFIG = {
        'data_root': 'model_eye',
        'batch_size': optimal_batch_size,
        'num_epochs': 40,
        'learning_rate': 2e-3,
        'weight_decay': 1e-4,
        'patience': 10,
        'num_workers': 4,
        'device': device,
        'pin_memory': True
    }
    
    logger.info(f"🚀 Starting ResNet18 Emotion Recognition Training (GPU)")
    logger.info(f"Configuration: {CONFIG}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    # Prepare datasets
    train_set, val_set, test_set, label_encoder = prepare_datasets(
        data_root=CONFIG['data_root'], 
        clean_data=True
    )
    
    # Calculate class weights
    train_labels = [train_set.labels[i] for i in range(len(train_set))]
    class_weights = calculate_class_weights(train_labels, label_encoder)
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    
    # Initialize ResNet18 model
    model = ResNet18EmotionModel(
        num_classes=len(label_encoder.classes_), 
        pretrained=True
    )
    model = model.to(device)
    
    logger.info(f"ResNet18 model initialized on {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    logger.info("🔥 Using Mixed Precision Training (AMP)")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    logger.info("🏋️ Starting GPU-optimized training...")
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logger.info(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # GPU memory usage
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"  GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Final evaluation on test set
    logger.info("🧪 Evaluating on test set...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device, 0
    )
    
    logger.info(f"Test Results:")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    report = classification_report(
        test_targets, test_preds, 
        target_names=class_names, 
        digits=4
    )
    logger.info(f"Classification Report:\n{report}")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_targets, test_preds, class_names)
    
    # Save model
    model_filename = 'resnet18_emotion_gpu.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'config': CONFIG,
        'test_accuracy': test_acc,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }, model_filename)
    
    logger.info(f"✅ GPU Training completed successfully!")
    logger.info(f"📁 Model saved as '{model_filename}'")
    logger.info(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    logger.info("🧹 GPU cache cleared")

if __name__ == "__main__":
    main()