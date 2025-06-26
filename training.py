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
import joblib
from tqdm import tqdm
import warnings
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generate comprehensive PDF reports for training results"""
    
    def __init__(self, filename="emotion_training_report.pdf"):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=A4)
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
    def add_title(self, title):
        """Add main title to report"""
        self.story.append(Paragraph(title, self.title_style))
        self.story.append(Spacer(1, 20))
        
    def add_section(self, title, content):
        """Add a section with title and content"""
        self.story.append(Paragraph(title, self.heading_style))
        if isinstance(content, str):
            self.story.append(Paragraph(content, self.styles['Normal']))
        elif isinstance(content, list):
            for item in content:
                self.story.append(Paragraph(f"• {item}", self.styles['Normal']))
        self.story.append(Spacer(1, 12))
        
    def add_table(self, data, title=None):
        """Add a table to the report"""
        if title:
            self.story.append(Paragraph(title, self.heading_style))
            
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 12))
        
    def add_image(self, image_path, title=None, width=6*inch):
        """Add an image to the report"""
        if title:
            self.story.append(Paragraph(title, self.heading_style))
            
        if os.path.exists(image_path):
            img = Image(image_path, width=width, height=width*0.6)
            self.story.append(img)
            self.story.append(Spacer(1, 12))
        
    def save_report(self):
        """Save the PDF report"""
        self.doc.build(self.story)
        logger.info(f"📄 PDF report saved as '{self.filename}'")

def check_and_optimize_device():
    """Check available device and optimize accordingly"""
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
        
        return device, True, 64  # GPU, use_amp, batch_size
    else:
        logger.info("💻 Using CPU - Optimizing for CPU performance")
        
        # CPU optimizations
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        
        logger.info(f"  CPU Cores: {num_cores}")
        logger.info(f"  PyTorch Threads: {torch.get_num_threads()}")
        
        return torch.device('cpu'), False, 32  # CPU, no_amp, smaller_batch_size

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

class ImprovedEmotionEyeDataset(Dataset):
    """Enhanced dataset class with error handling and brightness correction"""
    
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
            
            # FIX: Ensure label is LongTensor for CrossEntropyLoss
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            if self.transform:
                black_image = np.zeros((75, 75, 3), dtype=np.uint8)
                transformed = self.transform(image=black_image)
                return transformed['image'], torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                return torch.zeros(3, 224, 224), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def _correct_brightness(self, image):
        """Correct overly bright images"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l * 0.8, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return corrected

def get_transforms(split, image_size=224, device_type='cpu'):
    """Get transforms optimized for device type"""
    
    if split == 'train':
        if device_type == 'cuda':
            # More aggressive augmentation for GPU
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
            # Lighter augmentation for CPU
            return A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.GaussNoise(var_limit=(5, 25), p=0.1),
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

def prepare_datasets(data_root='model_eye', clean_data=True, device_type='cpu'):
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
    
    train_set = ImprovedEmotionEyeDataset(
        train_imgs, train_labels_enc, 
        get_transforms('train', device_type=device_type),
        brightness_issues=[item[0] for item in train_brightness]
    )
    val_set = ImprovedEmotionEyeDataset(
        val_imgs, val_labels_enc, 
        get_transforms('val', device_type=device_type),
        brightness_issues=[item[0] for item in val_brightness]
    )
    test_set = ImprovedEmotionEyeDataset(
        test_imgs, test_labels_enc, 
        get_transforms('test', device_type=device_type),
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

class MultiModelEmotionCNN:
    """Factory class for creating different CNN architectures"""
    
    @staticmethod
    def create_resnet18(num_classes=8, pretrained=True):
        """Create ResNet18 model"""
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model
    
    @staticmethod
    def create_efficientnet_b0(num_classes=8, pretrained=True):
        """Create EfficientNet-B0 model"""
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model
    
    @staticmethod
    def create_densenet121(num_classes=8, pretrained=True):
        """Create DenseNet121 model"""
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model

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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_amp=False, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler and device.type == 'cuda':
            # Mixed precision training for GPU
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
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

def validate_epoch(model, val_loader, criterion, device, epoch, use_amp=False):
    """Validate for one epoch with optional mixed precision"""
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
            
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
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

def train_single_model(model_name, model, train_loader, val_loader, test_loader, 
                      criterion, device, config, use_amp, scaler, label_encoder):
    """Train a single model and return results"""
    logger.info(f"🚀 Training {model_name}...")
    
    # Optimizer and scheduler for this model
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Early stopping for this model
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp, scaler
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, epoch, use_amp
        )
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logger.info(f"📉 {model_name} Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log epoch results
        if epoch % 5 == 0 or epoch == config['num_epochs'] - 1:
            logger.info(f"{model_name} - Epoch {epoch+1}/{config['num_epochs']}")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f"{model_name} - Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, device, 0, use_amp
    )
    
    logger.info(f"{model_name} Training completed in {training_time/60:.2f} minutes")
    logger.info(f"{model_name} Test Accuracy: {test_acc:.2f}%")
    
    # Calculate additional metrics
    class_names = label_encoder.classes_
    report_text = classification_report(
        test_targets, test_preds, 
        target_names=class_names, 
        digits=4,
        output_dict=True
    )
    
    return {
        'model': model,
        'model_name': model_name,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_preds': test_preds,
        'test_targets': test_targets,
        'classification_report': report_text,
        'epochs_completed': len(train_losses)
    }

def plot_model_comparison(results_dict, save_prefix="model_comparison"):
    """Plot comprehensive comparison of all models"""
    
    # 1. Training History Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    colors = ['blue', 'red', 'green']
    
    # Training Loss Comparison
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax1.plot(results['train_losses'], label=f'{model_name} Train', 
                color=colors[i], linestyle='-', alpha=0.8)
        ax1.plot(results['val_losses'], label=f'{model_name} Val', 
                color=colors[i], linestyle='--', alpha=0.8)
    
    ax1.set_title('Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training Accuracy Comparison
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax2.plot(results['train_accs'], label=f'{model_name} Train', 
                color=colors[i], linestyle='-', alpha=0.8)
        ax2.plot(results['val_accs'], label=f'{model_name} Val', 
                color=colors[i], linestyle='--', alpha=0.8)
    
    ax2.set_title('Training and Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test Accuracy Bar Chart
    model_names = list(results_dict.keys())
    test_accs = [results_dict[name]['test_acc'] for name in model_names]
    
    bars = ax3.bar(model_names, test_accs, color=colors[:len(model_names)], alpha=0.7)
    ax3.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Training Time Comparison
    training_times = [results_dict[name]['training_time']/60 for name in model_names]
    
    bars = ax4.bar(model_names, training_times, color=colors[:len(model_names)], alpha=0.7)
    ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Training Time (minutes)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        cm = confusion_matrix(results['test_targets'], results['test_preds'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=label_encoder.classes_, 
                   yticklabels=label_encoder.classes_)
        axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Per-Class Performance Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    class_names = label_encoder.classes_
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric_idx, metric in enumerate(metrics):
        ax = [ax1, ax2, ax3][metric_idx]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            values = []
            for class_name in class_names:
                if class_name in results['classification_report']:
                    values.append(results['classification_report'][class_name][metric])
                else:
                    values.append(0)
            
            ax.bar(x + i*width, values, width, label=model_name, 
                  color=colors[i], alpha=0.7)
        
        ax.set_title(f'Per-Class {metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emotion Classes')
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_table(results_dict):
    """Create a comprehensive comparison table"""
    
    table_data = [['Model', 'Test Accuracy (%)', 'Training Time (min)', 
                   'Epochs', 'Best Val Acc (%)', 'Parameters']]
    
    for model_name, results in results_dict.items():
        # Count parameters
        total_params = sum(p.numel() for p in results['model'].parameters())
        trainable_params = sum(p.numel() for p in results['model'].parameters() if p.requires_grad)
        
        table_data.append([
            model_name,
            f"{results['test_acc']:.2f}",
            f"{results['training_time']/60:.1f}",
            str(results['epochs_completed']),
            f"{max(results['val_accs']):.2f}",
            f"{trainable_params:,}"
        ])
    
    return table_data

def main():
    """Main training function that compares multiple models"""
    # Check and optimize device
    device, use_amp, optimal_batch_size = check_and_optimize_device()
    device_type = device.type
    
    # Configuration
    CONFIG = {
        'data_root': 'model_eye',
        'batch_size': optimal_batch_size,
        'num_epochs': 30 if device_type == 'cuda' else 20,
        'learning_rate': 2e-3 if device_type == 'cuda' else 1e-3,
        'weight_decay': 1e-4,
        'patience': 8,
        'num_workers': 4 if device_type == 'cuda' else 2,
        'device': device,
        'use_amp': use_amp,
        'pin_memory': device_type == 'cuda'
    }
    
    logger.info(f"🚀 Starting Multi-Model Emotion Recognition Training ({device_type.upper()})")
    logger.info(f"Configuration: {CONFIG}")
    
    # Initialize PDF report
    report = PDFReportGenerator("multi_model_emotion_comparison_report.pdf")
    report.add_title("Multi-Model Emotion Recognition Comparison Report")
    report.add_section("Training Configuration", [
        f"Device: {device_type.upper()}",
        f"Models: ResNet18, EfficientNet-B0, DenseNet121",
        f"Batch Size: {CONFIG['batch_size']}",
        f"Learning Rate: {CONFIG['learning_rate']}",
        f"Number of Epochs: {CONFIG['num_epochs']}",
        f"Mixed Precision: {use_amp}",
        f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ])
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if device_type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Prepare datasets
    train_set, val_set, test_set, label_encoder = prepare_datasets(
        data_root=CONFIG['data_root'], 
        clean_data=True,
        device_type=device_type
    )
    
    # Add dataset info to report
    report.add_section("Dataset Information", [
        f"Training samples: {len(train_set)}",
        f"Validation samples: {len(val_set)}",
        f"Test samples: {len(test_set)}",
        f"Number of classes: {len(label_encoder.classes_)}",
        f"Classes: {', '.join(label_encoder.classes_)}"
    ])
    
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
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    
    # Define models to train
    models_to_train = {
        'ResNet18': MultiModelEmotionCNN.create_resnet18(len(label_encoder.classes_)),
        'EfficientNet-B0': MultiModelEmotionCNN.create_efficientnet_b0(len(label_encoder.classes_)),
        'DenseNet121': MultiModelEmotionCNN.create_densenet121(len(label_encoder.classes_))
    }
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    # Mixed precision scaler for GPU
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("🔥 Using Mixed Precision Training (AMP)")
    
    # Train all models
    results_dict = {}
    total_start_time = time.time()
    
    for model_name, model in models_to_train.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")
        
        # Move model to device
        model = model.to(device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{model_name} - Total parameters: {total_params:,}")
        logger.info(f"{model_name} - Trainable parameters: {trainable_params:,}")
        
        # Train the model
        results = train_single_model(
            model_name, model, train_loader, val_loader, test_loader,
            criterion, device, CONFIG, use_amp, scaler, label_encoder
        )
        
        results_dict[model_name] = results
        
        # Save individual model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'results': results,
            'label_encoder': label_encoder
        }, f'{model_name.lower().replace("-", "_")}_emotion_model.pth')
        
        logger.info(f"✅ {model_name} training completed and saved!")
        
        # Clear GPU cache
        if device_type == 'cuda':
            torch.cuda.empty_cache()
    
    total_training_time = time.time() - total_start_time
    
    # Generate comprehensive comparison plots
    logger.info("\n📊 Generating comparison visualizations...")
    plot_model_comparison(results_dict)
    
    # Create comparison table
    comparison_table = create_comparison_table(results_dict)
    
    # Print final results
    logger.info(f"\n{'='*80}")
    logger.info("🏆 FINAL RESULTS COMPARISON")
    logger.info(f"{'='*80}")
    
    for model_name, results in results_dict.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Test Accuracy: {results['test_acc']:.2f}%")
        logger.info(f"  Training Time: {results['training_time']/60:.2f} minutes")
        logger.info(f"  Best Val Accuracy: {max(results['val_accs']):.2f}%")
        logger.info(f"  Epochs Completed: {results['epochs_completed']}")
    
    logger.info(f"\nTotal Training Time: {total_training_time/60:.2f} minutes")
    
    # Find best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_acc'])
    logger.info(f"\n🥇 Best Model: {best_model[0]} with {best_model[1]['test_acc']:.2f}% test accuracy")
    
    # Add results to PDF report
    report.add_section("Training Results Summary", [
        f"Total Training Time: {total_training_time/60:.2f} minutes",
        f"Best Model: {best_model[0]} ({best_model[1]['test_acc']:.2f}% accuracy)",
        f"All models completed training successfully"
    ])
    
    # Add comparison table to report
    report.add_table(comparison_table, "Model Comparison Table")
    
    # Add plots to report
    report.add_image("model_comparison_overview.png", "Training Overview Comparison")
    report.add_image("model_comparison_confusion_matrices.png", "Confusion Matrices Comparison")
    report.add_image("model_comparison_per_class_metrics.png", "Per-Class Metrics Comparison")
    
    # Add detailed results for each model
    for model_name, results in results_dict.items():
        report.add_section(f"{model_name} Detailed Results", [
            f"Test Accuracy: {results['test_acc']:.2f}%",
            f"Test Loss: {results['test_loss']:.4f}",
            f"Training Time: {results['training_time']/60:.2f} minutes",
            f"Best Validation Accuracy: {max(results['val_accs']):.2f}%",
            f"Epochs Completed: {results['epochs_completed']}"
        ])
    
    # Generate PDF report
    report.save_report()
    
    # Save combined results
    combined_results = {
        'config': CONFIG,
        'results': results_dict,
        'label_encoder': label_encoder,
        'comparison_table': comparison_table,
        'best_model': best_model[0],
        'total_training_time': total_training_time
    }
    
    joblib.dump(combined_results, 'multi_model_comparison_results.pkl')
    
    logger.info(f"\n✅ Multi-model training completed successfully!")
    logger.info(f"📁 Individual models saved as: resnet18_emotion_model.pth, efficientnet_b0_emotion_model.pth, densenet121_emotion_model.pth")
    logger.info(f"📊 Combined results saved as: multi_model_comparison_results.pkl")
    logger.info(f"📄 PDF report generated: multi_model_emotion_comparison_report.pdf")
    logger.info(f"🎯 Best performing model: {best_model[0]} with {best_model[1]['test_acc']:.2f}% accuracy")
    
    # Clear GPU cache
    if device_type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("🧹 GPU cache cleared")

if __name__ == "__main__":
    main()
