import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import logging
from collections import defaultdict, Counter
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                # Resize to 8x8 and convert to grayscale for consistent hashing
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
        
        # Create set of images to remove (keep first occurrence)
        to_remove = set()
        for dup_path, original_path in duplicates:
            to_remove.add(dup_path)
            
        # Filter out duplicates
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
                    
                # Check brightness
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                if brightness > 235:
                    brightness_issues.append((img_path, brightness))
                    # Still include but flag for special handling
                    
                valid_paths.append(img_path)
                
            except Exception as e:
                logger.warning(f"Error validating {img_path}: {e}")
                
        if brightness_issues:
            logger.warning(f"Found {len(brightness_issues)} images with brightness issues")
            for path, brightness in brightness_issues[:5]:
                logger.warning(f"  {path}: brightness={brightness:.1f}")
                
        return valid_paths, brightness_issues

class ImprovedEmotionEyeDataset(Dataset):
    """Enhanced dataset class with better error handling and validation"""
    
    def __init__(self, image_paths, labels, transform=None, brightness_issues=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.brightness_issues = set(brightness_issues) if brightness_issues else set()
        
        # Validate inputs
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
            
            # Handle brightness issues
            if img_path in self.brightness_issues:
                # Apply brightness correction
                image = self._correct_brightness(image)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = np.zeros((75, 75, 3), dtype=np.uint8)
                transformed = self.transform(image=black_image)
                return transformed['image'], self.labels[idx]
            else:
                return torch.zeros(3, 224, 224), self.labels[idx]
    
    def _correct_brightness(self, image):
        """Correct overly bright images"""
        # Convert to LAB color space for better brightness control
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Reduce brightness in L channel
        l = np.clip(l * 0.8, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return corrected

def get_enhanced_transforms(split, image_size=224):
    """Enhanced transforms with better augmentation for small images"""
    
    if split == 'train':
        return A.Compose([
            # Resize with better interpolation for small images
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.4
            ),
            
            # Photometric augmentations - enhanced for brightness issues
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
            
            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Normalization
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

def load_and_clean_dataset_split(split_dir, cleaner=None):
    """Load dataset split with cleaning and validation"""
    logger.info(f"📂 Loading {split_dir}...")
    
    image_paths = []
    labels = []
    
    if not os.path.exists(split_dir):
        logger.error(f"Directory not found: {split_dir}")
        return [], []
    
    # Load all images
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
    
    # Clean duplicates if cleaner provided
    if cleaner:
        duplicates = cleaner.find_duplicates(image_paths)
        if duplicates:
            image_paths, labels = cleaner.remove_duplicates(image_paths, labels, duplicates)
        
        # Validate images
        image_paths, brightness_issues = cleaner.validate_images(image_paths)
        
        # Update labels to match cleaned paths
        # Re-extract labels from cleaned paths
        cleaned_labels = []
        for path in image_paths:
            emotion = os.path.basename(os.path.dirname(path))
            cleaned_labels.append(emotion)
        labels = cleaned_labels
        
        return image_paths, labels, brightness_issues
    
    return image_paths, labels, []

def analyze_dataset_balance(labels, split_name):
    """Analyze and report class distribution"""
    label_counts = Counter(labels)
    total = len(labels)
    
    logger.info(f"📊 {split_name} class distribution:")
    for emotion, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        logger.info(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 2:
        logger.warning(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    return label_counts, imbalance_ratio

def save_cleaning_report(cleaner, output_dir):
    """Save data cleaning report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'problematic_images': cleaner.problematic_images,
        'total_duplicates_found': len(cleaner.duplicate_map),
        'cleaning_timestamp': str(pd.Timestamp.now())
    }
    
    with open(os.path.join(output_dir, 'cleaning_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Cleaning report saved to {output_dir}/cleaning_report.json")

def prepare_enhanced_datasets(data_root='model_eye', clean_data=True, save_report=True):
    """Enhanced dataset preparation with cleaning and validation"""
    logger.info("🚀 Starting enhanced dataset preparation...")
    
    # Initialize cleaner
    cleaner = DataCleaner(data_root) if clean_data else None
    
    # Load datasets
    train_imgs, train_labels, train_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'train'), cleaner
    )
    val_imgs, val_labels, val_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'val'), cleaner
    )
    test_imgs, test_labels, test_brightness = load_and_clean_dataset_split(
        os.path.join(data_root, 'test'), cleaner
    )
    
    # Analyze class distributions
    train_dist, train_imbalance = analyze_dataset_balance(train_labels, 'Train')
    val_dist, val_imbalance = analyze_dataset_balance(val_labels, 'Validation')
    test_dist, test_imbalance = analyze_dataset_balance(test_labels, 'Test')
    
    # Encode labels
    logger.info("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    all_labels = train_labels + val_labels + test_labels
    label_encoder.fit(all_labels)
    
    train_labels_enc = label_encoder.transform(train_labels)
    val_labels_enc = label_encoder.transform(val_labels)
    test_labels_enc = label_encoder.transform(test_labels)
    
    logger.info(f"Label classes: {list(label_encoder.classes_)}")
    
    # Create datasets with enhanced transforms
    logger.info("🔧 Creating datasets with enhanced transforms...")
    
    train_set = ImprovedEmotionEyeDataset(
        train_imgs, train_labels_enc, 
        get_enhanced_transforms('train'),
        brightness_issues=[item[0] for item in train_brightness]
    )
    val_set = ImprovedEmotionEyeDataset(
        val_imgs, val_labels_enc, 
        get_enhanced_transforms('val'),
        brightness_issues=[item[0] for item in val_brightness]
    )
    test_set = ImprovedEmotionEyeDataset(
        test_imgs, test_labels_enc, 
        get_enhanced_transforms('test'),
        brightness_issues=[item[0] for item in test_brightness]
    )
    
    # Save cleaning report
    if save_report and cleaner:
        save_cleaning_report(cleaner, 'reports')
    
    # Final summary
    logger.info("✅ Dataset preparation completed!")
    logger.info(f"📊 Final dataset sizes:")
    logger.info(f"  Train: {len(train_set)} samples")
    logger.info(f"  Validation: {len(val_set)} samples") 
    logger.info(f"  Test: {len(test_set)} samples")
    logger.info(f"  Total: {len(train_set) + len(val_set) + len(test_set)} samples")
    
    return train_set, val_set, test_set, label_encoder

def validate_preprocessing_pipeline(train_set, val_set, test_set, label_encoder):
    """Validate the preprocessing pipeline"""
    logger.info("🔍 Validating preprocessing pipeline...")
    
    # Test data loading
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    
    try:
        batch = next(iter(train_loader))
        images, labels = batch
        
        logger.info("✅ Preprocessing validation:")
        logger.info(f"  Batch shape: {images.shape}")
        logger.info(f"  Image dtype: {images.dtype}")
        logger.info(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Unique labels: {torch.unique(labels).tolist()}")
        
        # Check for issues
        if torch.isnan(images).any():
            logger.error("❌ NaN values found in images!")
        if torch.isinf(images).any():
            logger.error("❌ Infinite values found in images!")
            
        # Check normalization
        mean_vals = images.mean(dim=[0, 2, 3])
        std_vals = images.std(dim=[0, 2, 3])
        logger.info(f"  Actual mean: {mean_vals.tolist()}")
        logger.info(f"  Actual std: {std_vals.tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing validation failed: {e}")
        return False

# Utility function for calculating class weights
def calculate_class_weights(labels, label_encoder):
    """Calculate class weights for handling any remaining imbalance"""
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    # Convert to torch tensor
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info("⚖️ Class weights calculated:")
    for i, (class_name, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
        logger.info(f"  {class_name}: {weight:.3f}")
    
    return class_weights_tensor

if __name__ == "__main__":
    # Example usage
    train_set, val_set, test_set, label_encoder = prepare_enhanced_datasets(
        data_root='model_eye',
        clean_data=True,
        save_report=True
    )
    
    # Validate pipeline
    validate_preprocessing_pipeline(train_set, val_set, test_set, label_encoder)
    
    # Calculate class weights (optional)
    train_labels = [train_set.labels[i] for i in range(len(train_set))]
    class_weights = calculate_class_weights(train_labels, label_encoder)
    
    logger.info("🎉 All preprocessing completed successfully!")
