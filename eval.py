import os
import cv2
import numpy as np
import pandas as pd
import logging
import time
import json
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.serialization
import torchvision.models as tv_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import scipy.stats as stats
from itertools import combinations

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedEmotionEyeDataset(Dataset):
    """Enhanced dataset class with error handling"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
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
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = np.zeros((224, 224, 3), dtype=np.uint8)
                transformed = self.transform(image=black_image)
                return transformed['image'], torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                return torch.zeros(3, 224, 224), torch.tensor(self.labels[idx], dtype=torch.long)

def get_transforms(split, image_size=224):
    """Get transforms for evaluation"""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_dataset_from_directory(data_root):
    """Load dataset and create label encoder from directory structure"""
    logger.info(f"📂 Loading dataset from {data_root}...")
    
    # Check if organized structure exists
    test_dir = os.path.join(data_root, 'test')
    
    if not os.path.exists(test_dir):
        logger.info("No test directory found. Looking for emotion folders...")
        
        # Look for emotion folders directly in data_root
        emotion_folders = []
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
                # Check if this folder contains images
                has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                               for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
                if has_images:
                    emotion_folders.append(item)
        
        if not emotion_folders:
            logger.error("❌ No valid dataset structure found!")
            return None, None, None
        
        # Create test split from emotion folders (use last 20% of each emotion)
        logger.info(f"Creating test split from emotion folders: {emotion_folders}")
        
        test_imgs = []
        test_labels = []
        
        for emotion in sorted(emotion_folders):
            emotion_path = os.path.join(data_root, emotion)
            images = [f for f in os.listdir(emotion_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Take last 20% for testing
            test_count = max(1, len(images) // 5)
            test_images = images[-test_count:]
            
            for img in test_images:
                img_path = os.path.join(emotion_path, img)
                test_imgs.append(img_path)
                test_labels.append(emotion)
            
            logger.info(f"  {emotion}: {len(test_images)} test images")
    
    else:
        # Load from organized test directory
        logger.info("Loading from organized test directory...")
        test_imgs = []
        test_labels = []
        
        for emotion in sorted(os.listdir(test_dir)):
            emotion_dir = os.path.join(test_dir, emotion)
            if os.path.isdir(emotion_dir):
                emotion_count = 0
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        test_imgs.append(img_path)
                        test_labels.append(emotion)
                        emotion_count += 1
                logger.info(f"  {emotion}: {emotion_count} images")
    
    if not test_imgs:
        logger.error("❌ No test images found!")
        return None, None, None
    
    # Create label encoder
    logger.info("🏷️ Creating label encoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(test_labels)
    test_labels_enc = label_encoder.transform(test_labels)
    
    logger.info(f"Label classes: {list(label_encoder.classes_)}")
    logger.info(f"Total test images: {len(test_imgs)}")
    
    # Create dataset
    test_set = ImprovedEmotionEyeDataset(
        test_imgs, test_labels_enc, 
        get_transforms('test')
    )
    
    return test_set, label_encoder, test_imgs

def extract_label_encoder_from_model(model_path):
    """Extract label encoder information from saved model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if label encoder is saved in checkpoint
        if 'label_encoder' in checkpoint:
            return checkpoint['label_encoder']
        
        # If not, try to infer from results
        if 'results' in checkpoint and 'checkpoint' in checkpoint['results']:
            results_checkpoint = checkpoint['results']['checkpoint']
            if 'label_encoder' in results_checkpoint:
                return results_checkpoint['label_encoder']
        
        return None
    except Exception as e:
        logger.warning(f"Could not extract label encoder from {model_path}: {e}")
        return None

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model_paths, test_loader, label_encoder, device):
        self.model_paths = model_paths
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.device = device
        self.models = {}
        self.results = {}
        
    def load_models(self):
        logger.info("📥 Loading trained models...")

        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                continue

            try:
            # Allow unsafe deserialization (only if you trust the checkpoint)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Create model architecture
                num_classes = len(self.label_encoder.classes_)
                if 'resnet' in model_name.lower():
                    model = self._create_resnet18(num_classes)
                elif 'efficientnet' in model_name.lower():
                    model = self._create_efficientnet_b0(num_classes)
                elif 'densenet' in model_name.lower():
                    model = self._create_densenet121(num_classes)
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue

                # Load model state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()

                self.models[model_name] = {
                    'model': model,
                    'checkpoint': checkpoint
                }

                logger.info(f"✅ Loaded {model_name}")

            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {e}")



    
    def _create_resnet18(self, num_classes):
        """Create ResNet18 model"""
        model = models.resnet18(pretrained=False)
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
    
    def _create_efficientnet_b0(self, num_classes):
        """Create EfficientNet-B0 model"""
        model = models.efficientnet_b0(pretrained=False)
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
    
    def _create_densenet121(self, num_classes):
        """Create DenseNet121 model"""
        model = models.densenet121(pretrained=False)
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
    
    def evaluate_model(self, model_name, model_info):
        """Comprehensive evaluation of a single model"""
        logger.info(f"🔍 Evaluating {model_name}...")
        
        model = model_info['model']
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc=f"Evaluating {model_name}")):
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                probs = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, labels=range(len(self.label_encoder.classes_))
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_targets, all_preds,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # ROC curves and AUC (for multiclass)
        roc_data = self._calculate_multiclass_roc(all_targets, all_probs)
        
        # Inference statistics
        avg_inference_time = np.mean(inference_times)
        fps = len(all_targets) / sum(inference_times)
        
        # Model complexity
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Store results
        self.results[model_name] = {
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'class_report': class_report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'checkpoint': model_info['checkpoint']
        }
        
        logger.info(f"✅ {model_name} evaluation completed")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Avg Inference Time: {avg_inference_time*1000:.2f}ms")
        logger.info(f"   FPS: {fps:.2f}")
    
    def _calculate_multiclass_roc(self, y_true, y_probs):
        """Calculate ROC curves for multiclass classification"""
        n_classes = len(self.label_encoder.classes_)
        y_true_bin = np.eye(n_classes)[y_true]
        y_probs = np.array(y_probs)
        
        roc_data = {}
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[self.label_encoder.classes_[i]] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        
        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        roc_data['micro'] = {
            'fpr': fpr_micro,
            'tpr': tpr_micro,
            'auc': roc_auc_micro
        }
        
        return roc_data
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("🚀 Starting comprehensive model evaluation...")
        
        self.load_models()
        
        if not self.models:
            logger.error("❌ No models loaded successfully!")
            return
        
        # Evaluate each model
        for model_name, model_info in self.models.items():
            self.evaluate_model(model_name, model_info)
        
        logger.info("✅ All models evaluated successfully!")

class EvaluationVisualizer:
    """Create comprehensive visualizations for model evaluation"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.results = evaluator.results
        self.label_encoder = evaluator.label_encoder
        
    def plot_accuracy_comparison(self, save_path="accuracy_comparison.png"):
        """Plot accuracy comparison across models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Overall accuracy
        bars = ax1.bar(models, accuracies, color=colors[:len(models)], alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Per-class accuracy heatmap
        class_names = self.label_encoder.classes_
        accuracy_matrix = []
        
        for model in models:
            class_accs = []
            cm = self.results[model]['confusion_matrix']
            for i in range(len(class_names)):
                if cm[i].sum() > 0:
                    class_acc = cm[i, i] / cm[i].sum()
                else:
                    class_acc = 0
                class_accs.append(class_acc)
            accuracy_matrix.append(class_accs)
        
        im = ax2.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Emotion Classes')
        ax2.set_ylabel('Models')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Accuracy', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(class_names)):
                text = ax2.text(j, i, f'{accuracy_matrix[i][j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, save_path="confusion_matrices_detailed.png"):
        """Plot detailed confusion matrices"""
        models = list(self.results.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        class_names = self.label_encoder.classes_
        
        for i, model in enumerate(models):
            cm = self.results[model]['confusion_matrix']
            
            # Raw confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i],
                       xticklabels=class_names, yticklabels=class_names)
            axes[0, i].set_title(f'{model}\nConfusion Matrix (Raw Counts)', 
                               fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, i],
                       xticklabels=class_names, yticklabels=class_names)
            axes[1, i].set_title(f'{model}\nNormalized Confusion Matrix', 
                               fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, save_path="performance_metrics.png"):
        """Plot comprehensive performance metrics"""
        models = list(self.results.keys())
        class_names = self.label_encoder.classes_
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. F1-Score comparison
        f1_data = []
        for model in models:
            f1_scores = self.results[model]['f1_score']
            f1_data.append(f1_scores)
        
        x = np.arange(len(class_names))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (model, f1_scores) in enumerate(zip(models, f1_data)):
            ax1.bar(x + i*width, f1_scores, width, label=model, 
                   color=colors[i], alpha=0.8)
        
        ax1.set_title('F1-Score Comparison by Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotion Classes')
        ax1.set_ylabel('F1-Score')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(class_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Precision vs Recall scatter
        for i, model in enumerate(models):
            precision = self.results[model]['precision']
            recall = self.results[model]['recall']
            ax2.scatter(recall, precision, label=model, s=100, 
                       color=colors[i], alpha=0.7)
            
            # Add class labels
            for j, class_name in enumerate(class_names):
                ax2.annotate(class_name, (recall[j], precision[j]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 3. Inference time comparison
        models_list = list(models)
        inference_times = [self.results[model]['avg_inference_time']*1000 for model in models_list]
        
        bars = ax3.bar(models_list, inference_times, color=colors[:len(models)], alpha=0.8)
        ax3.set_title('Average Inference Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model complexity vs accuracy
        params = [self.results[model]['trainable_params']/1e6 for model in models_list]  # in millions
        accuracies = [self.results[model]['accuracy'] for model in models_list]
        
        scatter = ax4.scatter(params, accuracies, s=200, c=colors[:len(models)], alpha=0.7)
        
        for i, model in enumerate(models_list):
            ax4.annotate(model, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold')
        
        ax4.set_title('Model Complexity vs Accuracy', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Parameters (Millions)')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class StatisticalAnalyzer:
    """Perform statistical analysis on model results"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.results = evaluator.results
        self.label_encoder = evaluator.label_encoder
    
    def perform_significance_tests(self):
        """Perform statistical significance tests between models"""
        logger.info("📊 Performing statistical significance tests...")
        
        models = list(self.results.keys())
        if len(models) < 2:
            logger.warning("Need at least 2 models for significance testing")
            return {}
        
        significance_results = {}
        
        # McNemar's test for paired model comparison
        for model1, model2 in combinations(models, 2):
            preds1 = np.array(self.results[model1]['predictions'])
            preds2 = np.array(self.results[model2]['predictions'])
            targets = np.array(self.results[model1]['targets'])
            
            # Create contingency table
            correct1 = (preds1 == targets)
            correct2 = (preds2 == targets)
            
            # McNemar's test contingency table
            both_correct = np.sum(correct1 & correct2)
            model1_only = np.sum(correct1 & ~correct2)
            model2_only = np.sum(~correct1 & correct2)
            both_wrong = np.sum(~correct1 & ~correct2)
            
            # McNemar's test statistic
            if model1_only + model2_only > 0:
                mcnemar_stat = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                mcnemar_stat = 0
                p_value = 1.0
            
            significance_results[f"{model1}_vs_{model2}"] = {
                'mcnemar_statistic': mcnemar_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'contingency_table': {
                    'both_correct': both_correct,
                    'model1_only_correct': model1_only,
                    'model2_only_correct': model2_only,
                    'both_wrong': both_wrong
                }
            }
            
            logger.info(f"{model1} vs {model2}: p-value = {p_value:.4f}, significant = {p_value < 0.05}")
        
        return significance_results
    
    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals for model accuracies"""
        logger.info(f"📊 Calculating {confidence*100}% confidence intervals...")
        
        ci_results = {}
        alpha = 1 - confidence
        
        for model_name, results in self.results.items():
            n = len(results['targets'])
            accuracy = results['accuracy']
            
            # Wilson score interval for binomial proportion
            z = stats.norm.ppf(1 - alpha/2)
            
            # Wilson score interval
            center = (accuracy + z**2/(2*n)) / (1 + z**2/n)
            margin = z * np.sqrt((accuracy*(1-accuracy) + z**2/(4*n)) / n) / (1 + z**2/n)
            
            ci_lower = center - margin
            ci_upper = center + margin
            
            ci_results[model_name] = {
                'accuracy': accuracy,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin_of_error': margin,
                'sample_size': n
            }
            
            logger.info(f"{model_name}: {accuracy:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return ci_results

def generate_summary_table(results):
    """Generate summary table of all metrics"""
    models = list(results.keys())
    
    # Create comprehensive summary
    summary_data = []
    headers = ['Model', 'Accuracy', 'Precision (Avg)', 'Recall (Avg)', 'F1-Score (Avg)', 
              'Inference Time (ms)', 'FPS', 'Parameters (M)', 'AUC (Micro)']
    
    summary_data.append(headers)
    
    for model in models:
        model_results = results[model]
        
        row = [
            model,
            f"{model_results['accuracy']:.4f}",
            f"{np.mean(model_results['precision']):.4f}",
            f"{np.mean(model_results['recall']):.4f}",
            f"{np.mean(model_results['f1_score']):.4f}",
            f"{model_results['avg_inference_time']*1000:.2f}",
            f"{model_results['fps']:.2f}",
            f"{model_results['trainable_params']/1e6:.2f}",
            f"{model_results['roc_data']['micro']['auc']:.4f}"
        ]
        summary_data.append(row)
    
    return summary_data

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting comprehensive model evaluation...")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model paths (adjust these based on your saved models)
    model_paths = {
        'ResNet18': 'resnet18_emotion_model.pth',
        'EfficientNet-B0': 'efficientnet_b0_emotion_model.pth',
        'DenseNet121': 'densenet121_emotion_model.pth'
    }
    
    # Check if models exist
    existing_models = {k: v for k, v in model_paths.items() if os.path.exists(v)}
    
    if not existing_models:
        logger.error("❌ No trained models found! Please run the training script first.")
        logger.info("Expected model files:")
        for model_name, path in model_paths.items():
            logger.info(f"  {model_name}: {path}")
        return
    
    logger.info(f"Found {len(existing_models)} trained models: {list(existing_models.keys())}")
    
    # Load dataset from directory structure
    data_root = 'model_eye'
    test_set, label_encoder, test_image_paths = load_dataset_from_directory(data_root)
    
    if test_set is None:
        logger.error("❌ Failed to load test dataset!")
        return
    
    # Create test data loader
    test_loader = DataLoader(
        test_set, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2,
        pin_memory=device.type == 'cuda'
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(existing_models, test_loader, label_encoder, device)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    if not evaluator.results:
        logger.error("❌ No evaluation results generated!")
        return
    
    # Initialize visualizer and analyzer
    visualizer = EvaluationVisualizer(evaluator)
    analyzer = StatisticalAnalyzer(evaluator)
    
    # Generate visualizations
    logger.info("📊 Generating evaluation visualizations...")
    visualizer.plot_accuracy_comparison()
    visualizer.plot_confusion_matrices()
    visualizer.plot_performance_metrics()
    
    # Perform statistical analysis
    logger.info("📊 Performing statistical analysis...")
    significance_results = analyzer.perform_significance_tests()
    confidence_intervals = analyzer.calculate_confidence_intervals()
    
    # Generate summary table
    summary_table = generate_summary_table(evaluator.results)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🏆 EVALUATION SUMMARY")
    logger.info("="*80)
    
    for row in summary_table:
        logger.info(" | ".join(f"{cell:>15}" for cell in row))
        if row == summary_table[0]:  # After header
            logger.info("-" * (len(row) * 18))
    
    # Best model
    best_model = max(evaluator.results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\n🥇 Best Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
    
    # Statistical significance
    if significance_results:
        logger.info("\n📊 Statistical Significance Tests:")
        for comparison, result in significance_results.items():
            significance = "✅ Significant" if result['significant'] else "❌ Not significant"
            logger.info(f"  {comparison}: p-value = {result['p_value']:.4f} ({significance})")
    
    # Confidence intervals
    logger.info("\n📊 95% Confidence Intervals:")
    for model_name, ci_data in confidence_intervals.items():
        logger.info(f"  {model_name}: {ci_data['accuracy']:.4f} "
                   f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]")
    
    # Create and save the multi_model_comparison_results.pkl file
    logger.info("💾 Creating multi_model_comparison_results.pkl...")
    
    # Reconstruct the expected structure
    multi_model_results = {
        'config': {
            'data_root': data_root,
            'device': device,
            'evaluation_timestamp': datetime.now().isoformat()
        },
        'results': {},
        'label_encoder': label_encoder,
        'comparison_table': summary_table,
        'best_model': best_model[0],
        'significance_tests': significance_results,
        'confidence_intervals': confidence_intervals
    }
    
    # Add model results in the expected format
    for model_name, results in evaluator.results.items():
        multi_model_results['results'][model_name] = {
            'model_name': model_name,
            'test_acc': results['accuracy'],
            'test_loss': 0.0,  # Not calculated in evaluation
            'test_preds': results['predictions'],
            'test_targets': results['targets'],
            'classification_report': results['class_report'],
            'training_time': 0.0,  # Not available in evaluation
            'epochs_completed': 0,  # Not available in evaluation
            'val_accs': [],  # Not available in evaluation
            'train_losses': [],  # Not available in evaluation
            'val_losses': [],  # Not available in evaluation
            'train_accs': []  # Not available in evaluation
        }
    
    # Save the file
    joblib.dump(multi_model_results, 'multi_model_comparison_results.pkl')
    
    # Save comprehensive evaluation results
    evaluation_results = {
        'model_results': evaluator.results,
        'significance_tests': significance_results,
        'confidence_intervals': confidence_intervals,
        'summary_table': summary_table,
        'label_encoder': label_encoder,
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_image_paths': test_image_paths
    }
    
    joblib.dump(evaluation_results, 'comprehensive_evaluation_results.pkl')
    
    logger.info("\n✅ Comprehensive evaluation completed successfully!")
    logger.info("📁 Files generated:")
    logger.info("  - accuracy_comparison.png")
    logger.info("  - confusion_matrices_detailed.png")
    logger.info("  - performance_metrics.png")
    logger.info("  - multi_model_comparison_results.pkl")
    logger.info("  - comprehensive_evaluation_results.pkl")
    logger.info(f"🎯 Best performing model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")

if __name__ == "__main__":
    main()
