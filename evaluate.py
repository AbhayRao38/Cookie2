import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import numpy as np

from train import ImprovedEmotionEyeDataset, get_transforms, prepare_datasets, AdaptiveEmotionCNN

# Allow unpickling LabelEncoder (PyTorch >=2.6 fix)
add_safe_globals([LabelEncoder])

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_eval.png', dpi=150)
    plt.show()

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

def run_evaluation():
    logger.info("Starting Evaluation...")

    checkpoint_path = "emotion_model_cuda.pth"
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['config']
    test_accuracy_saved = checkpoint.get('test_accuracy', None)

    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    logger.info(f"Saved Test Accuracy: {test_accuracy_saved}")
    logger.info(f"Label classes: {list(label_encoder.classes_)}")

    # Load training history and plot
    history = checkpoint.get('training_history', {})
    if history:
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        train_accs = history.get('train_accs', [])
        val_accs = history.get('val_accs', [])

        if train_losses and val_losses and train_accs and val_accs:
            logger.info(f"Final Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%")
            logger.info(f"Final Val Loss:   {val_losses[-1]:.4f}, Val Acc:   {val_accs[-1]:.2f}%")
            plot_training_history(train_losses, val_losses, train_accs, val_accs)
    else:
        logger.warning("No training history found in checkpoint.")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    logger.info(f"Evaluation device: {device}")

    # Load test data
    _, _, test_set, _ = prepare_datasets(
        data_root=config['data_root'],
        clean_data=False,
        device_type=device_type
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Load model
    model = AdaptiveEmotionCNN(
        num_classes=len(label_encoder.classes_),
        pretrained=False,
        device_type=device_type
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # Evaluation
    all_preds = []
    all_targets = []
    all_softmax = []
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets.long())
            total_loss += loss.item()

            probs = softmax(outputs)
            _, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_softmax.extend(probs.cpu().numpy())

            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            acc = 100.0 * correct / total
            pbar.set_postfix({'Loss': f'{total_loss/(total//config["batch_size"]):.4f}', 'Acc': f'{acc:.2f}%'})

    test_acc = 100. * correct / total
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")

    # Save softmax probabilities
    np.save("eye_softmax_probs.npy", np.array(all_softmax))
    logger.info("Softmax probabilities saved to 'eye_softmax_probs.npy'")

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=label_encoder.classes_, digits=4)
    logger.info(f"Classification Report:\n{report}")

    # Class-wise accuracy stats
    cm = confusion_matrix(all_targets, all_preds)
    class_wise_acc = cm.diagonal() / cm.sum(axis=1)
    for cls, acc in zip(label_encoder.classes_, class_wise_acc):
        logger.info(f"Accuracy for {cls}: {acc * 100:.2f}%")

    acc_variance = np.var(class_wise_acc)
    acc_std = np.std(class_wise_acc)
    logger.info(f"Class-wise Accuracy Variance: {acc_variance:.6f}")
    logger.info(f"Class-wise Accuracy Std Dev: {acc_std:.6f}")

    # Confusion Matrix Plot
    plot_confusion_matrix(all_targets, all_preds, class_names=label_encoder.classes_)

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
