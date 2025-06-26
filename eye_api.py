from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import joblib
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------ Model Definition ------------------------ #

class AdaptiveEmotionCNN(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(AdaptiveEmotionCNN, self).__init__()

        # Use only ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        num_features = self.backbone.fc.in_features
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

# ------------------------ Model Initialization ------------------------ #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'emotion_model_cpu.pth' if device.type == 'cpu' else 'emotion_model_cuda.pth')
    label_encoder_path = os.path.join(base_dir, f'label_encoder_{device.type}.pkl')

    logging.info(f"📦 Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Force backbone to be ResNet only
    eye_model = AdaptiveEmotionCNN(num_classes=8, pretrained=True)
    eye_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    eye_model.to(device)
    eye_model.eval()

    # Load the label encoder
    label_encoder = joblib.load(label_encoder_path)

    logging.info(f"✅ Successfully loaded ResNet18-based eye model on {device}")

except Exception as e:
    logging.error(f"❌ Failed to load eye model from {checkpoint_path}: {e}")
    eye_model = None
    label_encoder = None

# ------------------------ Image Preprocessing ------------------------ #

def get_transforms():
    return A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ------------------------ API Endpoints ------------------------ #

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if eye_model else 'error',
        'model_loaded': eye_model is not None,
        'device': str(device),
        'model_type': 'AdaptiveEmotionCNN (ResNet18)'
    })

@app.route('/predict/eye', methods=['POST'])
def predict_eye():
    if eye_model is None:
        return jsonify({'success': False, 'error': 'Eye model not loaded'}), 500

    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        image = Image.open(request.files['file'].stream).convert('RGB')
        image_np = np.array(image)
        transform = get_transforms()
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = eye_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

        emotion_labels = label_encoder.classes_.tolist() if label_encoder else [
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]
        predicted_emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else 'Unknown'

        mci_relevant_emotions = ['Anger', 'Fear', 'Disgust', 'Sadness']
        mci_probability = confidence if predicted_emotion in mci_relevant_emotions else (1 - confidence)
        binary_probs = [1 - mci_probability, mci_probability]

        return jsonify({
            'success': True,
            'probabilities': binary_probs,
            'confidence': confidence,
            'predicted_emotion': predicted_emotion,
            'all_probabilities': probabilities.tolist(),
            'emotion_labels': emotion_labels
        })

    except Exception as e:
        logging.error(f"❌ Error in eye prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ------------------------ App Start ------------------------ #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
