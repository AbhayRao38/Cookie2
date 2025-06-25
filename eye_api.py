from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import joblib
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing._label import LabelEncoder
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class AdaptiveEmotionCNN(nn.Module):
    """Adaptive CNN model from your eye_model_training.py"""
    def __init__(self, num_classes=8, pretrained=True, device_type='cpu'):
        super(AdaptiveEmotionCNN, self).__init__()
        
        if device_type == 'cuda':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
    def forward(self, x):
        return self.backbone(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type = device.type

try:
    # Load the model weights
    model_data = torch.load(f'emotion_model_{device_type}.pth', map_location=device)

    # Build and load model
    eye_model = AdaptiveEmotionCNN(num_classes=8, device_type=device_type)
    eye_model.load_state_dict(model_data['model_state_dict'])
    eye_model.to(device)
    eye_model.eval()

    # ✅ Load LabelEncoder using joblib (safe)
    label_encoder = joblib.load(f'label_encoder_{device_type}.pkl')

    logging.info(f"Eye model loaded successfully on {device}")

except Exception as e:
    logging.error(f"Failed to load eye model: {e}")
    eye_model = None
    label_encoder = None

def get_transforms(device_type='cpu'):
    """Get transforms matching your training pipeline"""
    return A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': eye_model is not None,
        'device': str(device),
        'model_type': 'AdaptiveEmotionCNN'
    })

@app.route('/predict/eye', methods=['POST'])
def predict_eye():
    if eye_model is None:
        return jsonify({
            'success': False,
            'error': 'Eye model not loaded'
        }), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
            
        image_file = request.files['file']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Load and preprocess image
        image = Image.open(image_file.stream).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        transform = get_transforms(device_type)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = eye_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
        
        # Get emotion labels
        emotion_labels = label_encoder.classes_.tolist() if label_encoder else [
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]
        predicted_emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else 'Unknown'
        
        # MCI-relevant probability mapping
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
        logging.error(f"Error in eye prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
