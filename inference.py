import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Model architecture - must match training script
def net(dropout_rate=0.3):
    num_classes = 5
    model = models.resnet50(pretrained=False)  # Set to False for inference
    
    # Freeze layers (same as training)
    for name, param in model.named_parameters():
        if 'layer1' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Modified classifier with tunable dropout
    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    return model

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    logger.info("Loading model.")
    
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the model
    model = net()
    
    # Load the saved state dict
    model_path = os.path.join(model_dir, 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Model loaded successfully from model.pth")
    else:
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.to(device)
    model.eval()
    
    logger.info("Model loading complete.")
    return model

def input_fn(request_body, content_type='application/json'):
    """
    Deserialize the invoke request body into an object we can perform prediction on.
    """
    logger.info(f"Deserializing the input data with content type: {content_type}")
    
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle base64 encoded image
        if 'image' in input_data:
            image_data = input_data['image']
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                raise ValueError("Image data should be base64 encoded string")
        else:
            raise ValueError("No 'image' key found in input data")
            
    elif content_type == 'image/jpeg' or content_type == 'image/png':
        # Handle direct image upload
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    # Apply preprocessing transforms (same as test transform in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    logger.info("Input data deserialized successfully.")
    return image_tensor

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    logger.info("Generating prediction based on input data.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class
        _, predicted_class = torch.max(outputs, 1)
        
        # Convert to numpy for JSON serialization
        predicted_class = predicted_class.cpu().numpy()[0]
        probabilities = probabilities.cpu().numpy()[0]
    
    logger.info(f"Prediction generated: class {predicted_class}")
    return {
        'predicted_class': int(predicted_class),
        'probabilities': probabilities.tolist(),
        'confidence': float(probabilities[predicted_class])
    }

def output_fn(prediction, accept):
    """
    Serialize the prediction result into the desired response content type.
    """
    logger.info(f"Serializing the prediction result with accept type: {accept}")
    
    if accept == 'application/json':
        # Add class names mapping
        class_names = ["1", "2", "3", "4", "5"]
        
        result = {
            'predicted_class': prediction['predicted_class'],
            'predicted_class_name': class_names[prediction['predicted_class']],
            'confidence': prediction['confidence'],
            'probabilities': {
                class_names[i]: prob for i, prob in enumerate(prediction['probabilities'])
            }
        }
        
        logger.info("Prediction serialized successfully.")
        return json.dumps(result), accept
    
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
