#!/usr/bin/env python3
"""
PlantusAI - Flask Web App with Real-time Plant Disease Detection
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import json
import os
import io
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
CONFIDENCE_THRESHOLD = 0.5

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
MODEL_PATH = "models/plantusAI_model_FINAL.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    print(f"✅ Model loaded with {len(class_names)} classes")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(pil_image):
    """Preprocess a PIL Image for model prediction"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(pil_image):
    """Make prediction on a PIL Image"""
    if not MODEL_LOADED:
        return {'error': 'Model not loaded'}

    img_batch = preprocess_image(pil_image)
    predictions = model.predict(img_batch, verbose=0)

    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            'success': False,
            'error': f'Unable to identify — please upload a clear, well-lit photo of a plant leaf (confidence too low: {confidence*100:.1f}%)'
        }

    disease_name = class_names[class_idx]

    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5 = [
        {
            'disease': class_names[idx],
            'confidence': float(predictions[0][idx]),
            'confidence_percent': float(predictions[0][idx] * 100)
        }
        for idx in top_5_idx
    ]

    return {
        'disease': disease_name,
        'confidence': confidence,
        'confidence_percent': confidence * 100,
        'top_5_predictions': top_5,
        'success': True
    }

def image_to_base64(img_bytes, mime_type='image/jpeg'):
    """Convert raw image bytes to a base64 data URL"""
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"

# Routes

@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'classes': len(class_names) if MODEL_LOADED else 0
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image — processed entirely in memory"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, bmp, gif'}), 400

        # Read file into memory — no disk writes
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result = predict_disease(img)

        if result.get('success'):
            mime_type = file.content_type or 'image/jpeg'
            result['image_url'] = image_to_base64(img_bytes, mime_type)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/camera', methods=['POST'])
def predict_camera():
    """Predict from camera capture (base64) — no disk writes"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 directly into memory
        raw = data['image']
        img_bytes = base64.b64decode(raw.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result = predict_disease(img)

        if result.get('success'):
            result['image_url'] = raw  # already a base64 data URL

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({
        'total_classes': len(class_names),
        'classes': sorted(list(class_names.values()))
    }), 200

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'model_name': 'PlantusAI',
        'version': '1.0.0',
        'description': 'Real-time plant disease detection using CNN',
        'total_classes': len(class_names) if MODEL_LOADED else 0,
        'accuracy': '95%+',
        'training_data': '70,000+ labeled images',
        'framework': 'TensorFlow/Keras'
    }), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌱 PlantusAI Web Interface")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("\nFeatures:")
    print("  📤 File Upload")
    print("  📷 Camera Capture")
    print("  🤖 Real-time Prediction")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)