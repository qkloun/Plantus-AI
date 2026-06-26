# PlantusAI - Real-Time Plant Disease Detection

## Overview
PlantusAI is a deep learning application that uses convolutional neural networks (CNN) to detect plant diseases in real-time. It achieves 95%+ accuracy across 38 plant disease classes.

## Features
- Deep Learning Model (TensorFlow/Keras)
- Real-time webcam detection (OpenCV)
- REST API for predictions
- 95%+ accuracy on 70,000+ training images
- 38 disease classes detection

## Tech Stack
- **Framework:** TensorFlow 2.16
- **Frontend:** OpenCV
- **Backend:** Flask
- **Language:** Python 3.11
- **Model:** MobileNetV2 with Transfer Learning

## Project Structure

```
├── Plantus-AI/
│   ├── Models
│   │   ├── class_indices.json
│   │   ├── plantusAI_model_FINAL.keras
│   ├── Venv/
│   ├── static/
│   │   ├── css/
│   │   │   ├── style.css
│   ├── templates/
│   │   ├── about.html
│   │   ├── base.html
│   │   ├── index.html
├── Plantus.py
└── plantus_app.py
```
