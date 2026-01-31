# 🎤 Voice-Based Intelligent Customer Service System

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)

An end-to-end intelligent customer service system that handles user queries through **natural voice interaction**, powered by a custom-trained **SVM (Support Vector Machine)** classification model.

## 🚀 Key Features

- **Voice-First Interaction**: Uses Google TTS and Whisper/SpeechRecognition for seamless voice communication.
- **Intelligent Classification**: Custom SVM model trained on 5,000+ synthetic samples with **100% accuracy**.
- **10 Core Intents**: Handles Order Status, Cancellations, Payment Issues, Complaints, and more.
- **Decision Engine**: Confidence-based routing (Auto-respond, Clarify, or Escalate).
- **Web Interface**: Clean, modern Flask-based UI for real-time interaction.

## 🏗️ Project Structure

```text
├── app.py                # Flask Web Application
├── src/                  # Core logic modules
│   ├── model_training.py # SVM training pipeline
│   ├── generate_dataset.py # Synthetic data generation
│   ├── data_preprocessing.py # Text NLP cleaning
│   ├── speech_to_text.py # Audio transcription
│   └── text_to_speech.py # Voice synthesis
├── models/               # Saved PKL models & vectorizers
├── data/                 # Raw & processed datasets
└── templates/            # Web UI HTML templates
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Voice-Based-Intelligent-Customer-Service-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Dataset & Train Model**:
   ```bash
   # Generate 5000+ samples
   python src/generate_dataset.py
   
   # Train SVM model with Trigrams (ngram_range=(1,3))
   python src/model_training.py
   ```

4. **Run the App**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5001` in your browser.

## 📊 Model Performance

| Metric | Score |
| --- | --- |
| **Training Samples** | 4,000 |
| **Testing Samples** | 1,000 |
| **Algorithm** | SVM (Linear Kernel) |
| **Features** | TF-IDF (Unigrams, Bigrams, Trigrams) |
| **Overall Accuracy** | **100%** |

## 🎯 Intent Categories Supported

- `order_status`, `order_cancellation`
- `payment_issues`, `refund_status`
- `product_information`, `shipping_inquiry`
- `return_exchange`, `complaint`
- `account_issues`, `general_inquiry`

## 🎓 Academic Credits
This project was developed as part of a Graduation Project focusing on Machine Learning and Voice Computing. Detailed documentation can be found in `PROJECT_DOCUMENTATION.md`.
