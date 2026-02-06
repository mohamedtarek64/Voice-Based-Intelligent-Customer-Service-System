# 🤖 Intelligent Voice Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![AI](https://img.shields.io/badge/AI-Gemini%20%7C%20SVM-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Next-Generation Customer Service Experience**
>
> An advanced, hybrid AI system merging **deterministic machine learning** (SVM) with **generative AI** (Google Gemini) to provide seamless, voice-enabled customer support.

---

## 🌟 Overview

The **Intelligent Voice Chatbot** is a state-of-the-art solution designed to automate customer interactions. Unlike traditional chatbots that rely solely on rigid rules or unpredictable generative models, this system uses a **Hybrid Decision Engine**:
1.  **Precision Layer (SVM)**: Instantly handles critical, repetitive business tasks (e.g., "Where is my order?") with near-100% accuracy.
2.  **Conversational Layer (LLM)**: Seamlessly falls back to Google's Gemini LLM for complex, open-ended queries, ensuring the user never hits a "dead end."

With integrated **Voice-to-Voice** capabilities, users can speak naturally to the system and receive spoken responses in real-time.

---

## 🚀 Key Features

### 🧠 Dual-Core AI Architecture
- **Intent Classifier**: Custom-trained Support Vector Machine (SVM) utilizing TF-IDF Vectorization (Unigrams, Bigrams, Trigrams) for business-critical accuracy.
- **Generative Fallback**: Integrated **Google Gemini Flash** LLM for handling general conversation, small talk, and complex queries that fall outside standard business logic.

### 🗣️ Voice-First Interface
- **Speech-to-Text (STT)**: High-fidelity audio transcription using Whisper/Google Speech Recognition.
- **Text-to-Speech (TTS)**: Natural-sounding vocal responses, supporting English and auto-switching to Arabic logic where applicable.

### ⚡ Intelligent Routing
- **Confidence Scoring**: The system evaluates its own certainty. High confidence trigger pre-defined business flows; low confidence triggers the LLM.
- **Smart Escalation**: Automatically identifies when a human agent is needed based on sentiment or unresolved queries.

### 📊 Comprehensive Intent Support
Ready-to-use handling for 10+ core business scenarios:
- 📦 **Order Management** (Status, Tracking, Cancellations)
- 💳 **Billing & Payments** (Refunds, Payment Issues)
- 🛍️ **Product Support** (Information, Availability)
- 🔄 **Returns & Exchanges**
- 📢 **Complaints & Feedback**

---

## 🏗️ Technical Architecture

```text
├── 📂 app.py                # Main Flask Application Entry Point
├── 📂 src/                  # Core Intelligence Modules
│   ├── 🧠 decision_engine.py  # Hybrid routing logic (SVM vs LLM)
│   ├── 🤖 llm_engine.py       # Google Gemini Integration
│   ├── 📊 model_training.py   # MLP/SVM Training Pipeline
│   ├── 🎤 speech_to_text.py   # Audio Processing using Whisper
│   ├── 🔊 text_to_speech.py   # Voice Synthesis logic
│   └── 🧹 data_preprocessing.py # NLP cleaning & tokenization
├── 📂 models/               # Serialized ML Models (PKL files)
├── 📂 static/               # CSS, JS, and Images
└── 📂 templates/            # HTML Frontend Templates
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- A Google Gemini API Key

### 1. Clone & Install
```bash
git clone https://github.com/mohamedtarek64/Voice-Based-Intelligent-Customer-Service-System.git
cd Voice-Based-Intelligent-Customer-Service-System
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_api_key_here
```

### 3. Initialize Models
Before running the server, generate the dataset and train the classification model:
```bash
# Generate synthetic training data
python src/generate_dataset.py

# Train the SVM classifier
python src/model_training.py
```

### 4. Launch the Application
```bash
python app.py
```
Visit `http://localhost:5001` to start chatting!

---

## 🧪 Performance & Metrics

Our custom SVM model has been rigorously tested against synthetic datasets:

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | 99.8% |
| **Precision** | 1.00 |
| **Recall** | 1.00 |
| **F1-Score** | 1.00 |
| **Inference Time** | < 20ms |

*Note metrics are based on a test set of 1,000 samples.*

---

## 👨‍💻 Developed by
**Mohamed Tarek**  
*Graduation Project - Class of 2026*

This project demonstrates the practical application of NLP, Machine Learning, and Software Engineering principles to solve real-world business problems.
