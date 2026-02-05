"""
Flask Web Application
=====================
Web interface for the Voice Customer Service System.
"""

import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.data_preprocessing import preprocess_text, download_nltk_data
from src.decision_engine import DecisionEngine
from src.text_to_speech import TextToSpeech
from src.llm_engine import LLMEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm', 'm4a', 'flac'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
model = None
vectorizer = None
stt = None
tts = None
decision_engine = None
llm_engine = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load ML models and initialize components."""
    global model, vectorizer, stt, tts, decision_engine
    
    model_path = 'models/intent_classifier.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✓ ML models loaded")
    else:
        print("⚠ ML models not found. Train the model first.")
        model = None
        vectorizer = None
    
    # Initialize TTS
    tts = TextToSpeech(language='en')
    print("✓ TTS initialized")
    
    # Initialize Decision Engine
    decision_engine = DecisionEngine()
    print("✓ Decision Engine initialized")
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize LLM Engine
    llm_engine = LLMEngine()
    print("✓ LLM Engine initialized")
    
    # Try to initialize STT (optional - might not work on all systems)
    try:
        from src.speech_to_text import create_stt
        stt = create_stt(use_whisper=True, model_size='base')
        print("✓ STT initialized")
    except Exception as e:
        print(f"⚠ STT initialization failed: {e}")
        stt = None


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check system status."""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'stt_available': stt is not None,
        'tts_available': tts is not None,
        'llm_available': llm_engine.is_available() if llm_engine else False,
        'intents': list(model.classes_) if model else []
    })


@app.route('/api/process_text', methods=['POST'])
def process_text():
    """Process text query."""
    if model is None or vectorizer is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'No query provided'
        }), 400
    
    try:
        # Preprocess
        processed_query = preprocess_text(query)
        
        # Predict
        query_vector = vectorizer.transform([processed_query])
        predicted_intent = model.predict(query_vector)[0]
        confidence_scores = model.predict_proba(query_vector)[0]
        max_confidence = float(confidence_scores.max())
        
        # Get top intents
        top_indices = confidence_scores.argsort()[-3:][::-1]
        top_intents = [
            {'intent': model.classes_[i], 'confidence': float(confidence_scores[i])}
            for i in top_indices
        ]
        
        prediction = {
            'predicted_intent': predicted_intent,
            'confidence': max_confidence,
            'top_intents': top_intents
        }
        
        # Generate response
        # If it's a general intent or low confidence, use LLM
        service_intents = ['order_cancellation', 'order_status', 'payment_issues', 'return_exchange', 'refund_status', 'shipping_inquiry']
        
        if predicted_intent not in service_intents or max_confidence < 0.7:
            if llm_engine.is_available():
                print(f"Using LLM for query: {query}")
                llm_response = llm_engine.generate_response(query)
                result = {
                    'original_query': query,
                    'intent': predicted_intent,
                    'confidence': max_confidence,
                    'action': 'llm_respond',
                    'response': llm_response,
                    'top_intents': top_intents
                }
            else:
                result = decision_engine.process_query(prediction, query)
        else:
            result = decision_engine.process_query(prediction, query)
            
        result['success'] = True
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """Process audio file."""
    if model is None or vectorizer is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    if stt is None:
        return jsonify({
            'success': False,
            'error': 'Speech-to-Text not available.'
        }), 500
    
    if 'audio' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    try:
        # Save uploaded file
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Transcribe
        stt_result = stt.transcribe(filepath)
        
        if not stt_result.get('success', False):
            return jsonify({
                'success': False,
                'error': f"Transcription failed: {stt_result.get('error', 'Unknown')}"
            }), 500
        
        query = stt_result['text']
        
        # Preprocess
        processed_query = preprocess_text(query)
        
        # Predict
        query_vector = vectorizer.transform([processed_query])
        predicted_intent = model.predict(query_vector)[0]
        confidence_scores = model.predict_proba(query_vector)[0]
        max_confidence = float(confidence_scores.max())
        
        # Get top intents
        top_indices = confidence_scores.argsort()[-3:][::-1]
        top_intents = [
            {'intent': model.classes_[i], 'confidence': float(confidence_scores[i])}
            for i in top_indices
        ]
        
        prediction = {
            'predicted_intent': predicted_intent,
            'confidence': max_confidence,
            'top_intents': top_intents
        }
        
        # Generate response
        service_intents = ['order_cancellation', 'order_status', 'payment_issues', 'return_exchange', 'refund_status', 'shipping_inquiry']
        
        if predicted_intent not in service_intents or max_confidence < 0.7:
            if llm_engine.is_available():
                print(f"Using LLM for query: {query}")
                llm_response = llm_engine.generate_response(query)
                result = {
                    'original_query': query,
                    'intent': predicted_intent,
                    'confidence': max_confidence,
                    'action': 'llm_respond',
                    'response': llm_response,
                    'top_intents': top_intents
                }
            else:
                result = decision_engine.process_query(prediction, query)
        else:
            result = decision_engine.process_query(prediction, query)
            
        result['success'] = True
        result['transcription'] = query
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/speak', methods=['POST'])
def speak():
    """Generate speech from text."""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({
            'success': False,
            'error': 'No text provided'
        }), 400
    
    try:
        # Detect language for TTS
        # Simple check for Arabic characters
        lang = 'en'
        if any(u'\u0600' <= c <= u'\u06FF' for c in text):
            lang = 'ar'
            
        filename = f"response_{uuid.uuid4()}.mp3"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create a temporary TTS instance for the specific language if needed
        temp_tts = TextToSpeech(language=lang)
        temp_tts.speak(text, output_file=filepath, play_audio=False)
        
        return send_file(filepath, mimetype='audio/mpeg')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/intents')
def get_intents():
    """Get list of supported intents."""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'intents': list(model.classes_)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 Voice Customer Service - Web Server")
    print("="*60)
    
    # Load models
    load_models()
    
    print("\n" + "-"*60)
    print("Starting web server...")
    print("Open http://localhost:5001 in your browser")
    print("-"*60 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')
