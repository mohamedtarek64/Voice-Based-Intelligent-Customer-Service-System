"""
Main Application - Voice Customer Service Pipeline
===================================================
End-to-end voice-based customer service system.
"""

import os
import sys
import joblib
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import preprocess_text, download_nltk_data
from speech_to_text import SpeechToText, create_stt
from text_to_speech import TextToSpeech, create_tts
from decision_engine import DecisionEngine
from audio_recorder import record_audio, record_audio_with_countdown


class VoiceCustomerService:
    """
    Complete Voice-Based Customer Service System.
    Integrates STT, intent classification, decision engine, and TTS.
    """
    
    def __init__(self, model_dir: str = 'models', 
                 stt_model: str = 'base',
                 use_whisper: bool = True,
                 use_gtts: bool = True):
        """
        Initialize the voice customer service system.
        
        Args:
            model_dir: Directory containing trained models
            stt_model: Whisper model size
            use_whisper: Use Whisper for STT
            use_gtts: Use gTTS for TTS
        """
        self.model_dir = model_dir
        self.is_ready = False
        
        # Download NLTK data
        download_nltk_data()
        
        # Initialize components
        print("\n" + "="*60)
        print("🎤 Voice Customer Service System")
        print("="*60)
        
        # Load ML model
        self._load_models()
        
        # Initialize STT
        print("\n[1/3] Initializing Speech-to-Text...")
        self.stt = create_stt(use_whisper=use_whisper, model_size=stt_model)
        
        # Initialize TTS
        print("\n[2/3] Initializing Text-to-Speech...")
        self.tts = create_tts(use_gtts=use_gtts)
        
        # Initialize Decision Engine
        print("\n[3/3] Initializing Decision Engine...")
        self.decision_engine = DecisionEngine()
        
        self.is_ready = True
        print("\n" + "="*60)
        print("✅ System Ready!")
        print("="*60 + "\n")
    
    def _load_models(self):
        """Load the trained intent classification models."""
        model_path = os.path.join(self.model_dir, 'intent_classifier.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print("⚠️ Trained models not found!")
            print("Please train the model first:")
            print("  1. Run: python src/generate_dataset.py")
            print("  2. Run: python src/model_training.py")
            self.model = None
            self.vectorizer = None
            return
        
        print("\n[0/3] Loading ML Models...")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"  ✓ Model loaded from: {model_path}")
        print(f"  ✓ Vectorizer loaded from: {vectorizer_path}")
    
    def process_audio(self, audio_path: str, 
                      play_response: bool = True) -> Dict[str, Any]:
        """
        Process audio file through the complete pipeline.
        
        Args:
            audio_path: Path to audio file
            play_response: Whether to play the voice response
            
        Returns:
            Dictionary with complete results
        """
        if not self.is_ready or self.model is None:
            return {
                'success': False,
                'error': 'System not ready. Check model files.'
            }
        
        print("\n" + "-"*50)
        print("🎤 Processing Voice Query")
        print("-"*50)
        
        # Step 1: Speech to Text
        print("\n[Step 1/4] Converting speech to text...")
        stt_result = self.stt.transcribe(audio_path)
        
        if not stt_result.get('success', False):
            return {
                'success': False,
                'error': f"STT Error: {stt_result.get('error', 'Unknown')}"
            }
        
        customer_query = stt_result['text']
        print(f"  📝 Transcription: \"{customer_query}\"")
        
        # Step 2: Preprocess
        print("\n[Step 2/4] Preprocessing query...")
        processed_query = preprocess_text(customer_query)
        print(f"  📝 Processed: \"{processed_query}\"")
        
        # Step 3: Predict intent
        print("\n[Step 3/4] Analyzing intent...")
        query_vector = self.vectorizer.transform([processed_query])
        predicted_intent = self.model.predict(query_vector)[0]
        confidence_scores = self.model.predict_proba(query_vector)[0]
        max_confidence = float(confidence_scores.max())
        
        # Get top intents
        top_indices = confidence_scores.argsort()[-3:][::-1]
        top_intents = [
            {'intent': self.model.classes_[i], 'confidence': float(confidence_scores[i])}
            for i in top_indices
        ]
        
        prediction = {
            'predicted_intent': predicted_intent,
            'confidence': max_confidence,
            'top_intents': top_intents
        }
        
        # Generate response using decision engine
        result = self.decision_engine.process_query(prediction, customer_query)
        
        print(f"  🎯 Intent: {result['intent']}")
        print(f"  📊 Confidence: {result['confidence']:.2%}")
        print(f"  ⚡ Action: {result['action']}")
        print(f"  💬 Response: {result['response']}")
        
        # Step 4: Text to Speech
        print("\n[Step 4/4] Generating voice response...")
        if play_response:
            audio_file = self.tts.speak(
                result['response'], 
                output_file="response.mp3",
                play_audio=True
            )
            result['response_audio'] = audio_file
        
        result['success'] = True
        result['transcription'] = customer_query
        
        print("\n" + "-"*50)
        print("✅ Processing Complete!")
        print("-"*50)
        
        return result
    
    def process_text(self, text_query: str, 
                     speak_response: bool = False) -> Dict[str, Any]:
        """
        Process text query (skip STT).
        
        Args:
            text_query: Text query from user
            speak_response: Whether to speak the response
            
        Returns:
            Dictionary with results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        # Preprocess
        processed_query = preprocess_text(text_query)
        
        # Predict
        query_vector = self.vectorizer.transform([processed_query])
        predicted_intent = self.model.predict(query_vector)[0]
        confidence_scores = self.model.predict_proba(query_vector)[0]
        max_confidence = float(confidence_scores.max())
        
        top_indices = confidence_scores.argsort()[-3:][::-1]
        top_intents = [
            {'intent': self.model.classes_[i], 'confidence': float(confidence_scores[i])}
            for i in top_indices
        ]
        
        prediction = {
            'predicted_intent': predicted_intent,
            'confidence': max_confidence,
            'top_intents': top_intents
        }
        
        # Generate response
        result = self.decision_engine.process_query(prediction, text_query)
        result['success'] = True
        
        # Optionally speak
        if speak_response:
            self.tts.speak(result['response'], play_audio=True)
        
        return result
    
    def record_and_process(self, duration: int = 5, 
                           play_response: bool = True) -> Dict[str, Any]:
        """
        Record audio and process through pipeline.
        
        Args:
            duration: Recording duration in seconds
            play_response: Whether to play the response
            
        Returns:
            Dictionary with results
        """
        audio_path = record_audio_with_countdown(
            duration=duration,
            countdown=3,
            output_file="user_query.wav"
        )
        
        if not audio_path:
            return {
                'success': False,
                'error': 'Failed to record audio'
            }
        
        return self.process_audio(audio_path, play_response)


def cli_interface():
    """Command-line interface for the voice customer service system."""
    print("\n" + "="*60)
    print("🎤 Voice Customer Service System - CLI Mode")
    print("="*60)
    
    # Initialize system
    vcs = VoiceCustomerService(model_dir='models')
    
    if not vcs.is_ready:
        print("\n❌ System initialization failed!")
        return
    
    print("\nCommands:")
    print("  'record' - Record voice and process")
    print("  'file <path>' - Process audio file")
    print("  'text' - Enter text query")
    print("  'quit' - Exit")
    print("-"*60)
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print("👋 Goodbye!")
                break
            
            elif command == 'record':
                result = vcs.record_and_process(duration=5)
                if not result['success']:
                    print(f"❌ Error: {result.get('error', 'Unknown')}")
            
            elif command.startswith('file '):
                file_path = command[5:].strip()
                if os.path.exists(file_path):
                    result = vcs.process_audio(file_path)
                    if not result['success']:
                        print(f"❌ Error: {result.get('error', 'Unknown')}")
                else:
                    print(f"❌ File not found: {file_path}")
            
            elif command == 'text':
                query = input("Enter your query: ").strip()
                if query:
                    result = vcs.process_text(query, speak_response=True)
                    print(f"\n🎯 Intent: {result['intent']}")
                    print(f"📊 Confidence: {result['confidence']:.2%}")
                    print(f"💬 Response: {result['response']}")
            
            else:
                print("Unknown command. Try 'record', 'text', 'file <path>', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    cli_interface()
