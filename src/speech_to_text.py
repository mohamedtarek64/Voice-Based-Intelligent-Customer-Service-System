"""
Speech-to-Text Module
=====================
Handles audio transcription using OpenAI Whisper.
"""

import os
import warnings
from typing import Optional, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")


class SpeechToText:
    """
    Speech-to-Text converter using OpenAI Whisper.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize the STT module.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print("Whisper model loaded successfully!")
        except ImportError:
            print("Warning: Whisper not installed. Using fallback STT.")
            print("Install with: pip install openai-whisper")
            self.model = None
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe(self, audio_file_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (e.g., 'en', 'ar')
            
        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(audio_file_path):
            return {
                'text': '',
                'success': False,
                'error': f'Audio file not found: {audio_file_path}'
            }
        
        if self.model is None:
            return {
                'text': '',
                'success': False,
                'error': 'Whisper model not loaded'
            }
        
        try:
            print(f"Transcribing audio: {audio_file_path}")
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                fp16=False  # Use FP32 for CPU compatibility
            )
            
            transcribed_text = result["text"].strip()
            print(f"Transcription: {transcribed_text}")
            
            return {
                'text': transcribed_text,
                'success': True,
                'language': result.get('language', language),
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e)
            }
    
    def transcribe_with_timestamps(self, audio_file_path: str, 
                                    language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with transcription and timestamps
        """
        result = self.transcribe(audio_file_path, language)
        
        if not result['success']:
            return result
        
        # Extract timestamps from segments
        timestamps = []
        for segment in result.get('segments', []):
            timestamps.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
        
        result['timestamps'] = timestamps
        return result


class SpeechToTextFallback:
    """
    Fallback STT using speech_recognition library.
    Used when Whisper is not available.
    """
    
    def __init__(self):
        """Initialize the fallback STT."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            print("Using speech_recognition as fallback STT")
        except ImportError:
            print("Warning: speech_recognition not installed")
            self.recognizer = None
    
    def transcribe(self, audio_file_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe audio using Google Speech Recognition.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with transcription results
        """
        if self.recognizer is None:
            return {
                'text': '',
                'success': False,
                'error': 'speech_recognition not installed'
            }
        
        try:
            import speech_recognition as sr
            
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio, language=language)
            
            return {
                'text': text,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e)
            }


def create_stt(use_whisper: bool = True, model_size: str = "base") -> SpeechToText:
    """
    Factory function to create appropriate STT instance.
    
    Args:
        use_whisper: Whether to use Whisper (True) or fallback (False)
        model_size: Whisper model size
        
    Returns:
        SpeechToText instance
    """
    if use_whisper:
        stt = SpeechToText(model_size=model_size)
        if stt.model is not None:
            return stt
    
    print("Falling back to speech_recognition...")
    return SpeechToTextFallback()


if __name__ == "__main__":
    # Test STT
    stt = create_stt(use_whisper=True, model_size="base")
    
    test_audio = "data/audio_samples/test.wav"
    if os.path.exists(test_audio):
        result = stt.transcribe(test_audio)
        print(f"Result: {result}")
    else:
        print(f"Test audio not found: {test_audio}")
        print("Record a test audio file first.")
