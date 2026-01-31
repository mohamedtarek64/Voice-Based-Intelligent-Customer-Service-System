"""
Text-to-Speech Module
=====================
Handles speech synthesis using gTTS and pyttsx3.
"""

import os
import platform
from typing import Optional


class TextToSpeech:
    """
    Text-to-Speech converter using gTTS (Google Text-to-Speech).
    """
    
    def __init__(self, language: str = 'en', slow: bool = False):
        """
        Initialize the TTS module.
        
        Args:
            language: Language code ('en', 'ar', etc.)
            slow: Whether to speak slowly
        """
        self.language = language
        self.slow = slow
        self._check_gtts()
    
    def _check_gtts(self):
        """Check if gTTS is available."""
        try:
            from gtts import gTTS
            self.gtts_available = True
        except ImportError:
            print("Warning: gTTS not installed. Install with: pip install gTTS")
            self.gtts_available = False
    
    def speak(self, text: str, output_file: str = "response.mp3", 
              play_audio: bool = True) -> str:
        """
        Convert text to speech and optionally play it.
        
        Args:
            text: Text to convert to speech
            output_file: Output audio file path
            play_audio: Whether to play the audio
            
        Returns:
            Path to the generated audio file
        """
        if not text:
            print("No text to speak")
            return ""
        
        if not self.gtts_available:
            print("gTTS not available. Cannot generate speech.")
            return ""
        
        try:
            from gtts import gTTS
            
            print(f"Generating speech: '{text[:50]}...'")
            
            # Generate speech
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(output_file)
            print(f"Audio saved to: {output_file}")
            
            # Play audio
            if play_audio:
                self._play_audio(output_file)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return ""
    
    def _play_audio(self, audio_file: str):
        """
        Play audio file using system default player.
        
        Args:
            audio_file: Path to audio file
        """
        system = platform.system()
        
        try:
            if system == 'Windows':
                os.system(f'start "" "{audio_file}"')
            elif system == 'Darwin':  # macOS
                os.system(f'afplay "{audio_file}"')
            elif system == 'Linux':
                # Try different players
                players = ['mpg123', 'mpg321', 'ffplay', 'aplay']
                for player in players:
                    if os.system(f'which {player} > /dev/null 2>&1') == 0:
                        os.system(f'{player} "{audio_file}" > /dev/null 2>&1')
                        break
            print("Audio playback started")
        except Exception as e:
            print(f"Could not play audio: {e}")


class TextToSpeechOffline:
    """
    Offline Text-to-Speech using pyttsx3.
    Works without internet connection.
    """
    
    def __init__(self, rate: int = 150, voice_index: int = 0):
        """
        Initialize offline TTS.
        
        Args:
            rate: Speech rate (words per minute)
            voice_index: Index of voice to use
        """
        self.rate = rate
        self.voice_index = voice_index
        self.engine = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
                print(f"Using voice: {voices[self.voice_index].name}")
            
            print("pyttsx3 engine initialized")
            
        except ImportError:
            print("Warning: pyttsx3 not installed. Install with: pip install pyttsx3")
            self.engine = None
        except Exception as e:
            print(f"Error initializing pyttsx3: {e}")
            self.engine = None
    
    def speak(self, text: str, output_file: Optional[str] = None) -> bool:
        """
        Speak text directly or save to file.
        
        Args:
            text: Text to speak
            output_file: Optional file to save audio
            
        Returns:
            True if successful
        """
        if not text:
            return False
        
        if self.engine is None:
            print("pyttsx3 engine not available")
            return False
        
        try:
            if output_file:
                # Save to file
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                print(f"Audio saved to: {output_file}")
            else:
                # Speak directly
                print(f"Speaking: '{text[:50]}...'")
                self.engine.say(text)
                self.engine.runAndWait()
            
            return True
            
        except Exception as e:
            print(f"Error speaking: {e}")
            return False
    
    def list_voices(self):
        """List available voices."""
        if self.engine is None:
            return []
        
        voices = self.engine.getProperty('voices')
        voice_list = []
        
        for i, voice in enumerate(voices):
            voice_info = {
                'index': i,
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages
            }
            voice_list.append(voice_info)
            print(f"[{i}] {voice.name}")
        
        return voice_list


def create_tts(use_gtts: bool = True, language: str = 'en') -> TextToSpeech:
    """
    Factory function to create appropriate TTS instance.
    
    Args:
        use_gtts: Whether to use gTTS (True) or pyttsx3 (False)
        language: Language code
        
    Returns:
        TTS instance
    """
    if use_gtts:
        tts = TextToSpeech(language=language)
        if tts.gtts_available:
            return tts
    
    print("Using offline TTS (pyttsx3)...")
    return TextToSpeechOffline()


if __name__ == "__main__":
    # Test TTS
    print("Testing Text-to-Speech...")
    
    # Test gTTS
    tts = TextToSpeech(language='en')
    tts.speak("Hello! Welcome to the Voice Customer Service System.", 
              output_file="test_response.mp3", play_audio=False)
    
    # Test offline TTS
    print("\nTesting offline TTS...")
    offline_tts = TextToSpeechOffline()
    offline_tts.list_voices()
    offline_tts.speak("This is a test of the offline text to speech system.")
