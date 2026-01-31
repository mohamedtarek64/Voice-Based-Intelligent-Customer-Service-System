"""
Audio Recording Module
======================
Handles audio recording from microphone.
"""

import os
import time
from typing import Optional
import numpy as np


def record_audio(duration: int = 5, sample_rate: int = 16000, 
                 output_file: str = "recording.wav") -> str:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate (16000 recommended for speech)
        output_file: Output WAV file path
        
    Returns:
        Path to recorded audio file
    """
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        
        print(f"🎤 Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()  # Wait until recording is finished
        
        print("✅ Recording complete!")
        
        # Save to WAV file
        wavfile.write(output_file, sample_rate, audio_data)
        print(f"💾 Audio saved to: {output_file}")
        
        return output_file
        
    except ImportError as e:
        print(f"Error: Required libraries not installed. Run: pip install sounddevice scipy")
        print(f"Details: {e}")
        return ""
    except Exception as e:
        print(f"Error recording audio: {e}")
        return ""


def record_audio_with_countdown(duration: int = 5, countdown: int = 3,
                                 sample_rate: int = 16000,
                                 output_file: str = "recording.wav") -> str:
    """
    Record audio with countdown before starting.
    
    Args:
        duration: Recording duration in seconds
        countdown: Countdown duration before recording
        sample_rate: Audio sample rate
        output_file: Output WAV file path
        
    Returns:
        Path to recorded audio file
    """
    print("Get ready to speak...")
    
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    return record_audio(duration, sample_rate, output_file)


def record_until_silence(max_duration: int = 30, silence_threshold: float = 0.01,
                          silence_duration: float = 1.5, sample_rate: int = 16000,
                          output_file: str = "recording.wav") -> str:
    """
    Record audio until silence is detected.
    
    Args:
        max_duration: Maximum recording duration
        silence_threshold: RMS threshold for silence detection
        silence_duration: Duration of silence to stop recording
        sample_rate: Audio sample rate
        output_file: Output WAV file path
        
    Returns:
        Path to recorded audio file
    """
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        
        print("🎤 Recording... (will stop on silence)")
        print("Speak now!")
        
        chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        recorded_chunks = []
        silent_chunks = 0
        max_chunks = int(max_duration * 10)  # Convert to 100ms chunks
        silent_chunks_threshold = int(silence_duration * 10)
        
        for i in range(max_chunks):
            chunk = sd.rec(chunk_size, samplerate=sample_rate, 
                          channels=1, dtype=np.float32)
            sd.wait()
            recorded_chunks.append(chunk)
            
            # Check for silence
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= silent_chunks_threshold:
                    print("🔇 Silence detected, stopping...")
                    break
            else:
                silent_chunks = 0
        
        print("✅ Recording complete!")
        
        # Combine chunks and save
        audio_data = np.concatenate(recorded_chunks)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(output_file, sample_rate, audio_int16)
        
        print(f"💾 Audio saved to: {output_file}")
        return output_file
        
    except ImportError as e:
        print(f"Error: Required libraries not installed")
        return ""
    except Exception as e:
        print(f"Error recording audio: {e}")
        return ""


def check_microphone() -> bool:
    """
    Check if microphone is available and working.
    
    Returns:
        True if microphone is available
    """
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("❌ No microphone detected!")
            return False
        
        print("✅ Microphone available:")
        for d in input_devices:
            print(f"   - {d['name']}")
        
        # Quick test recording
        print("\n🎤 Testing microphone...")
        test_audio = sd.rec(int(0.5 * 16000), samplerate=16000, 
                           channels=1, dtype=np.int16)
        sd.wait()
        
        if np.max(np.abs(test_audio)) > 0:
            print("✅ Microphone is working!")
            return True
        else:
            print("⚠️ Microphone detected but no audio received")
            return False
            
    except Exception as e:
        print(f"❌ Error checking microphone: {e}")
        return False


if __name__ == "__main__":
    print("Audio Recording Test")
    print("=" * 40)
    
    # Check microphone
    if check_microphone():
        print("\nStarting test recording...")
        audio_file = record_audio_with_countdown(
            duration=5,
            countdown=3,
            output_file="test_recording.wav"
        )
        
        if audio_file:
            print(f"\n✅ Test recording saved: {audio_file}")
        else:
            print("\n❌ Test recording failed")
