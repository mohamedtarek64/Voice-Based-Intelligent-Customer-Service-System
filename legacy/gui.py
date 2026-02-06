"""
Desktop GUI Application
=======================
Tkinter-based desktop interface for the Voice Customer Service System.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


class VoiceCustomerServiceGUI:
    """
    Desktop GUI for the Voice Customer Service System.
    """
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("🎤 Voice Customer Service System")
        self.root.geometry("800x700")
        self.root.configure(bg='#1a1a2e')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # State
        self.is_recording = False
        self.vcs = None
        
        # Build UI
        self.build_ui()
        
        # Initialize voice service in background
        self.status_var.set("Initializing...")
        threading.Thread(target=self.init_voice_service, daemon=True).start()
    
    def configure_styles(self):
        """Configure ttk styles for dark theme."""
        self.style.configure('TFrame', background='#1a1a2e')
        self.style.configure('TLabel', background='#1a1a2e', foreground='#ffffff',
                            font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'),
                            foreground='#6366f1')
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12),
                            foreground='#94a3b8')
        self.style.configure('Result.TLabel', font=('Segoe UI', 11),
                            foreground='#ffffff', wraplength=700)
        self.style.configure('TButton', font=('Segoe UI', 11, 'bold'),
                            padding=10)
        self.style.configure('Record.TButton', font=('Segoe UI', 14, 'bold'),
                            padding=20)
    
    def build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(header_frame, text="🎤 Voice Customer Service",
                         style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(header_frame, text="AI-Powered Intelligent Support System",
                            style='Subtitle.TLabel')
        subtitle.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(header_frame, textvariable=self.status_var,
                                style='Subtitle.TLabel')
        status_label.pack(pady=(10, 0))
        
        # Recording section
        record_frame = ttk.Frame(main_frame)
        record_frame.pack(fill=tk.X, pady=20)
        
        self.record_btn = tk.Button(
            record_frame,
            text="🎤 Start Recording",
            font=('Segoe UI', 14, 'bold'),
            bg='#6366f1',
            fg='white',
            activebackground='#4f46e5',
            activeforeground='white',
            relief=tk.FLAT,
            padx=40,
            pady=15,
            cursor='hand2',
            command=self.toggle_recording
        )
        self.record_btn.pack()
        
        self.record_status = ttk.Label(record_frame, text="Click to start recording",
                                       style='Subtitle.TLabel')
        self.record_status.pack(pady=(10, 0))
        
        # Text input section
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(text_frame, text="Or type your query:",
                 style='TLabel').pack(anchor=tk.W)
        
        input_container = ttk.Frame(text_frame)
        input_container.pack(fill=tk.X, pady=(5, 0))
        
        self.text_input = tk.Entry(
            input_container,
            font=('Segoe UI', 12),
            bg='#2d2d44',
            fg='white',
            insertbackground='white',
            relief=tk.FLAT
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.text_input.bind('<Return>', lambda e: self.process_text())
        
        send_btn = tk.Button(
            input_container,
            text="Send →",
            font=('Segoe UI', 11, 'bold'),
            bg='#10b981',
            fg='white',
            activebackground='#059669',
            activeforeground='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2',
            command=self.process_text
        )
        send_btn.pack(side=tk.RIGHT)
        
        # Quick queries
        quick_frame = ttk.Frame(main_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 20))
        
        quick_queries = [
            ("Cancel Order", "I want to cancel my order"),
            ("Track Order", "Where is my package?"),
            ("Refund", "I need a refund"),
            ("Complaint", "I have a complaint")
        ]
        
        for label, query in quick_queries:
            btn = tk.Button(
                quick_frame,
                text=label,
                font=('Segoe UI', 9),
                bg='#374151',
                fg='white',
                activebackground='#4b5563',
                activeforeground='white',
                relief=tk.FLAT,
                padx=15,
                pady=5,
                cursor='hand2',
                command=lambda q=query: self.send_quick_query(q)
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Results section
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(results_frame, text="📊 Results",
                 style='TLabel').pack(anchor=tk.W)
        
        # Results text area with scrollbar
        result_container = ttk.Frame(results_frame)
        result_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        scrollbar = ttk.Scrollbar(result_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            result_container,
            font=('Consolas', 11),
            bg='#2d2d44',
            fg='#e2e8f0',
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=15,
            pady=15,
            yscrollcommand=scrollbar.set
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
        
        # Configure text tags for formatting
        self.result_text.tag_config('header', foreground='#6366f1', font=('Consolas', 11, 'bold'))
        self.result_text.tag_config('success', foreground='#10b981')
        self.result_text.tag_config('warning', foreground='#f59e0b')
        self.result_text.tag_config('error', foreground='#ef4444')
        self.result_text.tag_config('intent', foreground='#8b5cf6', font=('Consolas', 12, 'bold'))
        
        # Footer buttons
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        file_btn = tk.Button(
            footer_frame,
            text="📁 Load Audio File",
            font=('Segoe UI', 10),
            bg='#374151',
            fg='white',
            activebackground='#4b5563',
            activeforeground='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.load_audio_file
        )
        file_btn.pack(side=tk.LEFT)
        
        speak_btn = tk.Button(
            footer_frame,
            text="🔊 Speak Response",
            font=('Segoe UI', 10),
            bg='#374151',
            fg='white',
            activebackground='#4b5563',
            activeforeground='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.speak_response
        )
        speak_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        clear_btn = tk.Button(
            footer_frame,
            text="🗑️ Clear",
            font=('Segoe UI', 10),
            bg='#374151',
            fg='white',
            activebackground='#4b5563',
            activeforeground='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.clear_results
        )
        clear_btn.pack(side=tk.RIGHT)
    
    def init_voice_service(self):
        """Initialize the voice service in background."""
        try:
            from src.main import VoiceCustomerService
            self.vcs = VoiceCustomerService(model_dir='models')
            self.status_var.set("✅ System Ready")
            self.log_result("System initialized successfully!", 'success')
        except Exception as e:
            self.status_var.set("⚠️ Initialization Error")
            self.log_result(f"Error: {e}", 'error')
    
    def toggle_recording(self):
        """Toggle audio recording."""
        if self.vcs is None:
            messagebox.showwarning("Not Ready", "System is still initializing...")
            return
        
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.config(text="⏹️ Stop Recording", bg='#ef4444')
            self.record_status.config(text="Recording... Click to stop")
            threading.Thread(target=self.record_and_process, daemon=True).start()
        else:
            self.is_recording = False
            self.record_btn.config(text="🎤 Start Recording", bg='#6366f1')
            self.record_status.config(text="Processing...")
    
    def record_and_process(self):
        """Record audio and process."""
        try:
            from src.audio_recorder import record_audio
            
            # Record for 5 seconds
            audio_path = record_audio(duration=5, output_file="recording.wav")
            
            if audio_path and self.vcs:
                result = self.vcs.process_audio(audio_path, play_response=False)
                self.display_result(result)
        except Exception as e:
            self.log_result(f"Recording error: {e}", 'error')
        finally:
            self.is_recording = False
            self.root.after(0, lambda: self.record_btn.config(
                text="🎤 Start Recording", bg='#6366f1'))
            self.root.after(0, lambda: self.record_status.config(
                text="Click to start recording"))
    
    def process_text(self):
        """Process text input."""
        query = self.text_input.get().strip()
        if not query:
            return
        
        if self.vcs is None:
            messagebox.showwarning("Not Ready", "System is still initializing...")
            return
        
        self.text_input.delete(0, tk.END)
        threading.Thread(target=self._process_text_thread, args=(query,), daemon=True).start()
    
    def _process_text_thread(self, query):
        """Process text in background thread."""
        try:
            result = self.vcs.process_text(query, speak_response=False)
            self.display_result(result, query)
        except Exception as e:
            self.log_result(f"Error: {e}", 'error')
    
    def send_quick_query(self, query):
        """Send a quick query."""
        self.text_input.delete(0, tk.END)
        self.text_input.insert(0, query)
        self.process_text()
    
    def load_audio_file(self):
        """Load and process an audio file."""
        if self.vcs is None:
            messagebox.showwarning("Not Ready", "System is still initializing...")
            return
        
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.m4a"),
                ("All Files", "*.*")
            ]
        )
        
        if filepath:
            threading.Thread(target=self._process_audio_thread, 
                           args=(filepath,), daemon=True).start()
    
    def _process_audio_thread(self, filepath):
        """Process audio file in background."""
        try:
            result = self.vcs.process_audio(filepath, play_response=False)
            self.display_result(result)
        except Exception as e:
            self.log_result(f"Error: {e}", 'error')
    
    def display_result(self, result, query=None):
        """Display processing result."""
        self.result_text.delete(1.0, tk.END)
        
        if not result.get('success', False):
            self.log_result(f"Error: {result.get('error', 'Unknown error')}", 'error')
            return
        
        # Query
        if query or result.get('transcription'):
            self.log_result("📝 Query:", 'header')
            self.log_result(f"   {query or result.get('transcription', '')}\n")
        
        # Intent
        self.log_result("🎯 Intent:", 'header')
        intent = result.get('intent', 'unknown').replace('_', ' ').title()
        self.log_result(f"   {intent}\n", 'intent')
        
        # Confidence
        confidence = result.get('confidence', 0) * 100
        self.log_result("📊 Confidence:", 'header')
        if confidence >= 85:
            self.log_result(f"   {confidence:.1f}% (High)\n", 'success')
        elif confidence >= 60:
            self.log_result(f"   {confidence:.1f}% (Medium)\n", 'warning')
        else:
            self.log_result(f"   {confidence:.1f}% (Low)\n", 'error')
        
        # Action
        action = result.get('action', 'unknown')
        self.log_result("⚡ Action:", 'header')
        self.log_result(f"   {action.replace('_', ' ').title()}\n")
        
        # Response
        self.log_result("💬 Response:", 'header')
        self.log_result(f"   {result.get('response', '')}\n")
        
        # Top intents
        if result.get('top_intents'):
            self.log_result("\n📈 Top Predictions:", 'header')
            for item in result['top_intents']:
                intent_name = item['intent'].replace('_', ' ').title()
                score = item['confidence'] * 100
                self.log_result(f"   • {intent_name}: {score:.1f}%")
        
        # Store response for TTS
        self.last_response = result.get('response', '')
    
    def log_result(self, text, tag=None):
        """Log text to results area."""
        self.result_text.insert(tk.END, text + '\n', tag)
        self.result_text.see(tk.END)
    
    def speak_response(self):
        """Speak the last response."""
        if hasattr(self, 'last_response') and self.last_response:
            if self.vcs and self.vcs.tts:
                threading.Thread(
                    target=lambda: self.vcs.tts.speak(self.last_response),
                    daemon=True
                ).start()
    
    def clear_results(self):
        """Clear results area."""
        self.result_text.delete(1.0, tk.END)
        self.last_response = ''
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = VoiceCustomerServiceGUI()
    app.run()


if __name__ == "__main__":
    main()
