
import os
import google.generativeai as genai
from typing import Dict, Any

class LLMEngine:
    """
    LLM Engine using Google Gemini to handle general queries and any topic.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini model.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use gemini-1.5-flash for speed and efficiency
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ Gemini LLM Engine initialized")
        else:
            print("⚠ Gemini API Key not found. Generative AI features will be disabled.")

    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate a response for a general query.
        """
        if not self.model:
            return "I'm sorry, my Generative AI brain is not connected right now. Please provide an API key."
            
        try:
            # System prompt to keep it concise and friendly
            prompt = f"You are a friendly and helpful AI Voice Assistant. Respond concisely to the user's query.\nContext: {context}\nUser: {query}"
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "I encountered an error while thinking about that. Could you try asking again?"

    def is_available(self) -> bool:
        return self.model is not None
