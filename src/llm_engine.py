
import os
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv
import socket

load_dotenv()

class LLMEngine:
    """
    LLM Engine using the stable google-generativeai with network diagnostics.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        # Network Diagnostic
        print("🔍 Checking network connectivity...")
        try:
            # Try to resolve Google's DNS to see if Python can see the internet
            socket.gethostbyname("generativelanguage.googleapis.com")
            print("✅ DNS resolution successful")
        except Exception as e:
            print(f"❌ DNS resolution failed: {e}")
            print("💡 Tip: Try to disable any VPN or Proxy, or check Firewall settings.")

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # 'gemini-flash-latest' is the most stable name for the high-quota free tier
                self.model_name = 'gemini-flash-latest'
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✓ Gemini LLM Engine initialized with {self.model_name}")
            except Exception as e:
                print(f"❌ Init Error: {e}")

    def generate_response(self, query: str, context: str = "") -> str:
        if not self.model:
            return "I'm sorry, my AI brain is not connected."
            
        try:
            # We use a very simple prompt to save tokens (Quota preservation)
            prompt = f"User: {query}\nAssistant: Respond concisely."
            
            response = self.model.generate_content(
                prompt,
                safety_settings={
                    'HATE': 'BLOCK_NONE',
                    'HARASSMENT': 'BLOCK_NONE',
                    'SEXUAL': 'BLOCK_NONE',
                    'DANGEROUS': 'BLOCK_NONE'
                }
            )
            return response.text.strip()
                
        except Exception as e:
            print(f"❌ LLM Runtime Error with {self.model_name}: {e}")
            # If 429 Quota or 404 Not Found, try the Pro version
            if "429" in str(e) or "not found" in str(e).lower():
                try:
                    print("🔄 Quota full or Model not found. Trying fallback 'gemini-pro-latest'...")
                    self.model_name = 'gemini-pro-latest'
                    self.model = genai.GenerativeModel(self.model_name)
                    # Simple prompt for fallback
                    response = self.model.generate_content(query)
                    return response.text.strip()
                except Exception as e2:
                    print(f"❌ Final Fallback Error: {e2}")
            return "I'm sorry, I'm a bit overwhelmed right now. Please try again in 30 seconds."

    def is_available(self) -> bool:
        return self.model is not None
