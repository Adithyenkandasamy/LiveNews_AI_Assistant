"""
Google Gemini API Client for LiveNews AI Assistant
Advanced AI client using Google's Gemini 2.0 Flash model
"""

import google.generativeai as genai
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            self.logger.error("GEMINI_API_KEY not found in environment variables")
            self.available = False
            return
            
        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.available = True
            self.logger.info(f"✅ Gemini {self.model_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            self.available = False
    
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate response using Gemini API"""
        if not self.available:
            return "Gemini API not available. Please check your API key configuration."
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                self.logger.warning("Gemini returned empty response")
                return "I couldn't generate a response at this time."
                
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        if not self.available:
            return False
            
        try:
            response = self.generate_response("Hello", max_tokens=10)
            return len(response) > 0 and "error" not in response.lower()
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'available': self.available,
            'api_configured': self.api_key is not None
        }

# Global instance for easy import
gemini_client = GeminiClient()
