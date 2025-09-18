#!/usr/bin/env python3
"""
Enhanced summarizer with multiple LLM options
"""

import requests
from transformers import pipeline
import logging

class EnhancedSummarizer:
    def __init__(self, model_type="ollama", model_name="llama3.2:3b"):
        self.model_type = model_type
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Initialize based on type
        if model_type == "transformers":
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def summarize_article(self, content: str, max_length: int = 150) -> str:
        """Summarize article content using selected model"""
        
        if self.model_type == "ollama":
            return self._ollama_summarize(content, max_length)
        elif self.model_type == "transformers":
            return self._transformers_summarize(content, max_length)
        
    def _ollama_summarize(self, content: str, max_length: int) -> str:
        """Use Ollama models for summarization"""
        prompt = f"""Summarize this news article in {max_length} words or less. Focus on key facts and main points:

Article:
{content}

Summary:"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_length + 50
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return "Summary unavailable"
                
        except Exception as e:
            logging.error(f"Ollama summarization failed: {e}")
            return "Summary unavailable"
    
    def _transformers_summarize(self, content: str, max_length: int) -> str:
        """Use Hugging Face transformers for summarization"""
        try:
            # Truncate if too long for model
            if len(content) > 1024:
                content = content[:1024]
                
            summary = self.summarizer(content, max_length=max_length, min_length=30)
            return summary[0]['summary_text']
            
        except Exception as e:
            logging.error(f"Transformers summarization failed: {e}")
            return "Summary unavailable"
    
    def process_full_article(self, article_content: str) -> dict:
        """Process full article content intelligently"""
        
        # Use more content based on model capabilities
        if self.model_type == "ollama" and "llama3.2" in self.model_name:
            max_input = 2000  # 32K context window
        elif self.model_type == "ollama":
            max_input = 1500  # Standard models
        else:
            max_input = 1000  # Transformers
            
        # Smart truncation
        if len(article_content) > max_input:
            processed_content = article_content[:max_input] + "..."
        else:
            processed_content = article_content
            
        return {
            'full_content': processed_content,
            'summary': self.summarize_article(processed_content),
            'word_count': len(processed_content.split()),
            'char_count': len(processed_content)
        }

# Example usage
if __name__ == "__main__":
    # Test with different models
    models_to_test = [
        ("ollama", "llama3.2:3b"),
        ("ollama", "mistral:7b"),
        ("transformers", "facebook/bart-large-cnn")
    ]
    
    sample_article = """
    OpenAI has announced a major breakthrough in artificial intelligence with the release of GPT-5, 
    which demonstrates unprecedented capabilities in reasoning and problem-solving. The new model 
    shows significant improvements over previous versions, particularly in mathematical reasoning 
    and code generation. Industry experts believe this could revolutionize how AI is used in 
    scientific research and software development.
    """
    
    for model_type, model_name in models_to_test:
        print(f"\n--- Testing {model_type}: {model_name} ---")
        summarizer = EnhancedSummarizer(model_type, model_name)
        result = summarizer.process_full_article(sample_article)
        print(f"Summary: {result['summary']}")
        print(f"Processed: {result['char_count']} chars, {result['word_count']} words")
