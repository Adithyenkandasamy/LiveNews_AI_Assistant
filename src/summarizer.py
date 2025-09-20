#!/usr/bin/env python3
"""
Enhanced summarizer with multiple LLM options
"""

import requests
from transformers import pipeline
import logging

class EnhancedSummarizer:
    def __init__(self, model_type="gemini", model_config=None):
        self.model_type = model_type
        if model_type == "gemini":
            self.gemini_client = model_config
        else:
            self.model_name = model_config
        
        # Initialize based on type
        if model_type == "transformers":
            self.summarizer = pipeline("summarization", model="google/pegasus-xsum")
        
    def summarize_article(self, content: str, max_length: int = 150) -> str:
        """Summarize article content using selected model"""
        
        if self.model_type == "gemini":
            return self._summarize_with_gemini(content, max_length)
        elif self.model_type == "transformers":
            return self._transformers_summarize(content, max_length)
        
    def _summarize_with_gemini(self, text, max_length=150):
        """Summarize using Gemini API with better content handling"""
        if not self.gemini_client or not self.gemini_client.available:
            return "Gemini API not available for summarization."
            
        # Check for content availability
        if not text or len(text.strip()) < 20:
            return "No meaningful content available to summarize."
            
        prompt = f"""Please provide a comprehensive summary of the following news article in approximately {max_length} words. If the content is insufficient or unclear, please indicate that:

{text}

Summary:"""
        
        try:
            summary = self.gemini_client.generate_response(
                prompt,
                max_tokens=max_length + 100,  # More generous buffer
                temperature=0.3
            )
            
            if not summary or len(summary.strip()) < 10:
                return "Unable to generate meaningful summary from the provided content."
                
            return summary.strip()
        except Exception as e:
            logging.error(f"Gemini summarization failed: {e}")
            return f"Summarization error: {str(e)}"
    
    
    def _transformers_summarize(self, content: str, max_length: int) -> str:
        """Use Hugging Face transformers for summarization"""
        try:
            # Check for content availability
            if not content or len(content.strip()) < 20:
                return "No content available to summarize"
                
            # Use larger content window for transformers
            if len(content) > 2048:  # Increased from 1024
                content = self._smart_content_selection(content, 2048)
                
            summary = self.summarizer(content, max_length=max_length, min_length=30)
            return summary[0]['summary_text']
            
        except Exception as e:
            logging.error(f"Transformers summarization failed: {e}")
            return "Summary unavailable"
    
    def process_full_article(self, article_content: str) -> dict:
        """Process full article content intelligently without artificial limits"""
        
        # Validate content exists and is meaningful
        if not article_content or len(article_content.strip()) < 20:
            return {
                'full_content': '[No content available]',
                'summary': 'No content available to summarize',
                'word_count': 0,
                'char_count': 0,
                'content_status': 'missing'
            }
        
        # Use much larger content windows based on model capabilities
        if self.model_type == "gemini":
            max_input = 8000  # Gemini has very large context window
        else:
            max_input = 2000  # Even transformers can handle more
            
        # More intelligent content handling - preserve full content when possible
        if len(article_content) > max_input:
            # Instead of simple truncation, try to keep the most important parts
            processed_content = self._smart_content_selection(article_content, max_input)
        else:
            processed_content = article_content
            
        return {
            'full_content': processed_content,
            'summary': self.summarize_article(processed_content),
            'word_count': len(processed_content.split()),
            'char_count': len(processed_content),
            'content_status': 'full' if len(article_content) <= max_input else 'truncated',
            'original_length': len(article_content)
        }
    
    def _smart_content_selection(self, content: str, max_chars: int) -> str:
        """Intelligently select the most important parts of content"""
        if len(content) <= max_chars:
            return content
            
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) == 1:
            # If no paragraph breaks, split by sentences
            sentences = content.split('. ')
            if len(sentences) <= 3:
                # Just truncate if very few sentences
                return content[:max_chars] + "..."
            
            # Take first few and last few sentences
            first_part = '. '.join(sentences[:2]) + '.'
            last_part = '. '.join(sentences[-1:]) + '.'
            middle_indicator = "\n\n[... content continues ...]\n\n"
            
            combined = first_part + middle_indicator + last_part
            if len(combined) <= max_chars:
                return combined
            else:
                return content[:max_chars] + "..."
        
        # Take first paragraph, middle summary, and last paragraph if possible
        first_para = paragraphs[0]
        last_para = paragraphs[-1] if len(paragraphs) > 1 else ""
        middle_indicator = "\n\n[... article continues ...]\n\n" if len(paragraphs) > 2 else "\n\n"
        
        combined = first_para + middle_indicator + last_para
        
        if len(combined) <= max_chars:
            return combined
        else:
            # If still too long, just take the beginning
            return content[:max_chars] + "..."

# Example usage
if __name__ == "__main__":
    # Test with different models
    # Import gemini client for testing
    from gemini_client import GeminiClient
    gemini_client = GeminiClient()
    
    models_to_test = [
        ("gemini", gemini_client),
        ("transformers", "google/pegasus-xsum")
    ]
    
    sample_article = """
    OpenAI has announced a major breakthrough in artificial intelligence with the release of GPT-5, 
    which demonstrates unprecedented capabilities in reasoning and problem-solving. The new model 
    shows significant improvements over previous versions, particularly in mathematical reasoning 
    and code generation. Industry experts believe this could revolutionize how AI is used in 
    scientific research and software development.
    """
    
    for model_type, model_config in models_to_test:
        print(f"\n=== Testing {model_type} ===")
        if model_type == "gemini":
            summarizer = EnhancedSummarizer(model_type, model_config)
        else:
            summarizer = EnhancedSummarizer(model_type, model_config)
        result = summarizer.process_full_article(sample_article)
        print(f"Summary: {result['summary']}")
        print(f"Processed: {result['char_count']} chars, {result['word_count']} words")
