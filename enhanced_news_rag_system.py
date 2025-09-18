#!/usr/bin/env python3
"""
Enhanced News RAG System with Llama 2/3 Support
Supports multiple AI models including Llama 2, Llama 3, and original BART/SentenceTransformer
"""

import os
import logging
import psycopg2
import feedparser
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from dotenv import load_dotenv

# AI/ML imports
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    model_type: str = "bart"  # "bart", "llama2", "llama3", "openai"
    model_name: str = "facebook/bart-large-cnn"
    embedder_model: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # "auto", "cpu", "cuda"
    max_length: int = 512
    temperature: float = 0.7
    use_gpu: bool = True

@dataclass
class CleanupConfig:
    """Configuration for cleanup policies"""
    retention_days: int = 30
    max_articles_per_category: int = 1000

class EnhancedNewsRAGSystem:
    """Enhanced News RAG System with multiple AI model support"""
    
    def __init__(self, db_config: Dict[str, str], model_config: ModelConfig = None, cleanup_config: CleanupConfig = None):
        self.db_config = db_config
        self.model_config = model_config or ModelConfig()
        self.cleanup_config = cleanup_config or CleanupConfig()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize AI models
        self.logger.info("🤖 Loading AI models...")
        self.init_models()
        
        # RSS feeds and Reddit sources
        self.setup_news_sources()
        
        # Category keywords for classification
        self.setup_categories()
        
        # Initialize database
        self.init_database()
        
        self.logger.info("✅ Enhanced News RAG System initialized successfully")

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/enhanced_news_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_models(self):
        """Initialize AI models based on configuration"""
        # Determine device
        if self.model_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() and self.model_config.use_gpu else "cpu"
        else:
            self.device = self.model_config.device
        
        self.logger.info(f"Use pytorch device_name: {self.device}")
        
        # Initialize embedder (always use SentenceTransformer for now)
        self.logger.info(f"Load pretrained SentenceTransformer: {self.model_config.embedder_model}")
        self.embedder = SentenceTransformer(self.model_config.embedder_model)
        
        # Initialize main model based on type
        if self.model_config.model_type == "llama2":
            self.init_llama2()
        elif self.model_config.model_type == "llama3":
            self.init_llama3()
        elif self.model_config.model_type == "openai":
            self.init_openai()
        else:  # Default to BART
            self.init_bart()

    def init_bart(self):
        """Initialize BART model"""
        self.logger.info("Loading BART model for summarization...")
        self.summarizer = pipeline(
            "summarization", 
            model=self.model_config.model_name,
            device=0 if self.device == "cuda" else -1
        )
        self.qa_model = None  # BART doesn't do Q&A directly

    def init_llama2(self):
        """Initialize Llama 2 model"""
        self.logger.info("Loading Llama 2 model...")
        model_name = self.model_config.model_name or "meta-llama/Llama-2-7b-chat-hf"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.qa_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.tokenizer,
                max_length=self.model_config.max_length,
                temperature=self.model_config.temperature,
                do_sample=True,
                device=0 if self.device == "cuda" else -1
            )
            
            self.summarizer = self.qa_pipeline  # Use same model for summarization
            self.logger.info("✅ Llama 2 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Llama 2: {e}")
            self.logger.info("Falling back to BART...")
            self.model_config.model_type = "bart"
            self.init_bart()

    def init_llama3(self):
        """Initialize Llama 3 model"""
        self.logger.info("Loading Llama 3 model...")
        model_name = self.model_config.model_name or "meta-llama/Meta-Llama-3-8B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.qa_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.tokenizer,
                max_length=self.model_config.max_length,
                temperature=self.model_config.temperature,
                do_sample=True,
                device=0 if self.device == "cuda" else -1
            )
            
            self.summarizer = self.qa_pipeline  # Use same model for summarization
            self.logger.info("✅ Llama 3 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Llama 3: {e}")
            self.logger.info("Falling back to BART...")
            self.model_config.model_type = "bart"
            self.init_bart()

    def init_openai(self):
        """Initialize OpenAI API"""
        self.logger.info("Setting up OpenAI API...")
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            self.openai_client = openai
            self.summarizer = None  # Will use OpenAI API
            self.qa_model = None
            self.logger.info("✅ OpenAI API configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenAI: {e}")
            self.logger.info("Falling back to BART...")
            self.model_config.model_type = "bart"
            self.init_bart()

    def setup_news_sources(self):
        """Setup RSS feeds and Reddit sources"""
        self.rss_feeds = {
            'BBC_World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC_Business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'BBC_Technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'CNN_World': 'http://rss.cnn.com/rss/edition.rss',
            'CNN_Business': 'http://rss.cnn.com/rss/money_latest.rss',
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Reuters_World': 'https://www.reuters.com/rssfeed/worldnews',
        }
        
        self.reddit_sources = [
            'worldnews', 'technology', 'business', 'politics'
        ]

    def setup_categories(self):
        """Setup category classification keywords"""
        self.category_keywords = {
            'Politics': ['election', 'government', 'president', 'minister', 'parliament', 'congress', 'senate', 'vote', 'policy', 'political'],
            'Technology': ['AI', 'artificial intelligence', 'tech', 'software', 'hardware', 'startup', 'innovation', 'digital', 'cyber', 'data'],
            'Business': ['economy', 'market', 'stock', 'finance', 'investment', 'company', 'corporate', 'trade', 'economic', 'financial'],
            'Sports': ['football', 'basketball', 'soccer', 'tennis', 'olympics', 'championship', 'team', 'player', 'match', 'game'],
            'Entertainment': ['movie', 'music', 'celebrity', 'film', 'actor', 'actress', 'entertainment', 'show', 'concert', 'album'],
            'General': []
        }

    def summarize_with_llama(self, text: str, max_length: int = 100) -> str:
        """Summarize text using Llama model"""
        prompt = f"""Please provide a concise summary of the following news article in 1-2 sentences:

Article: {text}

Summary:"""
        
        try:
            response = self.qa_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract summary from response
            full_response = response[0]['generated_text']
            summary = full_response.split("Summary:")[-1].strip()
            
            # Clean up the summary
            summary = summary.split('\n')[0].strip()
            return summary if summary else text[:200] + "..."
            
        except Exception as e:
            self.logger.error(f"Llama summarization failed: {e}")
            return text[:200] + "..."

    def summarize_with_openai(self, text: str, max_length: int = 100) -> str:
        """Summarize text using OpenAI API"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a news summarization assistant. Provide concise 1-2 sentence summaries."},
                    {"role": "user", "content": f"Summarize this news article: {text}"}
                ],
                max_tokens=max_length,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI summarization failed: {e}")
            return text[:200] + "..."

    def generate_summary(self, text: str) -> str:
        """Generate summary using the configured model"""
        if self.model_config.model_type in ["llama2", "llama3"]:
            return self.summarize_with_llama(text)
        elif self.model_config.model_type == "openai":
            return self.summarize_with_openai(text)
        else:
            # Use BART
            try:
                summary = self.summarizer(text, max_length=60, min_length=10, do_sample=False)
                return summary[0]['summary_text']
            except Exception as e:
                self.logger.error(f"BART summarization failed: {e}")
                return text[:200] + "..."

    def answer_question_with_llama(self, question: str, context: str) -> str:
        """Answer question using Llama model"""
        prompt = f"""Based on the following news articles, please answer the question. Be concise and factual.

News Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.qa_pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract answer from response
            full_response = response[0]['generated_text']
            answer = full_response.split("Answer:")[-1].strip()
            
            # Clean up the answer
            answer = answer.split('\n')[0].strip()
            return answer if answer else "I couldn't find relevant information to answer your question."
            
        except Exception as e:
            self.logger.error(f"Llama Q&A failed: {e}")
            return "Sorry, I encountered an error while processing your question."

    def answer_question_with_openai(self, question: str, context: str) -> str:
        """Answer question using OpenAI API"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful news assistant. Answer questions based on the provided news context. Be concise and factual."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI Q&A failed: {e}")
            return "Sorry, I encountered an error while processing your question."

    # ... (rest of the methods remain the same as the original system)
    # I'll include the key methods needed for functionality

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def init_database(self):
        """Initialize database schema"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    url TEXT UNIQUE,
                    source TEXT,
                    category TEXT,
                    published_date TIMESTAMP,
                    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding FLOAT8[],
                    keywords TEXT[],
                    sentiment TEXT,
                    sentiment_score FLOAT,
                    importance_score FLOAT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles(published_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✅ Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
            raise

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer question using RAG with enhanced models"""
        start_time = datetime.now()
        
        # Search for relevant articles
        relevant_articles = self.search_articles(question, top_k)
        
        if not relevant_articles:
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'response_time': f"{(datetime.now() - start_time).total_seconds() * 1000:.0f}ms"
            }
        
        # Prepare context
        context = "\n\n".join([
            f"Title: {article['title']}\nSummary: {article['summary']}\nCategory: {article['category']}"
            for article in relevant_articles
        ])
        
        # Generate answer based on model type
        if self.model_config.model_type in ["llama2", "llama3"]:
            answer = self.answer_question_with_llama(question, context)
        elif self.model_config.model_type == "openai":
            answer = self.answer_question_with_openai(question, context)
        else:
            # Fallback to simple context-based answer
            answer = f"Based on recent news: {relevant_articles[0]['summary']}"
        
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'answer': answer,
            'sources': relevant_articles,
            'response_time': f"{response_time:.0f}ms",
            'articles_used': len(relevant_articles)
        }

def main():
    """Main function with model selection"""
    print("🚀 Enhanced Live News AI Assistant")
    print("==================================================")
    
    # Model selection
    print("\nSelect AI Model:")
    print("1. BART (Fast, CPU-friendly)")
    print("2. Llama 2 (Better reasoning, requires GPU)")
    print("3. Llama 3 (Best performance, requires GPU)")
    print("4. OpenAI GPT (Requires API key)")
    
    choice = input("Enter choice (1-4, default=1): ").strip() or "1"
    
    # Configure model
    if choice == "2":
        model_config = ModelConfig(
            model_type="llama2",
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )
    elif choice == "3":
        model_config = ModelConfig(
            model_type="llama3",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        )
    elif choice == "4":
        model_config = ModelConfig(
            model_type="openai",
            model_name="gpt-3.5-turbo"
        )
    else:
        model_config = ModelConfig(
            model_type="bart",
            model_name="facebook/bart-large-cnn"
        )
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'livenews'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    # Initialize system
    try:
        news_system = EnhancedNewsRAGSystem(db_config, model_config)
        print(f"\n✅ System initialized with {model_config.model_type.upper()} model!")
        
        # Interactive mode (same as before)
        # ... rest of interactive code
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
