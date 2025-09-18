#!/usr/bin/env python3
"""
Real-Time News RAG System with Nemotron-mini:4b
Features: 1-minute collection, real-time chatbot, daily briefings
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
import threading
import time
from dotenv import load_dotenv

# AI/ML imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

@dataclass
class NemotronConfig:
    model_name: str = "nvidia/nemotron-mini-4b-instruct"
    device: str = "auto"
    max_length: int = 1024
    temperature: float = 0.7
    collection_interval: int = 60  # 1 minute

class RealTimeNewsRAG:
    def __init__(self, db_config: Dict[str, str], nemotron_config: NemotronConfig = None):
        self.db_config = db_config
        self.nemotron_config = nemotron_config or NemotronConfig()
        
        self.setup_logging()
        self.init_models()
        self.setup_news_sources()
        self.init_database()
        
        self.collection_running = False
        self.collection_thread = None
        
        self.logger.info("✅ Real-Time News RAG System initialized")

    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/nemotron_news_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Nemotron
        try:
            self.logger.info("Loading Nemotron-mini:4b...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.nemotron_config.model_name)
            self.nemotron_model = AutoModelForCausalLM.from_pretrained(
                self.nemotron_config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.nemotron_pipeline = pipeline(
                "text-generation",
                model=self.nemotron_model,
                tokenizer=self.tokenizer,
                max_length=self.nemotron_config.max_length,
                temperature=self.nemotron_config.temperature,
                device=0 if self.device == "cuda" else -1
            )
            
            self.logger.info("✅ Nemotron-mini:4b loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Nemotron: {e}")
            self.nemotron_pipeline = None

    def setup_news_sources(self):
        self.rss_feeds = {
            'BBC_World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC_Business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'BBC_Technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'CNN_World': 'http://rss.cnn.com/rss/edition.rss',
            'CNN_Business': 'http://rss.cnn.com/rss/money_latest.rss',
            'TechCrunch': 'https://techcrunch.com/feed/',
        }

    def init_database(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
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
                    collection_minute INTEGER
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_collected_date ON articles(collected_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_collection_minute ON articles(collection_minute)")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✅ Database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise

    def collect_and_process_news(self):
        """Collect and process news in one cycle"""
        current_minute = datetime.now().minute
        new_articles = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    if hasattr(entry, 'link') and not self.article_exists(entry.link):
                        article = {
                            'title': entry.get('title', 'No title'),
                            'content': entry.get('description', '') or entry.get('summary', ''),
                            'url': entry.get('link', ''),
                            'source': source_name,
                            'published_date': self.parse_date(entry.get('published', '')),
                            'collection_minute': current_minute
                        }
                        
                        # Process with AI
                        article['summary'] = self.generate_summary(article['content'])
                        article['embedding'] = self.embedder.encode([f"{article['title']} {article['summary']}"]).tolist()[0]
                        article['category'] = self.detect_category(article['title'] + ' ' + article['content'])
                        
                        new_articles.append(article)
                        self.logger.info(f"📰 New: {article['title'][:50]}...")
                        
            except Exception as e:
                self.logger.warning(f"Error fetching from {source_name}: {e}")
        
        # Store articles
        if new_articles:
            self.store_articles(new_articles)
            self.logger.info(f"💾 Stored {len(new_articles)} articles")
        
        return len(new_articles)

    def article_exists(self, url: str) -> bool:
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM articles WHERE url = %s LIMIT 1", (url,))
            exists = cursor.fetchone() is not None
            cursor.close()
            conn.close()
            return exists
        except:
            return False

    def parse_date(self, date_string: str) -> datetime:
        try:
            from dateutil import parser
            return parser.parse(date_string)
        except:
            return datetime.now()

    def detect_category(self, text: str) -> str:
        categories = {
            'Technology': ['AI', 'tech', 'software', 'digital', 'cyber'],
            'Politics': ['election', 'government', 'president', 'political'],
            'Business': ['economy', 'market', 'finance', 'economic'],
            'Sports': ['football', 'basketball', 'sports', 'game'],
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'General'

    def generate_summary(self, text: str) -> str:
        if not self.nemotron_pipeline:
            return text[:200] + "..."
        
        prompt = f"""Summarize this news article in 1-2 sentences:

{text[:500]}

Summary:"""
        
        try:
            response = self.nemotron_pipeline(prompt, max_new_tokens=100, temperature=0.3)
            summary = response[0]['generated_text'].split("Summary:")[-1].strip()
            return summary if len(summary) > 10 else text[:200] + "..."
        except:
            return text[:200] + "..."

    def store_articles(self, articles: List[Dict[str, Any]]):
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute("""
                    INSERT INTO articles (title, content, summary, url, source, category,
                                        published_date, embedding, collection_minute)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                """, (
                    article['title'], article['content'], article['summary'],
                    article['url'], article['source'], article['category'],
                    article['published_date'], article['embedding'], article['collection_minute']
                ))
            except Exception as e:
                self.logger.error(f"Error storing article: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()

    def search_articles(self, query: str, date_filter: str = "today", top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedder.encode([query])[0]
            
            sql = "SELECT * FROM articles WHERE embedding IS NOT NULL"
            if date_filter == "today":
                sql += " AND DATE(collected_date) = CURRENT_DATE"
            elif date_filter == "week":
                sql += " AND collected_date >= CURRENT_DATE - INTERVAL '7 days'"
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(sql)
            
            articles = []
            embeddings = []
            
            for row in cursor.fetchall():
                articles.append({
                    'title': row[1], 'summary': row[3], 'source': row[5],
                    'category': row[6], 'collected_date': row[8]
                })
                embeddings.append(np.array(row[9]))
            
            cursor.close()
            conn.close()
            
            if not embeddings:
                return []
            
            similarities = cosine_similarity([query_embedding], np.vstack(embeddings))[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [articles[i] for i in top_indices]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def chat_with_context(self, user_input: str) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if asking about current events
        if any(word in user_input.lower() for word in ['today', 'news', 'latest', 'current']):
            articles = self.search_articles(user_input, "today", 3)
            
            if articles:
                context = f"Current time: {current_time}\n\nRecent news:\n"
                for article in articles:
                    context += f"- {article['title']} ({article['source']})\n"
                
                prompt = f"""Based on this context, answer the question:

{context}

Question: {user_input}

Answer:"""
                
                if self.nemotron_pipeline:
                    try:
                        response = self.nemotron_pipeline(prompt, max_new_tokens=200, temperature=0.3)
                        return response[0]['generated_text'].split("Answer:")[-1].strip()
                    except:
                        pass
                
                return f"Based on today's news: {articles[0]['title']} - {articles[0]['summary']}"
            else:
                return f"No recent news found. Current time: {current_time}"
        else:
            # General chat
            if self.nemotron_pipeline:
                try:
                    response = self.nemotron_pipeline(user_input, max_new_tokens=150, temperature=0.7)
                    return response[0]['generated_text']
                except:
                    pass
            return "I'm here to help with news and general questions!"

    def get_daily_briefing(self) -> str:
        articles = self.search_articles("", "today", 10)
        
        if not articles:
            return f"No news articles found for today ({datetime.now().strftime('%Y-%m-%d')})."
        
        context = f"Today's top stories ({datetime.now().strftime('%B %d, %Y')}):\n\n"
        for i, article in enumerate(articles[:5], 1):
            context += f"{i}. {article['title']} - {article['summary']} ({article['source']})\n"
        
        prompt = f"""Create a daily news briefing based on these stories:

{context}

Provide a comprehensive briefing:"""
        
        if self.nemotron_pipeline:
            try:
                response = self.nemotron_pipeline(prompt, max_new_tokens=300, temperature=0.3)
                return response[0]['generated_text'].split("briefing:")[-1].strip()
            except:
                pass
        
        return context

    def start_continuous_collection(self):
        if self.collection_running:
            return
        
        self.collection_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("🔄 Started 1-minute collection intervals")

    def stop_continuous_collection(self):
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("⏹️ Stopped collection")

    def _collection_loop(self):
        cycle = 0
        while self.collection_running:
            try:
                cycle += 1
                self.logger.info(f"🔄 Collection cycle #{cycle}")
                count = self.collect_and_process_news()
                self.logger.info(f"✅ Cycle #{cycle}: {count} new articles")
                time.sleep(self.nemotron_config.collection_interval)
            except Exception as e:
                self.logger.error(f"Collection error: {e}")
                time.sleep(self.nemotron_config.collection_interval)

def main():
    print("🚀 Real-Time News RAG with Nemotron-mini:4b")
    print("=" * 50)
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'livenews'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    try:
        news_system = RealTimeNewsRAG(db_config)
        
        print("\n✅ System ready!")
        print("\nCommands:")
        print("  start - Start 1-minute collection")
        print("  briefing - Today's news briefing")
        print("  <message> - Chat with real-time context")
        print("  quit - Exit")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                news_system.stop_continuous_collection()
                break
            elif user_input.lower() == 'start':
                news_system.start_continuous_collection()
                print("🔄 Started continuous collection every minute")
            elif user_input.lower() == 'briefing':
                print("\n📰 Today's News Briefing:")
                print(news_system.get_daily_briefing())
            elif user_input:
                response = news_system.chat_with_context(user_input)
                print(f"\n🤖 {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
