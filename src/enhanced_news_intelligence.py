"""
Enhanced News Intelligence Pipeline
Advanced Pathway integration with comprehensive AI analysis
"""

import pathway as pw
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import json
import feedparser
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from .gemini_client import GeminiClient
from .rag_system import PathwayRAGSystem, PathwayRAGConfig
from .freshness_validator import NewsFreshnessValidator, FreshnessConfig
import time
import hashlib
import re

@dataclass 
class EnhancedNewsConfig:
    """Configuration for Enhanced News Intelligence"""
    gemini_client: Any = None
    embedding_model: str = "all-MiniLM-L6-v2"
    max_articles_per_source: int = 10
    real_time_update_interval: int = 60  # seconds
    sentiment_threshold: float = 0.1

class EnhancedNewsIntelligence:
    """
    Comprehensive News Intelligence Platform with:
    - Real-time news collection from multiple sources
    - AI-powered summaries, key points, sentiment analysis  
    - Advanced personalization and relevance scoring
    - Pathway integration for stream processing
    """
    
    def __init__(self, config: EnhancedNewsConfig):
        self.config = config
        self.setup_logging()
        self.setup_database()
        self.init_models()
        self.setup_news_sources()
        self.init_pathway_pipeline()
        
    def setup_logging(self):
        """Setup enhanced logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/enhanced_news_intelligence.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Setup enhanced database schema"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'news_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Create enhanced tables
        self.create_enhanced_schema()
        
    def create_enhanced_schema(self):
        """Create enhanced database schema with AI analysis fields"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Enhanced articles table with AI analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    full_content TEXT,
                    source TEXT,
                    category TEXT,
                    url TEXT,
                    author TEXT,
                    published_date TIMESTAMP,
                    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- AI Analysis Fields
                    ai_summary TEXT,
                    key_points TEXT,
                    sentiment_score FLOAT,
                    sentiment_label TEXT,
                    topics TEXT[],
                    entities TEXT[],
                    reading_time INTEGER,
                    quality_score FLOAT,
                    
                    -- Personalization Fields  
                    relevance_scores JSONB,
                    engagement_prediction FLOAT,
                    trending_score FLOAT,
                    
                    -- Technical Fields
                    embedding FLOAT[],
                    embedding_vec vector(384),
                    canonical_id TEXT UNIQUE,
                    
                    -- Metadata
                    word_count INTEGER,
                    image_url TEXT,
                    tags TEXT[],
                    is_breaking BOOLEAN DEFAULT FALSE,
                    is_trending BOOLEAN DEFAULT FALSE
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    interests TEXT[],
                    interest_weights JSONB,
                    reading_history JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Article interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_interactions (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES enhanced_articles(id),
                    interaction_type TEXT, -- view, click, share, like
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration INTEGER, -- reading time in seconds
                    user_id TEXT
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            self.logger.info("✅ Enhanced database schema created")
            
        except Exception as e:
            self.logger.error(f"Database schema creation failed: {e}")
            
    def init_models(self):
        """Initialize AI models and clients"""
        # Initialize embedder
        self.embedder = SentenceTransformer(self.config.embedding_model)
        
        # Initialize Gemini client
        self.gemini_client = self.config.gemini_client
        if self.gemini_client and self.gemini_client.available:
            self.logger.info("✅ Gemini client ready for AI analysis")
        else:
            self.logger.warning("⚠️ Gemini client not available")
            
    def setup_news_sources(self):
        """Setup enhanced news sources with real-time capabilities"""
        self.news_sources = {
            # Traditional RSS feeds
            'bbc_world': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'bbc_tech': 'http://feeds.bbci.co.uk/news/technology/rss.xml', 
            'bbc_business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'cnn_world': 'http://rss.cnn.com/rss/edition.rss',
            'reuters_world': 'https://feeds.reuters.com/reuters/topNews',
            'techcrunch': 'https://techcrunch.com/feed/',
            'wired': 'https://www.wired.com/feed/',
            'ars_technica': 'http://feeds.arstechnica.com/arstechnica/index',
            'hacker_news': 'https://hnrss.org/frontpage',
            
            # API endpoints (for real-time data)
            'newsapi_tech': {
                'type': 'api',
                'url': 'https://newsapi.org/v2/top-headlines',
                'params': {'category': 'technology', 'language': 'en'}
            },
            'newsapi_business': {
                'type': 'api', 
                'url': 'https://newsapi.org/v2/top-headlines',
                'params': {'category': 'business', 'language': 'en'}
            }
        }
        
    def init_pathway_pipeline(self):
        """Initialize Pathway pipeline for real-time processing"""
        try:
            # Create Pathway tables for real-time news processing
            self.setup_pathway_stream()
            self.logger.info("✅ Pathway pipeline initialized")
        except Exception as e:
            self.logger.error(f"Pathway pipeline setup failed: {e}")
            
    def setup_pathway_stream(self):
        """Setup Pathway streaming pipeline"""
        # This is a simplified version - in production, you'd use actual Pathway connectors
        self.news_stream = pw.Table.empty()
        
    def fetch_enhanced_articles(self, hours_back: int = 12) -> List[Dict[str, Any]]:
        """Fetch and enhance articles with comprehensive AI analysis"""
        all_articles = []
        
        for source_name, source_config in self.news_sources.items():
            try:
                if isinstance(source_config, str):
                    # RSS feed
                    articles = self._fetch_rss_articles(source_name, source_config)
                else:
                    # API endpoint
                    articles = self._fetch_api_articles(source_name, source_config)
                
                # Apply AI enhancements to each article
                enhanced_articles = []
                for article in articles:
                    enhanced = self.enhance_article_with_ai(article)
                    if enhanced:
                        enhanced_articles.append(enhanced)
                        
                all_articles.extend(enhanced_articles)
                self.logger.info(f"Enhanced {len(enhanced_articles)} articles from {source_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source_name}: {e}")
                
        return all_articles
        
    def _fetch_rss_articles(self, source_name: str, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feeds"""
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:self.config.max_articles_per_source]:
                # Parse publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                else:
                    pub_date = datetime.now()
                
                # Skip old articles
                if (datetime.now() - pub_date).days > 7:
                    continue
                
                # Get content
                content = entry.get('summary', '') or entry.get('description', '')
                if isinstance(content, list):
                    content = ' '.join(str(item) for item in content if item)
                
                # Skip articles without sufficient content
                if not content or len(content.strip()) < 50:
                    continue
                
                article = {
                    'title': entry.get('title', ''),
                    'content': content,
                    'url': entry.get('link', ''),
                    'source': source_name,
                    'published_date': pub_date,
                    'author': entry.get('author', 'Unknown')
                }
                
                articles.append(article)
                
        except Exception as e:
            self.logger.error(f"RSS fetch error for {source_name}: {e}")
            
        return articles
        
    def _fetch_api_articles(self, source_name: str, api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch articles from API endpoints"""
        articles = []
        
        try:
            # This is a placeholder - you'd implement actual API calls here
            # For NewsAPI, you'd need an API key and proper error handling
            pass
            
        except Exception as e:
            self.logger.error(f"API fetch error for {source_name}: {e}")
            
        return articles
        
    def enhance_article_with_ai(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply comprehensive AI analysis to article"""
        if not self.gemini_client or not self.gemini_client.available:
            return article
            
        try:
            content = article.get('content', '')
            if not content or len(content.strip()) < 50:
                return None
                
            # Generate AI summary
            ai_summary = self.generate_ai_summary(content)
            
            # Extract key points  
            key_points = self.extract_key_points(content)
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self.analyze_sentiment(content)
            
            # Extract topics and entities
            topics, entities = self.extract_topics_and_entities(content)
            
            # Calculate reading time
            reading_time = self.calculate_reading_time(content)
            
            # Assess quality
            quality_score = self.assess_article_quality(article)
            
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Create enhanced article
            enhanced_article = {
                **article,
                'ai_summary': ai_summary,
                'key_points': key_points,
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'topics': topics,
                'entities': entities,
                'reading_time': reading_time,
                'quality_score': quality_score,
                'embedding': embedding,
                'word_count': len(content.split()),
                'canonical_id': self._generate_canonical_id(article),
                'is_breaking': self._detect_breaking_news(article),
                'trending_score': self._calculate_trending_score(article)
            }
            
            return enhanced_article
            
        except Exception as e:
            self.logger.error(f"AI enhancement failed: {e}")
            return article
            
    def generate_ai_summary(self, content: str) -> str:
        """Generate compelling AI summary"""
        try:
            prompt = f"""
            Create a compelling 2-3 sentence summary of this news article that:
            1. Captures the main story and its significance  
            2. Highlights why readers should care
            3. Uses engaging, accessible language
            
            Article: {content[:1000]}...
            
            Summary:
            """
            
            response = self.gemini_client.generate_response(
                prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.strip() if response else "AI summary unavailable."
            
        except Exception as e:
            self.logger.error(f"AI summary generation failed: {e}")
            return "Summary generation failed."
            
    def extract_key_points(self, content: str) -> str:
        """Extract key bullet points"""
        try:
            prompt = f"""
            Extract 3-5 key bullet points from this article that highlight the most important information:
            
            {content[:1000]}...
            
            Format as:
            • Point 1
            • Point 2  
            • Point 3
            """
            
            response = self.gemini_client.generate_response(
                prompt,
                max_tokens=200,
                temperature=0.5
            )
            
            return response.strip() if response else "Key points unavailable."
            
        except Exception as e:
            self.logger.error(f"Key points extraction failed: {e}")
            return "Key points extraction failed."
            
    def analyze_sentiment(self, content: str) -> tuple[float, str]:
        """Analyze sentiment of article"""
        try:
            prompt = f"""
            Analyze the sentiment of this news article. Respond with just a number between -1.0 (very negative) and 1.0 (very positive):
            
            {content[:500]}...
            
            Sentiment score:
            """
            
            response = self.gemini_client.generate_response(
                prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            # Parse sentiment score
            try:
                score = float(response.strip())
                score = max(-1.0, min(1.0, score))  # Clamp to valid range
                
                if score > 0.1:
                    label = "Positive"
                elif score < -0.1:
                    label = "Negative"  
                else:
                    label = "Neutral"
                    
                return score, label
                
            except ValueError:
                return 0.0, "Neutral"
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return 0.0, "Neutral"
            
    def extract_topics_and_entities(self, content: str) -> tuple[List[str], List[str]]:
        """Extract topics and named entities"""
        try:
            prompt = f"""
            Extract key topics and named entities from this article.
            
            {content[:800]}...
            
            Respond in this exact format:
            Topics: topic1, topic2, topic3
            Entities: entity1, entity2, entity3
            """
            
            response = self.gemini_client.generate_response(
                prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            # Parse response
            topics = []
            entities = []
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('Topics:'):
                    topics = [t.strip() for t in line.replace('Topics:', '').split(',')]
                elif line.startswith('Entities:'):
                    entities = [e.strip() for e in line.replace('Entities:', '').split(',')]
                    
            return topics[:5], entities[:5]  # Limit to 5 each
            
        except Exception as e:
            self.logger.error(f"Topic/entity extraction failed: {e}")
            return [], []
            
    def calculate_reading_time(self, content: str) -> int:
        """Calculate estimated reading time in minutes"""
        words = len(content.split())
        reading_speed = 250  # average words per minute
        return max(1, round(words / reading_speed))
        
    def assess_article_quality(self, article: Dict[str, Any]) -> float:
        """Assess article quality based on various factors"""
        score = 0.5  # baseline
        
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Length factor
        if len(content) > 500:
            score += 0.2
        elif len(content) > 200:
            score += 0.1
            
        # Title quality  
        if len(title) > 20 and len(title) < 100:
            score += 0.1
            
        # Source credibility (simple heuristic)
        source = article.get('source', '').lower()
        if any(trusted in source for trusted in ['bbc', 'reuters', 'cnn', 'techcrunch']):
            score += 0.2
            
        return min(1.0, score)
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
            
    def _generate_canonical_id(self, article: Dict[str, Any]) -> str:
        """Generate canonical ID for deduplication"""
        title = article.get('title', '').lower().strip()
        if len(title) < 10:
            return ''
        return hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]
        
    def _detect_breaking_news(self, article: Dict[str, Any]) -> bool:
        """Detect if article is breaking news"""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        
        breaking_indicators = [
            'breaking', 'urgent', 'just in', 'developing', 
            'alert', 'flash', 'live updates'
        ]
        
        return any(indicator in title or indicator in content[:200] 
                  for indicator in breaking_indicators)
                  
    def _calculate_trending_score(self, article: Dict[str, Any]) -> float:
        """Calculate trending score based on recency and keywords"""
        # Simple trending calculation
        pub_date = article.get('published_date', datetime.now())
        age_hours = (datetime.now() - pub_date).total_seconds() / 3600
        
        # Newer articles get higher scores
        recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
        
        # Boost for trending keywords
        content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        trending_keywords = ['ai', 'breakthrough', 'revolutionary', 'major', 'significant']
        keyword_boost = sum(0.1 for keyword in trending_keywords if keyword in content)
        
        return min(1.0, recency_score + keyword_boost)
        
    def store_enhanced_articles(self, articles: List[Dict[str, Any]]):
        """Store enhanced articles in database"""
        stored_count = 0
        
        for article in articles:
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO enhanced_articles (
                        title, content, full_content, source, category, url, author,
                        published_date, ai_summary, key_points, sentiment_score, 
                        sentiment_label, topics, entities, reading_time, quality_score,
                        embedding, canonical_id, word_count, is_breaking, trending_score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (canonical_id) DO NOTHING
                """, (
                    article.get('title', ''),
                    article.get('content', ''),
                    article.get('full_content', article.get('content', '')),
                    article.get('source', ''),
                    article.get('category', 'General'),
                    article.get('url', ''),
                    article.get('author', 'Unknown'),
                    article.get('published_date', datetime.now()),
                    article.get('ai_summary', ''),
                    article.get('key_points', ''),
                    article.get('sentiment_score', 0.0),
                    article.get('sentiment_label', 'Neutral'),
                    article.get('topics', []),
                    article.get('entities', []),
                    article.get('reading_time', 5),
                    article.get('quality_score', 0.5),
                    article.get('embedding', []),
                    article.get('canonical_id', ''),
                    article.get('word_count', 0),
                    article.get('is_breaking', False),
                    article.get('trending_score', 0.0)
                ))
                
                conn.commit()
                cursor.close() 
                conn.close()
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to store article: {e}")
                
        self.logger.info(f"Stored {stored_count} enhanced articles")
        return stored_count
        
    def get_personalized_articles(self, user_preferences: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
        """Get personalized articles based on user preferences"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # For now, get recent high-quality articles
            cursor.execute("""
                SELECT * FROM enhanced_articles 
                WHERE quality_score > 0.6 
                AND published_date > %s
                ORDER BY trending_score DESC, quality_score DESC, published_date DESC
                LIMIT %s
            """, (datetime.now() - timedelta(days=7), limit))
            
            articles = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to list of dicts and add relevance scores
            result = []
            for article in articles:
                article_dict = dict(article)
                article_dict['relevance_score'] = self._calculate_relevance_score(
                    article_dict, user_preferences
                )
                result.append(article_dict)
                
            # Sort by relevance
            result.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get personalized articles: {e}")
            return []
            
    def _calculate_relevance_score(self, article: Dict[str, Any], user_preferences: Dict[str, Any]) -> float:
        """Calculate relevance score for user"""
        # Simple relevance calculation based on topics and interests
        base_score = article.get('quality_score', 0.5) * 100
        
        # Boost for user interests
        user_interests = user_preferences.get('interests', [])
        article_topics = article.get('topics', [])
        
        interest_boost = 0
        for interest in user_interests:
            for topic in article_topics:
                if any(word in topic.lower() for word in interest.lower().split()):
                    interest_boost += 10
                    
        return min(98, base_score + interest_boost)

def main():
    """Test the enhanced news intelligence system"""
    from .gemini_client import GeminiClient
    
    # Initialize components
    gemini_client = GeminiClient()
    config = EnhancedNewsConfig(gemini_client=gemini_client)
    
    # Create enhanced news system
    news_intelligence = EnhancedNewsIntelligence(config)
    
    # Fetch and enhance articles
    print("Fetching enhanced articles...")
    articles = news_intelligence.fetch_enhanced_articles()
    
    print(f"Enhanced {len(articles)} articles")
    
    # Store articles
    news_intelligence.store_enhanced_articles(articles)
    
    # Get personalized articles
    user_prefs = {
        'interests': ['AI', 'Technology', 'Business']
    }
    
    personalized = news_intelligence.get_personalized_articles(user_prefs, limit=10)
    print(f"Retrieved {len(personalized)} personalized articles")

if __name__ == "__main__":
    main()
