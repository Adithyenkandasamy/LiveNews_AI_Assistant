"""
Complete News RAG System with PostgreSQL
Real-time news aggregation with AI-powered question answering
"""

import feedparser
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
from typing import List, Dict, Any, Optional
import hashlib
import threading
from dataclasses import dataclass
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleanupConfig:
    """Configuration for automatic cleanup"""
    enabled: bool = True
    default_retention_days: int = 7
    cleanup_frequency: int = 10
    batch_size: int = 100
    emergency_threshold: float = 0.9
    category_retention: Dict[str, int] = None
    
    def __post_init__(self):
        if self.category_retention is None:
            self.category_retention = {
                'Breaking News': 14,
                'Politics': 10,
                'Technology': 7,
                'Business': 7,
                'Entertainment': 3,
                'Sports': 5,
                'Reddit': 1,
                'General': 7
            }

class CompleteNewsRAGSystem:
    def __init__(self, db_config: Dict[str, str], cleanup_config: CleanupConfig = None):
        """Complete News RAG System"""
        self.db_config = db_config
        self.cleanup_config = cleanup_config or CleanupConfig()
        self.cycle_count = 0
        self.is_running = False
        
        # Initialize AI models
        logger.info("🤖 Loading AI models...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # News sources
        self.rss_sources = {
            'BBC_World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC_Business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'BBC_Technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'CNN_World': 'http://rss.cnn.com/rss/edition.rss',
            'CNN_Business': 'http://rss.cnn.com/rss/money_latest.rss',
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Reuters_World': 'https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best',
        }
        
        self.reddit_sources = {
            'worldnews': 'https://www.reddit.com/r/worldnews.json',
            'technology': 'https://www.reddit.com/r/technology.json',
            'business': 'https://www.reddit.com/r/business.json',
            'news': 'https://www.reddit.com/r/news.json',
        }
        
        # Category keywords
        self.category_keywords = {
            'Politics': ['election', 'government', 'president', 'congress', 'senate', 'vote', 'policy'],
            'Technology': ['ai', 'artificial intelligence', 'tech', 'software', 'app', 'google', 'apple', 'microsoft'],
            'Business': ['stock', 'market', 'economy', 'finance', 'earnings', 'revenue', 'investment'],
            'Entertainment': ['movie', 'film', 'actor', 'actress', 'hollywood', 'netflix', 'disney'],
            'Sports': ['football', 'basketball', 'soccer', 'baseball', 'tennis', 'olympics'],
            'Science': ['research', 'study', 'discovery', 'space', 'nasa', 'climate', 'medicine'],
            'Breaking News': ['breaking', 'urgent', 'alert', 'developing', 'exclusive', 'live'],
        }
        
        # Initialize database
        self.init_database()
        logger.info("✅ Complete News RAG System initialized successfully")
    
    def get_db_connection(self):
        """Get database connection with error handling"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def init_database(self):
        """Initialize PostgreSQL database with complete schema"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Main articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id VARCHAR(255) PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source VARCHAR(100),
                    category VARCHAR(50),
                    url TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_timestamp TEXT,
                    entities TEXT[],
                    keywords TEXT[],
                    embedding REAL[],
                    sentiment VARCHAR(20),
                    importance_score REAL DEFAULT 0.0,
                    word_count INTEGER DEFAULT 0,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Activity log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    source VARCHAR(100),
                    category VARCHAR(50),
                    details JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON articles(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
                CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
                CREATE INDEX IF NOT EXISTS idx_articles_importance ON articles(importance_score DESC);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✅ Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            raise
    
    def generate_article_id(self, title: str, source: str) -> str:
        """Generate unique article ID"""
        return hashlib.md5(f"{source}_{title}".encode()).hexdigest()
    
    def article_exists(self, article_id: str) -> bool:
        """Check if article already exists"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM articles WHERE id = %s", (article_id,))
            exists = cursor.fetchone() is not None
            cursor.close()
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"❌ Error checking article existence: {e}")
            return False
    
    def detect_category(self, title: str, content: str) -> str:
        """Automatically detect article category"""
        text = f"{title} {content}".lower()
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'General'
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        important_terms = [
            'Tesla', 'Apple', 'Google', 'Microsoft', 'Amazon', 'Meta', 'Netflix',
            'Biden', 'Trump', 'Elon Musk', 'AI', 'Bitcoin', 'Stock Market',
            'COVID', 'Climate Change', 'Ukraine', 'Russia', 'China'
        ]
        keywords = []
        text_lower = text.lower()
        for term in important_terms:
            if term.lower() in text_lower:
                keywords.append(term)
        return list(set(keywords))
    
    def calculate_importance_score(self, title: str, content: str, source: str, category: str) -> float:
        """Calculate article importance score (0.0 to 1.0)"""
        score = 0.0
        source_weights = {'BBC': 0.9, 'CNN': 0.8, 'Reuters': 0.9, 'TechCrunch': 0.7, 'Reddit': 0.3}
        for source_key, weight in source_weights.items():
            if source_key in source:
                score += weight * 0.3
                break
        category_weights = {
            'Breaking News': 0.9, 'Politics': 0.8, 'Business': 0.7,
            'Technology': 0.7, 'Science': 0.7, 'Sports': 0.5,
            'Entertainment': 0.4, 'General': 0.5
        }
        score += category_weights.get(category, 0.5) * 0.2
        text = f"{title} {content}".lower()
        breaking_words = ['breaking', 'urgent', 'alert', 'developing', 'exclusive']
        breaking_count = sum(1 for word in breaking_words if word in text)
        score += min(breaking_count * 0.1, 0.3)
        return min(score, 1.0)
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'success', 'win']
        negative_words = ['bad', 'terrible', 'negative', 'loss', 'fail', 'crisis', 'drop']
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        if pos_count > neg_count + 1:
            return 'positive'
        elif neg_count > pos_count + 1:
            return 'negative'
        else:
            return 'neutral'
    
    def collect_rss_feeds(self) -> List[Dict[str, Any]]:
        """Collect articles from RSS feeds"""
        new_articles = []
        for source_name, rss_url in self.rss_sources.items():
            try:
                logger.info(f"📡 Fetching from {source_name}...")
                response = requests.get(rss_url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"⚠️ HTTP {response.status_code} from {source_name}")
                    continue
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:3]:
                    article_id = self.generate_article_id(entry.title, source_name)
                    if self.article_exists(article_id):
                        continue
                    content = getattr(entry, 'summary', '') or getattr(entry, 'description', '') or entry.title
                    category = self.detect_category(entry.title, content)
                    article = {
                        'id': article_id,
                        'title': entry.title,
                        'content': content,
                        'source': source_name,
                        'category': category,
                        'url': getattr(entry, 'link', ''),
                        'raw_timestamp': getattr(entry, 'published', str(datetime.now())),
                        'keywords': self.extract_keywords(f"{entry.title} {content}"),
                        'word_count': len(content.split()),
                        'importance_score': self.calculate_importance_score(entry.title, content, source_name, category)
                    }
                    new_articles.append(article)
                    logger.info(f"📰 New {category}: {article['title'][:50]}...")
            except Exception as e:
                logger.error(f"❌ Error fetching from {source_name}: {e}")
                continue
        return new_articles
    
    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process articles with AI"""
        processed_articles = []
        for article in articles:
            try:
                content = article['content']
                if len(content) > 100:
                    try:
                        summary = self.summarizer(content, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
                    except:
                        summary = content[:200] + "..." if len(content) > 200 else content
                else:
                    summary = content
                embedding_text = f"{article['title']} {content}"
                embedding = self.embedder.encode(embedding_text)
                sentiment = self.analyze_sentiment(f"{article['title']} {content}")
                article.update({
                    'summary': summary,
                    'embedding': embedding.tolist(),
                    'sentiment': sentiment,
                    'entities': article['keywords']
                })
                processed_articles.append(article)
                logger.info(f"✅ Processed {article['category']}: {article['title'][:40]}...")
            except Exception as e:
                logger.error(f"❌ Error processing article: {e}")
                continue
        return processed_articles
    
    def store_articles(self, articles: List[Dict[str, Any]]):
        """Store articles in PostgreSQL"""
        if not articles:
            return
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            stored_count = 0
            for article in articles:
                try:
                    cursor.execute("""
                        INSERT INTO articles (
                            id, title, content, summary, source, category, url, 
                            raw_timestamp, entities, keywords, embedding, 
                            sentiment, importance_score, word_count
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (id) DO NOTHING
                    """, (
                        article['id'], article['title'], article['content'], article['summary'],
                        article['source'], article['category'], article['url'], article['raw_timestamp'],
                        article['entities'], article['keywords'], article['embedding'],
                        article['sentiment'], article['importance_score'], article['word_count']
                    ))
                    if cursor.rowcount > 0:
                        stored_count += 1
                except Exception as e:
                    logger.error(f"❌ Error storing individual article: {e}")
                    continue
            conn.commit()
            cursor.close()
            conn.close()
            if stored_count > 0:
                logger.info(f"💾 Stored {stored_count} articles")
        except Exception as e:
            logger.error(f"❌ Error storing articles: {e}")
    
    def search_articles(self, query: str, limit: int = 5, hours: int = 24) -> List[Dict[str, Any]]:
        """Search articles using vector similarity"""
        try:
            query_embedding = self.embedder.encode(query)
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT id, title, content, summary, source, category, url, 
                       timestamp, entities, keywords, embedding, sentiment, importance_score
                FROM articles 
                WHERE timestamp >= %s 
                ORDER BY importance_score DESC, timestamp DESC
                LIMIT 100
            """, (datetime.now() - timedelta(hours=hours),))
            articles = cursor.fetchall()
            cursor.close()
            conn.close()
            if not articles:
                return []
            similarities = []
            for article in articles:
                if article['embedding']:
                    embedding = np.array(article['embedding'])
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    if similarity > 0.1:
                        similarities.append((similarity, dict(article)))
            similarities.sort(key=lambda x: x[0], reverse=True)
            results = [article for score, article in similarities[:limit]]
            return results
        except Exception as e:
            logger.error(f"❌ Error searching articles: {e}")
            return []
    
    def answer_question(self, question: str, max_articles: int = 3) -> str:
        """RAG-based question answering"""
        try:
            start_time = time.time()
            relevant_articles = self.search_articles(question, limit=max_articles)
            if not relevant_articles:
                return "I don't have recent information about that topic. Try asking about general topics like 'technology news', 'business updates', or 'sports news'."
            context_parts = []
            for i, article in enumerate(relevant_articles, 1):
                timestamp = article['timestamp'].strftime("%Y-%m-%d %H:%M")
                context_parts.append(
                    f"{i}. {article['source']} ({timestamp})\n"
                    f"   Title: {article['title']}\n"
                    f"   Summary: {article['summary']}\n"
                    f"   Category: {article['category']} | Sentiment: {article['sentiment']}"
                )
            context = "\n\n".join(context_parts)
            response_time = round((time.time() - start_time) * 1000)
            answer = f"""📰 Latest News Answer

Question: {question}

Current Information (Last 24 hours):
{context}

Summary:
• Main Story: {relevant_articles[0]['summary']}
• Sentiment: {relevant_articles[0]['sentiment']}
• Category: {relevant_articles[0]['category']}
• Importance: {('High' if relevant_articles[0]['importance_score'] > 0.7 else 'Medium' if relevant_articles[0]['importance_score'] > 0.4 else 'Low')}

Response time: {response_time}ms | Articles: {len(relevant_articles)}
"""
            return answer
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM articles WHERE timestamp >= %s", 
                         (datetime.now() - timedelta(hours=24),))
            recent_articles = cursor.fetchone()[0] or 0
            cursor.execute("""
                SELECT category, COUNT(*) FROM articles 
                WHERE timestamp >= %s GROUP BY category ORDER BY COUNT(*) DESC
            """, (datetime.now() - timedelta(hours=24),))
            by_category = dict(cursor.fetchall())
            cursor.close()
            conn.close()
            return {
                'total_articles': total_articles,
                'recent_articles_24h': recent_articles,
                'by_category': by_category,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {}
    
    def run_collection_cycle(self):
        """Run one complete collection cycle"""
        try:
            logger.info(f"🔄 Starting collection cycle #{self.cycle_count + 1}")
            rss_articles = self.collect_rss_feeds()
            all_new_articles = rss_articles
            if all_new_articles:
                processed_articles = self.process_articles(all_new_articles)
                self.store_articles(processed_articles)
                self.cycle_count += 1
                logger.info(f"✅ Cycle #{self.cycle_count} completed: {len(processed_articles)} articles")
            else:
                logger.info("⏸️ No new articles found in this cycle")
        except Exception as e:
            logger.error(f"❌ Error in collection cycle: {e}")
    
    def run_continuous_collection(self, interval_seconds: int = 30):
        """Run continuous news collection"""
        self.is_running = True
        logger.info(f"🚀 Starting continuous collection (every {interval_seconds}s)")
        try:
            while self.is_running:
                self.run_collection_cycle()
                if self.is_running:
                    logger.info(f"⏱️ Waiting {interval_seconds} seconds...")
                    time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("⏹️ Collection stopped by user")
        finally:
            self.is_running = False
            logger.info("🏁 Collection ended")
    
    def stop_collection(self):
        """Stop continuous collection"""
        self.is_running = False


def create_database_config():
    """Create database configuration"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'newsrag'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }

def create_cleanup_config():
    """Create cleanup configuration"""
    return CleanupConfig()

def interactive_mode():
    """Interactive command-line mode"""
    print("🚀 Live News AI Assistant - Interactive Mode")
    print("=" * 50)
    
    db_config = create_database_config()
    cleanup_config = create_cleanup_config()
    
    try:
        news_system = CompleteNewsRAGSystem(db_config, cleanup_config)
        print("\n✅ System initialized successfully!")
        print("\nCommands:")
        print("  collect - Run single collection cycle")
        print("  continuous - Start continuous collection")
        print("  ask <question> - Ask a question")
        print("  stats - Show system statistics")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command == "quit":
                    break
                elif command == "collect":
                    print("🔄 Running collection cycle...")
                    news_system.run_collection_cycle()
                    print("✅ Collection completed!")
                    
                elif command == "continuous":
                    print("🚀 Starting continuous collection (Ctrl+C to stop)...")
                    news_system.run_continuous_collection(interval_seconds=60)
                    
                elif command.startswith("ask "):
                    question = command[4:].strip()
                    if question:
                        print("🤖 Thinking...")
                        answer = news_system.answer_question(question)
                        print(f"\n{answer}")
                    else:
                        print("❌ Please provide a question")
                        
                elif command == "stats":
                    stats = news_system.get_system_stats()
                    print("\n📊 System Statistics:")
                    for key, value in stats.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for subkey, subvalue in list(value.items())[:5]:
                                print(f"    {subkey}: {subvalue}")
                        else:
                            print(f"  {key}: {value}")
                            
                else:
                    print("❌ Unknown command. Available: collect, continuous, ask <question>, stats, quit")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Running in TEST mode...")
        db_config = create_database_config()
        news_system = CompleteNewsRAGSystem(db_config)
        news_system.run_collection_cycle()
        print("\n💬 Testing Q&A...")
        test_questions = ["What's the latest technology news?", "Any business updates?"]
        for question in test_questions:
            print(f"\n🔍 {question}")
            answer = news_system.answer_question(question)
            print(f"📝 {answer[:200]}...")
        stats = news_system.get_system_stats()
        print(f"\n📊 Stats: {stats}")
    else:
        interactive_mode()
