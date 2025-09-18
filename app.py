#!/usr/bin/env python3
"""
Live News AI Flask Web Application
Real-time news chatbot using Ollama Nemotron-mini
"""

from flask import Flask, render_template, request, jsonify
import requests
import feedparser
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import threading
import time
from pathway_rag_system import PathwayRAGSystem, PathwayRAGConfig
from news_freshness_validator import NewsFreshnessValidator, FreshnessConfig
from news_comparison_tool import NewsComparisonTool
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

class LiveNewsAI:
    """Real-time news AI chatbot"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"  # Better summarization model
        
        # News sources
        self.news_feeds = {
            'BBC World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC Tech': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'CNN': 'http://rss.cnn.com/rss/edition.rss',
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Reuters': 'https://feeds.reuters.com/reuters/topNews'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.setup_database()
        
        # Initialize all systems
        self.setup_rag_system()
        self.setup_news_comparison()
        
        # Start background news collection
        self.collection_running = False
        self.start_news_collection()
        
    def setup_rag_system(self):
        """Initialize Pathway RAG and news freshness validator"""
        try:
            # Setup Pathway RAG
            pathway_config = PathwayRAGConfig(
                ollama_url=self.ollama_url,
                ollama_model=self.ollama_model,
                embedding_model='all-MiniLM-L6-v2',
                chunk_size=512,
                max_results=5
            )
            
            self.pathway_rag = PathwayRAGSystem(pathway_config)
            self.rag_available = True
            self.logger.info("✅ Pathway RAG system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pathway RAG: {e}")
            self.rag_available = False
            
        # Setup freshness validator
        try:
            freshness_config = FreshnessConfig(
                ollama_url=self.ollama_url,
                ollama_model=self.ollama_model
            )
            self.freshness_validator = NewsFreshnessValidator(freshness_config)
            self.logger.info("✅ News freshness validator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize freshness validator: {e}")
            self.freshness_validator = None
            
    def setup_database(self):
        """Setup database configuration"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'news_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Create tables if they don't exist
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source TEXT,
                    category TEXT,
                    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding FLOAT[],
                    url TEXT,
                    UNIQUE(title, source)
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            self.logger.info("✅ Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            
    def setup_news_comparison(self):
        """Initialize news comparison tool"""
        try:
            self.comparison_tool = NewsComparisonTool()
            self.logger.info("✅ News comparison tool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize comparison tool: {e}")
            self.comparison_tool = None
            
    def start_news_collection(self):
        """Start background news collection"""
        if not self.collection_running:
            self.collection_running = True
            collection_thread = threading.Thread(target=self._news_collection_worker, daemon=True)
            collection_thread.start()
            self.logger.info("🔄 Started background news collection")
            
    def _news_collection_worker(self):
        """Background worker for collecting news"""
        while self.collection_running:
            try:
                self.logger.info("📰 Collecting latest news...")
                articles = self.fetch_and_store_news()
                
                if articles and self.rag_available:
                    # Add to RAG system
                    self.pathway_rag.add_news_articles(articles)
                    
                self.logger.info(f"✅ Collected {len(articles)} articles")
                
            except Exception as e:
                self.logger.error(f"News collection error: {e}")
                
            # Wait 30 minutes before next collection
            time.sleep(1800)
            
    def fetch_and_store_news(self) -> List[Dict[str, Any]]:
        """Fetch news and store in database"""
        all_articles = []
        
        for source_name, feed_url in self.news_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Top 10 from each source
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                            
                        article = {
                            'title': entry.get('title', ''),
                            'content': entry.get('summary', '') or entry.get('description', ''),
                            'source': source_name,
                            'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'url': entry.get('link', ''),
                            'category': self._categorize_article(entry.get('title', '') + ' ' + entry.get('summary', ''))
                        }
                        
                        # Store in database
                        if self._store_article(article):
                            all_articles.append(article)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing article: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
                
        return all_articles
        
    def _categorize_article(self, text: str) -> str:
        """Categorize article based on content"""
        categories = {
            'Technology': ['AI', 'tech', 'software', 'digital', 'cyber', 'innovation', 'startup'],
            'Politics': ['election', 'government', 'president', 'political', 'congress', 'senate'],
            'Business': ['economy', 'market', 'finance', 'economic', 'stock', 'investment'],
            'Sports': ['football', 'basketball', 'sports', 'game', 'team', 'player'],
            'Health': ['health', 'medical', 'hospital', 'vaccine', 'covid'],
            'Science': ['science', 'research', 'study', 'climate', 'space']
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'General'
        
    def _store_article(self, article: Dict[str, Any]) -> bool:
        """Store article in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO news_articles (title, content, source, category, collected_date, url)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (title, source) DO NOTHING
                RETURNING id
            """, (
                article['title'],
                article['content'],
                article['source'],
                article['category'],
                article['date'],
                article['url']
            ))
            
            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Database storage error: {e}")
            return False
        
    def fetch_latest_news(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch latest news from multiple sources"""
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for source_name, feed_url in self.news_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Top 5 from each source
                    # Parse date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()  # Assume recent if no date
                    except:
                        pub_date = datetime.now()
                    
                    # Only include recent news
                    if pub_date > cutoff_time:
                        news_item = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', '')[:800] + ("..." if len(entry.get('summary', '')) > 800 else ""),
                            'source': source_name,
                            'date': pub_date.strftime('%Y-%m-%d %H:%M'),
                            'link': entry.get('link', '')
                        }
                        all_news.append(news_item)
                        
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source_name}: {e}")
                
        # Sort by date (newest first) and return top 10
        all_news.sort(key=lambda x: x['date'], reverse=True)
        
        # Apply freshness filtering if validator available
        if self.freshness_validator:
            all_news = self.freshness_validator.filter_fresh_articles(all_news)
            
        return all_news[:10]
        
    def is_news_query(self, user_input: str) -> bool:
        """Check if user is asking for news"""
        news_keywords = [
            'news', 'today', 'latest', 'current', 'recent', 'happening', 
            'update', 'breaking', 'what\'s new', 'headlines', 'world',
            'politics', 'technology', 'business', 'sports'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in news_keywords)
        
    def chat_with_news(self, user_input: str) -> str:
        """Main chat function with Pathway RAG and news integration"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if user wants news
        if self.is_news_query(user_input):
            # Try Pathway RAG first for better results
            if self.rag_available:
                try:
                    rag_result = self.pathway_rag.query(user_input)
                    
                    if rag_result['sources']:
                        # Use RAG results
                        news_context = f"Current Date & Time: {current_time}\n\n"
                        news_context += "Relevant News (via Pathway RAG):\n"
                        
                        for i, source in enumerate(rag_result['sources'][:5], 1):
                            news_context += f"{i}. {source.get('title', 'No title')}\n"
                            news_context += f"   Source: {source.get('source', 'Unknown')} | {source.get('date', 'Unknown date')}\n"
                            news_context += f"   Summary: {source.get('content', '')[:800]}...\n\n"
                        
                        # Add freshness info if available
                        if rag_result.get('freshness_report'):
                            freshness = rag_result['freshness_report']
                            fresh_percent = freshness.get('freshness_percentage', {}).get('fresh', 0)
                            news_context += f"News Freshness: {fresh_percent}% recent content\n\n"
                        
                        prompt = f"""You are a helpful AI assistant with access to real-time news via advanced RAG system. Based on the current news below, answer the user's question naturally and conversationally.

{news_context}

User: {user_input}

Provide a natural, informative response using the news information above. Be conversational and helpful."""

                    else:
                        # RAG found no results, fallback to direct news fetch
                        latest_news = self.fetch_latest_news(24)
                        
                        if latest_news:
                            news_context = f"Current Date & Time: {current_time}\n\n"
                            news_context += "Latest News Headlines:\n"
                            
                            for i, news in enumerate(latest_news[:5], 1):
                                news_context += f"{i}. {news['title']}\n"
                                news_context += f"   Source: {news['source']} | {news['date']}\n"
                                news_context += f"   Summary: {news['summary']}\n\n"
                            
                            prompt = f"""You are a helpful AI assistant with access to real-time news. Based on the current news below, answer the user's question naturally and conversationally.

{news_context}

User: {user_input}

Provide a natural, informative response using the news information above. Be conversational and helpful."""
                        else:
                            prompt = f"""You are a helpful AI assistant. The user asked about news but no recent articles were found. Current time: {current_time}

User: {user_input}

Explain that you don't have access to the very latest news and suggest they check news websites directly."""
                            
                except Exception as e:
                    self.logger.error(f"RAG query failed: {e}")
                    # Fallback to simple news fetch
                    latest_news = self.fetch_latest_news(24)
                    
                    if latest_news:
                        news_context = f"Current Date & Time: {current_time}\n\n"
                        news_context += "Latest News Headlines:\n"
                        
                        for i, news in enumerate(latest_news[:5], 1):
                            news_context += f"{i}. {news['title']}\n"
                            news_context += f"   Source: {news['source']} | {news['date']}\n"
                            news_context += f"   Summary: {news['summary']}\n\n"
                        
                        prompt = f"""You are a helpful AI assistant with access to real-time news. Based on the current news below, answer the user's question naturally and conversationally.

{news_context}

User: {user_input}

Provide a natural, informative response using the news information above. Be conversational and helpful."""
                    else:
                        prompt = f"""You are a helpful AI assistant. The user asked about news but no recent articles were found. Current time: {current_time}

User: {user_input}

Explain that you don't have access to the very latest news and suggest they check news websites directly."""
            else:
                # No RAG available, use simple news fetch
                latest_news = self.fetch_latest_news(24)
                
                if latest_news:
                    news_context = f"Current Date & Time: {current_time}\n\n"
                    news_context += "Latest News Headlines:\n"
                    
                    for i, news in enumerate(latest_news[:5], 1):
                        news_context += f"{i}. {news['title']}\n"
                        news_context += f"   Source: {news['source']} | {news['date']}\n"
                        news_context += f"   Summary: {news['summary']}\n\n"
                    
                    prompt = f"""You are a helpful AI assistant with access to real-time news. Based on the current news below, answer the user's question naturally and conversationally.

{news_context}

User: {user_input}

Provide a natural, informative response using the news information above. Be conversational and helpful."""
                else:
                    prompt = f"""You are a helpful AI assistant. The user asked about news but no recent articles were found. Current time: {current_time}

User: {user_input}

Explain that you don't have access to the very latest news and suggest they check news websites directly."""
                
        else:
            # Normal conversation (non-news)
            prompt = f"""You are a helpful, friendly AI assistant. Current time: {current_time}

User: {user_input}

Respond naturally and helpfully."""
        
        # Generate response using Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 300
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer if answer else "I'm not sure how to respond to that."
            else:
                return "Sorry, I'm having trouble connecting to my AI model right now."
                
        except Exception as e:
            self.logger.error(f"Ollama request failed: {e}")
            return "Sorry, I encountered an error while processing your message."

# Initialize the AI
news_ai = LiveNewsAI()

@app.route('/')
def home():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Get AI response
        ai_response = news_ai.chat_with_news(user_message)
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/news')
def get_news():
    """Get latest news headlines"""
    try:
        latest_news = news_ai.fetch_latest_news(24)
        return jsonify({
            'news': latest_news,
            'count': len(latest_news),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        app.logger.error(f"News fetch error: {e}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@app.route('/rag-search', methods=['POST'])
def rag_search():
    """Advanced RAG search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        if news_ai.rag_available:
            result = news_ai.pathway_rag.query(query)
            return jsonify(result)
        else:
            return jsonify({'error': 'RAG system not available'}), 503
            
    except Exception as e:
        app.logger.error(f"RAG search error: {e}")
        return jsonify({'error': 'RAG search failed'}), 500

@app.route('/news-analysis', methods=['POST'])
def news_analysis():
    """News freshness and quality analysis"""
    try:
        data = request.get_json()
        query = data.get('query', 'latest news')
        
        if news_ai.comparison_tool:
            analysis = news_ai.comparison_tool.compare_news_quality(query)
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Analysis tool not available'}), 503
            
    except Exception as e:
        app.logger.error(f"News analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/database-stats')
def database_stats():
    """Get database statistics"""
    try:
        conn = psycopg2.connect(**news_ai.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get article counts by category
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM news_articles 
            GROUP BY category 
            ORDER BY count DESC
        """)
        categories = cursor.fetchall()
        
        # Get recent articles count
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM news_articles 
            WHERE collected_date >= NOW() - INTERVAL '24 hours'
        """)
        recent_count = cursor.fetchone()['count']
        
        # Get total articles
        cursor.execute("SELECT COUNT(*) as total FROM news_articles")
        total_count = cursor.fetchone()['total']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'total_articles': total_count,
            'recent_articles': recent_count,
            'categories': [dict(cat) for cat in categories],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        app.logger.error(f"Database stats error: {e}")
        return jsonify({'error': 'Failed to get database stats'}), 500

@app.route('/cleanup-old-news', methods=['POST'])
def cleanup_old_news():
    """Clean up old news articles"""
    try:
        data = request.get_json()
        days_threshold = data.get('days', 90)
        
        if news_ai.freshness_validator:
            deleted_count = news_ai.freshness_validator.clean_old_articles_from_db(days_threshold)
            return jsonify({
                'deleted_count': deleted_count,
                'threshold_days': days_threshold,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({'error': 'Freshness validator not available'}), 503
            
    except Exception as e:
        app.logger.error(f"Cleanup error: {e}")
        return jsonify({'error': 'Cleanup failed'}), 500

@app.route('/force-news-collection', methods=['POST'])
def force_news_collection():
    """Manually trigger news collection"""
    try:
        articles = news_ai.fetch_and_store_news()
        
        if articles and news_ai.rag_available:
            news_ai.pathway_rag.add_news_articles(articles)
            
        return jsonify({
            'collected_count': len(articles),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'Successfully collected {len(articles)} new articles'
        })
        
    except Exception as e:
        app.logger.error(f"Manual collection error: {e}")
        return jsonify({'error': 'Collection failed'}), 500

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Test Ollama connection
        test_response = requests.post(
            news_ai.ollama_url,
            json={
                "model": news_ai.ollama_model,
                "prompt": "Hello",
                "stream": False
            },
            timeout=5
        )
        
        ollama_status = "connected" if test_response.status_code == 200 else "disconnected"
        
        # Test database connection
        try:
            conn = psycopg2.connect(**news_ai.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            db_status = "connected"
        except:
            db_status = "disconnected"
        
        return jsonify({
            'status': 'healthy' if ollama_status == 'connected' and db_status == 'connected' else 'partial',
            'ollama': ollama_status,
            'database': db_status,
            'rag_system': 'available' if news_ai.rag_available else 'unavailable',
            'freshness_validator': 'available' if news_ai.freshness_validator else 'unavailable',
            'comparison_tool': 'available' if news_ai.comparison_tool else 'unavailable',
            'news_collection': 'running' if news_ai.collection_running else 'stopped',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

if __name__ == '__main__':
    # Check Ollama connection on startup
    try:
        test_response = requests.post(
            news_ai.ollama_url,
            json={
                "model": news_ai.ollama_model,
                "prompt": "Hello",
                "stream": False
            },
            timeout=5
        )
        
        if test_response.status_code == 200:
            print("✅ Connected to Ollama Nemotron-mini")
        else:
            print("⚠️ Warning: Ollama connection failed")
            
    except Exception as e:
        print("❌ Error: Cannot connect to Ollama")
        print("Please make sure Ollama is running: ollama run nemotron-mini")
    
    print("🚀 Starting Live News AI Web App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
