#!/usr/bin/env python3
"""
வெளிச்சம் - Enhanced News Intelligence Platform - Flask Application
Real-time news with GNews API, News API, and Gemini RAG
"""

from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
from dotenv import load_dotenv
# Using direct HTTP requests for GNews.io API
from newsapi import NewsApiClient
from src.gemini_client import GeminiClient
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import re
import time
from threading import Lock

load_dotenv()
app = Flask(__name__)

class EnhancedNewsApp:
    """Enhanced News Application with multiple APIs and Gemini RAG"""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting for Gemini API
        self.gemini_request_times = []
        self.gemini_lock = Lock()
        self.max_requests_per_minute = 12  # Conservative limit
        self.quota_exceeded = False
        self.quota_reset_time = None
        
        # Initialize APIs
        self.setup_apis()
        
        # Initialize database
        self.setup_database()
        
        # Cache for articles
        self.cached_articles = []
        self.last_fetch_time = None
        
        # Fetch initial news
        self.fetch_all_news()
        
    def setup_apis(self):
        try:
            # Gemini AI Client
            self.gemini_client = GeminiClient(os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))
            self.gemini_available = self.gemini_client.available
            self.logger.info(f"✅ Gemini AI: {'Available' if self.gemini_available else 'Not Available'}")
            
            # GNews API
            self.gnews_api_key = os.getenv('GNEWS_API_KEY')
            if self.gnews_api_key:
                self.gnews_available = True
                self.logger.info("✅ GNews API key found")
            else:
                self.gnews_available = False
                self.logger.warning("⚠️ GNews API key not found")
            
            # News API Client
            newsapi_key = os.getenv('NEWS_API_KEY')
            if newsapi_key:
                self.newsapi_client = NewsApiClient(api_key=newsapi_key)
                self.newsapi_available = True
                self.logger.info("✅ News API Client initialized")
            else:
                self.newsapi_client = None
                self.newsapi_available = False
                self.logger.warning("⚠️ News API key not found")
                
        except Exception as e:
            self.logger.error(f"API setup error: {e}")
            
    def setup_database(self):
        """Setup PostgreSQL database"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'livenews_db'),
            'user': os.getenv('DB_USER', 'postgres'), 
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create enhanced articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source TEXT,
                    category TEXT,
                    author TEXT,
                    published_at TIMESTAMP,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    url TEXT,
                    image_url TEXT,
                    sentiment FLOAT,
                    reading_time INTEGER,
                    keywords TEXT[],
                    canonical_id TEXT UNIQUE
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            self.logger.info("✅ Database initialized")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            
    def fetch_gnews_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from GNews.io API"""
        articles = []
        
        if not self.gnews_available:
            return articles
            
        try:
            # Fetch top stories
            url = f"https://gnews.io/api/v4/top-headlines?lang=en&max=20&apikey={self.gnews_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('articles', []):
                article = {
                    'title': item.get('title', ''),
                    'content': item.get('description', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('name', 'GNews'),
                    'published_at': item.get('publishedAt', ''),
                    'category': 'General',
                    'image_url': item.get('image', ''),
                }
                if article['title'] and article['content']:
                    articles.append(article)
                    
            # Fetch technology news
            tech_url = f"https://gnews.io/api/v4/search?q=technology OR AI OR software&lang=en&max=15&apikey={self.gnews_api_key}"
            tech_response = requests.get(tech_url, timeout=10)
            tech_response.raise_for_status()
            
            tech_data = tech_response.json()
            
            for item in tech_data.get('articles', []):
                article = {
                    'title': item.get('title', ''),
                    'content': item.get('description', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('name', 'GNews'),
                    'published_at': item.get('publishedAt', ''),
                    'category': 'Technology',
                    'image_url': item.get('image', ''),
                }
                if article['title'] and article['content']:
                    articles.append(article)
                
            self.logger.info(f"Fetched {len(articles)} articles from GNews.io")
            
        except Exception as e:
            self.logger.error(f"GNews API fetch error: {e}")
            
        return articles
        
    def fetch_newsapi_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from News API"""
        articles = []
        
        if not self.newsapi_available:
            return articles
            
        try:
            # Fetch top headlines
            top_headlines = self.newsapi_client.get_top_headlines(
                country='us',
                page_size=50
            )
            
            for item in top_headlines.get('articles', []):
                if item.get('title') and item.get('description'):
                    article = {
                        'title': item.get('title', ''),
                        'content': item.get('description', '') + ' ' + (item.get('content', '') or ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', {}).get('name', 'News API'),
                        'author': item.get('author', ''),
                        'published_at': item.get('publishedAt', ''),
                        'category': 'Breaking News',
                        'image_url': item.get('urlToImage', ''),
                    }
                    articles.append(article)
                    
            # Fetch technology news
            tech_news = self.newsapi_client.get_everything(
                q='technology OR AI OR software',
                sort_by='publishedAt',
                page_size=20
            )
            
            for item in tech_news.get('articles', []):
                if item.get('title') and item.get('description'):
                    article = {
                        'title': item.get('title', ''),
                        'content': item.get('description', '') + ' ' + (item.get('content', '') or ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', {}).get('name', 'News API'),
                        'author': item.get('author', ''),
                        'published_at': item.get('publishedAt', ''),
                        'category': 'Technology',
                        'image_url': item.get('urlToImage', ''),
                    }
                    articles.append(article)
                    
            self.logger.info(f"Fetched {len(articles)} articles from News API")
            
        except Exception as e:
            self.logger.error(f"News API fetch error: {e}")
            
        return articles
        
    def can_make_gemini_request(self) -> bool:
        """Check if we can make a Gemini API request without hitting quota"""
        with self.gemini_lock:
            current_time = time.time()
            
            # Check if quota reset time has passed
            if self.quota_reset_time and current_time > self.quota_reset_time:
                self.quota_exceeded = False
                self.quota_reset_time = None
                self.gemini_request_times.clear()
            
            # If quota is exceeded, don't make request
            if self.quota_exceeded:
                return False
            
            # Remove requests older than 1 minute
            minute_ago = current_time - 60
            self.gemini_request_times = [t for t in self.gemini_request_times if t > minute_ago]
            
            # Check if we can make another request
            return len(self.gemini_request_times) < self.max_requests_per_minute
    
    def record_gemini_request(self, success: bool = True):
        """Record a Gemini API request"""
        with self.gemini_lock:
            current_time = time.time()
            if success:
                self.gemini_request_times.append(current_time)
            else:
                # If quota exceeded, set reset time
                self.quota_exceeded = True
                self.quota_reset_time = current_time + 65  # Wait 65 seconds
    
    def detect_fake_news(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential fake news using AI analysis"""
        if not self.model_available or self.quota_exceeded:
            return article
            
        try:
            title = article.get('title', '')
            content = article.get('content', article.get('description', ''))
            source = article.get('source', '')
            
            prompt = f"""Analyze this news article for potential fake news indicators. Consider:
1. Source credibility
2. Content consistency and factual accuracy
3. Sensational language or clickbait patterns
4. Logical consistency
5. Verifiable claims

Article Title: {title}
Source: {source}
Content: {content[:800]}...

Provide analysis in this format:
FAKE_NEWS_SCORE: [0-100, where 0 is definitely real, 100 is definitely fake]
CREDIBILITY: [HIGH/MEDIUM/LOW]
RED_FLAGS: [list any concerning elements]
REASONING: [brief explanation]"""

            response = self.gemini_client.generate_response(prompt)
            
            if response:
                lines = response.strip().split('\n')
                fake_score = 0
                credibility = "MEDIUM"
                red_flags = []
                reasoning = ""
                
                for line in lines:
                    if line.startswith('FAKE_NEWS_SCORE:'):
                        try:
                            fake_score = int(line.replace('FAKE_NEWS_SCORE:', '').strip())
                        except:
                            fake_score = 0
                    elif line.startswith('CREDIBILITY:'):
                        credibility = line.replace('CREDIBILITY:', '').strip()
                    elif line.startswith('RED_FLAGS:'):
                        flags_str = line.replace('RED_FLAGS:', '').strip()
                        red_flags = [flag.strip() for flag in flags_str.split(',') if flag.strip()]
                    elif line.startswith('REASONING:'):
                        reasoning = line.replace('REASONING:', '').strip()
                
                article['fake_news_score'] = fake_score
                article['credibility'] = credibility
                article['red_flags'] = red_flags
                article['fake_news_reasoning'] = reasoning
                
        except Exception as e:
            self.logger.error(f"Fake news detection error: {e}")
        
        return article

    def enhance_article_with_ai(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced article processing with Gemini AI including fake news detection"""
        if not self.model_available:
            return article
            
        if self.quota_exceeded:
            return article
            
        # Rate limiting check
        with self.gemini_lock:
            current_time = time.time()
            # Remove requests older than 1 minute
            self.gemini_request_times = [t for t in self.gemini_request_times if current_time - t < 60]
            
            if len(self.gemini_request_times) >= self.max_requests_per_minute:
                self.logger.warning("Rate limit reached, skipping AI enhancement")
                return article
                
            self.gemini_request_times.append(current_time)
        
        try:
            title = article.get('title', '')
            content = article.get('content', article.get('description', ''))
            source = article.get('source', '')
            
            prompt = f"""Analyze this news article and provide:
1. A concise 2-sentence summary
2. Key topics/tags (comma-separated)
3. Fake news analysis (score 0-100 where 0=real, 100=fake)
4. Source credibility (HIGH/MEDIUM/LOW)

Article Title: {title}
Source: {source}
Content: {content[:500]}...

Format your response as:
SUMMARY: [your summary]
TAGS: [tag1, tag2, tag3]
FAKE_SCORE: [0-100]
CREDIBILITY: [HIGH/MEDIUM/LOW]"""

            response = self.gemini_client.generate_response(prompt)
            
            if response:
                # Parse AI response
                lines = response.strip().split('\n')
                ai_summary = ""
                ai_tags = []
                fake_score = 0
                credibility = "MEDIUM"
                
                for line in lines:
                    if line.startswith('SUMMARY:'):
                        ai_summary = line.replace('SUMMARY:', '').strip()
                    elif line.startswith('TAGS:'):
                        tags_str = line.replace('TAGS:', '').strip()
                        ai_tags = [tag.strip() for tag in tags_str.split(',')]
                    elif line.startswith('FAKE_SCORE:'):
                        try:
                            fake_score = int(line.replace('FAKE_SCORE:', '').strip())
                        except:
                            fake_score = 0
                    elif line.startswith('CREDIBILITY:'):
                        credibility = line.replace('CREDIBILITY:', '').strip()
                
                # Update article with AI enhancements
                if ai_summary:
                    article['ai_summary'] = ai_summary
                if ai_tags:
                    article['ai_tags'] = ai_tags
                article['fake_news_score'] = fake_score
                article['credibility'] = credibility
                article['ai_enhanced'] = True
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                self.quota_exceeded = True
                self.quota_reset_time = time.time() + 60  # Reset after 1 minute
                self.logger.error(f"Gemini quota exceeded: {e}")
            else:
                self.logger.error(f"AI enhancement error: {e}")
        
        return article
        
    def store_article(self, article: Dict[str, Any]) -> bool:
        """Store enhanced article in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Generate canonical ID for deduplication
            canonical_id = hashlib.sha256(
                (article.get('title', '') + article.get('url', '')).encode('utf-8')
            ).hexdigest()
            
            # Parse published date
            published_at = None
            if article.get('published_at'):
                try:
                    if 'T' in str(article['published_at']):
                        published_at = datetime.fromisoformat(str(article['published_at']).replace('Z', '+00:00'))
                    else:
                        published_at = datetime.now()
                except:
                    published_at = datetime.now()
            else:
                published_at = datetime.now()
            
            cursor.execute("""
                INSERT INTO enhanced_articles 
                (title, content, summary, source, category, author, published_at, url, image_url, 
                 sentiment, reading_time, keywords, canonical_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (canonical_id) DO NOTHING
            """, (
                article.get('title', ''),
                article.get('content', ''),
                article.get('summary', ''),
                article.get('source', ''),
                article.get('category', ''),
                article.get('author', ''),
                published_at,
                article.get('url', ''),
                article.get('image_url', ''),
                article.get('sentiment', 0.0),
                article.get('reading_time', 1),
                article.get('keywords', []),
                canonical_id
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Storage error: {e}")
            return False
            
    def fetch_all_news(self):
        """Fetch news from all sources and enhance with AI"""
        self.logger.info("🔄 Fetching news from all sources...")
        
        all_articles = []
        
        # Fetch from GNews
        gnews_articles = self.fetch_gnews_articles()
        all_articles.extend(gnews_articles)
        
        # Fetch from News API
        newsapi_articles = self.fetch_newsapi_articles()
        all_articles.extend(newsapi_articles)
        
        # Enhance and store articles (with rate limiting)
        enhanced_articles = []
        ai_enhanced_count = 0
        
        for i, article in enumerate(all_articles):
            if article.get('title') and len(article.get('title', '')) > 10:
                # Only AI-enhance first 10 articles to avoid quota issues
                if ai_enhanced_count < 10:
                    enhanced_article = self.enhance_article_with_ai(article)
                    if enhanced_article.get('keywords'):  # Check if AI enhancement worked
                        ai_enhanced_count += 1
                else:
                    # Use fallback enhancement for remaining articles
                    enhanced_article = self.enhance_article_with_ai(article)
                
                if self.store_article(enhanced_article):
                    enhanced_articles.append(enhanced_article)
                    
        self.cached_articles = enhanced_articles[:50]  # Keep top 50 for display
        self.last_fetch_time = datetime.now()
        
        self.logger.info(f"✅ Enhanced and stored {len(enhanced_articles)} articles ({ai_enhanced_count} with AI)")
        
    def get_articles_from_db(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get articles from database for display"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM enhanced_articles 
                WHERE published_at > NOW() - INTERVAL '7 days'
                ORDER BY published_at DESC 
                LIMIT %s
            """, (limit,))
            
            articles = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to dict and format
            formatted_articles = []
            for article in articles:
                formatted_article = dict(article)
                # Format dates
                if formatted_article.get('published_at'):
                    formatted_article['published_at'] = formatted_article['published_at'].strftime('%Y-%m-%d %H:%M')
                if formatted_article.get('collected_at'):
                    formatted_article['collected_at'] = formatted_article['collected_at'].strftime('%Y-%m-%d %H:%M')
                    
                formatted_articles.append(formatted_article)
                
            return formatted_articles
            
        except Exception as e:
            self.logger.error(f"Database fetch error: {e}")
            return self.cached_articles[:limit] if self.cached_articles else []
            
    def chat_with_news(self, user_query: str) -> str:
        """Chat with news using Gemini RAG"""
        if not self.gemini_available:
            return "I'm currently working on enhancing my AI capabilities. Please try asking about specific news topics."
            
        try:
            # Get relevant articles from database
            articles = self.get_articles_from_db(10)
            
            # Create context for RAG
            context = "Recent News Articles:\n\n"
            for i, article in enumerate(articles[:5], 1):
                context += f"{i}. {article['title']}\n"
                context += f"   Source: {article['source']}\n"
                context += f"   Summary: {article['summary']}\n"
                context += f"   Published: {article.get('published_at', 'Unknown')}\n\n"
                
            # Generate RAG response
            prompt = f"""
            You are an intelligent news assistant. Answer the user's question based on the recent news articles provided.
            Be conversational, informative, and cite relevant information from the articles when appropriate.
            
            {context}
            
            User Question: {user_query}
            
            Provide a helpful response based on the available news information.
            """
            
            response = self.gemini_client.generate_response(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return "I'm having trouble processing your request right now. Please try again."

# Initialize the app
news_app = EnhancedNewsApp()

# Flask Routes
@app.route('/')
def index():
    """Main news landing page"""
    articles = news_app.get_articles_from_db(30)
    
    # Get stats
    total_articles = len(articles)
    sources = len(set(article['source'] for article in articles))
    categories = len(set(article['category'] for article in articles))
    ai_enhanced = len([a for a in articles if a.get('keywords')])
    
    stats = {
        'total_articles': total_articles,
        'sources_count': sources,
        'categories_count': categories,
        'ai_enhanced': ai_enhanced,
        'last_updated': news_app.last_fetch_time.strftime('%H:%M') if news_app.last_fetch_time else 'Loading...',
        'quota_status': 'OK' if not news_app.quota_exceeded else 'Limited'
    }  
    return render_template('news_index.html', articles=articles, stats=stats)

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    """Individual article page"""
    try:
        conn = psycopg2.connect(**news_app.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM enhanced_articles WHERE id = %s", (article_id,))
        article = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if article:
            article = dict(article)
            if article.get('published_at'):
                article['published_at'] = article['published_at'].strftime('%Y-%m-%d %H:%M')
            return render_template('article_detail.html', article=article)
        else:
            return "Article not found", 404
            
    except Exception as e:
        return f"Error loading article: {e}", 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for AI assistant"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Get AI response
        ai_response = news_app.chat_with_news(user_message)
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh')
def refresh_news():
    """Manually refresh news feed"""
    try:
        news_app.fetch_all_news()
        return jsonify({'status': 'success', 'message': 'News refreshed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """Get live statistics"""
    try:
        articles = news_app.get_articles_from_db(100)
        
        # Calculate stats
        total_articles = len(articles)
        today_articles = len([a for a in articles if a.get('published_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
        
        sentiment_scores = [a.get('sentiment', 0) for a in articles if a.get('sentiment') is not None]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        sources = list(set(article['source'] for article in articles))
        categories = list(set(article['category'] for article in articles))
        
        return jsonify({
            'total_articles': total_articles,
            'today_articles': today_articles,
            'avg_sentiment': round(avg_sentiment, 2),
            'sources': sources,
            'categories': categories,
            'last_updated': news_app.last_fetch_time.strftime('%H:%M:%S') if news_app.last_fetch_time else 'Loading...'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoints for Next.js frontend
@app.route('/api/news', methods=['GET'])
def api_get_news():
    """API endpoint for Next.js frontend to fetch news"""
    try:
        location = request.args.get('location')
        category = request.args.get('category')
        country = request.args.get('country')
        city = request.args.get('city')
        tags = request.args.get('tags')
        
        # Get articles from the news app
        articles = news_app.get_display_articles()
        
        # Filter by category if specified
        if category and category != 'all':
            articles = [a for a in articles if a.get('category', '').lower() == category.lower()]
        
        # Filter by location if specified
        if country:
            articles = [a for a in articles if country.lower() in a.get('title', '').lower() or 
                       country.lower() in a.get('summary', '').lower()]
        
        if city:
            articles = [a for a in articles if city.lower() in a.get('title', '').lower() or 
                       city.lower() in a.get('summary', '').lower()]
        
        # Filter by tags if specified
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',')]
            articles = [a for a in articles if any(tag in a.get('title', '').lower() or 
                       tag in a.get('summary', '').lower() for tag in tag_list)]
        
        # Format articles for frontend
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                'id': article.get('id', ''),
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'content': article.get('content', ''),
                'source': article.get('source', ''),
                'publishedAt': article.get('date', ''),
                'category': article.get('category', 'General'),
                'url': article.get('url', '#'),
                'readingTime': article.get('reading_time', 2),
                'sentiment': 0.0,  # Default sentiment
                'keywords': []
            })
        
        # Get available categories and tags
        all_articles = news_app.get_display_articles()
        categories = list(set(a.get('category', 'General') for a in all_articles))
        
        return jsonify({
            'articles': formatted_articles,
            'total': len(formatted_articles),
            'location': {'country': country, 'city': city},
            'categories': categories,
            'availableTags': []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/query', methods=['POST'])
def api_rag_query():
    """API endpoint for RAG queries from Next.js frontend"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use the news app's chat functionality with fake news detection
        response = news_app.chat_with_news(query)
        
        # Check if query is asking about specific news and add fake news analysis
        fake_news_warning = ""
        if any(keyword in query.lower() for keyword in ['news', 'article', 'story', 'report', 'claim']):
            # Get recent articles to check for fake news
            articles = news_app.get_display_articles()[:5]
            high_risk_articles = [a for a in articles if a.get('fake_news_score', 0) > 70]
            
            if high_risk_articles:
                fake_news_warning = f"\n\n⚠️ **Fake News Alert**: {len(high_risk_articles)} recent articles have high fake news risk scores. Please verify information from multiple reliable sources."
        
        return jsonify({
            'answer': response + fake_news_warning,
            'sources': [],
            'fake_news_warning': fake_news_warning if fake_news_warning else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/brief', methods=['POST'])
def api_ai_brief():
    """API endpoint for AI brief generation"""
    try:
        data = request.get_json()
        location = data.get('location', {})
        preferences = data.get('preferences', {})
        
        # Generate a brief using latest news
        articles = news_app.get_display_articles()[:5]  # Top 5 articles
        
        if not articles:
            return jsonify({
                'brief': 'No recent news articles available at the moment.',
                'articles': []
            })
        
        # Create a simple brief
        brief_text = f"Here's your வெளிச்சம் news brief:\n\n"
        for i, article in enumerate(articles, 1):
            brief_text += f"{i}. {article.get('title', '')}\n"
            brief_text += f"   {article.get('summary', '')[:100]}...\n\n"
        
        return jsonify({
            'brief': brief_text,
            'articles': articles
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Enable CORS for Next.js frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
