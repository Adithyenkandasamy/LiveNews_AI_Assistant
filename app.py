#!/usr/bin/env python3
"""
Live News AI Flask Web Application
Real-time news chatbot using Google Gemini AI
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
from src.rag_system import PathwayRAGSystem, PathwayRAGConfig
from src.freshness_validator import NewsFreshnessValidator, FreshnessConfig
from src.comparison_tool import NewsComparisonTool
# from src.gemini_client import GeminiClient  # Removed API dependency
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import hashlib
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

class வெளிச்சம்AI:
    """Real-time news AI chatbot"""
    
    def __init__(self):
        # No external API dependencies - work offline
        self.model_available = False
        self.gemini_client = None
        
        
        # News sources  
        self.news_feeds = {
            'BBC World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC Tech': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'BBC India': 'http://feeds.bbci.co.uk/news/world/asia/india/rss.xml',
            'CNN': 'http://rss.cnn.com/rss/edition.rss',
            'Reuters': 'https://feeds.reuters.com/reuters/topNews',
            'Reuters India': 'https://feeds.reuters.com/reuters/INtopNews',
            'The Hindu': 'https://www.thehindu.com/news/national/feeder/default.rss',
            'Indian Express': 'https://indianexpress.com/section/india/feed/',
            'NDTV India': 'https://feeds.feedburner.com/ndtvnews-india-news',
            'Times of India': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            'Al Jazeera India': 'https://www.aljazeera.com/xml/rss/all.xml',  # Will filter by query
            'Reddit WorldNews': 'https://www.reddit.com/r/worldnews.json',
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Google News Tech': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
            'Hacker News': 'https://hnrss.org/frontpage'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store articles in memory for Flask routes
        self.cached_articles = []
        self.last_fetch_time = None
        
        # Database configuration
        self.setup_database()
        
        # Skip RAG system setup - work without external APIs
        self.rag_available = False
        self.pathway_rag = None
        self.freshness_validator = None
        self.comparison_tool = None
        
        # Start background news collection
        self.collection_running = False
        self.start_news_collection()
        
    def setup_rag_system(self):
        """Skip RAG system - work offline without APIs"""
        self.rag_available = False
        self.pathway_rag = None
        self.freshness_validator = None
        self.logger.info("✅ Working offline - no external APIs required")
            
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
                    canonical_id TEXT
                )
            """)

            # Ensure canonical_id column exists (for older deployments)
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='news_articles' AND column_name='canonical_id'
                    ) THEN
                        ALTER TABLE news_articles ADD COLUMN canonical_id TEXT;
                    END IF;
                END$$;
            """)

            # Ensure canonical_id empty strings are normalized to NULL for uniqueness
            cursor.execute("UPDATE news_articles SET canonical_id = NULL WHERE canonical_id = ''")

            # Create a proper UNIQUE constraint usable by ON CONFLICT
            cursor.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_name = 'news_articles' AND constraint_name = 'uniq_news_canonical'
                    ) THEN
                        ALTER TABLE news_articles
                        ADD CONSTRAINT uniq_news_canonical UNIQUE (canonical_id);
                    END IF;
                END$$;
                """
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            self.logger.info("✅ Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
        
        # Backfill canonical IDs for existing rows (run outside transaction)
        try:
            self._backfill_canonical_ids()
        except Exception as e:
            self.logger.error(f"Canonical ID backfill failed: {e}")
            
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
            
            # Clear old articles from database first
            cleared_count = self.clear_old_articles()
            
            collection_thread = threading.Thread(target=self._news_collection_worker, daemon=True)
            collection_thread.start()
            self.logger.info("🔄 Started background news collection")
            
    def _news_collection_worker(self):
        """Background worker for collecting news"""
        while self.collection_running:
            try:
                self.logger.info("📰 Collecting latest news...")
                articles = self.fetch_and_store_news()
                
                # Cache articles for Flask routes
                self.cached_articles = articles[:30]  # Keep latest 30 for display
                self.last_fetch_time = datetime.now()
                    
                self.logger.info(f"✅ Collected {len(articles)} articles")
                
            except Exception as e:
                self.logger.error(f"News collection error: {e}")
                
            # Wait 1 minute before next collection for real-time updates
            time.sleep(60)
            
    def fetch_and_store_news(self) -> List[Dict[str, Any]]:
        """Fetch news and store in database"""
        all_articles = []
        
        for source_name, feed_url in self.news_feeds.items():
            try:
                self.logger.info(f"Fetching from {source_name}...")
                
                # Handle Reddit JSON API differently
                if 'reddit.com' in feed_url and feed_url.endswith('.json'):
                    articles_from_source = self._fetch_reddit_articles(source_name, feed_url)
                    all_articles.extend(articles_from_source)
                    continue
                
                # Handle RSS feeds
                feed = feedparser.parse(feed_url)
                self.logger.info(f"Found {len(feed.entries)} entries from {source_name}")
                
                for entry in feed.entries[:5]:  # Top 5 from each source for speed
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        # Get full content without artificial limits first
                        full_content = entry.get('summary', '') or entry.get('description', '') or entry.get('content', '')
                        
                        # Skip articles older than 7 days to prevent old news
                        age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                        if age_hours > 168:  # 7 days in hours
                            continue
                        
                        # Additional check for old news content patterns
                        title_and_content = (entry.get('title', '') + ' ' + str(full_content)).lower()
                        old_indicators = [
                            '2023', '2022', '2021', 'last year', 'years ago', 
                            'months ago', 'following a 2023', 'since 2023',
                            'last month', 'previous year'
                        ]
                        if any(indicator in title_and_content for indicator in old_indicators):
                            continue
                        
                        # Handle content that might be a list (fix Indian Express parsing errors)
                        if isinstance(full_content, list):
                            full_content = ' '.join(str(item) for item in full_content if item)
                        elif not isinstance(full_content, str):
                            full_content = str(full_content) if full_content else ''
                        
                        # Validate content exists and is meaningful
                        if not full_content or len(full_content.strip()) < 20:
                            self.logger.warning(f"Skipping article with insufficient content: {entry.get('title', '')[:50]}...")
                            continue
                            
                        article = {
                            'title': entry.get('title', ''),
                            'content': full_content,  # No artificial limit
                            'source': source_name,
                            'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'url': entry.get('link', ''),
                            'category': self._categorize_article(entry.get('title', '') + ' ' + full_content[:500])  # Use more content for categorization
                        }
                        
                        # Store in database
                        if self._store_article(article):
                            all_articles.append(article)
                            self.logger.info(f"Added article: {article['title'][:50]}...")
                        else:
                            self.logger.warning(f"Failed to store article: {article['title'][:50]}...")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing article: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
                
        self.logger.info(f"Total articles collected and stored: {len(all_articles)}")
        return all_articles
        
    def _fetch_reddit_articles(self, source_name: str, json_url: str) -> List[Dict[str, Any]]:
        """Fetch articles from Reddit JSON API"""
        try:
            headers = {'User-Agent': 'வெளிச்சம்-AI-Bot/1.0'}
            response = requests.get(json_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for post in data['data']['children'][:5]:  # Top 5 posts
                post_data = post['data']
                
                # Skip if it's a deleted/removed post
                if post_data.get('removed_by_category') or not post_data.get('title'):
                    continue
                
                # Get post content - use selftext for text posts, or url for link posts
                content = post_data.get('selftext', '') or post_data.get('url', '')
                if not content or len(content.strip()) < 20:
                    continue
                
                # Convert Reddit timestamp to datetime
                created_utc = post_data.get('created_utc', 0)
                pub_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
                
                article = {
                    'title': post_data.get('title', ''),
                    'content': content,
                    'source': 'Reddit WorldNews',  # Use consistent source name
                    'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'category': 'World News'
                }
                articles.append(article)
                
            self.logger.info(f"Fetched {len(articles)} articles from Reddit WorldNews")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Reddit articles: {e}")
            return []
            
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
        """Store article in database. Treat duplicates as success (idempotent)."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Compute canonical_id for robust deduplication
            canonical_id = self._make_canonical_id(article)
            if not canonical_id:
                canonical_id = None

            cursor.execute(
                """
                INSERT INTO news_articles (title, content, source, category, collected_date, url, canonical_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT uniq_news_canonical DO NOTHING
                """,
                (
                    article['title'],
                    article['content'],
                    article['source'],
                    article['category'],
                    article['date'],
                    article['url'],
                    canonical_id,
                ),
            )

            conn.commit()
            cursor.close()
            conn.close()

            # If no exception, insertion succeeded or was a duplicate (idempotent)
            return True

        except Exception as e:
            self.logger.error(f"Database storage error: {e}")
            return False

    def _make_canonical_id(self, article: Dict[str, Any]) -> str:
        """Create a stable hash for an article to deduplicate across runs.
        Uses normalized title only as primary key - same story from different sources should be considered duplicates.
        """
        try:
            title = (article.get('title') or '').strip().lower()
            
            if not title or len(title) < 10:
                return ''
            
            # Clean and normalize title for better deduplication
            # Remove common prefixes, quotes, and normalize whitespace
            title = re.sub(r'^["\']*', '', title)  # Remove leading quotes
            title = re.sub(r'["\']*$', '', title)  # Remove trailing quotes
            title = re.sub(r'\s+', ' ', title)    # Normalize whitespace
            title = title.strip()
            
            # Use only title for canonical ID so same story from different sources is detected
            return hashlib.sha256(title.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating canonical ID: {e}")
            return ''

    def clear_old_articles(self):
        """Clear old articles from database to refresh storage"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Delete articles older than 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            cursor.execute(
                """
                DELETE FROM news_articles 
                WHERE collected_date < %s
                """,
                (cutoff_date,)
            )
            
            old_articles_deleted = cursor.rowcount
            
            # Remove duplicate articles (keep the most recent one for each canonical_id)
            cursor.execute(
                """
                DELETE FROM news_articles 
                WHERE id NOT IN (
                    SELECT DISTINCT ON (canonical_id) id
                    FROM news_articles 
                    WHERE canonical_id IS NOT NULL
                    ORDER BY canonical_id, collected_date DESC
                )
                AND canonical_id IS NOT NULL
                """
            )
            
            duplicates_deleted = cursor.rowcount
            
            # Also delete from pathway_articles if it exists
            try:
                cursor.execute(
                    """
                    DELETE FROM pathway_articles 
                    WHERE date < %s
                    """,
                    (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),)
                )
            except:
                pass  # Table might not exist
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"🗑️ Cleared {old_articles_deleted} old articles and {duplicates_deleted} duplicates from database")
            return old_articles_deleted + duplicates_deleted
            
        except Exception as e:
            self.logger.error(f"Failed to clear old articles: {e}")
            return 0

    def _backfill_canonical_ids(self, batch_size: int = 1000):
        """Populate canonical_id for rows where it is NULL, in batches."""
        processed = 0
        while True:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, source, url, content
                FROM news_articles
                WHERE canonical_id IS NULL
                LIMIT %s
                """,
                (batch_size,)
            )
            rows = cursor.fetchall()
            if not rows:
                cursor.close()
                conn.close()
                break
            for row in rows:
                row_id, title, source, url, content = row
                art = {'title': title or '', 'source': source or '', 'url': url or '', 'content': content or ''}
                cid = self._make_canonical_id(art)
                cursor.execute(
                    "UPDATE news_articles SET canonical_id = %s WHERE id = %s",
                    (cid, row_id)
                )
                processed += 1
            conn.commit()
            cursor.close()
            conn.close()
        if processed:
            self.logger.info(f"🔧 Backfilled canonical_id for {processed} existing articles")
        
    def fetch_latest_news(self, hours_back: int = 12, query: str | None = None) -> List[Dict[str, Any]]:
        """Fetch latest news from multiple sources. If query provided, prioritize matching items (e.g., India)."""
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        q = (query or '').lower()
        want_india = any(term in q for term in ['india', 'indian', 'delhi', 'mumbai', 'bengaluru', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'modi'])
        
        for source_name, feed_url in self.news_feeds.items():
            try:
                self.logger.info(f"Fetching from {source_name}...")
                
                # Handle Reddit JSON API differently
                if 'reddit.com' in feed_url and feed_url.endswith('.json'):
                    articles_from_source = self._fetch_reddit_articles(source_name, feed_url)
                    for article in articles_from_source:
                        # Check freshness for Reddit articles too
                        try:
                            article_date = datetime.strptime(article['date'], '%Y-%m-%d %H:%M:%S')
                            age_hours = (datetime.now() - article_date).total_seconds() / 3600
                            if age_hours > 168:  # 7 days in hours
                                continue
                        except:
                            pass
                        all_news.append(article)
                    continue
                
                # Handle RSS feeds
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Top 5 from each source
                    # Parse date - be more lenient
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            # If no date, assume it's recent
                            pub_date = datetime.now() - timedelta(hours=1)
                    except:
                        pub_date = datetime.now() - timedelta(hours=1)
                    
                    # Get full content without limits first
                    full_content = entry.get('summary', '') or entry.get('description', '') or entry.get('content', '')
                    
                    # Handle content that might be a list (fix Indian Express parsing errors)
                    if isinstance(full_content, list):
                        full_content = ' '.join(str(item) for item in full_content if item)
                    elif not isinstance(full_content, str):
                        full_content = str(full_content) if full_content else ''
                    
                    # Skip articles older than 7 days to prevent old news
                    age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                    if age_hours > 168:  # 7 days in hours
                        continue
                    
                    # Additional check for old news content patterns
                    title_and_content = (entry.get('title', '') + ' ' + str(full_content)).lower()
                    old_indicators = [
                        '2023', '2022', '2021', 'last year', 'years ago', 
                        'months ago', 'following a 2023', 'since 2023',
                        'last month', 'previous year', 'adani group'
                    ]
                    if any(indicator in title_and_content for indicator in old_indicators):
                        continue
                    
                    # Skip if no meaningful content
                    if not full_content or len(full_content.strip()) < 20:
                        self.logger.warning(f"Skipping article with no content: {entry.get('title', '')[:50]}...")
                        continue
                        
                    # Enhanced date formatting with age indication
                    date_str = pub_date.strftime('%Y-%m-%d %H:%M')
                    age = datetime.now() - pub_date
                    if age.days > 7:
                        age_indicator = f" ({age.days} days old)"
                    elif age.days > 0:
                        age_indicator = f" ({age.days}d ago)"
                    elif age.seconds > 3600:
                        hours = age.seconds // 3600
                        age_indicator = f" ({hours}h ago)"
                    else:
                        age_indicator = " (recent)"
                    
                    news_item = {
                        'title': entry.get('title', 'No title'),
                        'summary': full_content,  # Full content available
                        'source': source_name,
                        'date': date_str,
                        'date_with_age': date_str + age_indicator,
                        'link': entry.get('link', ''),
                        'content': full_content,  # No artificial limit
                        'age_days': age.days,
                        'is_old': age.days > 3  # Mark as old if > 3 days
                    }
                    all_news.append(news_item)
                    self.logger.info(f"Added article: {news_item['title'][:50]}...")
                        
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source_name}: {e}")
                
        # Query-aware prioritization
        if want_india:
            cities = ['india', 'indian', 'delhi', 'mumbai', 'bengaluru', 'bangalore', 'chennai', 'kolkata', 'hyderabad']
            def is_india_item(item: Dict[str, Any]) -> bool:
                text = (item.get('title','') + ' ' + item.get('summary','')).lower()
                if any(c in text for c in cities):
                    return True
                return item.get('source','').lower() in [
                    'bbc india','reuters india','the hindu','indian express','ndtv india','times of india'
                ]
            all_news.sort(key=lambda x: (not is_india_item(x), x['date']), reverse=False)
        else:
            # Sort by date (newest first)
            all_news.sort(key=lambda x: x['date'], reverse=True)
        self.logger.info(f"Total articles collected: {len(all_news)}")
        
        # Simple freshness filtering without external APIs
        # Remove articles older than 3 days
        fresh_news = []
        for article in all_news:
            try:
                article_date = datetime.strptime(article['date'], '%Y-%m-%d %H:%M')
                if (datetime.now() - article_date).days <= 3:
                    fresh_news.append(article)
            except:
                fresh_news.append(article)  # Keep if date parsing fails
        all_news = fresh_news
            
        return all_news[:15]
        
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
        """Simple chat function without external APIs"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Simple keyword-based responses without external AI
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return f"Hello! I'm your FlashFeed AI assistant. Ask me about recent news or type 'latest news' to see current headlines."
            
        if any(word in user_lower for word in ['news', 'latest', 'headlines', 'today']):
            # Return latest cached articles as simple text
            if self.cached_articles:
                response = f"📰 Latest News ({current_time}):\n\n"
                for i, article in enumerate(self.cached_articles[:5], 1):
                    response += f"{i}. {article['title']}\n   Source: {article['source']} | {article.get('date_with_age', article['date'])}\n\n"
                return response
            else:
                return "I'm currently collecting the latest news. Please try again in a moment."
                
        return "I can help you with the latest news headlines. Try asking 'What's the latest news?' or 'Show me today's headlines'."
    
    def get_articles_for_display(self) -> List[Dict[str, Any]]:
        """Get articles formatted for web display"""
        if not self.cached_articles:
            # Try to fetch fresh articles if cache is empty
            articles = self.fetch_and_store_news()
            self.cached_articles = articles[:30]
        
        # Format articles for display
        display_articles = []
        for article in self.cached_articles:
            display_article = {
                'title': article.get('title', 'No Title'),
                'content': article.get('content', '')[:500] + '...',
                'summary': article.get('content', '')[:200] + '...',
                'source': article.get('source', 'Unknown'),
                'category': article.get('category', 'General'),
                'date': article.get('date', ''),
                'url': article.get('url', '#'),
                'reading_time': len(article.get('content', '').split()) // 200 + 1  # Rough estimate
            }
            display_articles.append(display_article)
        
        return display_articles

    def chat_with_news(self, user_input: str) -> str:
        """Enhanced chat with news context using Gemini"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if it's a news-related query
        if self._is_news_query(user_input):
            # Try RAG first
            if hasattr(self, 'rag_system') and self.rag_system:
                try:
                    rag_response = self.rag_system.query(user_input)
                    if rag_response and rag_response.strip():
                        return rag_response
                except Exception as e:
                    self.logger.error(f"RAG query failed: {e}")
            
            # Fallback to simple news fetch
            topic = self._extract_topic(user_input)
            latest_news = self.fetch_latest_news(24, user_input)
            
            if latest_news:
                news_context = f"Current Date & Time: {current_time}\n\n"
                news_context += "Latest News Headlines:\n"
                
                for i, news in enumerate(latest_news[:5], 1):
                    news_context += f"{i}. {news['title']}\n"
                    news_context += f"   Source: {news['source']} | {news['date']}\n"
                    news_context += f"   Summary: {news['summary']}\n\n"
                
                if not self.model_available:
                    bullets = [
                        f"- {n['title']} — {n['source']} ({n['date']})" for n in latest_news[:3]
                    ]
                    return "Here are the latest updates:\n" + "\n".join(bullets)
                
                prompt = f"""You are a helpful AI assistant with access to real-time news. Based on the current news below, answer the user's question naturally and conversationally.

{news_context}

User: {user_input}

Provide a natural, informative response using the news information above. Be conversational and helpful."""
            else:
                if topic:
                    return f"No recent news found on {topic}."
                prompt = f"""You are a helpful AI assistant. The user asked about news but no recent articles were found. Current time: {current_time}

User: {user_input}

Explain that you don't have access to the very latest news and suggest they check news websites directly."""
                
        else:
            # Normal conversation (non-news)
            prompt = f"""You are a helpful AI assistant. Current time: {current_time}

User: {user_input}

Give a brief, direct response (1-2 sentences)."""
        
        # Generate response using Gemini
        try:
            if not self.model_available:
                return "Sorry, AI model is not available right now."
                
            answer = self.gemini_client.generate_response(
                prompt, 
                max_tokens=120, 
                temperature=0.7
            )
            return answer if answer else "I'm not sure how to respond to that."
                
        except Exception as e:
            self.logger.error(f"Gemini request failed: {e}")
            # Fallback: return bullet summary if we have recent news
            latest_news = self.fetch_latest_news(24, user_input)
            if latest_news:
                bullets = []
                bullets = [
                    f"- {n['title']} — {n['source']} ({n['date']})" for n in latest_news[:3]
                ]
                return "Here are the latest updates:\n" + "\n".join(bullets)
            # If user asked for a specific topic and nothing found, respond clearly
            if topic:
                return f"No hot news recently on {topic}."
            return "Sorry, I encountered an error while processing your message."

    def _extract_topic(self, text: str) -> str | None:
        """Extract a specific topic from user query only when explicitly mentioned.
        Returns normalized name only for very specific topic requests, else None.
        """
        if not text:
            return None
        t = text.lower()
        
        # Only extract topic if user explicitly mentions it with specific context
        specific_patterns = [
            ('karnataka', ['karnataka news', 'news from karnataka', 'about karnataka']),
            ('bengaluru', ['bengaluru news', 'bangalore news', 'news from bengaluru', 'news from bangalore']),
            ('india', ['india news', 'indian news', 'news from india', 'about india']),
            ('delhi', ['delhi news', 'news from delhi', 'about delhi']),
            ('mumbai', ['mumbai news', 'news from mumbai', 'about mumbai']),
            ('technology', ['technology news', 'tech news', 'about technology', 'tech updates']),
            ('politics', ['political news', 'politics news', 'about politics', 'political updates']),
            ('business', ['business news', 'economic news', 'about business', 'market news']),
            ('sports', ['sports news', 'about sports', 'sports updates']),
            ('health', ['health news', 'medical news', 'about health']),
            ('science', ['science news', 'scientific news', 'about science'])
        ]
        
        for topic, patterns in specific_patterns:
            if any(pattern in t for pattern in patterns):
                return topic.capitalize()
        return None

    def _is_existence_query(self, text: str) -> bool:
        """Detect if the user is asking whether there is any news about a topic (yes/no intent)."""
        if not text:
            return False
        tl = text.lower()
        patterns = [
            'any news', 'hot news', 'is there news', 'is there any news', 'updates on', 'anything new', 'recent news', 'latest on'
        ]
        return any(p in tl for p in patterns)

    def _wants_summary(self, text: str) -> bool:
        """Detect if the user asked for a brief/summarized response."""
        if not text:
            return False
        tl = text.lower()
        return any(k in tl for k in ['summary', 'summarised', 'summarized', 'brief', 'short'])
    
    def _wants_full_text(self, text: str) -> bool:
        """Detect if the user wants full article text."""
        if not text:
            return False
        tl = text.lower()
        return any(k in tl for k in ['full text', 'full article', 'complete article', 'entire article', 'full content', 'whole article'])

# Initialize the AI
news_ai = வெளிச்சம்AI()

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
        # Test Gemini connection
        gemini_status = "connected" if news_ai.model_available else "disconnected"
        
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
            'status': 'healthy' if gemini_status == 'connected' and db_status == 'connected' else 'partial',
            'gemini': gemini_status,
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
    # Check Gemini connection on startup
    if news_ai.model_available:
        print("✅ Connected to Google Gemini AI")
    else:
        print("❌ Error: Cannot connect to Gemini API")
        print("Please make sure GEMINI_API_KEY is set in your .env file")
    
    print("🚀 Starting Live News AI Web App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
