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
from src.rag_system import PathwayRAGSystem, PathwayRAGConfig
from src.freshness_validator import NewsFreshnessValidator, FreshnessConfig
from src.comparison_tool import NewsComparisonTool
from src.gemini_client import GeminiClient
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import hashlib
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

class LiveNewsAI:
    """Real-time news AI chatbot"""
    
    def __init__(self):
        # Initialize Gemini client
        self.gemini_client = GeminiClient(os.getenv('APP_GEMINI_MODEL', 'gemini-2.0-flash'))
        self.model_available = self.gemini_client.available
        
        # Keep ollama config for backward compatibility (will be removed)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"
        
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
            'Times of India': 'https://timesofindia.indiatimes.com/india/rss.cms',
            'Al Jazeera India': 'https://www.aljazeera.com/xml/rss/all.xml',  # Will filter by query
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Google News Tech': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en',
            'Hacker News': 'https://hnrss.org/frontpage'
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
                gemini_client=self.gemini_client,
                embedding_model='all-MiniLM-L6-v2',
                chunk_size=512,
                max_results=5
            )
            
            self.pathway_rag = PathwayRAGSystem(pathway_config)
            # Mark RAG as available only if a backend is ready
            self.rag_available = bool(getattr(self.pathway_rag, 'pathway_available', False) or getattr(self.pathway_rag, 'langchain_available', False))
            self.logger.info("✅ Pathway RAG system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pathway RAG: {e}")
            self.rag_available = False
            
        # Setup freshness validator
        try:
            freshness_config = FreshnessConfig(
                gemini_client=self.gemini_client
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
                
            # Wait 1 minute before next collection for real-time updates
            time.sleep(60)
            
    def fetch_and_store_news(self) -> List[Dict[str, Any]]:
        """Fetch news and store in database"""
        all_articles = []
        
        for source_name, feed_url in self.news_feeds.items():
            try:
                self.logger.info(f"Fetching from {source_name}...")
                feed = feedparser.parse(feed_url)
                self.logger.info(f"Found {len(feed.entries)} entries from {source_name}")
                
                for entry in feed.entries[:5]:  # Top 5 from each source for speed
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                            
                        # Get full content without artificial limits
                        full_content = entry.get('summary', '') or entry.get('description', '') or entry.get('content', '')
                        
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
        Uses normalized title + source + normalized URL path. Returns empty string if cannot compute.
        """
        try:
            title = (article.get('title') or '').strip().lower()
            source = (article.get('source') or '').strip().lower()
            url = (article.get('url') or '').strip()

            # Normalize URL to domain + path (ignore query params which often change)
            if url:
                p = urlparse(url)
                norm_url = (p.netloc + p.path).lower()
            else:
                norm_url = ''

            # If title is very short, include a bit of content to stabilize hash
            content = (article.get('content') or '').strip().lower()
            content_snippet = content[:120]

            key = '\n'.join([title, source, norm_url, content_snippet])
            return hashlib.sha256(key.encode('utf-8')).hexdigest()
        except Exception:
            return ''

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
                    
                    # Be more lenient with time filtering for now
                    # Get full content without limits
                    full_content = entry.get('summary', '') or entry.get('description', '') or entry.get('content', '')
                    
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
        
        # Apply freshness filtering when available to avoid outdated news
        if self.freshness_validator:
            all_news = self.freshness_validator.filter_fresh_articles(all_news)
            
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
        """Main chat function with Pathway RAG and news integration"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_available = self.model_available
        topic = self._extract_topic(user_input)
        wants_summary = self._wants_summary(user_input)
        wants_full_text = self._wants_full_text(user_input)
        
        # Check if user wants news
        if self.is_news_query(user_input):
            # Try Pathway RAG first for better results
            if self.rag_available:
                try:
                    # Fast existence check using private Pathway RAG method
                    if topic:
                        try:
                            has_any = self.pathway_rag.has_recent_news(topic, hours=24, min_similarity=0.5)
                        except Exception:
                            has_any = True  # fail open to avoid blocking
                        # For existence-style questions, reply immediately with yes/no using Pathway-only check
                        if self._is_existence_query(user_input):
                            return (
                                f"Yes, there are recent updates on {topic}."
                                if has_any else
                                f"No hot news recently on {topic}."
                            )
                        if not has_any and topic:  # Only return "no news" if specific topic was requested
                            return f"No hot news recently on {topic}."

                    rag_result = self.pathway_rag.query(user_input)
                    
                    if rag_result['sources']:
                        # Use RAG results
                        news_context = f"Current Date & Time: {current_time}\n\n"
                        news_context += "Relevant News (via Pathway RAG):\n"
                        
                        # Sort sources by recency when possible
                        def _parse_dt(d):
                            try:
                                return datetime.strptime(d, '%Y-%m-%d %H:%M')
                            except Exception:
                                try:
                                    return datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    return datetime.min
                        sorted_sources = sorted(rag_result['sources'], key=lambda s: _parse_dt(str(s.get('date',''))), reverse=True)
                        for i, source in enumerate(sorted_sources[:5], 1):
                            title = source.get('title', 'No title')
                            content = source.get('content', '')
                            
                            # Check for missing content
                            if not content or len(content.strip()) < 20:
                                content = "[Content not available or too brief]"
                            
                            # Enhanced date display with age
                            date_info = source.get('date', 'Unknown date')
                            try:
                                source_date = datetime.strptime(date_info, '%Y-%m-%d %H:%M')
                                age = datetime.now() - source_date
                                if age.days > 7:
                                    age_str = f" ({age.days} days old - may be outdated)"
                                elif age.days > 3:
                                    age_str = f" ({age.days} days ago)"
                                elif age.days > 0:
                                    age_str = f" ({age.days}d ago)"
                                else:
                                    age_str = " (recent)"
                                date_display = date_info + age_str
                            except:
                                date_display = date_info
                                
                            news_context += f"{i}. {title}\n"
                            news_context += f"   Source: {source.get('source', 'Unknown')} | {date_display}\n"
                            # Add clickable link if available
                            if source.get('url'):
                                news_context += f"   Link: {source.get('url')}\n"
                            news_context += f"   Content: {content}\n\n"  # Full content, no truncation
                        
                        # Add freshness info if available
                        if rag_result.get('freshness_report'):
                            freshness = rag_result['freshness_report']
                            fresh_percent = freshness.get('freshness_percentage', {}).get('fresh', 0)
                            news_context += f"News Freshness: {fresh_percent}% recent content\n\n"
                        
                        # If Gemini isn't available, return concise bullet points immediately
                        if wants_summary or not self.model_available:
                            bullets = []
                            for src in sorted_sources[:3]:
                                title = src.get('title', 'No title')
                                source = src.get('source', '?')
                                date = src.get('date', '?')
                                link = src.get('url', '')
                                bullet = f"- {title} — {source} ({date})"
                                if link:
                                    bullet += f"\n  Read more: {link}"
                                bullets.append(bullet)
                            return "Here are the latest updates:\n" + "\n".join(bullets)
                        
                        # Enhanced prompt with link instructions
                        if wants_full_text:
                            prompt = f"""You are a news AI assistant. Provide detailed information using the news below. Include relevant links for users to read the full articles.

{news_context}

User: {user_input}

Provide a comprehensive response with details and encourage users to click the links for full articles."""
                        else:
                            prompt = f"""You are a news AI assistant. Answer concisely using the news below. Always mention that links are available for full articles.

{news_context}

User: {user_input}

Give a brief, direct response (2-3 sentences max) and remind users they can click the links for full articles."""

                    else:
                        # RAG found no results, fallback to direct news fetch
                        latest_news = self.fetch_latest_news(24, user_input)
                        
                        if latest_news:
                            news_context = f"Current Date & Time: {current_time}\n\n"
                            news_context += "Latest News Headlines:\n"
                            
                            for i, news in enumerate(latest_news[:5], 1):
                                # Check for missing content
                                summary = news.get('summary', '')
                                if not summary or len(summary.strip()) < 20:
                                    summary = "[No content available for this article]"
                                
                                # Use enhanced date display if available
                                date_display = news.get('date_with_age', news.get('date', 'Unknown date'))
                                
                                # Add warning for old content
                                warning = ""
                                if news.get('is_old', False):
                                    warning = " ⚠️ OLD NEWS"
                                    
                                news_context += f"{i}. {news['title']}{warning}\n"
                                news_context += f"   Source: {news['source']} | {date_display}\n"
                                # Add clickable link if available
                                if news.get('link'):
                                    news_context += f"   Link: {news['link']}\n"
                                news_context += f"   Content: {summary}\n\n"
                            if wants_summary or not self.model_available:
                                bullets = []
                                for n in latest_news[:3]:
                                    bullet = f"- {n['title']} — {n['source']} ({n['date']})"
                                    if n.get('link'):
                                        bullet += f"\n  Read more: {n['link']}"
                                    bullets.append(bullet)
                                return "Here are the latest updates:\n" + "\n".join(bullets)
                            
                            if wants_full_text:
                                prompt = f"""Provide detailed analysis using the news below. Include information about article links for readers who want the complete stories.

{news_context}

User: {user_input}

Provide comprehensive coverage and mention the article links."""
                            else:
                                prompt = f"""Answer briefly using news below. Remind users that full article links are provided.

{news_context}

User: {user_input}

Keep response short (2-3 sentences) but mention the article links."""
                        else:
                            # If user asked about a specific topic/location and nothing found, respond clearly
                            if topic:
                                return f"No hot news recently on {topic}."
                            prompt = f"""No recent news found. Current time: {current_time}

User: {user_input}

Brief response: I don't have current news. Check BBC, CNN, or Reuters for latest updates."""
                            
                except Exception as e:
                    self.logger.error(f"RAG query failed: {e}")
                    # Fallback to simple news fetch
                    latest_news = self.fetch_latest_news(24, user_input)
                    
                    if latest_news:
                        news_context = f"Current Date & Time: {current_time}\n\n"
                        news_context += "Latest News Headlines:\n"
                        
                        for i, news in enumerate(latest_news[:5], 1):
                            news_context += f"{i}. {news['title']}\n"
                            news_context += f"   Source: {news['source']} | {news['date']}\n"
                            news_context += f"   Summary: {news['summary']}\n\n"
                        if wants_summary or not ollama_available:
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
                            return f"No hot news recently on {topic}."
                        prompt = f"""You are a helpful AI assistant. The user asked about news but no recent articles were found. Current time: {current_time}

User: {user_input}

Explain that you don't have access to the very latest news and suggest they check news websites directly."""
            else:
                # No RAG available, use simple news fetch
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
                        return f"No hot news recently on {topic}."
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
                for n in latest_news[:3]:
                    bullet = f"- {n['title']} — {n['source']} ({n['date']})"
                    if n.get('link'):
                        bullet += f"\n  Read full article: {n['link']}"
                    bullets.append(bullet)
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
            print("✅ Connected to Ollama Llama 3.2:3b")
        else:
            print("⚠️ Warning: Ollama connection failed")
            
    except Exception as e:
        print("❌ Error: Cannot connect to Ollama")
        print("Please make sure GEMINI_API_KEY is set in your .env file")
    
    print("🚀 Starting Live News AI Web App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
