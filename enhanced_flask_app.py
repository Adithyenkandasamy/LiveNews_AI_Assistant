from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import google.generativeai as genai
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import re
import time
from threading import Lock
from werkzeug.security import generate_password_hash, check_password_hash
# from pathway_rag_enhanced import rag_system  # Temporarily disabled

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'velicham-secret-key-2024')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNewsIntelligence:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_apis()
        self.setup_database()
        
        # Rate limiting for Gemini
        self.gemini_request_times = []
        self.max_requests_per_minute = 8
        self.gemini_lock = Lock()
        self.quota_exceeded = False
        self.quota_reset_time = 0
        
    def setup_apis(self):
        # Disable all APIs for hardcoded mode
        self.model = None
        self.gemini_available = False
        self.gnews_api_key = None
        self.gnews_available = False
        self.newsapi_client = None
        self.newsapi_available = False
        
        self.logger.info("📝 All APIs disabled - using hardcoded sample data only")
            
    def setup_database(self):
        """Database setup disabled for hardcoded mode"""
        self.logger.info("📝 Database setup skipped - using hardcoded data only")
    
    def fetch_all_news(self):
        """Disabled - using hardcoded sample articles only"""
        self.logger.info("📝 Live news fetching disabled - using hardcoded sample articles")
        return []
    
    def fetch_gnews_articles(self, limit=20):
        """Fetch articles from GNews API"""
        try:
            topics = ['technology', 'business', 'science', 'health']
            articles = []
            
            for topic in topics:
                try:
                    url = f"https://gnews.io/api/v4/search"
                    params = {
                        'q': topic,
                        'lang': 'en',
                        'country': 'us',
                        'max': 5,
                        'apikey': self.gnews_api_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    self.logger.info(f"GNews API response for '{topic}': {len(data.get('articles', []))} articles")
                    
                    for idx, article in enumerate(data.get('articles', []), 1):
                        processed_article = {
                            'id': f"gnews_{topic}_{idx}_{int(time.time())}",
                            'title': article.get('title', 'No Title'),
                            'content': article.get('description', '') + '\n\n' + article.get('content', ''),
                            'summary': article.get('description', ''),
                            'image_url': article.get('image', f'https://picsum.photos/600/400?random={len(articles)}'),
                            'published_at': article.get('publishedAt', datetime.now().isoformat()),
                            'source': article.get('source', {}).get('name', 'GNews'),
                            'author': 'GNews Reporter',
                            'url': article.get('url', '#'),
                            'keywords': topic,
                            'category': topic,
                            'fake_news_score': 0.1,
                            'credibility': 'HIGH',
                            'ai_enhanced': True,
                            'sentiment': 0.1,
                            'reading_time': max(2, len(str(article.get('content', ''))) // 200),
                            'relevance_score': 90 - (len(articles) * 2),
                            'time_ago': self.calculate_time_ago(article.get('publishedAt')),
                            'views': f"{(len(articles) * 1000) + 500}"
                        }
                        articles.append(processed_article)
                        
                        if len(articles) >= limit:
                            break
                                
                except Exception as query_error:
                    self.logger.warning(f"GNews topic '{topic}' failed: {query_error}")
                    continue
                    
                if len(articles) >= limit:
                    break
                    
            self.logger.info(f"✅ Fetched {len(articles)} articles from GNews")
            return articles[:limit]
            
        except Exception as e:
            self.logger.error(f"GNews fetch error: {e}")
            return []
    
    def fetch_newsapi_articles(self, max_articles=20):
        """Enhanced NewsAPI fetching with multiple categories"""
        if not self.newsapi_available:
            return []
            
        try:
            all_articles = []
            
            # Multiple categories to get diverse content
            categories = ['general', 'technology', 'business', 'health', 'science']
            articles_per_category = max(4, max_articles // len(categories))
            
            for cat_idx, category in enumerate(categories):
                try:
                    # Get articles from this category
                    headlines = self.newsapi_client.get_top_headlines(
                        category=category,
                        language='en',
                        page_size=articles_per_category
                    )
                    
                    articles_data = headlines.get('articles', [])
                    if articles_data is None:
                        continue
                        
                    for idx, article in enumerate(articles_data, 1):
                        if article is None:
                            continue
                            
                        processed_article = {
                            'id': f"newsapi_{cat_idx}_{idx}_{int(time.time())}",
                            'title': article.get('title', 'No Title'),
                            'content': article.get('content', '') or article.get('description', ''),
                            'summary': article.get('description', ''),
                            'image_url': article.get('urlToImage') or f'https://picsum.photos/600/400?random={cat_idx}{idx}',
                            'published_at': article.get('publishedAt', datetime.now().isoformat()),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'author': article.get('author', 'NewsAPI Reporter'),
                            'url': article.get('url', '#'),
                            'keywords': f'{category},breaking news',
                            'category': category,
                            'fake_news_score': 0.05,
                            'credibility': 'HIGH',
                            'ai_enhanced': True,
                            'sentiment': 0.0,
                            'reading_time': max(2, len(str(article.get('content', ''))) // 200),
                            'relevance_score': 95 - (len(all_articles) * 2),
                            'time_ago': self.calculate_time_ago(article.get('publishedAt')),
                            'views': f"{((len(all_articles) + 1) * 1200) + 800}"
                        }
                        all_articles.append(processed_article)
                        
                        if len(all_articles) >= max_articles:
                            break
                            
                except Exception as cat_error:
                    self.logger.warning(f"NewsAPI category '{category}' failed: {cat_error}")
                    continue
                    
                if len(all_articles) >= max_articles:
                    break
                
            self.logger.info(f"✅ Fetched {len(all_articles)} articles from NewsAPI")
            return all_articles[:max_articles]
            
        except Exception as e:
            self.logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    def calculate_time_ago(self, published_at):
        """Calculate time ago from published date"""
        try:
            if isinstance(published_at, str):
                pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_date = published_at
                
            now = datetime.now()
            diff = now - pub_date.replace(tzinfo=None)
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            else:
                return f"{diff.seconds // 60}m ago"
        except:
            return "1h ago"

    def fetch_all_news(self):
        """Disabled - using hardcoded sample articles only"""
        self.logger.info("📝 Live news fetching disabled - using hardcoded sample articles")
        return []
    
    def process_articles(self, articles):
        """Process articles with AI enhancement"""
        enhanced_articles = []
        ai_enhanced_count = 0
        
        for article in articles:
            if not article.get('title') or not article.get('url'):
                continue
                
            # Add metadata
            article['relevance_score'] = self.calculate_relevance_score(article)
            article['reading_time'] = self.calculate_reading_time(article.get('content', ''))
            article['sentiment'] = 'NEUTRAL'
            
            # Try AI enhancement (with rate limiting)
            enhanced_article = self.enhance_article_with_ai(article)
            if enhanced_article.get('ai_enhanced'):
                ai_enhanced_count += 1
                
            enhanced_articles.append(enhanced_article)
        
        self.logger.info(f"✅ Enhanced and stored {len(enhanced_articles)} articles ({ai_enhanced_count} with AI)")
        return enhanced_articles
    
    def calculate_relevance_score(self, article):
        """Calculate relevance score based on keywords"""
        tech_keywords = ['ai', 'artificial intelligence', 'technology', 'innovation', 'startup', 'microsoft', 'google', 'apple']
        title = str(article.get('title') or '').lower()
        content = str(article.get('content') or '').lower()
        
        score = 75  # Base score
        for keyword in tech_keywords:
            if keyword in title:
                score += 5
            if keyword in content:
                score += 2
                
        return min(score, 100)
    
    def calculate_reading_time(self, content):
        """Calculate estimated reading time"""
        words = len(content.split()) if content else 0
        return max(1, words // 200)  # 200 words per minute
    
    def enhance_article_with_ai(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced article processing with Gemini AI"""
        if not self.model_available or self.quota_exceeded:
            return article
            
        # Rate limiting check
        with self.gemini_lock:
            current_time = time.time()
            self.gemini_request_times = [t for t in self.gemini_request_times if current_time - t < 60]
            
            if len(self.gemini_request_times) >= self.max_requests_per_minute:
                self.logger.warning("Rate limit reached, skipping AI enhancement")
                return article
                
            self.gemini_request_times.append(current_time)
    
        try:
            title = article.get('title', '')
            content = article.get('content', '')
            
            prompt = f"""Analyze this news article and provide:
1. A compelling 2-sentence summary
2. Key topics/tags (comma-separated)
3. Sentiment (POSITIVE/NEGATIVE/NEUTRAL)
4. Fake news score (0-100 where 0=real, 100=fake)

Article Title: {title}
Content: {content[:500]}...

Format your response as:
SUMMARY: [your summary]
TAGS: [tag1, tag2, tag3]
SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
FAKE_SCORE: [0-100]"""

            response = self.gemini_model.generate_content(prompt).text
            
            if response:
                lines = response.strip().split('\n')
                ai_summary = ""
                ai_tags = []
                sentiment = "NEUTRAL"
                fake_score = 0
                
                for line in lines:
                    if line.startswith('SUMMARY:'):
                        ai_summary = line.replace('SUMMARY:', '').strip()
                    elif line.startswith('TAGS:'):
                        tags_str = line.replace('TAGS:', '').strip()
                        ai_tags = [tag.strip() for tag in tags_str.split(',')]
                    elif line.startswith('SENTIMENT:'):
                        sentiment = line.replace('SENTIMENT:', '').strip()
                    elif line.startswith('FAKE_SCORE:'):
                        try:
                            fake_score = int(line.replace('FAKE_SCORE:', '').strip())
                        except:
                            fake_score = 0
                
                # Update article with AI enhancements
                if ai_summary:
                    article['ai_summary'] = ai_summary
                if ai_tags:
                    article['ai_tags'] = ai_tags
                article['sentiment'] = sentiment
                article['fake_news_score'] = fake_score
                article['ai_enhanced'] = True
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                self.quota_exceeded = True
                self.logger.error(f"Gemini quota exceeded: {e}")
            else:
                self.logger.error(f"AI enhancement error: {e}")
        
        return article
    
    def store_articles_in_db(self, articles):
        """Store articles in PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            for article in articles:
                cursor.execute("""
                    INSERT INTO enhanced_articles 
                    (title, content, url, image_url, published_at, source, author, 
                     ai_summary, ai_tags, fake_news_score, credibility, ai_enhanced, 
                     sentiment, reading_time, relevance_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                    ai_summary = EXCLUDED.ai_summary,
                    ai_tags = EXCLUDED.ai_tags,
                    fake_news_score = EXCLUDED.fake_news_score,
                    ai_enhanced = EXCLUDED.ai_enhanced,
                    sentiment = EXCLUDED.sentiment,
                    relevance_score = EXCLUDED.relevance_score
                """, (
                    article.get('title'),
                    article.get('content'),
                    article.get('url'),
                    article.get('image_url'),
                    article.get('published_at'),
                    article.get('source'),
                    article.get('author'),
                    article.get('ai_summary'),
                    article.get('ai_tags', []),
                    article.get('fake_news_score', 0),
                    article.get('credibility', 'MEDIUM'),
                    article.get('ai_enhanced', False),
                    article.get('sentiment', 'NEUTRAL'),
                    article.get('reading_time', 5),
                    article.get('relevance_score', 75)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database storage error: {e}")
    
    def get_articles_from_db(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get articles from database for display"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM enhanced_articles 
                ORDER BY relevance_score DESC, published_at DESC 
                LIMIT %s
            """, (limit,))
            
            articles = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts and add time_ago
            result = []
            for article in articles:
                article_dict = dict(article)
                if article_dict.get('published_at'):
                    time_diff = datetime.now() - article_dict['published_at']
                    if time_diff.days > 0:
                        article_dict['time_ago'] = f"{time_diff.days} days ago"
                    elif time_diff.seconds > 3600:
                        article_dict['time_ago'] = f"{time_diff.seconds // 3600} hours ago"
                    else:
                        article_dict['time_ago'] = f"{time_diff.seconds // 60} minutes ago"
                else:
                    article_dict['time_ago'] = "Recently"
                    
                result.append(article_dict)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Database fetch error: {e}")
            return self.get_sample_articles(limit)
    
    def get_sample_articles(self, count=30):
        """Get hardcoded sample articles only"""
        return self.get_hardcoded_samples(count)
    
    def get_hardcoded_samples(self, count=30):
        """Generate hardcoded sample articles for demo"""
        hardcore_articles = [
            {
                'id': 1,
                'title': 'Trump announces new tariffs on trucks, furniture and pharmaceuticals - The Washington Post',
                'content': '''<p>Pharmaceuticals may face tariffs of up to 100 percent, kitchen cabinets 50 percent, upholstered furniture 30 percent and heavy trucks 25 percent from Oct. 1.</p>
                
                <p>The Trump administration announced sweeping new tariffs targeting key industries including pharmaceuticals, furniture manufacturing, and heavy truck imports. The measures, set to take effect October 1st, represent one of the most significant trade policy shifts in recent years.</p>
                
                <p>According to senior administration officials, the tariffs are designed to protect American manufacturing jobs and reduce dependency on foreign supply chains. "We're bringing manufacturing back to America," said Trade Representative Katherine Walsh during a press briefing.</p>
                
                <p>The pharmaceutical industry faces the steepest penalties, with tariffs reaching up to 100 percent on certain drug imports. This move has sparked concern among healthcare advocates who worry about potential impacts on drug prices and availability.</p>
                
                <p>Kitchen cabinet manufacturers will see a 50 percent tariff, while upholstered furniture faces 30 percent duties. Heavy truck imports, primarily from European and Asian manufacturers, will be subject to 25 percent tariffs.</p>
                
                <p>Industry groups have expressed mixed reactions, with domestic manufacturers largely supporting the measures while importers and retailers voice concerns about increased costs for consumers.</p>''',
                'summary': 'Trump administration announces major tariffs on pharmaceuticals (100%), kitchen cabinets (50%), furniture (30%), and heavy trucks (25%) starting October 1st to protect American manufacturing.',
                'image_url': 'https://picsum.photos/600/400?random=1',
                'published_at': '2025-09-27 07:42:19',
                'source': 'The Washington Post',
                'author': 'Andrew Restuccia',
                'url': 'https://washingtonpost.com/politics/trump-tariffs-announcement',
                'keywords': 'tariffs,trade,pharmaceuticals,furniture,trucks,manufacturing',
                'category': 'politics',
                'fake_news_score': 0.15,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 0.2,
                'reading_time': 3,
                'relevance_score': 95,
                'time_ago': '1 min read',
                'views': '15.2K'
            },
            {
                'id': 2,
                'title': 'Marvel Studios Announces Phase 6: Spider-Man 4 and Fantastic Four Details',
                'content': 'Marvel Studios reveals exciting details about Phase 6 of the MCU, including the highly anticipated Spider-Man 4 starring Tom Holland and a new Fantastic Four reboot featuring cutting-edge visual effects. The announcement came during a surprise presentation at Comic-Con, sending fans into a frenzy.',
                'ai_summary': 'Marvel Studios unveils Phase 6 plans with Spider-Man 4 and Fantastic Four reboot, promising groundbreaking storytelling and visual effects.',
                'image_url': 'https://picsum.photos/400/250?random=2',
                'published_at': datetime.now() - timedelta(hours=2),
                'source': 'Entertainment Weekly',
                'author': 'Mike Rodriguez',
                'url': 'https://example.com/marvel-phase6',
                'ai_tags': ['movies', 'marvel', 'spiderman', 'entertainment'],
                'tags': ['movies', 'marvel', 'spiderman', 'entertainment'],
                'category': 'entertainment',
                'age_group': 'young_adult',
                'fake_news_score': 10,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 3,
                'relevance_score': 88,
                'time_ago': '2h ago',
                'views': '12.8K'
            },
            {
                'id': 3,
                'title': 'React 19 Released: New Features Transform Web Development',
                'content': 'Meta releases React 19 with revolutionary features including automatic batching, concurrent rendering improvements, and enhanced developer tools. The update includes built-in support for server components and improved TypeScript integration, making it easier than ever to build modern web applications.',
                'ai_summary': 'React 19 introduces game-changing features for web developers, improving performance and developer experience significantly.',
                'image_url': 'https://picsum.photos/400/250?random=3',
                'published_at': datetime.now() - timedelta(hours=3),
                'source': 'Dev.to',
                'author': 'Alex Thompson',
                'url': 'https://example.com/react19-release',
                'ai_tags': ['react', 'development', 'javascript', 'web'],
                'tags': ['react', 'development', 'javascript', 'web'],
                'category': 'technology',
                'age_group': 'young_adult',
                'fake_news_score': 8,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 5,
                'relevance_score': 90,
                'time_ago': '3h ago',
                'views': '8.9K'
            },
            {
                'id': 4,
                'title': 'Dune: Part Three Confirmed - Denis Villeneuve Returns as Director',
                'content': 'Warner Bros confirms Dune: Part Three with Denis Villeneuve returning to direct. The film will adapt the second half of Frank Herbert\'s epic science fiction novel, continuing the story of Paul Atreides. Production is set to begin in 2025 with a planned 2027 release date.',
                'ai_summary': 'Dune: Part Three officially announced with Denis Villeneuve directing, continuing the epic sci-fi saga with 2027 release.',
                'image_url': 'https://picsum.photos/400/250?random=4',
                'published_at': datetime.now() - timedelta(hours=4),
                'source': 'Variety',
                'author': 'Jessica Park',
                'url': 'https://example.com/dune-part3',
                'ai_tags': ['movies', 'scifi', 'dune', 'villeneuve'],
                'tags': ['movies', 'scifi', 'dune', 'villeneuve'],
                'category': 'entertainment',
                'age_group': 'adult',
                'fake_news_score': 12,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 4,
                'relevance_score': 85,
                'time_ago': '4h ago',
                'views': '7.3K'
            },
            {
                'id': 5,
                'title': 'NASA Discovers Water on Mars: Potential for Life Increases',
                'content': 'NASA\'s Perseverance rover discovers significant water deposits beneath Mars surface using ground-penetrating radar. The discovery includes both frozen and liquid water reservoirs, raising exciting possibilities for past or present microbial life on the Red Planet.',
                'ai_summary': 'NASA makes groundbreaking discovery of water on Mars, significantly increasing potential for finding extraterrestrial life.',
                'image_url': 'https://picsum.photos/400/250?random=5',
                'published_at': datetime.now() - timedelta(hours=5),
                'source': 'NASA',
                'author': 'Dr. Maria Santos',
                'url': 'https://example.com/mars-water-discovery',
                'ai_tags': ['nasa', 'mars', 'space', 'science'],
                'tags': ['nasa', 'mars', 'space', 'science'],
                'category': 'science',
                'age_group': 'adult',
                'fake_news_score': 5,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 6,
                'relevance_score': 94,
                'time_ago': '5h ago',
                'views': '22.1K'
            },
            {
                'id': 6,
                'title': 'GitHub Copilot X: AI-Powered Coding Assistant Gets Major Upgrade',
                'content': 'GitHub releases Copilot X with enhanced AI capabilities, supporting 50+ programming languages and offering intelligent code suggestions for complex development tasks. The update includes voice commands, automated testing, and real-time collaboration features.',
                'ai_summary': 'GitHub Copilot X brings advanced AI assistance to developers with improved code generation and multi-language support.',
                'image_url': 'https://picsum.photos/400/250?random=6',
                'published_at': datetime.now() - timedelta(hours=6),
                'source': 'GitHub Blog',
                'author': 'Rachel Green',
                'url': 'https://example.com/github-copilot-x',
                'ai_tags': ['github', 'ai', 'development', 'programming'],
                'tags': ['github', 'ai', 'development', 'programming'],
                'category': 'technology',
                'age_group': 'young_adult',
                'fake_news_score': 7,
                'credibility': 'HIGH',
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 4,
                'relevance_score': 93,
                'time_ago': '6h ago',
                'views': '11.5K'
            }
        ]
        
        # Extend the list to reach the requested count
        articles = []
        for i in range(count):
            base_article = hardcore_articles[i % len(hardcore_articles)].copy()
            base_article['id'] = i + 1
            if i >= len(hardcore_articles):
                base_article['title'] = f"{base_article['title']} - Update {i + 1}"
                base_article['published_at'] = datetime.now() - timedelta(hours=i + 1)
                base_article['time_ago'] = f"{i + 1}h ago"
                base_article['views'] = f"{max(1000, 15000 - i*200)}"
            articles.append(base_article)
        
        return articles
    
    def chat_with_ai(self, user_query: str, context_articles: List[Dict] = None) -> str:
        """Chat with AI about news"""
        try:
            if context_articles:
                context = "\n".join([
                    f"Title: {article.get('title', '')}\nSummary: {article.get('ai_summary', article.get('content', ''))[:200]}..."
                    for article in context_articles[:5]
                ])
            else:
                context = "No specific articles provided."
            
            prompt = f"""
            You are an intelligent news assistant. Answer the user's question based on recent news.
            Be conversational, informative, and cite relevant information when appropriate.
            
            Recent News Context:
            {context}
            
            User Question: {user_query}
            
            Provide a helpful response based on the available news information.
            """
            
            response = self.gemini_model.generate_content(prompt).text
            return response
            
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return "I'm having trouble processing your request right now. Please try again."

# Initialize the app
news_app = EnhancedNewsIntelligence()

# Removed authentication routes - keeping simple news feed

# Flask Routes
@app.route('/')
def index():
    """Enhanced landing page with news feed"""
    try:
        # Use hardcoded sample articles as primary source
        articles = news_app.get_sample_articles(30)
        
        # Calculate stats
        total_articles = len(articles)
        ai_enhanced = sum(1 for article in articles if article.get('ai_enhanced'))
        fake_news_detected = sum(1 for article in articles if article.get('fake_news_score', 0) > 70)
        sources_count = len(set(article.get('source', 'Unknown') for article in articles))
        
        return render_template('enhanced_main.html', 
                             articles=articles,
                             total_articles=total_articles,
                             ai_enhanced=ai_enhanced,
                             fake_news_detected=fake_news_detected,
                             sources_count=sources_count)
    except Exception as e:
        logger.error(f"Homepage error: {e}")
        # Use basic sample data and render template anyway
        articles = [
            {
                'id': 1,
                'title': 'OpenAI Releases GPT-5: Revolutionary AI Model Breaks New Ground',
                'ai_summary': 'OpenAI launches GPT-5 with enhanced reasoning and multimodal capabilities, setting new benchmarks in AI performance.',
                'image_url': 'https://picsum.photos/400/250?random=1',
                'time_ago': '1h ago',
                'views': '15.2K',
                'source': 'TechCrunch',
                'author': 'Sarah Chen',
                'fake_news_score': 5,
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 4,
                'relevance_score': 95
            },
            {
                'id': 2,
                'title': 'Marvel Studios Announces Phase 6: Spider-Man 4 Details',
                'ai_summary': 'Marvel Studios unveils Phase 6 plans with Spider-Man 4 and Fantastic Four reboot.',
                'image_url': 'https://picsum.photos/400/250?random=2',
                'time_ago': '2h ago',
                'views': '12.8K',
                'source': 'Entertainment Weekly',
                'author': 'Mike Rodriguez',
                'fake_news_score': 10,
                'ai_enhanced': True,
                'sentiment': 'POSITIVE',
                'reading_time': 3,
                'relevance_score': 88
            }
        ]
        
        return render_template('enhanced_main.html', 
                             articles=articles,
                             total_articles=len(articles),
                             ai_enhanced=len(articles),
                             fake_news_detected=0,
                             sources_count=2)

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    """Immersive article reading page"""
    try:
        # Use sample articles as primary source for consistency
        sample_articles = news_app.get_sample_articles(30)
        article = next((a for a in sample_articles if a['id'] == article_id), None)
        
        if not article:
            # Fallback to database if not in samples
            try:
                conn = psycopg2.connect(**news_app.db_config)
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("SELECT * FROM enhanced_articles WHERE id = %s", (article_id,))
                article = cursor.fetchone()
                
                conn.close()
            except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                article = None
            
        if not article:
            return "Article not found", 404
        
        # Ensure fake_news_score is a float
        if article and 'fake_news_score' in article and article['fake_news_score'] is not None:
            try:
                article['fake_news_score'] = float(article['fake_news_score'])
            except (ValueError, TypeError):
                article['fake_news_score'] = 0.0
        
        # Get related articles from samples
        related_articles = [a for a in sample_articles if a['id'] != article_id][:3]
        
        return render_template('article_detail.html', 
                             article=dict(article) if hasattr(article, 'keys') else article,
                             related_articles=related_articles)
    except Exception as e:
        return f"Error loading article: {e}", 500

@app.route('/api/search', methods=['POST'])
def rag_search():
    """RAG-based semantic search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = session.get('user_id')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Use sample articles for search results
        all_articles = news_app.get_sample_articles(30)
        results = [article for article in all_articles if query.lower() in str(article.get('title', '')).lower() or query.lower() in str(article.get('content', '')).lower()][:10]
        
        return jsonify({
            'results': results,
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def check_fake_news(text):
    """
    Check if the given text contains potential fake news indicators
    Returns a tuple of (is_fake, confidence, explanation)
    """
    # This is a simplified version - in production, you'd want to use a more sophisticated model
    fake_indicators = [
        ('clickbait', 0.3, 'Contains clickbait phrases'),
        ('breaking', 0.2, 'Uses sensationalist language'),
        ('shocking', 0.4, 'Uses emotionally charged language'),
        ('unbelievable', 0.5, 'Makes extraordinary claims'),
        ('miracle', 0.6, 'Makes extraordinary claims'),
        ('cure', 0.4, 'Makes health claims'),
        ('guaranteed', 0.5, 'Makes absolute claims'),
        ('conspiracy', 0.7, 'References conspiracy theories'),
        ('hoax', 0.8, 'Discusses hoaxes')
    ]
    
    text_lower = text.lower()
    total_score = 0
    indicators_found = []
    
    for indicator, score, explanation in fake_indicators:
        if indicator in text_lower:
            total_score += score
            indicators_found.append(explanation)
    
    # Cap the score at 1.0
    confidence = min(total_score, 1.0)
    
    # If no indicators found, return not fake with low confidence
    if not indicators_found:
        return (False, 0.1, 'No clear indicators of fake news found')
    
    return (confidence > 0.5, confidence, 
            f"Potential issues found: {', '.join(set(indicators_found))}")

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Simple chat endpoint with hardcoded responses
    Expected JSON payload:
    {
        'message': 'user message',
        'current_article': 'article_id'  # optional
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip().lower()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Hardcoded responses based on common queries
        responses = {
            'hello': "Hi there! I'm your AI news assistant. How can I help you today?",
            'hi': "Hello! I'm here to help with any news-related questions. What would you like to know?",
            'help': "I can help you with:\n- Latest news updates\n- News summaries\n- Background information on current events\n- Fact-checking claims\n\nWhat would you like to know?",
            'latest news': "Here are the latest headlines:\n1. Breaking: Major developments in global markets\n2. Tech giant announces new AI breakthrough\n3. Climate summit reaches historic agreement\n\nWould you like more details on any of these?",
            'weather': "I can see you're asking about weather. For the most accurate weather updates, I recommend checking a dedicated weather service. Would you like help finding weather information?",
            'thank': "You're welcome! Is there anything else I can help you with?",
            'bye': "Goodbye! Feel free to come back if you have more questions.",
            'default': "I'm not sure I understand. Could you rephrase your question? I can help with news, summaries, and general information about current events."
        }
        
        # Check for keywords in the user's message
        response = responses['default']
        for keyword, reply in responses.items():
            if keyword in user_message:
                response = reply
                break
        
        # Get article context if available
        article_context = {}
        if 'current_article' in data and data['current_article']:
            try:
                conn = psycopg2.connect(**app.config['DATABASE'])
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    "SELECT title, summary FROM enhanced_articles WHERE id = %s",
                    (data['current_article'],)
                )
                article = cursor.fetchone()
                if article:
                    article_context = {
                        'title': article['title'],
                        'summary': article['summary'][:200] + '...' if article['summary'] else ''
                    }
                    response = f"I see you're reading about: {article['title']}. {response}"
                conn.close()
            except Exception as e:
                print(f"Error getting article context: {e}")
        
        return jsonify({
            'response': response,
            'article_context': article_context,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_preferences', methods=['POST'])
def update_preferences():
    """Update user preferences"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        interests = data.get('interests', [])
        preferred_tags = data.get('preferred_tags', [])
        
        conn = psycopg2.connect(**news_app.db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET interests = %s, preferred_tags = %s 
            WHERE id = %s
        """, (interests, preferred_tags, user_id))
        
        conn.commit()
        conn.close()
        
        # Update session
        session['user_interests'] = interests
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/<int:article_id>')
def summarize_article(article_id):
    """Get AI summary for specific article"""
    try:
        # Try RAG articles first
        conn = psycopg2.connect(**news_app.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM rag_articles WHERE id = %s", (article_id,))
        article = cursor.fetchone()
        
        if not article:
            cursor.execute("SELECT * FROM enhanced_articles WHERE id = %s", (article_id,))
            article = cursor.fetchone()
        
        conn.close()
        
        if not article:
            # Try sample articles
            sample_articles = news_app.get_sample_articles(30)
            article = next((a for a in sample_articles if a['id'] == article_id), None)
            
        if not article:
            return jsonify({'error': 'Article not found'}), 404
        
        # Return existing summary or generate new one
        if article.get('ai_summary'):
            summary = article['ai_summary']
        else:
            summary = "This article discusses important developments in technology and their potential impact on society. The breakthrough represents a significant advancement in the field."
        
        return jsonify({
            'summary': summary,
            'article_id': article_id,
            'tags': article.get('tags', []),
            'category': article.get('category', 'general')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
