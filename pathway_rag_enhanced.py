import pathway as pw
from pathway.xpacks.llm import embedders, llms
import google.generativeai as genai
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime

load_dotenv()

class PathwayRAGNewsSystem:
    """Enhanced news system with Pathway real-time processing and RAG capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_pathway()
        self.setup_gemini()
        self.setup_database()
        
    def setup_pathway(self):
        """Initialize Pathway for real-time news processing"""
        try:
            # Define news sources for real-time streaming
            self.news_sources = {
                'gnews': pw.io.http.rest_connector(
                    host="gnews.io",
                    route="/api/v4/search",
                    autocommit_duration_ms=30000  # Update every 30 seconds
                ),
                'newsapi': pw.io.http.rest_connector(
                    host="newsapi.org", 
                    route="/v2/everything",
                    autocommit_duration_ms=30000
                ),
                # RSS feeds for continuous updates
                'rss_feeds': pw.io.rss.read([
                    "https://feeds.bbci.co.uk/news/technology/rss.xml",
                    "https://techcrunch.com/feed/",
                    "https://www.wired.com/feed/",
                    "https://www.theverge.com/rss/index.xml"
                ])
            }
            
            # Initialize embedder for semantic search
            self.embedder = embedders.SentenceTransformerEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.logger.info("✅ Pathway initialized for real-time processing")
            
        except Exception as e:
            self.logger.error(f"Pathway setup error: {e}")
            
    def setup_gemini(self):
        """Initialize Gemini for AI processing"""
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.logger.info("✅ Gemini AI initialized")
        except Exception as e:
            self.logger.error(f"Gemini setup error: {e}")
            
    def setup_database(self):
        """Setup enhanced database with user profiles and embeddings"""
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
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    age INTEGER,
                    interests TEXT[],
                    preferred_tags TEXT[],
                    reading_level VARCHAR(20) DEFAULT 'intermediate',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create enhanced articles table with embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT UNIQUE,
                    image_url TEXT,
                    published_at TIMESTAMP,
                    source VARCHAR(100),
                    author VARCHAR(200),
                    tags TEXT[],
                    category VARCHAR(50),
                    age_group VARCHAR(20),
                    reading_difficulty VARCHAR(20),
                    embedding FLOAT[],
                    ai_summary TEXT,
                    sentiment VARCHAR(20),
                    relevance_scores JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user reading history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_reading_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    article_id INTEGER REFERENCES rag_articles(id),
                    read_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reading_time INTEGER,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5)
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("✅ Enhanced database schema created")
            
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
    
    def process_real_time_articles(self):
        """Process articles in real-time using Pathway"""
        try:
            # Combine all news sources
            all_articles = pw.Table.empty()
            
            for source_name, source in self.news_sources.items():
                articles = source.select(
                    title=pw.this.title,
                    content=pw.this.description,
                    full_content=pw.this.content if hasattr(pw.this, 'content') else pw.this.description,
                    image_url=pw.this.urlToImage,
                    published_at=pw.this.publishedAt,
                    source_name=source_name,
                    url=pw.this.url,
                    author=pw.this.author if hasattr(pw.this, 'author') else "Unknown"
                )
                all_articles = all_articles.concat(articles)
            
            # Enhanced AI processing pipeline
            enhanced_articles = all_articles.select(
                *pw.this,
                
                # AI Enhancements
                ai_summary=pw.this.content | self.generate_ai_summary,
                tags=pw.this.content | self.extract_tags,
                category=pw.this.content | self.classify_category,
                age_group=pw.this.content | self.determine_age_group,
                reading_difficulty=pw.this.content | self.assess_reading_difficulty,
                sentiment=pw.this.content | self.analyze_sentiment,
                
                # Embeddings for RAG
                embedding=pw.this.content | self.embedder,
                
                # Personalization scores
                relevance_scores=pw.this.content | self.calculate_multi_user_relevance
            )
            
            return enhanced_articles
            
        except Exception as e:
            self.logger.error(f"Real-time processing error: {e}")
            return None
    
    def generate_ai_summary(self, content):
        """Generate AI summary with age-appropriate language"""
        try:
            prompt = f"""
            Create a compelling summary of this news article:
            1. Make it engaging and accessible
            2. Highlight key points and significance
            3. Use clear, modern language
            
            Article: {content[:800]}
            
            Summary (2-3 sentences):
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except:
            return "AI summary temporarily unavailable"
    
    def extract_tags(self, content):
        """Extract relevant tags from content"""
        try:
            prompt = f"""
            Extract 5-8 relevant tags from this article. Focus on:
            - Technology (AI, ML, blockchain, etc.)
            - Entertainment (movies, games, music, etc.)
            - Development (programming, software, etc.)
            - Science (research, discoveries, etc.)
            - Business (startups, finance, etc.)
            
            Article: {content[:500]}
            
            Tags (comma-separated):
            """
            
            response = self.gemini_model.generate_content(prompt)
            tags = [tag.strip().lower() for tag in response.text.split(',')]
            return tags[:8]  # Limit to 8 tags
        except:
            return ['technology', 'news']
    
    def classify_category(self, content):
        """Classify article into main category"""
        try:
            prompt = f"""
            Classify this article into ONE main category:
            - technology
            - entertainment
            - science
            - business
            - health
            - sports
            - politics
            - lifestyle
            
            Article: {content[:300]}
            
            Category:
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip().lower()
        except:
            return 'general'
    
    def determine_age_group(self, content):
        """Determine appropriate age group for content"""
        keywords_analysis = {
            'teen': ['gaming', 'social media', 'tiktok', 'instagram', 'memes'],
            'young_adult': ['career', 'startup', 'college', 'dating', 'fitness'],
            'adult': ['business', 'finance', 'politics', 'family', 'health'],
            'senior': ['retirement', 'healthcare', 'grandchildren', 'travel']
        }
        
        content_lower = content.lower()
        scores = {}
        
        for age_group, keywords in keywords_analysis.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[age_group] = score
        
        return max(scores, key=scores.get) if scores else 'adult'
    
    def assess_reading_difficulty(self, content):
        """Assess reading difficulty level"""
        word_count = len(content.split())
        avg_word_length = sum(len(word) for word in content.split()) / max(word_count, 1)
        
        if avg_word_length < 4.5 and word_count < 200:
            return 'easy'
        elif avg_word_length < 6 and word_count < 500:
            return 'intermediate'
        else:
            return 'advanced'
    
    def analyze_sentiment(self, content):
        """Analyze sentiment of the article"""
        positive_words = ['breakthrough', 'success', 'innovation', 'growth', 'positive']
        negative_words = ['crisis', 'problem', 'failure', 'decline', 'concern']
        
        content_lower = content.lower()
        positive_score = sum(1 for word in positive_words if word in content_lower)
        negative_score = sum(1 for word in negative_words if word in content_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_multi_user_relevance(self, content):
        """Calculate relevance scores for different user types"""
        # This would normally query user preferences from database
        # For now, using sample user types
        user_types = {
            'tech_enthusiast': ['ai', 'machine learning', 'programming', 'startup'],
            'movie_lover': ['film', 'cinema', 'actor', 'director', 'hollywood'],
            'sci_fi_fan': ['science fiction', 'space', 'future', 'robot', 'alien'],
            'developer': ['coding', 'programming', 'software', 'github', 'api']
        }
        
        content_lower = content.lower()
        relevance_scores = {}
        
        for user_type, keywords in user_types.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            relevance_scores[user_type] = min(score * 20, 100)  # Scale to 0-100
        
        return relevance_scores
    
    def rag_search(self, query: str, user_id: int = None, limit: int = 10) -> List[Dict]:
        """RAG-based semantic search for articles"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get user preferences if user_id provided
            user_preferences = {}
            if user_id:
                cursor.execute("SELECT interests, preferred_tags, age FROM users WHERE id = %s", (user_id,))
                user_data = cursor.fetchone()
                if user_data:
                    user_preferences = {
                        'interests': user_data.get('interests', []),
                        'preferred_tags': user_data.get('preferred_tags', []),
                        'age': user_data.get('age')
                    }
            
            # Retrieve articles with embeddings
            cursor.execute("""
                SELECT *, 
                       array_length(embedding, 1) as embedding_dim
                FROM rag_articles 
                WHERE embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
            """)
            
            articles = cursor.fetchall()
            
            if not articles:
                return []
            
            # Calculate semantic similarity
            article_embeddings = []
            valid_articles = []
            
            for article in articles:
                if article['embedding'] and len(article['embedding']) > 0:
                    article_embeddings.append(article['embedding'])
                    valid_articles.append(dict(article))
            
            if not article_embeddings:
                return []
            
            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], article_embeddings)[0]
            
            # Combine with user preferences
            scored_articles = []
            for i, article in enumerate(valid_articles):
                base_score = similarities[i]
                
                # Boost score based on user preferences
                preference_boost = 0
                if user_preferences:
                    # Age-based boost
                    if user_preferences.get('age'):
                        age = user_preferences['age']
                        if age < 25 and article.get('age_group') == 'young_adult':
                            preference_boost += 0.1
                        elif 25 <= age < 50 and article.get('age_group') == 'adult':
                            preference_boost += 0.1
                        elif age >= 50 and article.get('age_group') == 'senior':
                            preference_boost += 0.1
                    
                    # Interest-based boost
                    article_tags = article.get('tags', [])
                    user_interests = user_preferences.get('interests', [])
                    common_interests = set(article_tags) & set(user_interests)
                    preference_boost += len(common_interests) * 0.05
                
                final_score = base_score + preference_boost
                
                scored_articles.append({
                    **article,
                    'similarity_score': float(final_score),
                    'base_similarity': float(base_score),
                    'preference_boost': float(preference_boost)
                })
            
            # Sort by final score and return top results
            scored_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            conn.close()
            return scored_articles[:limit]
            
        except Exception as e:
            self.logger.error(f"RAG search error: {e}")
            return []
    
    def get_personalized_feed(self, user_id: int, limit: int = 30) -> List[Dict]:
        """Get personalized news feed based on user profile"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get user profile
            cursor.execute("""
                SELECT age, interests, preferred_tags, reading_level 
                FROM users WHERE id = %s
            """, (user_id,))
            
            user_profile = cursor.fetchone()
            if not user_profile:
                return self.get_general_feed(limit)
            
            # Get articles matching user preferences
            age_group = self.map_age_to_group(user_profile['age'])
            interests = user_profile['interests'] or []
            preferred_tags = user_profile['preferred_tags'] or []
            
            # Build dynamic query based on preferences
            conditions = []
            params = []
            
            if age_group:
                conditions.append("age_group = %s")
                params.append(age_group)
            
            if preferred_tags:
                conditions.append("tags && %s")
                params.append(preferred_tags)
            
            if interests:
                conditions.append("category = ANY(%s)")
                params.append(interests)
            
            where_clause = " OR ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT *, 
                       CASE 
                           WHEN age_group = %s THEN 10
                           ELSE 0
                       END +
                       CASE 
                           WHEN tags && %s THEN 5
                           ELSE 0
                       END +
                       CASE 
                           WHEN category = ANY(%s) THEN 3
                           ELSE 0
                       END as relevance_score
                FROM rag_articles 
                WHERE {where_clause}
                ORDER BY relevance_score DESC, created_at DESC
                LIMIT %s
            """
            
            params.extend([age_group, preferred_tags or [], interests or [], limit])
            cursor.execute(query, params)
            
            articles = cursor.fetchall()
            conn.close()
            
            return [dict(article) for article in articles]
            
        except Exception as e:
            self.logger.error(f"Personalized feed error: {e}")
            return self.get_general_feed(limit)
    
    def map_age_to_group(self, age):
        """Map age to age group"""
        if age < 18:
            return 'teen'
        elif age < 30:
            return 'young_adult'
        elif age < 60:
            return 'adult'
        else:
            return 'senior'
    
    def get_general_feed(self, limit: int = 30) -> List[Dict]:
        """Get general news feed when no user preferences available"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM rag_articles 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            
            articles = cursor.fetchall()
            conn.close()
            
            return [dict(article) for article in articles]
            
        except Exception as e:
            self.logger.error(f"General feed error: {e}")
            return []
    
    def store_article_with_rag(self, article_data: Dict):
        """Store article with RAG processing"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Generate embedding
            content = article_data.get('content', '')
            embedding = self.embedder.embed_query(content) if content else None
            
            cursor.execute("""
                INSERT INTO rag_articles 
                (title, content, url, image_url, published_at, source, author, 
                 tags, category, age_group, reading_difficulty, embedding, 
                 ai_summary, sentiment, relevance_scores)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                tags = EXCLUDED.tags,
                category = EXCLUDED.category,
                age_group = EXCLUDED.age_group,
                embedding = EXCLUDED.embedding,
                ai_summary = EXCLUDED.ai_summary
            """, (
                article_data.get('title'),
                article_data.get('content'),
                article_data.get('url'),
                article_data.get('image_url'),
                article_data.get('published_at'),
                article_data.get('source'),
                article_data.get('author'),
                article_data.get('tags', []),
                article_data.get('category'),
                article_data.get('age_group'),
                article_data.get('reading_difficulty'),
                embedding.tolist() if embedding is not None else None,
                article_data.get('ai_summary'),
                article_data.get('sentiment'),
                json.dumps(article_data.get('relevance_scores', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Article storage error: {e}")

# Initialize the RAG system
rag_system = PathwayRAGNewsSystem()
