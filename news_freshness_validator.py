"""
News Freshness Validator - Ensures RAG system only returns recent, relevant news
Prevents old headlines like "Trump pleads not guilty" (April 2023) from appearing in current results
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class FreshnessConfig:
    """Configuration for news freshness validation"""
    max_age_hours: int = 72  # 3 days max for "recent" news
    stale_threshold_days: int = 30  # Mark as stale after 30 days
    breaking_news_hours: int = 6  # Breaking news window
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"

class NewsFreshnessValidator:
    """
    Validates news freshness and filters out old/stale content
    """
    
    def __init__(self, config: FreshnessConfig = None):
        self.config = config or FreshnessConfig()
        self.setup_logging()
        self.setup_database()
        
        # Known old events that should be filtered out
        self.known_old_events = {
            "trump pleads not guilty": "2023-04-04",
            "trump indictment": "2023-03-30", 
            "covid pandemic declared": "2020-03-11",
            "biden inauguration": "2021-01-20",
            "queen elizabeth death": "2022-09-08",
            "twitter elon musk": "2022-10-27"
        }
        
        # Patterns that indicate old news
        self.old_news_patterns = [
            r"pleads? not guilty",
            r"was (arrested|charged|indicted)",
            r"died (last|in) \d{4}",
            r"(announced|declared) (last|in) \d{4}",
            r"(former|ex-) (president|ceo|minister)"
        ]
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Setup database connection"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'news_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
    def calculate_news_age(self, article_date: str) -> Tuple[int, str]:
        """Calculate age of news article in hours and categorize freshness"""
        try:
            if isinstance(article_date, str):
                # Try multiple date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%a, %d %b %Y %H:%M:%S %Z']:
                    try:
                        article_datetime = datetime.strptime(article_date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format works, assume it's recent
                    return 1, "unknown"
            else:
                article_datetime = article_date
                
            now = datetime.now()
            age_hours = (now - article_datetime).total_seconds() / 3600
            
            # Categorize freshness
            if age_hours < self.config.breaking_news_hours:
                category = "breaking"
            elif age_hours < self.config.max_age_hours:
                category = "recent"
            elif age_hours < (self.config.stale_threshold_days * 24):
                category = "old"
            else:
                category = "stale"
                
            return int(age_hours), category
            
        except Exception as e:
            self.logger.error(f"Date parsing error: {e}")
            return 999999, "unknown"  # Treat as very old if can't parse
            
    def is_known_old_event(self, title: str, content: str) -> bool:
        """Check if this is a known old event that should be filtered"""
        text_lower = f"{title} {content}".lower()
        
        for event_key, event_date in self.known_old_events.items():
            if event_key in text_lower:
                # Check if the event date is old
                event_datetime = datetime.strptime(event_date, "%Y-%m-%d")
                age_days = (datetime.now() - event_datetime).days
                if age_days > 30:  # More than 30 days old
                    self.logger.info(f"Filtered known old event: {event_key}")
                    return True
                    
        return False
        
    def has_old_news_patterns(self, title: str, content: str) -> bool:
        """Check for patterns that indicate old news"""
        text = f"{title} {content}".lower()
        
        for pattern in self.old_news_patterns:
            if re.search(pattern, text):
                self.logger.info(f"Found old news pattern: {pattern}")
                return True
                
        return False
        
    def validate_with_ai(self, title: str, content: str) -> Dict[str, Any]:
        """Use AI to validate if news is current and relevant"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""Current date: {current_date}

Analyze this news headline and determine:
1. Is this current/recent news (within last 3 days)?
2. Is this old news being recycled?
3. What's the likely original date of this event?

News: "{title}"
Content snippet: "{content[:200]}..."

Respond in JSON format:
{{
    "is_current": true/false,
    "is_old_recycled": true/false,
    "estimated_date": "YYYY-MM-DD",
    "confidence": 0.0-1.0,
    "reason": "explanation"
}}"""

        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                
                # Try to parse JSON response
                try:
                    import json
                    ai_analysis = json.loads(ai_response)
                    return ai_analysis
                except json.JSONDecodeError:
                    # Fallback analysis
                    return {
                        "is_current": "recent" in ai_response.lower(),
                        "is_old_recycled": "old" in ai_response.lower(),
                        "estimated_date": current_date,
                        "confidence": 0.5,
                        "reason": "AI parsing failed"
                    }
            else:
                return {"is_current": True, "confidence": 0.3, "reason": "AI unavailable"}
                
        except Exception as e:
            self.logger.error(f"AI validation failed: {e}")
            return {"is_current": True, "confidence": 0.3, "reason": f"Error: {e}"}
            
    def filter_fresh_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles to only include fresh, relevant news"""
        fresh_articles = []
        filtered_count = 0
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            article_date = article.get('collected_date') or article.get('date', '')
            
            # Calculate age
            age_hours, freshness_category = self.calculate_news_age(article_date)
            
            # Skip if too old
            if freshness_category in ['stale']:
                filtered_count += 1
                self.logger.info(f"Filtered stale article: {title[:50]}...")
                continue
                
            # Skip known old events
            if self.is_known_old_event(title, content):
                filtered_count += 1
                continue
                
            # Skip articles with old news patterns
            if self.has_old_news_patterns(title, content):
                filtered_count += 1
                continue
                
            # For borderline cases, use AI validation
            if freshness_category == 'old':
                ai_analysis = self.validate_with_ai(title, content)
                if not ai_analysis.get('is_current', True) or ai_analysis.get('is_old_recycled', False):
                    filtered_count += 1
                    self.logger.info(f"AI filtered old news: {title[:50]}...")
                    continue
                    
            # Add freshness metadata
            article['freshness_category'] = freshness_category
            article['age_hours'] = age_hours
            fresh_articles.append(article)
            
        if filtered_count > 0:
            self.logger.info(f"✅ Filtered {filtered_count} old/stale articles, kept {len(fresh_articles)} fresh ones")
            
        return fresh_articles
        
    def get_news_comparison_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report comparing news freshness"""
        if not articles:
            return {"error": "No articles to analyze"}
            
        categories = {"breaking": 0, "recent": 0, "old": 0, "stale": 0, "unknown": 0}
        total_articles = len(articles)
        examples = {"fresh": [], "old": []}
        
        for article in articles:
            article_date = article.get('collected_date') or article.get('date', '')
            age_hours, category = self.calculate_news_age(article_date)
            
            categories[category] += 1
            
            if category in ["breaking", "recent"] and len(examples["fresh"]) < 3:
                examples["fresh"].append({
                    "title": article.get('title', '')[:60] + "...",
                    "age_hours": age_hours,
                    "category": category
                })
            elif category in ["old", "stale"] and len(examples["old"]) < 3:
                examples["old"].append({
                    "title": article.get('title', '')[:60] + "...",
                    "age_hours": age_hours,
                    "category": category
                })
                
        return {
            "total_articles": total_articles,
            "freshness_breakdown": categories,
            "freshness_percentage": {
                "fresh": round((categories["breaking"] + categories["recent"]) / total_articles * 100, 1),
                "old": round((categories["old"] + categories["stale"]) / total_articles * 100, 1)
            },
            "examples": examples,
            "recommendation": self._get_freshness_recommendation(categories, total_articles)
        }
        
    def _get_freshness_recommendation(self, categories: Dict[str, int], total: int) -> str:
        """Get recommendation based on freshness analysis"""
        fresh_percent = (categories["breaking"] + categories["recent"]) / total * 100
        
        if fresh_percent >= 80:
            return "✅ Excellent news freshness - mostly recent content"
        elif fresh_percent >= 60:
            return "⚠️ Good freshness but some old content detected"
        elif fresh_percent >= 40:
            return "⚠️ Mixed freshness - consider filtering older articles"
        else:
            return "❌ Poor freshness - many old articles detected, filtering recommended"
            
    def clean_old_articles_from_db(self, days_threshold: int = 90):
        """Clean very old articles from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            cursor.execute("""
                DELETE FROM news_articles 
                WHERE collected_date < %s
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"🗑️ Cleaned {deleted_count} articles older than {days_threshold} days")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {e}")
            return 0

def main():
    """Test the freshness validator"""
    validator = NewsFreshnessValidator()
    
    # Test articles (including the old Trump headline)
    test_articles = [
        {
            "title": "Trump pleads not guilty to 34 felony counts",
            "content": "Former President Donald Trump pleaded not guilty...",
            "collected_date": "2023-04-04 14:30:00",
            "source": "CNN"
        },
        {
            "title": "AI breakthrough announced today",
            "content": "Scientists today announced a major breakthrough...",
            "collected_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "TechCrunch"
        }
    ]
    
    print("=== News Freshness Analysis ===")
    
    # Get comparison report
    report = validator.get_news_comparison_report(test_articles)
    print(f"Total articles: {report['total_articles']}")
    print(f"Fresh content: {report['freshness_percentage']['fresh']}%")
    print(f"Old content: {report['freshness_percentage']['old']}%")
    print(f"Recommendation: {report['recommendation']}")
    
    # Filter fresh articles
    fresh_articles = validator.filter_fresh_articles(test_articles)
    print(f"\nAfter filtering: {len(fresh_articles)} articles remain")
    
    for article in fresh_articles:
        print(f"✅ {article['title'][:50]}... ({article.get('freshness_category', 'unknown')})")

if __name__ == "__main__":
    main()
