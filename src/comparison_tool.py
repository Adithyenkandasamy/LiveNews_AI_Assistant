"""
News Comparison Tool - Compare RAG results and evaluate news freshness
Shows exactly what news is being retrieved and filters old content
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from .rag_system import PathwayRAGSystem, PathwayRAGConfig
from .freshness_validator import NewsFreshnessValidator
import json

class NewsComparisonTool:
    """Tool to compare and evaluate RAG system results"""
    
    def __init__(self):
        self.setup_logging()
        
        # Initialize RAG system
        config = PathwayRAGConfig()
        self.rag_system = PathwayRAGSystem(config)
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def compare_news_quality(self, query: str) -> Dict[str, Any]:
        """Compare news quality and freshness for a given query"""
        print(f"\n🔍 Analyzing news quality for: '{query}'")
        print("=" * 60)
        
        # Get RAG results
        result = self.rag_system.query(query)
        
        # Extract information
        sources = result.get('sources', [])
        freshness_report = result.get('freshness_report', {})
        answer = result.get('answer', '')
        
        print(f"\n📊 FRESHNESS ANALYSIS:")
        if freshness_report:
            print(f"Total articles found: {freshness_report.get('total_articles', 0)}")
            breakdown = freshness_report.get('freshness_breakdown', {})
            print(f"Breaking news: {breakdown.get('breaking', 0)}")
            print(f"Recent news: {breakdown.get('recent', 0)}")
            print(f"Old news: {breakdown.get('old', 0)}")
            print(f"Stale news: {breakdown.get('stale', 0)}")
            
            percentages = freshness_report.get('freshness_percentage', {})
            print(f"\nFresh content: {percentages.get('fresh', 0)}%")
            print(f"Old content: {percentages.get('old', 0)}%")
            print(f"\n{freshness_report.get('recommendation', '')}")
        
        print(f"\n📰 ARTICLES RETRIEVED:")
        for i, source in enumerate(sources[:5], 1):
            title = source.get('title', 'No title')
            date = source.get('date', source.get('collected_date', 'Unknown date'))
            category = source.get('freshness_category', 'unknown')
            age_hours = source.get('age_hours', 0)
            
            # Calculate age display
            if age_hours < 24:
                age_display = f"{int(age_hours)}h ago"
            else:
                age_display = f"{int(age_hours/24)}d ago"
                
            status_emoji = {
                'breaking': '🔥',
                'recent': '✅', 
                'old': '⚠️',
                'stale': '❌',
                'unknown': '❓'
            }.get(category, '❓')
            
            print(f"{i}. {status_emoji} [{category.upper()}] {title}")
            print(f"   📅 {date} ({age_display})")
            print(f"   🏢 {source.get('source', 'Unknown source')}")
            print()
            
        print(f"\n🤖 AI RESPONSE:")
        print(f"{answer[:300]}{'...' if len(answer) > 300 else ''}")
        
        return {
            'query': query,
            'total_sources': len(sources),
            'freshness_report': freshness_report,
            'sources': sources,
            'answer': answer
        }
        
    def test_old_news_filtering(self):
        """Test the system with known old news to see if it gets filtered"""
        print("\n🧪 TESTING OLD NEWS FILTERING")
        print("=" * 60)
        
        test_queries = [
            "Trump pleads not guilty",  # Should be filtered (April 2023)
            "latest AI developments",   # Should show recent news
            "Biden inauguration",       # Should be filtered (January 2021)
            "current technology news"   # Should show recent news
        ]
        
        results = {}
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            result = self.compare_news_quality(query)
            
            fresh_percent = 0
            if result['freshness_report']:
                fresh_percent = result['freshness_report'].get('freshness_percentage', {}).get('fresh', 0)
                
            print(f"Result: {result['total_sources']} sources, {fresh_percent}% fresh")
            results[query] = result
            
        return results
        
    def generate_freshness_report(self, queries: List[str]) -> str:
        """Generate a comprehensive freshness report"""
        report = []
        report.append("📊 NEWS FRESHNESS EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_articles = 0
        total_fresh = 0
        
        for query in queries:
            result = self.rag_system.query(query)
            sources = result.get('sources', [])
            freshness_report = result.get('freshness_report', {})
            
            if freshness_report:
                articles = freshness_report.get('total_articles', 0)
                fresh_percent = freshness_report.get('freshness_percentage', {}).get('fresh', 0)
                
                total_articles += articles
                total_fresh += (articles * fresh_percent / 100)
                
                report.append(f"Query: '{query}'")
                report.append(f"  Articles: {articles}")
                report.append(f"  Fresh: {fresh_percent}%")
                report.append(f"  Status: {freshness_report.get('recommendation', 'Unknown')}")
                report.append("")
                
        overall_fresh_percent = (total_fresh / total_articles * 100) if total_articles > 0 else 0
        
        report.append("OVERALL SUMMARY:")
        report.append(f"Total articles analyzed: {total_articles}")
        report.append(f"Overall freshness: {overall_fresh_percent:.1f}%")
        
        if overall_fresh_percent >= 80:
            report.append("✅ EXCELLENT - RAG system is returning fresh, current news")
        elif overall_fresh_percent >= 60:
            report.append("⚠️ GOOD - Some old content detected, filtering working")
        else:
            report.append("❌ POOR - Too much old content, review filtering settings")
            
        return "\n".join(report)

def main():
    """Run news comparison analysis"""
    tool = NewsComparisonTool()
    
    print("🚀 NEWS RAG COMPARISON TOOL")
    print("Analyzing news freshness and quality...")
    
    # Test specific queries
    test_queries = [
        "latest technology news",
        "Trump pleads not guilty",  # This should be filtered as old news
        "current AI developments",
        "recent business updates"
    ]
    
    print("\n1. INDIVIDUAL QUERY ANALYSIS:")
    for query in test_queries:
        tool.compare_news_quality(query)
        print("\n" + "-" * 60)
        
    print("\n2. OLD NEWS FILTERING TEST:")
    tool.test_old_news_filtering()
    
    print("\n3. OVERALL FRESHNESS REPORT:")
    report = tool.generate_freshness_report(test_queries)
    print(report)
    
    print("\n✅ Analysis complete!")
    print("\nKey indicators to watch:")
    print("🔥 Breaking = Less than 6 hours old")
    print("✅ Recent = Less than 3 days old") 
    print("⚠️ Old = 3-30 days old")
    print("❌ Stale = More than 30 days old")

if __name__ == "__main__":
    main()
