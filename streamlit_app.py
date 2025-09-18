import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import asyncio
import threading
import time
from complete_news_rag_system import CompleteNewsRAGSystem, CleanupConfig

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Live News AI Assistant",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .news-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .politics { background-color: #ff6b6b; color: white; }
    .technology { background-color: #4ecdc4; color: white; }
    .business { background-color: #45b7d1; color: white; }
    .sports { background-color: #96ceb4; color: white; }
    .entertainment { background-color: #feca57; color: white; }
    .general { background-color: #a55eea; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_news_system():
    """Initialize the news system with caching"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'livenews'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    cleanup_config = CleanupConfig(
        retention_days=30,
        max_articles_per_category=1000
    )
    
    try:
        return CompleteNewsRAGSystem(db_config, cleanup_config)
    except Exception as e:
        st.error(f"Failed to initialize news system: {str(e)}")
        return None

def get_category_color(category):
    """Get color for category badge"""
    colors = {
        'Politics': '#ff6b6b',
        'Technology': '#4ecdc4', 
        'Business': '#45b7d1',
        'Sports': '#96ceb4',
        'Entertainment': '#feca57',
        'General': '#a55eea'
    }
    return colors.get(category, '#a55eea')

def format_article_card(article):
    """Format article as a card"""
    category = article.get('category', 'General')
    color = get_category_color(category)
    
    return f"""
    <div class="news-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span class="category-badge" style="background-color: {color};">{category}</span>
            <small style="color: #666;">{article.get('published_date', 'Unknown date')}</small>
        </div>
        <h4 style="margin: 0.5rem 0; color: #333;">{article.get('title', 'No title')}</h4>
        <p style="color: #666; margin: 0.5rem 0;">{article.get('summary', 'No summary available')}</p>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
            <small style="color: #888;">Source: {article.get('source', 'Unknown')}</small>
            <small style="color: #888;">Sentiment: {article.get('sentiment', 'neutral').title()}</small>
        </div>
    </div>
    """

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Live News AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize system
    news_system = initialize_news_system()
    if not news_system:
        st.error("Failed to initialize the news system. Please check your database configuration.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Collection controls
    st.sidebar.subheader("📥 News Collection")
    
    if st.sidebar.button("🔄 Collect News Now", type="primary"):
        with st.spinner("Collecting and processing news..."):
            try:
                articles = news_system.collect_news()
                st.sidebar.success(f"✅ Collected {len(articles)} new articles!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Collection failed: {str(e)}")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Search & Ask", "📰 Latest News", "⚙️ Settings"])
    
    with tab1:
        st.header("📊 System Dashboard")
        
        # Get system statistics
        try:
            stats = news_system.get_system_stats()
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📰 Total Articles", stats.get('total_articles', 0))
            
            with col2:
                st.metric("📅 Today's Articles", stats.get('articles_today', 0))
            
            with col3:
                st.metric("🏷️ Categories", stats.get('total_categories', 0))
            
            with col4:
                st.metric("📈 Avg Sentiment", f"{stats.get('avg_sentiment_score', 0):.2f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Articles by Category")
                if 'category_distribution' in stats:
                    df_cat = pd.DataFrame(list(stats['category_distribution'].items()), 
                                        columns=['Category', 'Count'])
                    fig_pie = px.pie(df_cat, values='Count', names='Category', 
                                   title="Distribution by Category")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📈 Articles Over Time")
                if 'daily_articles' in stats:
                    df_daily = pd.DataFrame(list(stats['daily_articles'].items()), 
                                          columns=['Date', 'Count'])
                    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
                    fig_line = px.line(df_daily, x='Date', y='Count', 
                                     title="Daily Article Count")
                    st.plotly_chart(fig_line, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load dashboard: {str(e)}")
    
    with tab2:
        st.header("🔍 Search & Ask Questions")
        
        # Search section
        st.subheader("🔍 Search Articles")
        search_query = st.text_input("Enter search terms:", placeholder="e.g., artificial intelligence, politics, technology")
        
        if search_query:
            try:
                with st.spinner("Searching articles..."):
                    results = news_system.search_articles(search_query, top_k=10)
                
                if results:
                    st.success(f"Found {len(results)} relevant articles")
                    for article in results:
                        st.markdown(format_article_card(article), unsafe_allow_html=True)
                else:
                    st.info("No articles found for your search query.")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
        
        st.divider()
        
        # Q&A section
        st.subheader("🤖 Ask AI Questions")
        question = st.text_input("Ask a question about the news:", 
                                placeholder="e.g., What are the latest developments in AI?")
        
        if question:
            try:
                with st.spinner("AI is thinking..."):
                    answer = news_system.answer_question(question)
                
                if answer:
                    st.success("🤖 AI Answer:")
                    st.write(answer['answer'])
                    
                    if answer.get('sources'):
                        st.subheader("📚 Sources:")
                        for source in answer['sources']:
                            st.markdown(format_article_card(source), unsafe_allow_html=True)
                else:
                    st.info("Sorry, I couldn't find relevant information to answer your question.")
            except Exception as e:
                st.error(f"Question answering failed: {str(e)}")
    
    with tab3:
        st.header("📰 Latest News")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_filter = st.selectbox("Category", 
                                         ["All"] + ["Politics", "Technology", "Business", "Sports", "Entertainment", "General"])
        
        with col2:
            sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])
        
        with col3:
            limit = st.slider("Number of articles", 5, 50, 20)
        
        # Get articles
        try:
            # Build filters
            filters = {}
            if category_filter != "All":
                filters['category'] = category_filter
            if sentiment_filter != "All":
                filters['sentiment'] = sentiment_filter.lower()
            
            articles = news_system.get_recent_articles(limit=limit, filters=filters)
            
            if articles:
                st.success(f"Showing {len(articles)} articles")
                for article in articles:
                    st.markdown(format_article_card(article), unsafe_allow_html=True)
            else:
                st.info("No articles found with the selected filters.")
                
        except Exception as e:
            st.error(f"Failed to load articles: {str(e)}")
    
    with tab4:
        st.header("⚙️ Settings & Configuration")
        
        # System info
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Database Status:** ✅ Connected")
            st.info("**AI Models:** ✅ Loaded")
            st.info("**RSS Sources:** ✅ Active")
        
        with col2:
            if st.button("🧹 Cleanup Old Articles"):
                try:
                    with st.spinner("Cleaning up old articles..."):
                        cleaned = news_system.cleanup_old_articles()
                    st.success(f"✅ Cleaned up {cleaned} old articles")
                except Exception as e:
                    st.error(f"Cleanup failed: {str(e)}")
            
            if st.button("📊 Refresh Statistics"):
                st.cache_resource.clear()
                st.success("✅ Statistics refreshed")
                st.rerun()
        
        # Configuration
        st.subheader("🔧 Configuration")
        st.json({
            "Database": os.getenv('DB_NAME', 'livenews'),
            "Host": os.getenv('DB_HOST', 'localhost'),
            "Port": os.getenv('DB_PORT', '5432'),
            "Auto-collection": "Disabled (Manual mode)"
        })
        
        # Logs
        st.subheader("📋 Recent Logs")
        try:
            log_file = "logs/news_rag.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = f.readlines()[-20:]  # Last 20 lines
                
                log_text = "".join(logs)
                st.text_area("System Logs", log_text, height=200)
            else:
                st.info("No log file found.")
        except Exception as e:
            st.error(f"Failed to load logs: {str(e)}")

if __name__ == "__main__":
    main()
