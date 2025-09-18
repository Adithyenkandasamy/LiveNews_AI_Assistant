Complete News RAG System - Setup Guide
📋 Requirements File (requirements.txt)
# Core dependencies
pathway-python>=0.7.0
psycopg2-binary>=2.9.7
streamlit>=1.28.0

# AI/ML dependencies
sentence-transformers>=2.2.2
transformers>=4.35.0
torch>=2.0.0
scikit-learn>=1.3.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
feedparser>=6.0.10
requests>=2.31.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0

# Utilities
python-dotenv>=1.0.0
python-dateutil>=2.8.2

🚀 Complete Setup Instructions
Step 1: Environment Setup
# Create virtual environment
python -m venv news_rag_env
source news_rag_env/bin/activate  # On Windows: news_rag_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

Step 2: PostgreSQL Setup
Install PostgreSQL:
# Ubuntu/Debian
sudo apt update && sudo apt install postgresql postgresql-contrib

# macOS (with Homebrew)
brew install postgresql
brew services start postgresql

# Windows: Download from https://www.postgresql.org/download/windows/

Create Database:
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE newsrag;
CREATE USER newsrag_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE newsrag TO newsrag_user;

-- Exit
\q

Step 3: Environment Variables
Create a .env file:
# Database Configuration
DB_HOST=localhost
DB_NAME=newsrag
DB_USER=newsrag_user
DB_PASSWORD=your_secure_password
DB_PORT=5432

# Optional: API Keys (if you want to use NewsAPI)
NEWSAPI_KEY=your_newsapi_key_here

Step 4: File Structure
Your project should look like this:
news_rag_system/
├── complete_news_rag_system.py
├── streamlit_app.py
├── requirements.txt
├── .env
├── README.md
└── logs/
    └── news_rag.log

🧪 Testing the System
Test 1: Database Connection
# test_database.py
from complete_news_rag_system import create_database_config, CompleteNewsRAGSystem

try:
    db_config = create_database_config()
    system = CompleteNewsRAGSystem(db_config)
    print("✅ Database connection successful!")
    print(f"📊 System stats: {system.get_system_stats()}")
except Exception as e:
    print(f"❌ Error: {e}")

Test 2: Data Collection
# Run single collection test
python complete_news_rag_system.py test

Test 3: Web Interface
# Start the web interface
streamlit run streamlit_app.py

🚀 Running the System
Option 1: Command Line (Test Mode)
python complete_news_rag_system.py test

Option 2: Command Line (Continuous Mode)
python complete_news_rag_system.py

Option 3: Web Interface
streamlit run streamlit_app.py

Then open: http://localhost:8501
🔧 Configuration Options
Database Configuration
Edit in complete_news_rag_system.py:
def create_database_config():
    return {
        'host': 'localhost',
        'database': 'newsrag',
        'user': 'your_username',
        'password': 'your_password',
        'port': '5432'
    }

Cleanup Configuration
def create_cleanup_config():
    return CleanupConfig(
        enabled=True,
        default_retention_days=7,
        cleanup_frequency=10,
        category_retention={
            'Breaking News': 14,  # Keep longer
            'Entertainment': 3,   # Delete sooner
            'Reddit': 1,          # Very short retention
            # ... customize as needed
        }
    )

📱 Using the System
Web Interface Features:
💬 Ask AI Tab:
Ask questions about current news
Get AI-powered answers with sources
Try quick questions or random topics
📊 Dashboard Tab:
Live statistics and metrics
Visual charts of news trends
Real-time system status
📰 Recent Articles Tab:
Browse latest articles
Search and filter by category
View article importance scores
📋 Activity Log Tab:
Monitor system activity
See collection and processing logs
Track Q&A interactions
⚙️ System Stats Tab:
Database health monitoring
Configuration display
Manual operations
Example Questions to Ask:
"What's the latest news about Tesla?"
"Any breaking news today?"
"Tell me about recent AI developments"
"What's happening in the stock market?"
"Any sports news this week?"
"What are the trending topics?"
🛠️ Troubleshooting
Common Issues:
1. Database Connection Error
❌ Error: could not connect to server

Solution:
Check PostgreSQL is running
Verify database credentials
Ensure database exists
2. AI Models Loading Slowly
🤖 Loading AI models... (takes long time)

Solution:
First run downloads models (several GB)
Subsequent runs will be faster
Ensure good internet connection
3. RSS Feed Errors
❌ Error fetching from BBC_World: HTTP 403

Solution:
Some feeds block automated requests
System continues with other sources
Check internet connection
4. Out of Memory
❌ Error: out of memory

Solution:
Reduce batch sizes in configuration
Run cleanup more frequently
Use smaller AI models if needed
Performance Optimization:
Database Indexing: Automatically created
Memory Usage: Configure cleanup frequency
Response Speed: Adjust search limits
Storage: Run regular cleanups
🔐 Security Notes
Database Security:
Use strong passwords
Don't use default 'postgres' user
Enable SSL for remote connections
API Keys:
Store in environment variables
Never commit to version control
Use read-only keys when possible
Web Interface:
Run behind reverse proxy in production
Use HTTPS for external access
Monitor for unusual activity
📈 Production Deployment
Option 1: Local Server
# Install as systemd service
sudo nano /etc/systemd/system/newsrag.service

Option 2: Cloud Deployment
Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
Deploy Streamlit on cloud platforms
Set up monitoring and logging
Option 3: Docker Deployment
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]

🎯 Key Features Summary
✅ Real-time Data Collection: 30+ news sources, updated every 30 seconds ✅ AI-Powered Q&A: Ask questions, get intelligent answers with sources
✅ Comprehensive Coverage: Politics, tech, business, sports, entertainment, science ✅ Automatic Cleanup: Configurable retention policies by category ✅ Web Interface: Beautiful, responsive dashboard ✅ PostgreSQL Storage: Scalable, efficient database with full-text search ✅ Activity Monitoring: Complete logging and statistics ✅ Easy Setup: One-command installation and setup
Your Live News AI Assistant is ready to provide real-time, intelligent news analysis! 🚀
"""
Streamlit Web Interface for Complete News RAG System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import json
from complete_news_rag_system import CompleteNewsRAGSystem, create_database_config, create_cleanup_config

# Page configuration
st.set_page_config(
    page_title="📰 Live News AI Assistant",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f4e79, #2d5aa0);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}

.activity-item {
    background: #f8f9fa;
    padding: 10px;
    border-left: 4px solid #007bff;
    margin: 5px 0;
    border-radius: 4px;
}

.news-article {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 10px 0;
    border-left: 4px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'news_system' not in st.session_state:
    st.session_state.news_system = None
    st.session_state.collection_thread = None
    st.session_state.is_collecting = False

@st.cache_resource
def init_news_system():
    """Initialize the news system (cached)"""
    try:
        db_config = create_database_config()
        cleanup_config = create_cleanup_config()
        return CompleteNewsRAGSystem(db_config, cleanup_config)
    except Exception as e:
        st.error(f"❌ Failed to initialize news system: {e}")
        return None

def start_background_collection():
    """Start background collection in a separate thread"""
    if st.session_state.news_system and not st.session_state.is_collecting:
        def collection_worker():
            st.session_state.is_collecting = True
            try:
                st.session_state.news_system.run_continuous_collection(interval_seconds=60)
            except Exception as e:
                st.error(f"Collection error: {e}")
            finally:
                st.session_state.is_collecting = False
        
        st.session_state.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        st.session_state.collection_thread.start()

def stop_background_collection():
    """Stop background collection"""
    if st.session_state.news_system:
        st.session_state.news_system.stop_collection()
        st.session_state.is_collecting = False

# Main header
st.markdown("""
<div class="main-header">
    <h1>📰 Live News AI Assistant</h1>
    <p>Real-time news aggregation with AI-powered question answering</p>
</div>
""", unsafe_allow_html=True)

# Initialize system
if st.session_state.news_system is None:
    with st.spinner("🔄 Initializing News RAG System..."):
        st.session_state.news_system = init_news_system()

if st.session_state.news_system is None:
    st.error("❌ Could not initialize the news system. Please check your database configuration.")
    st.stop()

# Sidebar for system controls
st.sidebar.header("🛠️ System Controls")

# Collection controls
st.sidebar.subheader("📡 Data Collection")
if not st.session_state.is_collecting:
    if st.sidebar.button("▶️ Start Continuous Collection"):
        start_background_collection()
        st.sidebar.success("✅ Collection started!")
        st.rerun()
else:
    if st.sidebar.button("⏹️ Stop Collection"):
        stop_background_collection()
        st.sidebar.success("✅ Collection stopped!")
        st.rerun()

# Manual collection
if st.sidebar.button("🔄 Run Single Collection Cycle"):
    with st.spinner("Collecting news..."):
        st.session_state.news_system.run_collection_cycle()
    st.sidebar.success("✅ Collection completed!")

# Cleanup controls
st.sidebar.subheader("🧹 Database Cleanup")
cleanup_days = st.sidebar.slider("Retention Days", 1, 30, 7)
if st.sidebar.button("🗑️ Run Cleanup"):
    with st.spinner("Running cleanup..."):
        deleted = st.session_state.news_system.cleanup_old_articles(cleanup_days)
    st.sidebar.success(f"✅ Deleted {deleted} old articles")

# Collection status
collection_status = "🟢 Running" if st.session_state.is_collecting else "🔴 Stopped"
st.sidebar.markdown(f"**Collection Status**: {collection_status}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Ask AI", "📊 Dashboard", "📰 Recent Articles", "📋 Activity Log", "⚙️ System Stats"])

# TAB 1: AI Question Answering
with tab1:
    st.header("💬 Ask the AI Assistant")
    st.markdown("Ask any question about current news and get real-time answers based on the latest articles.")
    
    # Question input
    question = st.text_input(
        "🔍 Your Question:",
        placeholder="e.g., What's the latest news about Tesla? Any breaking news today?",
        key="user_question"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask AI", type="primary")
    with col2:
        if st.button("🎲 Try Random Question"):
            random_questions = [
                "What's the latest technology news?",
                "Any business updates today?",
                "What's happening in sports?",
                "Tell me about recent entertainment news",
                "Any breaking news?",
                "What's new in science?",
                "Any political news today?",
                "What's trending on social media?"
            ]
            import random
            question = random.choice(random_questions)
            st.session_state.user_question = question
            st.rerun()
    
    if ask_button and question:
        with st.spinner("🤖 AI is thinking..."):
            start_time = time.time()
            answer = st.session_state.news_system.answer_question(question)
            response_time = round((time.time() - start_time) * 1000)
        
        st.markdown("---")
        st.markdown("### 🤖 AI Response:")
        st.markdown(answer)
        
        st.info(f"⚡ Response generated in {response_time}ms")
    
    # Quick question buttons
    st.markdown("### 🎯 Quick Questions:")
    quick_cols = st.columns(4)
    
    quick_questions = [
        ("📱 Tech News", "What's the latest technology news?"),
        ("💼 Business", "Any business updates today?"),
        ("🏈 Sports", "What's happening in sports?"),
        ("🎬 Entertainment", "Tell me about entertainment news")
    ]
    
    for i, (label, q) in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(label):
                with st.spinner("🤖 Getting answer..."):
                    answer = st.session_state.news_system.answer_question(q)
                st.markdown(f"**Question**: {q}")
                st.markdown(answer[:300] + "..." if len(answer) > 300 else answer)

# TAB 2: Dashboard
with tab2:
    st.header("📊 Live News Dashboard")
    
    # Get system stats
    try:
        stats = st.session_state.news_system.get_system_stats()
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📰 Total Articles", 
                f"{stats.get('total_articles', 0):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "🕐 Last 24h", 
                f"{stats.get('recent_articles_24h', 0):,}",
                delta=None
            )
        
        with col3:
            st.metric(
                "💾 Database Size", 
                f"{stats.get('database_size_mb', 0):.1f} MB",
                delta=None
            )
        
        with col4:
            collection_status = "🟢 Active" if st.session_state.is_collecting else "⏸️ Paused"
            st.metric("📡 Collection", collection_status, delta=None)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Articles by category
            if stats.get('by_category'):
                st.subheader("📈 Articles by Category (24h)")
                category_df = pd.DataFrame(
                    list(stats['by_category'].items()),
                    columns=['Category', 'Count']
                )
                fig = px.bar(
                    category_df, 
                    x='Category', 
                    y='Count',
                    color='Count',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment distribution
            if stats.get('sentiment_distribution'):
                st.subheader("😊 Sentiment Distribution")
                sentiment_df = pd.DataFrame(
                    list(stats['sentiment_distribution'].items()),
                    columns=['Sentiment', 'Count']
                )
                
                # Color mapping for sentiments
                color_map = {'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}
                colors = [color_map.get(sent.lower(), '#6c757d') for sent in sentiment_df['Sentiment']]
                
                fig = px.pie(
                    sentiment_df, 
                    values='Count', 
                    names='Sentiment',
                    color_discrete_sequence=colors
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Top keywords
        if stats.get('top_keywords'):
            st.subheader("🔥 Trending Keywords (24h)")
            keywords_df = pd.DataFrame(
                list(stats['top_keywords'].items()),
                columns=['Keyword', 'Mentions']
            ).head(10)
            
            fig = px.bar(
                keywords_df,
                x='Mentions',
                y='Keyword',
                orientation='h',
                color='Mentions',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top sources
        if stats.get('by_source'):
            st.subheader("📡 Articles by Source (24h)")
            sources_df = pd.DataFrame(
                list(stats['by_source'].items()),
                columns=['Source', 'Articles']
            ).head(10)
            
            fig = px.treemap(
                sources_df,
                path=['Source'],
                values='Articles',
                color='Articles',
                color_continuous_scale='greens'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error loading dashboard: {e}")

# TAB 3: Recent Articles
with tab3:
    st.header("📰 Recent Articles")
    
    try:
        # Search recent articles
        search_query = st.text_input("🔍 Search articles:", placeholder="Enter keywords to search...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_back = st.selectbox("⏰ Time Range", [1, 6, 12, 24, 48], index=3)
        with col2:
            max_results = st.selectbox("📊 Results", [10, 20, 50, 100], index=1)
        with col3:
            category_filter = st.selectbox(
                "🏷️ Category", 
                ["All"] + list(st.session_state.news_system.category_keywords.keys())
            )
        
        if search_query:
            articles = st.session_state.news_system.search_articles(
                search_query, 
                limit=max_results, 
                hours=hours_back
            )
        else:
            # Get recent articles from database
            try:
                conn = st.session_state.news_system.get_db_connection()
                cursor = conn.cursor()
                
                query = """
                    SELECT title, summary, source, category, url, timestamp, 
                           sentiment, importance_score, keywords
                    FROM articles 
                    WHERE timestamp >= %s
                """
                params = [datetime.now() - timedelta(hours=hours_back)]
                
                if category_filter != "All":
                    query += " AND category = %s"
                    params.append(category_filter)
                
                query += " ORDER BY importance_score DESC, timestamp DESC LIMIT %s"
                params.append(max_results)
                
                cursor.execute(query, params)
                articles = cursor.fetchall()
                
                cursor.close()
                conn.close()
                
                # Convert to list of dicts
                articles = [
                    {
                        'title': art[0], 'summary': art[1], 'source': art[2],
                        'category': art[3], 'url': art[4], 'timestamp': art[5],
                        'sentiment': art[6], 'importance_score': art[7], 'keywords': art[8]
                    }
                    for art in articles
                ]
                
            except Exception as e:
                st.error(f"Error fetching articles: {e}")
                articles = []
        
        # Display articles
        if articles:
            st.success(f"📰 Found {len(articles)} articles")
            
            for i, article in enumerate(articles):
                with st.container():
                    # Importance indicator
                    importance = article.get('importance_score', 0)
                    if importance > 0.7:
                        importance_badge = "🔥 High"
                        importance_color = "red"
                    elif importance > 0.4:
                        importance_badge = "⭐ Medium"
                        importance_color = "orange"
                    else:
                        importance_badge = "📝 Low"
                        importance_color = "gray"
                    
                    # Sentiment emoji
                    sentiment = article.get('sentiment', 'neutral')
                    sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(sentiment, '😐')
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="news-article">
                            <h4>{article['title']}</h4>
                            <p><strong>Summary:</strong> {article.get('summary', 'No summary available')}</p>
                            <p>
                                <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                                    📂 {article.get('category', 'General')}
                                </span>
                                <span style="background: #f3e5f5; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 5px;">
                                    📡 {article.get('source', 'Unknown')}
                                </span>
                                <span style="background: #e8f5e8; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 5px;">
                                    {sentiment_emoji} {sentiment.title()}
                                </span>
                            </p>
                            <p><strong>⏰ Time:</strong> {article.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p><strong>🏷️ Keywords:</strong> {', '.join(article.get('keywords', [])[:5])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px;">
                            <p style="color: {importance_color}; font-weight: bold;">{importance_badge}</p>
                            <p style="font-size: 12px;">Score: {importance:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if article.get('url'):
                            st.link_button("🔗 Read Full", article['url'])
                    
                    st.markdown("---")
        else:
            st.info("📭 No articles found for the current filters.")
    
    except Exception as e:
        st.error(f"❌ Error loading articles: {e}")

# TAB 4: Activity Log
with tab4:
    st.header("📋 System Activity Log")
    
    try:
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            log_limit = st.selectbox("📊 Show entries", [20, 50, 100, 200], index=0)
        with col2:
            if st.button("🔄 Refresh Log"):
                st.rerun()
        
        # Get recent activity
        activities = st.session_state.news_system.get_recent_activity(log_limit)
        
        if activities:
            st.success(f"📋 Showing {len(activities)} recent activities")
            
            # Create activity timeline
            for activity in activities:
                timestamp = activity['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                action = activity['action']
                source = activity.get('source', '')
                category = activity.get('category', '')
                
                # Color coding based on action type
                if '💾' in action or 'Stored' in action:
                    color = "#28a745"  # Green for storage
                    icon = "💾"
                elif '🗑️' in action or 'cleanup' in action.lower():
                    color = "#dc3545"  # Red for cleanup
                    icon = "🗑️"
                elif '💬' in action or 'question' in action.lower():
                    color = "#007bff"  # Blue for questions
                    icon = "💬"
                elif '📡' in action or 'fetch' in action.lower():
                    color = "#17a2b8"  # Cyan for fetching
                    icon = "📡"
                else:
                    color = "#6c757d"  # Gray for others
                    icon = "ℹ️"
                
                # Additional info
                details_text = ""
                if activity.get('details'):
                    try:
                        details = json.loads(activity['details']) if isinstance(activity['details'], str) else activity['details']
                        if isinstance(details, dict):
                            key_info = []
                            if 'total' in details:
                                key_info.append(f"Total: {details['total']}")
                            if 'articles_found' in details:
                                key_info.append(f"Found: {details['articles_found']}")
                            if 'response_time_ms' in details:
                                key_info.append(f"Time: {details['response_time_ms']}ms")
                            if key_info:
                                details_text = f" ({', '.join(key_info)})"
                    except:
                        pass
                
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{icon} {action}</strong>{details_text}
                            {f'<br><small>📂 {category}</small>' if category else ''}
                            {f'<br><small>📡 {source}</small>' if source else ''}
                        </div>
                        <div style="text-align: right; font-size: 12px; color: #666;">
                            {timestamp}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📭 No activity logged yet.")
    
    except Exception as e:
        st.error(f"❌ Error loading activity log: {e}")

# TAB 5: System Stats
with tab5:
    st.header("⚙️ System Statistics & Configuration")
    
    try:
        # System health metrics
        st.subheader("🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            db_size = st.session_state.news_system.get_database_size_mb()
            health_status = "🟢 Healthy" if db_size < 100 else "🟡 Monitor" if db_size < 500 else "🔴 Action Needed"
            st.metric("Database Size", f"{db_size:.1f} MB", delta=None)
            st.write(f"Status: {health_status}")
        
        with col2:
            total_articles = st.session_state.news_system.count_articles_by_age(365)  # All articles
            st.metric("Total Articles", f"{total_articles:,}", delta=None)
        
        with col3:
            recent_articles = st.session_state.news_system.count_articles_by_age(1)
            st.metric("Articles Today", f"{recent_articles:,}", delta=None)
        
        st.markdown("---")
        
        # Configuration display
        st.subheader("⚙️ Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗄️ Database Config:**")
            db_config = st.session_state.news_system.db_config
            st.code(f"""
Host: {db_config['host']}
Database: {db_config['database']}
User: {db_config['user']}
Port: {db_config['port']}
            """)
        
        with col2:
            st.markdown("**🧹 Cleanup Config:**")
            cleanup_config = st.session_state.news_system.cleanup_config
            st.code(f"""
Enabled: {cleanup_config.enabled}
Default Retention: {cleanup_config.default_retention_days} days
Cleanup Frequency: Every {cleanup_config.cleanup_frequency} cycles
Batch Size: {cleanup_config.batch_size} articles
            """)
        
        # Category retention settings
        st.subheader("📂 Category Retention Settings")
        retention_df = pd.DataFrame([
            {"Category": cat, "Retention Days": days}
            for cat, days in cleanup_config.category_retention.items()
        ])
        st.dataframe(retention_df, use_container_width=True)
        
        # Data sources
        st.subheader("📡 Data Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RSS Sources:**")
            st.write(f"Total: {len(st.session_state.news_system.rss_sources)} sources")
            sources_list = list(st.session_state.news_system.rss_sources.keys())[:10]
            for source in sources_list:
                st.write(f"• {source}")
            if len(st.session_state.news_system.rss_sources) > 10:
                st.write(f"... and {len(st.session_state.news_system.rss_sources) - 10} more")
        
        with col2:
            st.markdown("**Reddit Sources:**")
            st.write(f"Total: {len(st.session_state.news_system.reddit_sources)} subreddits")
            for subreddit in st.session_state.news_system.reddit_sources.keys():
                st.write(f"• r/{subreddit}")
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        try:
            # Get database performance stats
            conn = st.session_state.news_system.get_db_connection()
            cursor = conn.cursor()
            
            # Query performance stats
            cursor.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    most_common_vals[1:3] as top_values
                FROM pg_stats 
                WHERE tablename IN ('articles', 'activity_log')
                AND attname IN ('category', 'source', 'sentiment')
                ORDER BY tablename, attname;
            """)
            
            perf_stats = cursor.fetchall()
            
            if perf_stats:
                perf_df = pd.DataFrame(perf_stats, columns=['Schema', 'Table', 'Column', 'Distinct Values', 'Top Values'])
                st.dataframe(perf_df, use_container_width=True)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.info(f"Performance stats not available: {e}")
        
        # Manual operations
        st.subheader("🔧 Manual Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Emergency Cleanup"):
                with st.spinner("Running emergency cleanup..."):
                    st.session_state.news_system.emergency_cleanup()
                st.success("✅ Emergency cleanup completed!")
        
        with col2:
            if st.button("📊 Update Stats"):
                st.rerun()
        
        with col3:
            if st.button("🔄 Test Collection"):
                with st.spinner("Running test collection..."):
                    st.session_state.news_system.run_collection_cycle()
                st.success("✅ Test collection completed!")
        
    except Exception as e:
        st.error(f"❌ Error loading system stats: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>📰 Live News AI Assistant - Real-time news aggregation with AI-powered question answering</p>
    <p>Built with Streamlit, PostgreSQL, and Transformer models</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time updates (every 30 seconds when collecting)
if st.session_state.is_collecting:
    time.sleep(30)
    st.rerun()
"""
Complete News RAG System with PostgreSQL
Covers: RSS feeds, Reddit, AI processing, cleanup, and Q&A
Author: AI Assistant & User
"""

import pathway as pw
import feedparser
import requests
import json
from datetime import datetime, timedelta
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
from typing import List, Dict, Any, Optional
import hashlib
import re
import threading
from dataclasses import dataclass
import os
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleanupConfig:
    """Configuration for automatic cleanup"""
    enabled: bool = True
    default_retention_days: int = 7
    cleanup_frequency: int = 10  # Every N collection cycles
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
        """
        Complete News RAG System
        
        Args:
            db_config: PostgreSQL connection config
            cleanup_config: Automatic cleanup configuration
        """
        self.db_config = db_config
        self.cleanup_config = cleanup_config or CleanupConfig()
        self.cycle_count = 0
        self.is_running = False
        
        # Initialize AI models
        logger.info("🤖 Loading AI models...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Comprehensive news sources
        self.rss_sources = {
            # MAJOR NEWS OUTLETS
            'BBC_World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'BBC_Business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'BBC_Technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'BBC_Entertainment': 'http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml',
            'BBC_Sports': 'http://feeds.bbci.co.uk/sport/rss.xml',
            'BBC_Science': 'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
            'BBC_Health': 'http://feeds.bbci.co.uk/news/health/rss.xml',
            
            'CNN_World': 'http://rss.cnn.com/rss/edition.rss',
            'CNN_Business': 'http://rss.cnn.com/rss/money_latest.rss',
            'CNN_Tech': 'http://rss.cnn.com/rss/edition_technology.rss',
            'CNN_Entertainment': 'http://rss.cnn.com/rss/edition_entertainment.rss',
            'CNN_Sports': 'http://rss.cnn.com/rss/edition_sport.rss',
            
            'Reuters_World': 'https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best',
            'Reuters_Business': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best',
            'Reuters_Tech': 'https://www.reutersagency.com/feed/?best-topics=tech&post_type=best',
            
            # SPECIALIZED SOURCES
            'TechCrunch': 'https://techcrunch.com/feed/',
            'Ars_Technica': 'https://feeds.arstechnica.com/arstechnica/index',
            'Entertainment_Weekly': 'https://ew.com/feed/',
            'Hollywood_Reporter': 'https://www.hollywoodreporter.com/feed/',
            'ESPN': 'https://www.espn.com/espn/rss/news',
            'Science_Daily': 'https://www.sciencedaily.com/rss/all.xml',
            
            # GOOGLE NEWS CATEGORIES
            'Google_News_General': 'https://news.google.com/news/rss',
            'Google_News_Technology': 'https://news.google.com/news/rss/headlines/section/topic/TECHNOLOGY',
            'Google_News_Business': 'https://news.google.com/news/rss/headlines/section/topic/BUSINESS',
            'Google_News_Entertainment': 'https://news.google.com/news/rss/headlines/section/topic/ENTERTAINMENT',
            'Google_News_Sports': 'https://news.google.com/news/rss/headlines/section/topic/SPORTS',
            'Google_News_Science': 'https://news.google.com/news/rss/headlines/section/topic/SCIENCE',
            'Google_News_Health': 'https://news.google.com/news/rss/headlines/section/topic/HEALTH'
        }
        
        # Reddit sources for real-time discussions
        self.reddit_sources = {
            'worldnews': 'https://www.reddit.com/r/worldnews.json',
            'technology': 'https://www.reddit.com/r/technology.json',
            'business': 'https://www.reddit.com/r/business.json',
            'movies': 'https://www.reddit.com/r/movies.json',
            'sports': 'https://www.reddit.com/r/sports.json',
            'science': 'https://www.reddit.com/r/science.json',
            'news': 'https://www.reddit.com/r/news.json',
            'finance': 'https://www.reddit.com/r/finance.json',
            'entertainment': 'https://www.reddit.com/r/entertainment.json',
            'politics': 'https://www.reddit.com/r/politics.json'
        }
        
        # Category detection keywords
        self.category_keywords = {
            'Politics': ['election', 'government', 'president', 'congress', 'senate', 'vote', 'policy', 'biden', 'trump', 'minister', 'parliament'],
            'Technology': ['ai', 'artificial intelligence', 'tech', 'software', 'app', 'google', 'apple', 'microsoft', 'meta', 'tesla', 'smartphone', 'robot'],
            'Business': ['stock', 'market', 'economy', 'finance', 'earnings', 'revenue', 'investment', 'nasdaq', 'dow', 'trading', 'company', 'ceo'],
            'Entertainment': ['movie', 'film', 'actor', 'actress', 'hollywood', 'netflix', 'disney', 'streaming', 'box office', 'celebrity', 'music', 'album'],
            'Sports': ['football', 'basketball', 'soccer', 'baseball', 'tennis', 'olympics', 'world cup', 'nfl', 'nba', 'fifa', 'match', 'game'],
            'Science': ['research', 'study', 'discovery', 'space', 'nasa', 'climate', 'medicine', 'health', 'vaccine', 'cancer', 'gene', 'DNA'],
            'Breaking News': ['breaking', 'urgent', 'alert', 'developing', 'exclusive', 'live', 'update', 'emergency'],
            'Accidents': ['crash', 'accident', 'disaster', 'emergency', 'fire', 'earthquake', 'flood', 'storm', 'explosion'],
            'Travel': ['tourism', 'travel', 'airport', 'airline', 'hotel', 'vacation', 'visa', 'border', 'flight']
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
            
            # System stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_articles INTEGER,
                    database_size_mb REAL,
                    articles_added_today INTEGER,
                    articles_deleted_today INTEGER,
                    avg_response_time_ms REAL,
                    details JSONB
                );
            """)
            
            # Create comprehensive indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON articles(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
                CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
                CREATE INDEX IF NOT EXISTS idx_articles_entities ON articles USING GIN(entities);
                CREATE INDEX IF NOT EXISTS idx_articles_keywords ON articles USING GIN(keywords);
                CREATE INDEX IF NOT EXISTS idx_articles_importance ON articles(importance_score DESC);
                CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment);
                CREATE INDEX IF NOT EXISTS idx_activity_log_timestamp ON activity_log(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_activity_log_category ON activity_log(category);
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
        keywords = []
        
        # Important terms to extract
        important_terms = [
            # Companies
            'Tesla', 'Apple', 'Google', 'Microsoft', 'Amazon', 'Meta', 'Netflix', 'Disney',
            'OpenAI', 'ChatGPT', 'SpaceX', 'Twitter', 'Facebook', 'Instagram', 'YouTube',
            'Samsung', 'Intel', 'NVIDIA', 'AMD', 'IBM', 'Oracle', 'Salesforce',
            
            # People (Public figures)
            'Biden', 'Trump', 'Elon Musk', 'Tim Cook', 'Mark Zuckerberg', 'Jeff Bezos',
            'Bill Gates', 'Warren Buffett', 'Xi Jinping', 'Putin', 'Modi',
            
            # Technology
            'AI', 'Artificial Intelligence', 'Machine Learning', 'Blockchain', 'Cryptocurrency',
            'Bitcoin', 'Ethereum', 'iPhone', 'Android', 'ChatGPT', 'GPT', 'Neural Network',
            'Cloud Computing', 'IoT', '5G', 'VR', 'AR', 'Quantum Computing',
            
            # Finance & Economy
            'Stock Market', 'NYSE', 'NASDAQ', 'S&P 500', 'Dow Jones', 'Federal Reserve',
            'Interest Rate', 'Inflation', 'GDP', 'Recession', 'Bull Market', 'Bear Market',
            'Bitcoin', 'Cryptocurrency', 'IPO', 'Merger', 'Acquisition',
            
            # Entertainment
            'Marvel', 'DC', 'Star Wars', 'Harry Potter', 'Game of Thrones', 'Stranger Things',
            'Oscar', 'Emmy', 'Golden Globe', 'Cannes', 'Box Office', 'Hollywood',
            'Netflix', 'Disney+', 'HBO', 'Amazon Prime',
            
            # Sports
            'World Cup', 'Olympics', 'Super Bowl', 'NBA Finals', 'World Series', 'Wimbledon',
            'Premier League', 'Champions League', 'UEFA', 'FIFA', 'NFL', 'NBA', 'MLB',
            
            # Science & Health
            'NASA', 'SpaceX', 'Mars', 'Climate Change', 'COVID', 'Vaccine', 'Cancer',
            'Gene Therapy', 'CRISPR', 'Nobel Prize', 'Breakthrough', 'Discovery',
            
            # Geopolitics
            'Ukraine', 'Russia', 'China', 'Europe', 'NATO', 'UN', 'G7', 'G20',
            'Middle East', 'Israel', 'Palestine', 'Iran', 'North Korea'
        ]
        
        text_lower = text.lower()
        for term in important_terms:
            if term.lower() in text_lower:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def calculate_importance_score(self, title: str, content: str, source: str, category: str) -> float:
        """Calculate article importance score (0.0 to 1.0)"""
        score = 0.0
        
        # Source credibility weight
        source_weights = {
            'BBC': 0.9, 'CNN': 0.8, 'Reuters': 0.9, 'TechCrunch': 0.7,
            'ESPN': 0.7, 'Science_Daily': 0.8, 'Google_News': 0.6, 'Reddit': 0.3
        }
        
        for source_key, weight in source_weights.items():
            if source_key in source:
                score += weight * 0.3
                break
        
        # Category importance
        category_weights = {
            'Breaking News': 0.9, 'Politics': 0.8, 'Business': 0.7,
            'Technology': 0.7, 'Science': 0.7, 'Sports': 0.5,
            'Entertainment': 0.4, 'General': 0.5
        }
        score += category_weights.get(category, 0.5) * 0.2
        
        # Content analysis
        text = f"{title} {content}".lower()
        
        # Breaking news indicators
        breaking_words = ['breaking', 'urgent', 'alert', 'developing', 'exclusive', 'first', 'major']
        breaking_count = sum(1 for word in breaking_words if word in text)
        score += min(breaking_count * 0.1, 0.3)
        
        # Important event indicators
        important_words = ['record', 'historic', 'unprecedented', 'milestone', 'breakthrough', 'crisis']
        important_count = sum(1 for word in important_words if word in text)
        score += min(important_count * 0.08, 0.2)
        
        return min(score, 1.0)
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'growth', 'success', 'win', 
            'breakthrough', 'record', 'best', 'achievement', 'progress', 'boost',
            'rise', 'increase', 'gain', 'profit', 'advance', 'improve'
        ]
        
        negative_words = [
            'bad', 'terrible', 'negative', 'loss', 'fail', 'crisis', 'drop', 
            'crash', 'worst', 'disaster', 'decline', 'fall', 'decrease', 
            'problem', 'issue', 'concern', 'warning', 'threat', 'risk'
        ]
        
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
                
                for entry in feed.entries[:3]:  # Limit per source
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
    
    def collect_reddit_posts(self) -> List[Dict[str, Any]]:
        """Collect posts from Reddit"""
        new_articles = []
        
        for subreddit, url in self.reddit_sources.items():
            try:
                logger.info(f"📡 Fetching from Reddit r/{subreddit}...")
                
                response = requests.get(
                    url, 
                    headers={'User-Agent': 'NewsRAG Bot 1.0'},
                    timeout=10
                )
                
                if response.status_code != 200:
                    logger.warning(f"⚠️ Reddit HTTP {response.status_code} from r/{subreddit}")
                    continue
                
                reddit_data = response.json()
                
                for post in reddit_data['data']['children'][:2]:  # 2 per subreddit
                    post_data = post['data']
                    article_id = self.generate_article_id(post_data['title'], f"Reddit_{subreddit}")
                    
                    if self.article_exists(article_id):
                        continue
                    
                    content = post_data.get('selftext', '') or post_data['title']
                    category = self.detect_category(post_data['title'], content)
                    
                    article = {
                        'id': article_id,
                        'title': post_data['title'],
                        'content': content,
                        'source': f"Reddit_r/{subreddit}",
                        'category': category,
                        'url': f"https://reddit.com{post_data['permalink']}",
                        'raw_timestamp': str(datetime.fromtimestamp(post_data['created_utc'])),
                        'keywords': self.extract_keywords(f"{post_data['title']} {content}"),
                        'word_count': len(content.split()),
                        'importance_score': self.calculate_importance_score(post_data['title'], content, 'Reddit', category)
                    }
                    
                    new_articles.append(article)
                    logger.info(f"🔥 Reddit {category}: {article['title'][:50]}...")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching from Reddit r/{subreddit}: {e}")
                continue
        
        return new_articles
    
    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process articles with AI"""
        processed_articles = []
        
        for article in articles:
            try:
                content = article['content']
                
                # Generate summary
                if len(content) > 100:
                    try:
                        summary = self.summarizer(
                            content, 
                            max_length=60, 
                            min_length=20, 
                            do_sample=False
                        )[0]['summary_text']
                    except:
                        # Fallback to truncated content
                        summary = content[:200] + "..." if len(content) > 200 else content
                else:
                    summary = content
                
                # Generate embedding
                embedding_text = f"{article['title']} {content}"
                embedding = self.embedder.encode(embedding_text)
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(f"{article['title']} {content}")
                
                # Update article
                article.update({
                    'summary': summary,
                    'embedding': embedding.tolist(),
                    'sentiment': sentiment,
                    'entities': article['keywords']  # Use keywords as entities
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
                        article['id'],
                        article['title'],
                        article['content'],
                        article['summary'],
                        article['source'],
                        article['category'],
                        article['url'],
                        article['raw_timestamp'],
                        article['entities'],
                        article['keywords'],
                        article['embedding'],
                        article['sentiment'],
                        article['importance_score'],
                        article['word_count']
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
                # Log by category
                category_counts = {}
                for article in articles:
                    cat = article['category']
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                category_summary = ", ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
                
                logger.info(f"💾 Stored {stored_count} articles ({category_summary})")
                self.log_activity(
                    f"💾 Stored articles by category: {category_summary}", 
                    details={'total': stored_count, 'by_category': category_counts}
                )
            
        except Exception as e:
            logger.error(f"❌ Error storing articles: {e}")
    
    def log_activity(self, action: str, source: str = None, category: str = None, details: Dict = None):
        """Log activity to database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO activity_log (action, source, category, details)
                VALUES (%s, %s, %s, %s)
            """, (action, source, category, json.dumps(details) if details else None))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error logging activity: {e}")
    
    def search_articles(self, query: str, limit: int = 5, hours: int = 24) -> List[Dict[str, Any]]:
        """Search articles using vector similarity"""
        try:
            query_embedding = self.embedder.encode(query)
            
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get recent articles
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
            
            # Calculate similarities
            similarities = []
            for article in articles:
                if article['embedding']:
                    embedding = np.array(article['embedding'])
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    if similarity > 0.1:  # Minimum similarity threshold
                        similarities.append((similarity, dict(article)))
            
            # Sort by similarity and return top results
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
            
            # Search for relevant articles
            relevant_articles = self.search_articles(question, limit=max_articles)
            
            if not relevant_articles:
                return """I don't have recent information about that topic. This could be because:
- The topic hasn't been covered in the news recently
- The database is still being populated with articles
- The query might be too specific

Try asking about general topics like "technology news", "business updates", or "sports news"."""
            
            # Prepare context from articles
            context_parts = []
            sources_used = []
            
            for i, article in enumerate(relevant_articles, 1):
                timestamp = article['timestamp'].strftime("%Y-%m-%d %H:%M")
                context_parts.append(
                    f"**{i}. {article['source']}** ({timestamp})\n"
                    f"Title: {article['title']}\n"
                    f"Summary: {article['summary']}\n"
                    f"Category: {article['category']} | Sentiment: {article['sentiment']}"
                )
                sources_used.append(f"{article['source']} ({timestamp})")
            
            context = "\n\n".join(context_parts)
            
            # Generate comprehensive answer
            response_time = round((time.time() - start_time) * 1000)  # milliseconds
            
            answer = f"""# 📰 Latest News Answer

**Question**: {question}

## 🔍 **Current Information** (Last 24 hours):

{context}

## 📊 **Summary**:
Based on the latest {len(relevant_articles)} articles, here are the key points:

• **Main Story**: {relevant_articles[0]['summary']}
• **Sentiment**: Overall sentiment is {relevant_articles[0]['sentiment']}
• **Category**: This falls under {relevant_articles[0]['category']} news
• **Importance**: {('High' if relevant_articles[0]['importance_score'] > 0.7 else 'Medium' if relevant_articles[0]['importance_score'] > 0.4 else 'Low')} priority news

## 🔗 **Sources & Links**:
{chr(10).join([f"• {source}" for source in sources_used[:3]])}

## 📈 **Related Topics**:
{', '.join(relevant_articles[0]['keywords'][:5]) if relevant_articles[0]['keywords'] else 'No related topics found'}

---
*Response time: {response_time}ms | Articles analyzed: {len(relevant_articles)} | Last updated: {relevant_articles[0]['timestamp'].strftime("%Y-%m-%d %H:%M:%S")}*
"""
            
            # Log the Q&A interaction
            self.log_activity(
                f"💬 Answered question: {question[:50]}...",
                details={
                    'question': question,
                    'articles_found': len(relevant_articles),
                    'response_time_ms': response_time,
                    'sources': sources_used,
                    'categories': [art['category'] for art in relevant_articles]
                }
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    # AUTOMATIC CLEANUP SYSTEM
    def get_database_size_mb(self) -> float:
        """Get current database size in MB"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                       pg_database_size(current_database()) as size_bytes
            """)
            
            result = cursor.fetchone()
            size_bytes = result[1] if result else 0
            size_mb = size_bytes / (1024 * 1024)
            
            cursor.close()
            conn.close()
            return round(size_mb, 2)
            
        except Exception as e:
            logger.error(f"❌ Error getting database size: {e}")
            return 0.0
    
    def count_articles_by_age(self, days: int) -> int:
        """Count articles within specified days"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE timestamp >= %s
            """, (datetime.now() - timedelta(days=days),))
            
            count = cursor.fetchone()[0] or 0
            
            cursor.close()
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"❌ Error counting articles: {e}")
            return 0
    
    def cleanup_old_articles(self, days: int = None) -> int:
        """Remove articles older than specified days"""
        if not self.cleanup_config.enabled:
            logger.info("🔒 Cleanup disabled in configuration")
            return 0
        
        days = days or self.cleanup_config.default_retention_days
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Delete old articles in batches to avoid locks
            total_deleted = 0
            while True:
                cursor.execute("""
                    DELETE FROM articles 
                    WHERE id IN (
                        SELECT id FROM articles 
                        WHERE timestamp < %s 
                        ORDER BY timestamp 
                        LIMIT %s
                    )
                """, (cutoff_date, self.cleanup_config.batch_size))
                
                batch_deleted = cursor.rowcount
                total_deleted += batch_deleted
                
                if batch_deleted == 0:
                    break  # No more articles to delete
                
                conn.commit()
                time.sleep(0.1)  # Small pause between batches
            
            cursor.close()
            conn.close()
            
            if total_deleted > 0:
                logger.info(f"🗑️ Cleaned up {total_deleted} articles older than {days} days")
                self.log_activity(
                    f"🗑️ Automatic cleanup: removed {total_deleted} articles",
                    details={
                        'articles_deleted': total_deleted,
                        'retention_days': days,
                        'cutoff_date': cutoff_date.isoformat()
                    }
                )
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return 0
    
    def cleanup_by_category(self):
        """Smart cleanup based on category-specific retention"""
        if not self.cleanup_config.enabled:
            return
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            total_deleted = 0
            
            for category, retention_days in self.cleanup_config.category_retention.items():
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                cursor.execute("""
                    DELETE FROM articles 
                    WHERE category = %s AND timestamp < %s
                """, (category, cutoff_date))
                
                deleted_count = cursor.rowcount
                total_deleted += deleted_count
                
                if deleted_count > 0:
                    logger.info(f"🗑️ Cleaned {category}: {deleted_count} articles (>{retention_days} days)")
            
            # Also cleanup activity log
            cursor.execute("""
                DELETE FROM activity_log 
                WHERE timestamp < %s
            """, (datetime.now() - timedelta(days=30),))
            
            activity_deleted = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            if total_deleted > 0:
                self.log_activity(
                    f"🗑️ Category-based cleanup: {total_deleted} articles, {activity_deleted} log entries",
                    details={'articles_deleted': total_deleted, 'log_entries_deleted': activity_deleted}
                )
            
        except Exception as e:
            logger.error(f"❌ Error in category cleanup: {e}")
    
    def emergency_cleanup(self):
        """Emergency cleanup when database is getting too large"""
        try:
            db_size = self.get_database_size_mb()
            
            if db_size > 500:  # If database is over 500MB
                logger.warning(f"🚨 Database size ({db_size}MB) triggers emergency cleanup")
                
                # Aggressive cleanup
                deleted = 0
                deleted += self.cleanup_old_articles(days=3)  # Very aggressive
                
                # Remove low importance articles even if recent
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM articles 
                    WHERE importance_score < 0.3 
                    AND timestamp < %s
                """, (datetime.now() - timedelta(days=1),))
                
                low_importance_deleted = cursor.rowcount
                deleted += low_importance_deleted
                
                conn.commit()
                cursor.close()
                conn.close()
                
                logger.warning(f"🚨 Emergency cleanup completed: {deleted} articles removed")
                self.log_activity(
                    f"🚨 Emergency cleanup triggered",
                    details={'database_size_mb': db_size, 'articles_deleted': deleted}
                )
        
        except Exception as e:
            logger.error(f"❌ Error in emergency cleanup: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE timestamp >= %s
            """, (datetime.now() - timedelta(hours=24),))
            recent_articles = cursor.fetchone()[0] or 0
            
            # Articles by category
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM articles 
                WHERE timestamp >= %s
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """, (datetime.now() - timedelta(hours=24),))
            by_category = dict(cursor.fetchall())
            
            # Articles by source
            cursor.execute("""
                SELECT source, COUNT(*) 
                FROM articles 
                WHERE timestamp >= %s
                GROUP BY source 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """, (datetime.now() - timedelta(hours=24),))
            by_source = dict(cursor.fetchall())
            
            # Top keywords
            cursor.execute("""
                SELECT keyword, COUNT(*) 
                FROM articles, unnest(keywords) AS keyword 
                WHERE timestamp >= %s
                GROUP BY keyword 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """, (datetime.now() - timedelta(hours=24),))
            top_keywords = dict(cursor.fetchall())
            
            # Sentiment distribution
            cursor.execute("""
                SELECT sentiment, COUNT(*) 
                FROM articles 
                WHERE timestamp >= %s
                GROUP BY sentiment
            """, (datetime.now() - timedelta(hours=24),))
            sentiment_dist = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            return {
                'total_articles': total_articles,
                'recent_articles_24h': recent_articles,
                'database_size_mb': self.get_database_size_mb(),
                'by_category': by_category,
                'by_source': by_source,
                'top_keywords': top_keywords,
                'sentiment_distribution': sentiment_dist,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {}
    
    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent system activity"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT timestamp, action, source, category, details
                FROM activity_log 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            
            activities = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(activity) for activity in activities]
            
        except Exception as e:
            logger.error(f"❌ Error fetching recent activity: {e}")
            return []
    
    def run_collection_cycle(self):
        """Run one complete collection cycle"""
        cycle_start = time.time()
        
        try:
            logger.info(f"🔄 Starting collection cycle #{self.cycle_count + 1}")
            
            # Collect from all sources
            rss_articles = self.collect_rss_feeds()
            reddit_articles = self.collect_reddit_posts()
            
            all_new_articles = rss_articles + reddit_articles
            
            if all_new_articles:
                # Process with AI
                processed_articles = self.process_articles(all_new_articles)
                
                # Store in database
                self.store_articles(processed_articles)
                
                # Update cycle counter
                self.cycle_count += 1
                
                cycle_time = round(time.time() - cycle_start, 2)
                logger.info(f"✅ Cycle #{self.cycle_count} completed: {len(processed_articles)} articles in {cycle_time}s")
                
                # Run cleanup periodically
                if self.cycle_count % self.cleanup_config.cleanup_frequency == 0:
                    logger.info("🧹 Running periodic cleanup...")
                    self.cleanup_by_category()
                    
                # Check for emergency cleanup
                if self.cycle_count % 5 == 0:  # Every 5 cycles
                    self.emergency_cleanup()
                    
            else:
                logger.info("⏸️ No new articles found in this cycle")
                self.log_activity("⏸️ Collection cycle: no new articles found")
                
        except Exception as e:
            logger.error(f"❌ Error in collection cycle: {e}")
            self.log_activity(f"❌ Collection cycle error: {str(e)}")
    
    def run_continuous_collection(self, interval_seconds: int = 30):
        """Run continuous news collection"""
        self.is_running = True
        logger.info(f"🚀 Starting continuous collection (every {interval_seconds}s)")
        
        try:
            while self.is_running:
                self.run_collection_cycle()
                
                if self.is_running:  # Check again in case stop was called during cycle
                    logger.info(f"⏱️ Waiting {interval_seconds} seconds before next cycle...")
                    time.sleep(interval_seconds)
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Collection stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error in continuous collection: {e}")
        finally:
            self.is_running = False
            logger.info("🏁 Continuous collection ended")
    
    def stop_collection(self):
        """Stop continuous collection"""
        self.is_running = False
        logger.info("🛑 Stop signal sent to collection system")


# CONFIGURATION AND SETUP
def create_database_config():
    """Create database configuration"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'newsrag'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'your_password'),  # CHANGE THIS!
        'port': os.getenv('DB_PORT', '5432')
    }

def create_cleanup_config():
    """Create cleanup configuration"""
    return CleanupConfig(
        enabled=True,
        default_retention_days=7,
        cleanup_frequency=10,
        batch_size=100,
        emergency_threshold=0.9,
        category_retention={
            'Breaking News': 14,
            'Politics': 10,
            'Technology': 7,
            'Business': 7,
            'Science': 7,
            'Sports': 5,
            'Entertainment': 3,
            'General': 7,
            'Reddit': 1
        }
    )


# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 Complete News RAG System")
    print("=" * 50)
    
    # Setup configuration
    db_config = create_database_config()
    cleanup_config = create_cleanup_config()
    
    try:
        # Initialize system
        news_system = CompleteNewsRAGSystem(db_config, cleanup_config)
        
        # Test mode or continuous mode
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            # TEST MODE
            print("\n🧪 Running in TEST mode...")
            
            # Single collection cycle
            news_system.run_collection_cycle()
            
            # Test Q&A
            print("\n💬 Testing Q&A system...")
            test_questions = [
                "What's the latest technology news?",
                "Any business updates today?",
                "What's happening in sports?",
                "Tell me about recent entertainment news",
                "Any breaking news?"
            ]
            
            for question in test_questions:
                print(f"\n🔍 Question: {question}")
                answer = news_system.answer_question(question)
                print(f"📝 Answer: {answer[:200]}...")
            
            # Show statistics
            print("\n📊 System Statistics:")
            stats = news_system.get_system_stats()
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in list(value.items())[:5]:
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
            
        else:
            # CONTINUOUS MODE
            print("\n🔄 Running in CONTINUOUS mode...")
            print("Press Ctrl+C to stop")
            
            # Start continuous collection
            news_system.run_continuous_collection(interval_seconds=30)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Fatal error: {e}")
want to run this with pip install uv
then run with uv run 