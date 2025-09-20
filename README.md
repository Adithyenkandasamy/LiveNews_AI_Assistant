# 🤖 LiveNews AI Assistant

> A sophisticated real-time news chatbot powered by Google Gemini AI, Pathway RAG, and modern web technologies

## 📖 Overview

LiveNews AI Assistant is an intelligent news chatbot that provides real-time news updates, intelligent summarization, and contextual conversation about current events. Built with cutting-edge AI technologies including Google Gemini 2.0 Flash, Pathway RAG system, and a beautiful dark-themed glassmorphism UI.

## ✨ Key Features

- 🔄 **Real-time News Sync** - Updates every 60 seconds from multiple RSS sources
- 🤖 **AI-Powered Chat** - Natural conversations about current events using Google Gemini
- 📰 **Smart Summarization** - Intelligent article summarization with content validation
- 🔍 **RAG System** - Advanced Retrieval Augmented Generation using Pathway library
- 🌑 **Modern UI** - Dark theme with glassmorphism effects and smooth animations
- ⚡ **Duplicate Prevention** - Smart canonical ID system prevents duplicate news
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile
- 📅 **Article Timestamps** - Clear publication dates and freshness indicators

## 🚀 Quick Start with UV (Recommended)

### Prerequisites
- Python 3.11+ 
- PostgreSQL database
- Google Gemini API key

### 1. Install UV Package Manager
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 2. Clone and Setup Project
```bash
git clone <repository-url>
cd LiveNews_AI_Assistant

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Required environment variables:
```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Database Configuration  
DB_HOST=localhost
DB_NAME=livenews_ai
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=5432

```

### 4. Database Setup
```bash
# Create PostgreSQL database
createdb livenews_ai

# The app will automatically create required tables on first run
```

### 5. Run the Application
```bash
# Using UV
uv run python app.py

# Or with activated virtual environment
python app.py
```

Visit `http://localhost:5000` to access the chatbot interface.

## 📦 Libraries and Frameworks

### Core Framework Stack

#### **Flask** - Web Framework
```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    # Handle chat requests
    return jsonify(response)
```

#### Google Gemini AI - Primary LLM
```python
from src.gemini_client import GeminiClient

client = GeminiClient()
response = client.generate_response(
    prompt="Summarize today's news",
    max_tokens=1000
)
```

#### Pathway - RAG System
```python
from src.rag_system import PathwayRAGSystem, PathwayRAGConfig

config = PathwayRAGConfig()
rag = PathwayRAGSystem(config)
results = rag.search_articles("climate change news")
```

### Data Processing & AI

#### Sentence Transformers - Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["News article text"])
```

#### Transformers - Fallback Summarization
```python
from transformers import pipeline

summarizer = pipeline("summarization", 
                     model="google/pegasus-xsum")
summary = summarizer(text, max_length=130)
```

#### Feedparser - RSS Processing
```python
import feedparser

feed = feedparser.parse("https://feeds.bbci.co.uk/news/rss.xml")
articles = [entry for entry in feed.entries]
```

### Database & Storage

#### PostgreSQL + psycopg2 - Primary Database
```python
import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
```

#### LangChain + Chroma - Fallback Vector Store
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma(embedding_function=embeddings)
```

### Utility Libraries

#### Python-dotenv - Environment Variables
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
```

#### Requests - HTTP Client
```python
import requests

response = requests.get(
    'https://api.example.com/news',
    timeout=30
)
```

## 🏗️ Architecture Overview

```
LiveNews AI Assistant
├── app.py                 # Main Flask application & news aggregator
├── src/                   # Core modules package
│   ├── __init__.py       # Package initialization
│   ├── gemini_client.py  # Google Gemini AI client wrapper
│   ├── summarizer.py     # Enhanced multi-model summarization
│   ├── rag_system.py     # Pathway RAG system with vector search
│   ├── freshness_validator.py # News freshness validation & filtering
│   └── comparison_tool.py # News comparison utilities
├── templates/
│   └── index.html        # Modern dark theme UI with glassmorphism
├── pyproject.toml        # UV/pip dependencies configuration
├── uv.lock              # Locked dependency versions for reproducibility
├── .env.example         # Environment variables template
└── README.md            # This comprehensive documentation
```

## 📋 Detailed Code Structure

### 🚀 **app.py** - Main Application Hub
The heart of the application containing:

#### **NewsAggregator Class**
```python
class NewsAggregator:
    def __init__(self):
        # Database connection with PostgreSQL
        self.db_config = {...}
        # AI clients initialization
        self.gemini_client = GeminiClient()
        self.rag_system = PathwayRAGSystem()
        # News sources configuration
        self.RSS_FEEDS = {
            'BBC': 'https://feeds.bbci.co.uk/news/rss.xml',
            'CNN': 'http://rss.cnn.com/rss/edition.rss',
            # ... more sources
        }
```

#### **Key Methods:**
- `fetch_and_store_news()` - Aggregates from 8+ RSS sources including Reddit WorldNews
- `generate_canonical_id()` - Creates SHA256 hashes to prevent duplicate articles
- `clear_old_articles()` - Database cleanup removing articles older than 7 days
- `get_chat_response()` - RAG-powered conversational AI using Gemini

#### **Flask Routes:**
- `GET /` - Serves the main chat interface
- `POST /chat` - Handles user queries with RAG context
- `GET /news` - Returns latest news articles in JSON
- `GET /health` - Application health check endpoint

### 🤖 **src/gemini_client.py** - AI Client Wrapper
Handles Google Gemini 2.0 Flash API integration:

```python
class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        self.available = self._test_connection()
    
    def generate_response(self, prompt, max_tokens=1000):
        # Handles API calls with timeout and error handling
        # Includes safety settings and generation config
```

**Features:**
- Automatic fallback handling for API failures
- Configurable safety settings and generation parameters
- Connection testing and availability checking
- Timeout management (30s) to prevent hanging requests

### 🔍 **src/rag_system.py** - Advanced RAG Implementation  
Sophisticated Retrieval Augmented Generation system:

```python
class PathwayRAGSystem:
    def __init__(self, config: PathwayRAGConfig):
        # Pathway connector for real-time data processing
        self.pathway_available = self._check_pathway()
        # SentenceTransformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # LangChain + Chroma as fallback
        self.fallback_vectorstore = Chroma()
```

**Dual-Mode Architecture:**
- **Primary**: Pathway library for real-time vector operations
- **Fallback**: LangChain + Chroma for compatibility
- **Vector Store**: PostgreSQL with pgvector extension support
- **Embeddings**: 384-dimensional vectors using MiniLM model

### 📰 **src/summarizer.py** - Multi-Model Summarization
Flexible summarization with multiple AI backends:

```python
class EnhancedSummarizer:
    def __init__(self, model_type="gemini"):
        if model_type == "gemini":
            self.gemini_client = GeminiClient()
        elif model_type == "transformers":
            self.summarizer = pipeline("summarization", 
                                     model="google/pegasus-xsum")
```

**Supported Models:**
- **Gemini 2.0 Flash** - Primary, high-quality summaries
- **PEGASUS-XSUM** - Transformers fallback for offline use
- **Content Validation** - Filters low-quality or promotional content

### ✅ **src/freshness_validator.py** - News Quality Control
Ensures only fresh, relevant news reaches users:

```python
class FreshnessValidator:
    def validate_article(self, article):
        # Time-based filtering (7 days maximum)
        # Content pattern detection for old news
        # Publication date validation
        # Source credibility checking
```

**Smart Filtering:**
- **Time Window**: 7 days maximum article age
- **Pattern Detection**: Blocks articles mentioning "2023", "2022", "last year"
- **Content Analysis**: AI-powered relevance scoring
- **Source Validation**: Credible news source verification

### 🔧 **src/comparison_tool.py** - News Analysis Utilities
Advanced news comparison and analysis:

```python
class NewsComparisonTool:
    def compare_articles(self, articles):
        # Similarity detection using embeddings
        # Topic clustering and categorization
        # Duplicate content identification
        # Source bias analysis
```

## 🔄 Advanced Data Flow & Processing

### **1. Multi-Source News Ingestion Pipeline**
```python
# RSS Sources + Reddit JSON API
sources = ['BBC', 'CNN', 'Reuters', 'AP', 'NPR', 'Indian Express', 'TOI', 'Reddit WorldNews']
→ feedparser.parse() / requests.get()
→ Content validation & sanitization
→ Publication date parsing & validation
→ Age filtering (7 days maximum)
→ Content pattern detection (blocks old news references)
```

### **2. Intelligent Duplicate Detection**
```python
def generate_canonical_id(article):
    # SHA256 hash of normalized content
    key = '\n'.join([
        normalize_title(article['title']),
        article['source'],
        normalize_url(article['url']),
        article['content'][:120]  # Content snippet
    ])
    return hashlib.sha256(key.encode()).hexdigest()
```

### **3. RAG Processing Architecture**
```
User Query → Embedding Generation (MiniLM-L6-v2)
           ↓
Vector Search (Pathway/PostgreSQL with pgvector)
           ↓
Similarity Ranking (Cosine similarity > 0.7)
           ↓
Context Assembly (Top 10 relevant articles)
           ↓
Gemini API (Contextual response generation)
```

### **4. Real-time Database Operations**
- **Automatic Cleanup**: Removes articles older than 7 days on startup
- **Concurrent Processing**: Multi-threaded news collection
- **Fallback Systems**: LangChain + Chroma when Pathway unavailable
- **Connection Pooling**: Efficient PostgreSQL connection management

### **5. AI Response Generation**
```python
# Context-aware prompt construction
prompt = f"""
Based on these recent news articles:
{formatted_context}

User question: {user_query}
Provide accurate, current information...
"""
response = gemini_client.generate_response(prompt, max_tokens=1000)
```

## 🚀 Recent Technical Improvements

### **Migration from Ollama to Gemini (2024)**
- **Previous**: Local Ollama llama3.2:3b model with timeout issues
- **Current**: Google Gemini 2.0 Flash API for superior performance
- **Benefits**: 
  - Eliminated local model timeout problems
  - Improved response quality and speed
  - Better multilingual support
  - More reliable availability

### **Advanced RAG Implementation**
- **Pathway Integration**: Real-time vector operations for news processing
- **Dual-Mode System**: Pathway primary + LangChain fallback
- **PostgreSQL Vector Storage**: pgvector extension for database-side similarity
- **Smart Fallbacks**: Automatic Python cosine similarity when pgvector unavailable

### **Enhanced News Quality Control**
- **Freshness Filtering**: 7-day window with content pattern detection
- **Source Diversification**: Added Reddit WorldNews JSON API
- **Content Validation**: AI-powered relevance and quality scoring
- **Duplicate Prevention**: Sophisticated canonical ID system

## 🔧 Configuration

### Pathway RAG Configuration
```python
@dataclass
class PathwayRAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 10
    similarity_threshold: float = 0.7
    database_url: str = "postgresql://user:pass@localhost/db"
```

### Freshness Validation
```python
@dataclass  
class FreshnessConfig:
    max_age_days: int = 7
    stale_warning_days: int = 3
    enable_ai_validation: bool = True
```

## 📰 Supported News Sources

- 📺 **BBC News** - `https://feeds.bbci.co.uk/news/rss.xml`
- 📡 **CNN** - `http://rss.cnn.com/rss/edition.rss`
- 🏢 **Reuters** - `https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best`
- 📰 **Associated Press** - `https://apnews.com/apf-topnews`
- 🎙️ **NPR** - `https://feeds.npr.org/1001/rss.xml`

## 🎨 UI Features

The modern dark theme includes:
- ✨ **Glassmorphism effects** with backdrop blur
- 🌈 **Animated gradients** and floating particles
- 🔄 **Smooth transitions** and hover effects  
- 📱 **Responsive design** for all screen sizes
- 🎯 **Custom scrollbars** and enhanced typography
- 🎬 **Message animations** with slide-in effects

## 🔒 Security & Best Practices

- 🔐 Environment variables for sensitive data
- 🛡️ SQL injection prevention with parameterized queries
- ⏱️ Rate limiting on API endpoints
- ✅ Input validation and sanitization
- 🌐 CORS configuration for production

## 🚀 Deployment Options

### Development
```bash
uv run python app.py
```

### Production (Gunicorn)
```bash
uv add gunicorn
uv run gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install uv && uv sync
CMD ["uv", "run", "python", "app.py"]
```

## 🛠️ Development

### Adding New News Sources
```python
RSS_FEEDS = {
    'new_source': {
        'url': 'https://example.com/rss',
        'category': 'technology'
    }
}
```

### Customizing AI Models
```python
# Switch to different Gemini model
GEMINI_MODEL = "gemini-pro"  

# Add new summarization model
summarizer = pipeline("summarization", 
                     model="google/pegasus-xsum")
```

## 🐛 Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Test connection
psql -h localhost -U username -d livenews_ai
```

**Gemini API Errors**  
```bash
# Verify API key
echo $GEMINI_API_KEY

# Test connection
uv run python -c "from src.gemini_client import GeminiClient; client = GeminiClient(); print('Connected!')"
```

**Missing Dependencies**
```bash
# Reinstall dependencies
uv sync --reinstall
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`  
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Support

For support and questions:
- Create an issue on GitHub
- Check troubleshooting section
- Review configuration examples

---

**Built with ❤️ using modern AI and web technologies**