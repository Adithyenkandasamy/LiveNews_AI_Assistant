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
- 🔗 **Clickable Links** - Direct access to full original articles

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

# Ollama (Optional fallback)
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=nemotron-mini
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
                     model="facebook/bart-large-cnn")
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
├── app.py                 # Main Flask application
├── src/                   # Core modules package
│   ├── __init__.py       # Package initialization
│   ├── gemini_client.py  # Google Gemini AI client
│   ├── summarizer.py     # Enhanced summarization
│   ├── rag_system.py     # Pathway RAG system
│   ├── freshness_validator.py # News freshness validation
│   └── comparison_tool.py # News comparison utilities
├── templates/
│   └── index.html        # Dark theme UI with glassmorphism
├── pyproject.toml        # UV/pip dependencies
├── uv.lock              # Locked dependency versions
├── .env.example         # Environment variables template
└── README.md            # This documentation
```

### Data Flow

1. News Ingestion: RSS feeds → Feedparser → PostgreSQL
2. Duplicate Detection: Canonical ID hashing prevents duplicates  
3. RAG Processing: Pathway → Sentence Transformers → Vector similarity
4. AI Response: User query → RAG context → Gemini API → Response
5. UI Rendering: Flask → Jinja2 templates → Modern dark UI

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