# LiveNews AI Assistant - Gemini Migration Guide

## Overview
Your LiveNews AI Assistant has been successfully migrated from Ollama's `llama3.2:3b` to Google's `gemini-2.0-flash` model. This guide will help you complete the setup and verify everything works correctly.

## Prerequisites

### 1. Install Dependencies
```bash
# Install the Google Gemini API client
pip install google-generativeai>=0.3.0

# Or if using uv (recommended):
uv add google-generativeai>=0.3.0
```

### 2. Get Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 3. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your API key:
nano .env
```

Add these lines to your `.env` file:
```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_actual_api_key_here
APP_GEMINI_MODEL=gemini-2.0-flash

# Database Configuration (adjust as needed)
DB_HOST=localhost
DB_NAME=news_rag
DB_USER=postgres
DB_PASSWORD=password
DB_PORT=5432
```

## Testing the Migration

### 1. Test Gemini Client
```bash
python -c "
from gemini_client import GeminiClient
client = GeminiClient()
print('Gemini Available:', client.available)
if client.available:
    response = client.generate_response('Hello, how are you?')
    print('Test Response:', response)
"
```

### 2. Test Enhanced Summarizer
```bash
python enhanced_summarizer.py
```

### 3. Test Full Application
```bash
python app.py
```

Visit `http://localhost:5000` and try asking:
- "What's the latest news?"
- "Any updates on technology?"
- "Tell me about recent developments"

## What Changed

### Files Modified:
- ✅ `app.py` - Uses Gemini instead of Ollama
- ✅ `pathway_rag_system.py` - RAG system with Gemini
- ✅ `news_freshness_validator.py` - AI validation with Gemini
- ✅ `enhanced_summarizer.py` - Gemini-powered summarization
- ✅ `gemini_client.py` - New Gemini API wrapper
- ✅ `requirements.txt` - Added google-generativeai dependency
- ✅ `.env.example` - Configuration template

### Performance Improvements:
- **Faster responses** - API calls vs local inference
- **Better quality** - More accurate summaries and analysis
- **No GPU required** - Runs on any machine
- **Eliminated timeouts** - No more Ollama connection issues

## Troubleshooting

### Common Issues:

**1. "Gemini API not available"**
- Check your `GEMINI_API_KEY` in `.env`
- Verify API key is valid at [Google AI Studio](https://makersuite.google.com/)
- Ensure `google-generativeai` package is installed

**2. "ModuleNotFoundError: No module named 'google.generativeai'"**
```bash
pip install google-generativeai>=0.3.0
```

**3. Database connection errors**
- Update database credentials in `.env`
- Ensure PostgreSQL is running
- For pgvector support: `CREATE EXTENSION IF NOT EXISTS vector;`

**4. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Performance Notes:
- Gemini 2.0 Flash is optimized for speed and efficiency
- No local GPU or RAM requirements
- API rate limits apply (typically very generous)
- Responses are cached for similar queries

## Migration Benefits

### Before (Ollama):
- Required local GPU/CPU resources
- Potential timeout errors
- Limited to model capabilities
- DB similarity search issues

### After (Gemini):
- ✅ Cloud-based, no local compute needed
- ✅ Faster, more reliable responses
- ✅ Better quality summaries and analysis
- ✅ Eliminated database compatibility issues
- ✅ Access to latest AI capabilities

## API Usage & Costs

- Gemini 2.0 Flash has generous free tier
- Monitor usage at [Google AI Studio](https://makersuite.google.com/)
- Cost-effective for most use cases
- Pricing: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens

## Support

If you encounter any issues:
1. Check this guide first
2. Verify your `.env` configuration
3. Test the individual components as shown above
4. Ensure all dependencies are installed

Your LiveNews AI Assistant is now powered by Google's latest Gemini model!
