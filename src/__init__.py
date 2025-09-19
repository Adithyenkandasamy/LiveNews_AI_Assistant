"""
LiveNews AI Assistant - Core Modules

This package contains the core modules for the LiveNews AI Assistant:
- gemini_client: Google Gemini AI client wrapper
- summarizer: Advanced news article summarization 
- rag_system: Pathway-based Retrieval Augmented Generation
- freshness_validator: News content freshness validation
- comparison_tool: News comparison and similarity analysis
"""

__version__ = "1.0.0"
__author__ = "LiveNews AI Team"

from .gemini_client import GeminiClient
from .summarizer import EnhancedSummarizer  
from .rag_system import PathwayRAGSystem, PathwayRAGConfig
from .freshness_validator import NewsFreshnessValidator, FreshnessConfig
from .comparison_tool import NewsComparisonTool

__all__ = [
    "GeminiClient",
    "EnhancedSummarizer", 
    "PathwayRAGSystem",
    "PathwayRAGConfig",
    "NewsFreshnessValidator", 
    "FreshnessConfig",
    "NewsComparisonTool"
]
