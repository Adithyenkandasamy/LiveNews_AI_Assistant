"""
Enhanced RAG System using Pathway library with LangChain fallback
Integrates with Ollama Nemotron-mini for news analysis and Q&A
"""

import pathway as pw
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import feedparser
from dotenv import load_dotenv
from news_freshness_validator import NewsFreshnessValidator, FreshnessConfig

# LangChain imports as fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import SentenceTransformerEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available, using Pathway only")

load_dotenv()

@dataclass
class PathwayRAGConfig:
    """Configuration for Pathway RAG system"""
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_results: int = 5
    similarity_threshold: float = 0.7

class PathwayRAGSystem:
    """
    Advanced RAG system using Pathway for real-time data processing
    with LangChain fallback for traditional vector operations
    """
    
    def __init__(self, config: PathwayRAGConfig):
        self.config = config
        self.setup_logging()
        self.setup_database()
        self.init_models()
        self.setup_pathway_pipeline()
        
        # Initialize news freshness validator
        freshness_config = FreshnessConfig(
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model
        )
        self.freshness_validator = NewsFreshnessValidator(freshness_config)
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pathway_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Setup PostgreSQL connection"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'news_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
    def init_models(self):
        """Initialize embedding model and test Ollama connection"""
        # Initialize embedder
        self.embedder = SentenceTransformer(self.config.embedding_model)
        self.logger.info(f"✅ Loaded embedding model: {self.config.embedding_model}")
        
        # Test Ollama connection
        try:
            test_response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=5
            )
            
            if test_response.status_code == 200:
                self.logger.info("✅ Connected to Ollama Nemotron-mini successfully")
                self.ollama_available = True
            else:
                self.logger.error(f"Ollama connection failed: {test_response.status_code}")
                self.ollama_available = False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.ollama_available = False
            
    def setup_pathway_pipeline(self):
        """Setup Pathway data processing pipeline"""
        try:
            # Create Pathway table for news articles
            self.news_table = pw.Table.empty(
                title=pw.column_definition(dtype=str),
                content=pw.column_definition(dtype=str),
                source=pw.column_definition(dtype=str),
                date=pw.column_definition(dtype=str),
                category=pw.column_definition(dtype=str),
                embedding=pw.column_definition(dtype=list)
            )
            
            # Setup text processing pipeline
            self.processed_table = self.news_table.select(
                title=self.news_table.title,
                content=self.news_table.content,
                source=self.news_table.source,
                date=self.news_table.date,
                category=self.news_table.category,
                chunks=pw.apply(self.chunk_text, self.news_table.content),
                embedding=pw.apply(self.generate_embedding, self.news_table.content)
            )
            
            self.logger.info("✅ Pathway pipeline initialized successfully")
            self.pathway_available = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pathway: {e}")
            self.pathway_available = False
            
            # Fallback to LangChain if available
            if LANGCHAIN_AVAILABLE:
                self.setup_langchain_fallback()
            else:
                self.logger.error("No RAG backend available!")
                
    def setup_langchain_fallback(self):
        """Setup LangChain as fallback RAG system"""
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            self.langchain_embeddings = SentenceTransformerEmbeddings(
                model_name=self.config.embedding_model
            )
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                embedding_function=self.langchain_embeddings,
                persist_directory="./chroma_db"
            )
            
            self.logger.info("✅ LangChain fallback initialized successfully")
            self.langchain_available = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain fallback: {e}")
            self.langchain_available = False
            
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using Pathway"""
        if not text or len(text) < 50:
            return [text]
            
        # Simple chunking for Pathway
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(word)
            current_length += len(word) + 1
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
            
    def add_news_articles(self, articles: List[Dict[str, Any]]):
        """Add news articles to the RAG system"""
        if self.pathway_available:
            self._add_to_pathway(articles)
        elif self.langchain_available:
            self._add_to_langchain(articles)
        else:
            self.logger.error("No RAG backend available for adding articles")
            
    def _add_to_pathway(self, articles: List[Dict[str, Any]]):
        """Add articles to Pathway pipeline"""
        try:
            for article in articles:
                # Process article through Pathway
                article_data = {
                    'title': article.get('title', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', ''),
                    'date': article.get('date', ''),
                    'category': article.get('category', 'General'),
                    'embedding': self.generate_embedding(article.get('content', ''))
                }
                
                # Add to Pathway table (this would be done differently in real Pathway)
                # For now, we'll store in database and use Pathway for processing
                self._store_in_database(article_data)
                
            self.logger.info(f"Added {len(articles)} articles to Pathway system")
            
        except Exception as e:
            self.logger.error(f"Failed to add articles to Pathway: {e}")
            
    def _add_to_langchain(self, articles: List[Dict[str, Any]]):
        """Add articles to LangChain vector store"""
        try:
            documents = []
            for article in articles:
                content = f"Title: {article.get('title', '')}\n\nContent: {article.get('content', '')}"
                
                # Split into chunks
                chunks = self.text_splitter.split_text(content)
                
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'title': article.get('title', ''),
                            'source': article.get('source', ''),
                            'date': article.get('date', ''),
                            'category': article.get('category', 'General')
                        }
                    )
                    documents.append(doc)
                    
            # Add to vector store
            self.vector_store.add_documents(documents)
            self.logger.info(f"Added {len(documents)} document chunks to LangChain vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to add articles to LangChain: {e}")
            
    def _store_in_database(self, article_data: Dict[str, Any]):
        """Store article data in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pathway_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    source TEXT,
                    date TEXT,
                    category TEXT,
                    embedding FLOAT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert article
            cursor.execute("""
                INSERT INTO pathway_articles (title, content, source, date, category, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                article_data['title'],
                article_data['content'],
                article_data['source'],
                article_data['date'],
                article_data['category'],
                article_data['embedding']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            
    def search_similar_articles(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for similar articles using RAG with freshness filtering"""
        limit = limit or self.config.max_results
        
        # Get raw results
        if self.pathway_available:
            raw_results = self._search_with_pathway(query, limit * 2)  # Get more to filter
        elif self.langchain_available:
            raw_results = self._search_with_langchain(query, limit * 2)
        else:
            self.logger.error("No RAG backend available for search")
            return []
            
        # Filter for freshness
        fresh_results = self.freshness_validator.filter_fresh_articles(raw_results)
        
        # Return top results after filtering
        return fresh_results[:limit]
            
    def _search_with_pathway(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using Pathway (with database backend for now)"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search in database using cosine similarity
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use PostgreSQL's array operations for similarity
            cursor.execute("""
                SELECT title, content, source, date, category,
                       (1 - (embedding <=> %s::float[])) as similarity
                FROM pathway_articles
                WHERE (1 - (embedding <=> %s::float[])) > %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding, query_embedding, self.config.similarity_threshold, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Pathway search failed: {e}")
            return []
            
    def _search_with_langchain(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using LangChain vector store"""
        try:
            # Similarity search
            docs = self.vector_store.similarity_search(query, k=limit)
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'title': doc.metadata.get('title', ''),
                    'source': doc.metadata.get('source', ''),
                    'date': doc.metadata.get('date', ''),
                    'category': doc.metadata.get('category', 'General'),
                    'similarity': 0.8  # LangChain doesn't return similarity scores directly
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"LangChain search failed: {e}")
            return []
            
    def generate_rag_response(self, query: str, context_articles: List[Dict[str, Any]]) -> str:
        """Generate response using RAG with Ollama"""
        if not self.ollama_available:
            return "AI model not available for response generation."
            
        # Build context from articles
        context = "Relevant News Articles:\n\n"
        for i, article in enumerate(context_articles[:3], 1):
            context += f"{i}. **{article.get('title', 'No title')}**\n"
            context += f"   Source: {article.get('source', 'Unknown')}\n"
            context += f"   Date: {article.get('date', 'Unknown')}\n"
            context += f"   Content: {article.get('content', '')[:300]}...\n\n"
            
        # Create prompt
        prompt = f"""Based on the following news articles, please answer the user's question comprehensively and accurately.

{context}

User Question: {query}

Please provide a detailed, informative response based on the news articles above."""

        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 400
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer if answer else "I couldn't generate a response based on the available information."
            else:
                self.logger.error(f"Ollama response failed: {response.status_code}")
                return "Sorry, I encountered an error while generating the response."
                
        except Exception as e:
            self.logger.error(f"RAG response generation failed: {e}")
            return "Sorry, I encountered an error while processing your question."
            
    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface for the RAG system with freshness analysis"""
        # Search for relevant articles
        relevant_articles = self.search_similar_articles(question)
        
        if not relevant_articles:
            return {
                'answer': "I couldn't find any relevant recent news articles for your question.",
                'sources': [],
                'method': 'pathway' if self.pathway_available else 'langchain',
                'freshness_report': None
            }
            
        # Generate freshness report
        freshness_report = self.freshness_validator.get_news_comparison_report(relevant_articles)
        
        # Generate response
        answer = self.generate_rag_response(question, relevant_articles)
        
        return {
            'answer': answer,
            'sources': relevant_articles,
            'method': 'pathway' if self.pathway_available else 'langchain',
            'num_sources': len(relevant_articles),
            'freshness_report': freshness_report
        }

def main():
    """Test the Pathway RAG system"""
    config = PathwayRAGConfig()
    rag_system = PathwayRAGSystem(config)
    
    # Test query
    result = rag_system.query("What are the latest developments in AI technology?")
    print(f"Answer: {result['answer']}")
    print(f"Method: {result['method']}")
    print(f"Sources found: {result['num_sources']}")

if __name__ == "__main__":
    main()
