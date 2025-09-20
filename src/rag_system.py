"""
Enhanced RAG System using Pathway library with LangChain fallback
Integrates with Google Gemini AI for advanced news analysis and Q&A
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
from .freshness_validator import NewsFreshnessValidator, FreshnessConfig

# LangChain imports as fallback (updated for LangChain >= 0.1 / 0.3 split)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available, using Pathway only")

load_dotenv()

@dataclass
class PathwayRAGConfig:
    """Configuration for Pathway RAG system"""
    gemini_client: Any = None
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
        # Default backend flags to avoid attribute errors
        self.pathway_available: bool = False
        self.langchain_available: bool = False

        self.setup_logging()
        self.setup_database()
        self.init_models()
        self.setup_pathway_pipeline()
        
        # Initialize news freshness validator
        freshness_config = FreshnessConfig(
            gemini_client=config.gemini_client
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
        # Soft flags for pgvector capability (evaluated lazily when needed)
        self._pgvector_checked = False
        self._pgvector_available = False
        self._vector_column_ready = False
        
    def init_models(self):
        """Initialize embedding model and test Gemini connection"""
        # Initialize embedder
        self.embedder = SentenceTransformer(self.config.embedding_model)
        self.logger.info(f"✅ Loaded embedding model: {self.config.embedding_model}")
        
        # Test Gemini connection
        if self.config.gemini_client and self.config.gemini_client.available:
            if self.config.gemini_client.test_connection():
                self.logger.info("✅ Connected to Gemini successfully")
                self.gemini_available = True
            else:
                self.logger.error("Gemini connection test failed")
                self.gemini_available = False
        else:
            self.logger.error("Gemini client not available")
            self.gemini_available = False
            
    def setup_pathway_pipeline(self):
        """Setup Pathway data processing pipeline. If it fails, fall back to LangChain."""
        try:
            # Pathway schema API changes across versions; keep minimal usage here.
            # If this fails, we'll gracefully fall back to LangChain.
            self.news_table = pw.Table.empty()
            self.processed_table = self.news_table
            self.logger.info("✅ Pathway pipeline placeholder initialized")
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
        if getattr(self, "pathway_available", False):
            self._add_to_pathway(articles)
        elif getattr(self, "langchain_available", False):
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
        # First ensure schema is ready (separate transaction)
        self._ensure_database_schema()
        
        # Then insert data (separate transaction)
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if we can use vector column
            self._check_pgvector_capabilities()
            
            # Insert article with appropriate columns
            if self._pgvector_available and self._vector_column_ready:
                try:
                    cursor.execute(
                        """
                        INSERT INTO pathway_articles (title, content, source, date, category, embedding, embedding_vec)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            article_data['title'],
                            article_data['content'],
                            article_data['source'],
                            article_data['date'],
                            article_data['category'],
                            article_data['embedding'],
                            article_data['embedding'],
                        )
                    )
                except Exception as vec_err:
                    # Rollback and try witwhout vector
                    conn.rollback()
                    cursor.execute(
                        """
                        INSERT INTO pathway_articles (title, content, source, date, category, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            article_data['title'],
                            article_data['content'],
                            article_data['source'],
                            article_data['date'],
                            article_data['category'],
                            article_data['embedding']
                        )
                    )
            else:
                cursor.execute(
                    """
                    INSERT INTO pathway_articles (title, content, source, date, category, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        article_data['title'],
                        article_data['content'],
                        article_data['source'],
                        article_data['date'],
                        article_data['category'],
                        article_data['embedding']
                    )
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            import traceback
            traceback.print_exc()
            self.logger.error(f"Database storage failed: {e}")
            
    def _ensure_database_schema(self):
        """Ensure database schema exists (separate transaction)"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create base table
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
            conn.commit()
            cursor.close()
            conn.close()
            
            # Try pgvector in separate transaction
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='pathway_articles' AND column_name='embedding_vec'
                        ) THEN
                            ALTER TABLE pathway_articles ADD COLUMN embedding_vec vector(384);
                        END IF;
                    END$$;
                    """
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception:
                # pgvector not available, that's ok
                if 'conn' in locals():
                    conn.rollback()
                    conn.close()
                pass
                
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            self.logger.error(f"Schema setup failed: {e}")
            
    def _check_pgvector_capabilities(self) -> None:
        """Check once whether pgvector extension and vector column are available."""
        if self._pgvector_checked:
            return
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')")
            self._pgvector_available = bool(cursor.fetchone()[0])
            if self._pgvector_available:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='pathway_articles' AND column_name='embedding_vec'
                    )
                    """
                )
                self._vector_column_ready = bool(cursor.fetchone()[0])
            else:
                self._vector_column_ready = False
            cursor.close()
            conn.close()
        except Exception:
            self._pgvector_available = False
            self._vector_column_ready = False
        finally:
            self._pgvector_checked = True

    def search_similar_articles(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for similar articles using RAG with freshness filtering"""
        limit = limit or self.config.max_results
        
        # Check if user is asking for specific news source
        source_filter = self._extract_source_filter(query)
        
        # Get raw results
        if getattr(self, "pathway_available", False):
            raw_results = self._search_with_pathway(query, limit * 2, source_filter)  # Get more to filter
        elif getattr(self, "langchain_available", False):
            raw_results = self._search_with_langchain(query, limit * 2, source_filter)
        else:
            self.logger.error("No RAG backend available for search")
            return []
            
        # Filter for freshness
        fresh_results = self.freshness_validator.filter_fresh_articles(raw_results)
        
        # Return top results after filtering
        return fresh_results[:limit]
            
    def _extract_source_filter(self, query: str) -> str | None:
        """Extract specific news source from user query"""
        query_lower = query.lower()
        
        # Map of source variations to actual source names
        source_mappings = {
            'indian express': 'Indian Express',
            'express': 'Indian Express',
            'bbc': 'BBC',
            'cnn': 'CNN',
            'reuters': 'Reuters',
            'hindu': 'The Hindu',
            'ndtv': 'NDTV',
            'times of india': 'Times of India',
            'techcrunch': 'TechCrunch',
            'al jazeera': 'Al Jazeera',
            'reddit': 'Reddit WorldNews',
            'tech crunch': 'TechCrunch'
        }
        
        for pattern, source in source_mappings.items():
            if pattern in query_lower:
                return source
                
        return None

    def _search_with_pathway(self, query: str, limit: int, source_filter: str = None) -> List[Dict[str, Any]]:
        """Search using Pathway (with database backend for now)"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Check pgvector capability once
            self._check_pgvector_capabilities()

            if self._pgvector_available and self._vector_column_ready:
                # Use DB-side similarity with pgvector
                try:
                    conn = psycopg2.connect(**self.db_config)
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    
                    # Build query with optional source filter
                    if source_filter:
                        sql = """
                        SELECT title, content, source, date, category,
                               (1 - (embedding_vec <=> %s::vector)) as similarity
                        FROM pathway_articles
                        WHERE embedding_vec IS NOT NULL AND source ILIKE %s
                        ORDER BY similarity DESC
                        LIMIT %s
                        """
                        cursor.execute(sql, (query_embedding, f"%{source_filter}%", limit))
                    else:
                        sql = """
                        SELECT title, content, source, date, category,
                               (1 - (embedding_vec <=> %s::vector)) as similarity
                        FROM pathway_articles
                        WHERE embedding_vec IS NOT NULL
                        ORDER BY similarity DESC
                        LIMIT %s
                        """
                        cursor.execute(sql, (query_embedding, limit))
                    results = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    return [dict(row) for row in results]
                except Exception as sql_err:
                    # If anything goes wrong, gracefully fall back to Python similarity
                    self.logger.warning(f"DB-side similarity not available, falling back to Python. Error: {sql_err}")
                    return self._search_with_python_similarity(query_embedding, limit, source_filter)
            else:
                # No pgvector support; use Python similarity
                return self._search_with_python_similarity(query_embedding, limit, source_filter)
            
        except Exception as e:
            self.logger.error(f"Pathway search failed: {e}")
            return []

    def _search_with_python_similarity(self, query_embedding: List[float], limit: int, source_filter: str = None) -> List[Dict[str, Any]]:
        """Fallback similarity search using Python cosine similarity on embeddings stored as FLOAT[]."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            # Fetch a reasonable window of recent rows to keep it fast
            if source_filter:
                cursor.execute(
                    """
                    SELECT title, content, source, date, category, embedding
                    FROM pathway_articles
                    WHERE embedding IS NOT NULL AND source ILIKE %s
                    ORDER BY id DESC
                    LIMIT 1000
                    """,
                    (f"%{source_filter}%",)
                )
            else:
                cursor.execute(
                    """
                    SELECT title, content, source, date, category, embedding
                    FROM pathway_articles
                    WHERE embedding IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 1000
                    """
                )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if not rows:
                return []

            # Normalize query vector
            q = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q) + 1e-8
            q = q / q_norm

            scored = []
            for r in rows:
                emb = np.array(r.get('embedding') or [], dtype=np.float32)
                if emb.size == 0:
                    continue
                emb_norm = np.linalg.norm(emb) + 1e-8
                emb = emb / emb_norm
                sim = float(np.dot(q, emb))
                item = {k: r[k] for k in ['title', 'content', 'source', 'date', 'category']}
                item['similarity'] = sim
                scored.append(item)

            # Sort by similarity desc, then by recency if date parsable
            def _parse_dt(d):
                try:
                    return datetime.strptime(d, '%Y-%m-%d %H:%M')
                except Exception:
                    try:
                        return datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        return datetime.min

            scored.sort(key=lambda x: (x['similarity'], _parse_dt(str(x.get('date','')))), reverse=True)
            return scored[:limit]
        except Exception as e:
            self.logger.error(f"Python similarity fallback failed: {e}")
            return []
            
    def _search_with_langchain(self, query: str, limit: int, source_filter: str = None) -> List[Dict[str, Any]]:
        """Search using LangChain vector store"""
        try:
            # Similarity search
            docs = self.vector_store.similarity_search(query, k=limit)
            
            results = []
            for doc in docs:
                # Apply source filter if specified
                doc_source = doc.metadata.get('source', '')
                if source_filter and source_filter.lower() not in doc_source.lower():
                    continue
                    
                results.append({
                    'content': doc.page_content,
                    'title': doc.metadata.get('title', ''),
                    'source': doc_source,
                    'date': doc.metadata.get('date', ''),
                    'category': doc.metadata.get('category', 'General'),
                    'similarity': 0.8  # LangChain doesn't return similarity scores directly
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"LangChain search failed: {e}")
            return []
            
    def generate_rag_response(self, query: str, context_articles: List[Dict[str, Any]]) -> str:
        """Generate response using RAG with Gemini"""
        if not self.gemini_available:
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
            answer = self.config.gemini_client.generate_response(
                prompt,
                max_tokens=220,
                temperature=0.7
            )
            return answer if answer else "I couldn't generate a response based on the available information."
                
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

    def has_recent_news(self, topic: str, hours: int = 12, min_similarity: float = 0.5) -> bool:
        """Return True if there is at least one recent article about the topic meeting similarity threshold.
        Uses fast vector similarity. Applies freshness filtering for given time window.
        """
        try:
            # First, fetch more results than needed and then filter by time via freshness validator
            raw = []
            if getattr(self, "pathway_available", False):
                raw = self._search_with_pathway(topic, limit=20)
            elif getattr(self, "langchain_available", False):
                raw = self._search_with_langchain(topic, limit=20)
            else:
                return False

            if not raw:
                return False

            # Filter by similarity threshold
            candidates = [r for r in raw if float(r.get('similarity', 0.0)) >= min_similarity]
            if not candidates:
                return False

            # Time window filtering
            cutoff = datetime.now() - timedelta(hours=hours)
            fresh = []
            for r in candidates:
                d = str(r.get('date', ''))
                dt = None
                for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'):
                    try:
                        dt = datetime.strptime(d, fmt)
                        break
                    except Exception:
                        continue
                if dt is None:
                    # If no date, keep but treat as less reliable; let freshness validator handle it
                    fresh.append(r)
                else:
                    if dt >= cutoff:
                        fresh.append(r)

            if not fresh:
                # As a secondary check, use freshness validator across all candidates
                fresh = self.freshness_validator.filter_fresh_articles(candidates)

            return len(fresh) > 0
        except Exception as e:
            self.logger.error(f"has_recent_news failed: {e}")
            return False

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
