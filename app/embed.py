import os
import pickle
import json
import numpy as np
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini embedding model
embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

VECTOR_STORE_DIR = "vector_store"
METADATA_FILE = "chunk_metadata.pkl"

# Performance settings
MAX_WORKERS = min(8, (os.cpu_count() or 1) + 4)
EMBEDDING_BATCH_SIZE = 5
EMBED_DIM = 768

# Try to import optional dependencies
PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
    logger.info("Pinecone available")
except ImportError:
    logger.warning("Pinecone not available, using local storage only")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "existing-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "rag-hybrid")

pc = None
pinecone_index = None

def _init_pinecone():
    global pc, pinecone_index
    if not PINECONE_AVAILABLE or not PINECONE_API_KEY:
        logger.warning("Pinecone not available or API key not set")
        return
        
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")

class HybridRAGIndex:
    """Simplified RAG Index with local storage fallback"""
    
    def __init__(self):
        if not os.path.isdir(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        if PINECONE_AVAILABLE and pinecone_index is None:
            _init_pinecone()
        self.pinecone = pinecone_index
        
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.document_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.local_vectors: Dict[str, List[float]] = {}

    def create_or_load_index(self):
        """Create or load local metadata"""
        metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self.metadata = data.get('metadata', {})
                    self.document_chunks = data.get('document_chunks', {})
                    self.local_vectors = data.get('local_vectors', {})
                else:
                    self.metadata = data
                logger.info("Loaded existing index")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.metadata = {}
                self.document_chunks = {}
                self.local_vectors = {}
        else:
            self.metadata = {}
            self.document_chunks = {}
            self.local_vectors = {}
            logger.info("Created new index")
        
        return self

def is_valid_embedding(embedding: List[float], min_norm: float = 1e-6) -> bool:
    """Check if embedding is valid"""
    if not embedding or not isinstance(embedding, (list, tuple)):
        return False
    
    try:
        arr = np.array(embedding, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
        norm = np.linalg.norm(arr)
        return norm >= min_norm
    except Exception:
        return False

def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for a batch of texts"""
    embeddings = []
    
    for text in texts:
        if not text or len(text.strip()) < 3:
            embeddings.append(None)
            continue
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text.strip(),
                task_type="retrieval_document"
            )
            emb = result.get('embedding')
            
            if not emb or not is_valid_embedding(emb):
                embeddings.append(None)
                continue
                
            # Normalize embedding
            emb_arr = np.array(emb, dtype=float)
            norm = np.linalg.norm(emb_arr)
            if norm > 1e-6:
                emb_norm = (emb_arr / norm).tolist()
                embeddings.append(emb_norm)
            else:
                embeddings.append(None)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            embeddings.append(None)
    
    return embeddings

async def add_chunks_to_hybrid_index(
    hybrid_index: HybridRAGIndex,
    chunks: List[Dict[str, Any]],
    document_path: str = None
):
    """Add chunks to index"""
    if not chunks:
        return hybrid_index

    document_id = chunks[0].get('document_id')
    logger.info(f"Adding {len(chunks)} chunks for document {document_id}")

    # Remove existing document
    remove_document_from_hybrid_index(hybrid_index, document_id)

    # Store chunks
    hybrid_index.document_chunks[document_id] = chunks

    # Generate embeddings
    chunk_texts = [chunk.get('text', '') for chunk in chunks]
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(chunk_texts), EMBEDDING_BATCH_SIZE):
        batch = chunk_texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = generate_embeddings_batch(batch)
        all_embeddings.extend(batch_embeddings)
        
        # Small delay to avoid rate limits
        if i + EMBEDDING_BATCH_SIZE < len(chunk_texts):
            await asyncio.sleep(0.1)

    # Store successful embeddings
    vectors = []
    successful_count = 0
    
    for emb, chunk in zip(all_embeddings, chunks):
        if emb is None or not is_valid_embedding(emb):
            continue
            
        chunk_id = str(chunk.get('chunk_id'))
        hybrid_index.metadata[chunk_id] = chunk
        hybrid_index.local_vectors[chunk_id] = emb
        
        if hybrid_index.pinecone:
            vectors.append({
                "id": chunk_id,
                "values": emb,
                "metadata": {
                    "document_id": chunk.get("document_id"),
                    "document_name": chunk.get("document_name"),
                    "title": chunk.get("title"),
                    "page_number": chunk.get("page_number"),
                    "chunk_id": chunk_id,
                    "short_text": (chunk.get("text") or "")[:300],
                }
            })
        
        successful_count += 1

    # Upsert to Pinecone if available
    if hybrid_index.pinecone and vectors:
        try:
            hybrid_index.pinecone.upsert(
                vectors=vectors,
                namespace=PINECONE_NAMESPACE
            )
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")

    # Save local state
    save_hybrid_index(hybrid_index)
    logger.info(f"Successfully indexed {successful_count}/{len(chunks)} chunks")
    
    return hybrid_index

def hybrid_search(hybrid_index: HybridRAGIndex, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
    """Perform search using available methods"""
    logger.info(f"Searching for: {query[:50]}... in document {document_id}")
    
    # Generate query embedding
    query_embeddings = generate_embeddings_batch([query])
    if not query_embeddings[0] or not is_valid_embedding(query_embeddings[0]):
        logger.warning("Failed to generate query embedding")
        return []
    
    q_vec = query_embeddings[0]
    results = []

    # Try Pinecone first
    if hybrid_index.pinecone:
        try:
            res = hybrid_index.pinecone.query(
                vector=q_vec,
                top_k=top_k * 2,
                include_metadata=True,
                namespace=PINECONE_NAMESPACE,
                filter={"document_id": {"$eq": document_id}}
            )
            
            for match in res.get("matches", []):
                meta = match.get("metadata", {})
                chunk_id = meta.get("chunk_id")
                chunk_data = hybrid_index.metadata.get(chunk_id)
                
                if chunk_data:
                    results.append({
                        "score": float(match.get("score", 0.0)),
                        "chunk_data": chunk_data,
                        "retrieval_type": "dense"
                    })
                    
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")

    # Fallback to local search
    if not results:
        logger.info("Using local vector search")
        similarities = []
        chunk_ids = []
        
        for chunk_id, vec in hybrid_index.local_vectors.items():
            chunk_data = hybrid_index.metadata.get(chunk_id)
            if chunk_data and chunk_data.get('document_id') == document_id:
                similarity = np.dot(q_vec, vec)
                similarities.append(similarity)
                chunk_ids.append(chunk_id)
        
        if similarities:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            for idx in top_indices:
                chunk_id = chunk_ids[idx]
                chunk_data = hybrid_index.metadata[chunk_id]
                results.append({
                    "score": float(similarities[idx]),
                    "chunk_data": chunk_data,
                    "retrieval_type": "local"
                })

    # Add context to results
    expanded_results = []
    for result in results:
        result['context'] = result['chunk_data'].get('text', '')
        result['short_text'] = result['chunk_data'].get('text', '')
        expanded_results.append(result)

    logger.info(f"Found {len(expanded_results)} relevant chunks")
    return expanded_results

def remove_document_from_hybrid_index(hybrid_index: HybridRAGIndex, document_id: str):
    """Remove document from index"""
    # Remove from Pinecone
    if hybrid_index.pinecone:
        try:
            hybrid_index.pinecone.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=PINECONE_NAMESPACE
            )
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")

    # Remove from local storage
    to_remove = [cid for cid, data in hybrid_index.metadata.items() 
                if data.get('document_id') == document_id]
    
    for chunk_id in to_remove:
        hybrid_index.metadata.pop(chunk_id, None)
        hybrid_index.local_vectors.pop(chunk_id, None)
    
    hybrid_index.document_chunks.pop(document_id, None)

def save_hybrid_index(hybrid_index: HybridRAGIndex):
    """Save index to disk"""
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadata': hybrid_index.metadata,
            'document_chunks': hybrid_index.document_chunks,
            'local_vectors': hybrid_index.local_vectors
        }, f)

# Legacy compatibility functions
def create_or_load_index():
    hybrid_index = HybridRAGIndex()
    return hybrid_index.create_or_load_index(), {}

def add_chunks_to_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    if isinstance(index, HybridRAGIndex):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_hybrid_index(index, chunks, document_path))
    else:
        hybrid_index = HybridRAGIndex()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_hybrid_index(hybrid_index, chunks, document_path))

def search_document(index, metadata, query: str, document_id: str, top_k: int = 5):
    if isinstance(index, HybridRAGIndex):
        return hybrid_search(index, query, document_id, top_k)
    else:
        hybrid_index = HybridRAGIndex()
        hybrid_index.metadata = metadata
        return hybrid_search(hybrid_index, query, document_id, top_k)
