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
from rank_bm25 import BM25Okapi
import re

from pinecone import Pinecone

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
BM25_FILE = "bm25_index.pkl"

# Performance settings
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
EMBEDDING_BATCH_SIZE = 10
CHUNK_BATCH_SIZE = 50

# Pinecone configuration - Updated to use namespaces
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "existing-index")  # Use existing index
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "rag-hybrid")  # Use namespace for data separation
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMBED_DIM = 768

pc = None
pinecone_index = None

def _init_pinecone():
    global pc, pinecone_index
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set in environment")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        # Connect to existing index instead of creating new one
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to existing Pinecone index '{PINECONE_INDEX_NAME}' with namespace '{PINECONE_NAMESPACE}'")
    except Exception as e:
        logger.error(f"Failed to connect to index '{PINECONE_INDEX_NAME}': {e}")
        raise

class HybridRAGIndex:
    """Hybrid RAG Index combining dense (Pinecone) and sparse (BM25) retrieval with multi-level chunking"""
    
    def __init__(self):
        if not os.path.isdir(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        if pinecone_index is None:
            _init_pinecone()
        self.pinecone = pinecone_index
        self.sparse_index = None  # BM25
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.document_chunks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.bm25_corpus: List[List[str]] = []
        self.chunk_to_parent: Dict[str, str] = {}

    def create_or_load_index(self):
        """Create or load local metadata/BM25 (vectors live in Pinecone)"""
        start_time = time.time()
        metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
        bm25_path = os.path.join(VECTOR_STORE_DIR, BM25_FILE)

        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.metadata = data.get('metadata', {})
                self.document_chunks = data.get('document_chunks', {})
                self.bm25_corpus = data.get('bm25_corpus', [])
                self.chunk_to_parent = data.get('chunk_to_parent', {})
            else:
                self.metadata = data
                self.document_chunks = {}
                self.bm25_corpus = []
                self.chunk_to_parent = {}
            
            if os.path.exists(bm25_path) and self.bm25_corpus:
                try:
                    with open(bm25_path, 'rb') as f:
                        self.sparse_index = pickle.load(f)
                except Exception:
                    self.sparse_index = None
            logger.info("Loaded local metadata and BM25")
        else:
            self.metadata = {}
            self.document_chunks = {}
            self.bm25_corpus = []
            self.chunk_to_parent = {}
            logger.info("Initialized empty local state")
        
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' namespace '{PINECONE_NAMESPACE}' ready; init took {time.time() - start_time:.2f}s")
        return self

def process_chunk_parallel(chunk_data: Tuple[Dict[str, Any], int]) -> Tuple[List[Dict], List[Dict]]:
    """Process a single chunk for multi-level chunking (for parallel execution)"""
    chunk, _ = chunk_data
    text = chunk.get('text', '') or ''
    
    short_parts = split_into_short_chunks(text, max_chars=300)
    long_parts = split_into_long_chunks(text, max_chars=1000)
    
    short_chunks: List[Dict] = []
    long_chunks: List[Dict] = []
    
    for i, short_text in enumerate(short_parts):
        short_chunk = {
            **chunk,
            'text': short_text,
            'chunk_id': f"{chunk['chunk_id']}_short_{i}",
            'chunk_type': 'short',
            'parent_id': chunk['chunk_id']
        }
        short_chunks.append(short_chunk)
    
    for i, long_text in enumerate(long_parts):
        long_chunk = {
            **chunk,
            'text': long_text,
            'chunk_id': f"{chunk['chunk_id']}_long_{i}",
            'chunk_type': 'long',
            'parent_id': chunk['chunk_id']
        }
        long_chunks.append(long_chunk)
    
    return short_chunks, long_chunks

def create_multi_level_chunks_parallel(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Create both short (for precision) and long (for context) chunks using parallel processing"""
    start_time = time.time()
    logger.info(f"Starting parallel multi-level chunking for {len(chunks)} chunks")
    
    all_short_chunks: List[Dict] = []
    all_long_chunks: List[Dict] = []
    
    chunk_data = [(chunk, idx) for idx, chunk in enumerate(chunks)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {executor.submit(process_chunk_parallel, data): data for data in chunk_data}
        for future in as_completed(future_to_chunk):
            try:
                short_chunks, long_chunks = future.result()
                all_short_chunks.extend(short_chunks)
                all_long_chunks.extend(long_chunks)
            except Exception as exc:
                logger.error(f'Chunk processing error: {exc}')
    
    logger.info(f"Multi-level chunking completed in {time.time() - start_time:.2f}s")
    return all_short_chunks, all_long_chunks

def is_valid_embedding(embedding: List[float], min_norm: float = 1e-6) -> bool:
    """
    Check if embedding is valid (not None, not empty, not all zeros, has reasonable norm).
    
    Args:
        embedding: The embedding vector to validate
        min_norm: Minimum acceptable L2 norm for the embedding
    
    Returns:
        bool: True if embedding is valid, False otherwise
    """
    if not embedding or not isinstance(embedding, (list, tuple)):
        return False
    
    try:
        arr = np.array(embedding, dtype=float)
        
        # Check if array is empty
        if arr.size == 0:
            return False
            
        # Check if all values are finite (not NaN or inf)
        if not np.all(np.isfinite(arr)):
            return False
            
        # Check if norm is above minimum threshold
        norm = np.linalg.norm(arr)
        if norm < min_norm:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating embedding: {e}")
        return False

def process_embedding_batch(batch_texts: List[str], doc_id: Optional[str] = None) -> List[Optional[List[float]]]:
    """
    Synchronous embedding batch processing. Returns a list containing either:
    - normalized embedding (List[float]) for each non-empty text, or
    - None for skipped/errored items.
    """
    embeddings: List[Optional[List[float]]] = []
    for i, text in enumerate(batch_texts):
        # Skip empty or whitespace-only text
        if not (text and text.strip()):
            logger.warning(f"[Doc {doc_id}] Batch item {i}: Skipping empty chunk text for embedding")
            embeddings.append(None)
            continue
            
        # Skip very short text that might not generate meaningful embeddings
        if len(text.strip()) < 3:
            logger.warning(f"[Doc {doc_id}] Batch item {i}: Skipping very short text: '{text[:50]}'")
            embeddings.append(None)
            continue
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text.strip(),  # Ensure we strip whitespace
                task_type="retrieval_document"
            )
            emb = result.get('embedding')
            
            if not emb or not isinstance(emb, (list, tuple)):
                logger.warning(f"[Doc {doc_id}] Batch item {i}: Received invalid embedding (None or wrong type)")
                embeddings.append(None)
                continue
                
            # Validate embedding before normalization
            if not is_valid_embedding(emb):
                logger.warning(f"[Doc {doc_id}] Batch item {i}: Embedding failed validation (likely all zeros or invalid)")
                embeddings.append(None)
                continue
                
            # Normalize embedding
            emb_arr = np.array(emb, dtype=float)
            norm = np.linalg.norm(emb_arr)
            
            if norm > 1e-6:  # Use a reasonable threshold
                emb_norm = (emb_arr / norm).tolist()
                
                # Final validation after normalization
                if is_valid_embedding(emb_norm):
                    embeddings.append(emb_norm)
                else:
                    logger.warning(f"[Doc {doc_id}] Batch item {i}: Normalized embedding failed validation")
                    embeddings.append(None)
            else:
                logger.warning(f"[Doc {doc_id}] Batch item {i}: Embedding norm too small ({norm}), likely all zeros")
                embeddings.append(None)
                
        except Exception as e:
            logger.error(f"[Doc {doc_id}] Batch item {i}: Error generating embedding: {e}")
            # Log the problematic text (truncated)
            logger.error(f"[Doc {doc_id}] Problematic text: '{text[:100]}...'")
            embeddings.append(None)
            
    return embeddings

def generate_embeddings_parallel(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings in parallel batches using ThreadPoolExecutor.
    Returns a list aligned with `texts` where each item is either:
      - a normalized embedding list (List[float]) or
      - None (if skipped/failed)
    """
    if not texts:
        return []
    
    batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
    
    # Use a threadpool to call the synchronous process_embedding_batch
    with ThreadPoolExecutor(max_workers=min(4, len(batches) or 1)) as executor:
        future_to_idx = {}
        for idx, batch in enumerate(batches):
            future = executor.submit(process_embedding_batch, batch, f"batch_{idx}")
            future_to_idx[future] = idx
        
        for future in as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            try:
                batch_embeddings = future.result()
            except Exception as exc:
                logger.error(f"Embedding batch {batch_idx} error: {exc}")
                batch_embeddings = [None] * len(batches[batch_idx])
            # Put batch_embeddings into the right slice of all_embeddings
            start = batch_idx * EMBEDDING_BATCH_SIZE
            for offset, emb in enumerate(batch_embeddings):
                if start + offset < len(all_embeddings):
                    all_embeddings[start + offset] = emb
    
    return all_embeddings

def process_tokenization_batch(texts_batch: List[str]) -> List[List[str]]:
    return [tokenize_text(text) for text in texts_batch]

def tokenize_texts_parallel(texts: List[str]) -> List[List[str]]:
    if not texts:
        return []
    
    batches = [texts[i:i + CHUNK_BATCH_SIZE] for i in range(0, len(texts), CHUNK_BATCH_SIZE)]
    all_tokenized: List[List[str]] = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(process_tokenization_batch, batch): idx for idx, batch in enumerate(batches)}
        batch_results: List[List[List[str]] | None] = [None] * len(batches)
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_tokenized = future.result()
                batch_results[batch_idx] = batch_tokenized
            except Exception as exc:
                logger.error(f'Tokenization batch {batch_idx} error: {exc}')
                batch_results[batch_idx] = [[] for _ in batches[batch_idx]]
        
        for batch_tokenized in batch_results:
            if batch_tokenized:
                all_tokenized.extend(batch_tokenized)
    
    return all_tokenized

async def add_chunks_to_hybrid_index(
    hybrid_index: HybridRAGIndex,
    chunks: List[Dict[str, Any]],
    document_path: str = None
):
    """Add chunks to hybrid index with parallel multi-level chunking and upsert to Pinecone."""
    total_start_time = time.time()
    logger.info(f"Starting hybrid index update for {len(chunks)} chunks")
    
    if not chunks:
        logger.warning("No chunks to add to index")
        return hybrid_index

    document_id = chunks[0].get('document_id')

    # Remove existing vectors for this document from Pinecone and local caches
    remove_start = time.time()
    try:
        remove_document_from_hybrid_index(hybrid_index, document_id)
    except Exception as e:
        logger.error(f"Error removing existing document vectors: {e}")
    logger.info(f"Document removal completed in {time.time() - remove_start:.2f}s")

    # Multi-level chunking
    chunking_start = time.time()
    short_chunks, long_chunks = create_multi_level_chunks_parallel(chunks)
    all_chunks = short_chunks + long_chunks
    logger.info(f"Multi-level chunking completed in {time.time() - chunking_start:.2f}s")

    # Store document chunk structures locally (we do this early so we can reference them)
    storage_start = time.time()
    hybrid_index.document_chunks[document_id] = {
        'original': chunks,
        'short': short_chunks,
        'long': long_chunks
    }
    logger.info(f"Document chunks storage completed in {time.time() - storage_start:.2f}s")

    # Generate embeddings in parallel
    embedding_start = time.time()
    chunk_texts = [chunk.get('text', '') or '' for chunk in all_chunks]
    # generate_embeddings_parallel is synchronous (returns list), call it via asyncio.to_thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    embeddings: List[Optional[List[float]]] = await loop.run_in_executor(None, generate_embeddings_parallel, chunk_texts)
    
    # Log embedding statistics
    valid_embeddings = sum(1 for emb in embeddings if emb is not None)
    logger.info(f"Embedding generation completed in {time.time() - embedding_start:.2f}s")
    logger.info(f"Generated {valid_embeddings}/{len(embeddings)} valid embeddings")

    # Prepare vectors for Pinecone - filter out None (skipped/failed) embeddings
    upsert_start = time.time()
    vectors = []
    successful_chunk_ids: List[str] = []  # track chunks that will be upserted successfully (by id)
    skipped_count = 0
    
    for emb, chunk in zip(embeddings, all_chunks):
        if emb is None:
            # Skip: either empty text or embedding generation failed or embedding was all zeros
            skipped_count += 1
            chunk_id = chunk.get('chunk_id', 'unknown')
            chunk_text_preview = (chunk.get('text', '') or '')[:100]
            logger.warning(f"Skipping chunk {chunk_id} due to missing/invalid embedding. Text: '{chunk_text_preview}...'")
            continue
            
        # Double-check the embedding is valid before adding to Pinecone
        if not is_valid_embedding(emb):
            skipped_count += 1
            chunk_id = chunk.get('chunk_id', 'unknown')
            logger.warning(f"Skipping chunk {chunk_id} - final embedding validation failed")
            continue

        vid = str(chunk.get('chunk_id'))
        meta = {
            "document_id": chunk.get("document_id"),
            "document_name": chunk.get("document_name"),
            "file_path": chunk.get("file_path"),
            "title": chunk.get("title"),
            "page_number": chunk.get("page_number"),
            "type": chunk.get("type"),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type", "original"),
            "parent_id": chunk.get("parent_id"),
            "section_index": chunk.get("section_index"),
            "short_text": (chunk.get("text") or "")[:300],
        }
        vectors.append({"id": vid, "values": emb, "metadata": meta})
        successful_chunk_ids.append(vid)

    logger.info(f"Prepared {len(vectors)} vectors for Pinecone upsert (skipped {skipped_count})")

    # Upsert to Pinecone in batches with namespace
    if hybrid_index.pinecone and vectors:  # Only upsert if we have valid vectors
        try:
            batch = 100
            for i in range(0, len(vectors), batch):
                batch_vectors = vectors[i:i+batch]
                # Final validation of batch before upsert
                valid_batch = []
                for vec in batch_vectors:
                    if is_valid_embedding(vec["values"]):
                        valid_batch.append(vec)
                    else:
                        logger.error(f"Invalid vector detected in batch: {vec['id']}")
                
                if valid_batch:
                    hybrid_index.pinecone.upsert(
                        vectors=valid_batch,
                        namespace=PINECONE_NAMESPACE
                    )
                    logger.info(f"Upserted batch {i//batch + 1}: {len(valid_batch)} vectors")
        except Exception as e:
            logger.error(f"Pinecone upsert error: {e}")
            # On failure, we should avoid marking local metadata for unsuccessful chunks.
            # We'll continue and only update local caches for chunks that were part of the upsert request.
    else:
        if not hybrid_index.pinecone:
            logger.error("Pinecone index is not initialized. Skipping upsert.")
        if not vectors:
            logger.warning("No valid vectors to upsert to Pinecone.")

    logger.info(f"Pinecone upsert completed in {time.time() - upsert_start:.2f}s")

    # Cache successful chunk metadata locally for reconstructing text later
    metadata_start = time.time()
    # Only store metadata for chunks that had valid embeddings and were attempted to be upserted
    for chunk in all_chunks:
        cid = chunk.get('chunk_id')
        if cid and str(cid) in successful_chunk_ids:
            hybrid_index.metadata[cid] = chunk
    logger.info(f"Metadata update completed in {time.time() - metadata_start:.2f}s")

    # BM25 corpus and index - only include tokenized texts for successfully upserted chunks
    bm25_start = time.time()
    texts_for_bm25 = []
    for chunk in all_chunks:
        cid = chunk.get('chunk_id')
        if cid and str(cid) in successful_chunk_ids:
            text = chunk.get('text', '') or ''
            if text.strip():  # Only include non-empty texts
                texts_for_bm25.append(text)
    
    if texts_for_bm25:
        tokenized_chunks = tokenize_texts_parallel(texts_for_bm25)
        hybrid_index.bm25_corpus.extend(tokenized_chunks)
        try:
            hybrid_index.sparse_index = BM25Okapi(hybrid_index.bm25_corpus) if hybrid_index.bm25_corpus else None
        except Exception as e:
            logger.error(f"BM25 index build error: {e}")
            hybrid_index.sparse_index = None
    
    logger.info(f"BM25 processing completed in {time.time() - bm25_start:.2f}s")

    # Parent-child mapping
    mapping_start = time.time()
    for chunk in short_chunks:
        pid = chunk.get('parent_id')
        cid = chunk.get('chunk_id')
        if cid and str(cid) in successful_chunk_ids:
            hybrid_index.chunk_to_parent[cid] = pid
    logger.info(f"Parent-child mapping completed in {time.time() - mapping_start:.2f}s")

    # Save local state
    save_start = time.time()
    try:
        save_hybrid_index(hybrid_index)
    except Exception as e:
        logger.error(f"Error saving hybrid index to disk: {e}")
    logger.info(f"Index saving completed in {time.time() - save_start:.2f}s")

    total_time = time.time() - total_start_time
    logger.info(f"""
=== HYBRID INDEX UPDATE SUMMARY ===
Total base chunks: {len(chunks)} → {len(all_chunks)} generated chunks ({len(short_chunks)} short + {len(long_chunks)} long)
Successfully upserted chunks: {len(successful_chunk_ids)}
Skipped chunks: {skipped_count}
Document ID: {document_id}
TOTAL TIME: {total_time:.2f}s
=====================================
""")
    return hybrid_index

def split_into_short_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Split text into short, precise chunks"""
    if not text or not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[str] = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def split_into_long_chunks(text: str, max_chars: int = 1000) -> List[str]:
    """Split text into longer chunks for context"""
    if not text or not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks: List[str] = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for BM25"""
    if not text:
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [token for token in text.split() if token.strip()]
    return tokens

def hybrid_search(hybrid_index: HybridRAGIndex, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
    """Perform hybrid search combining dense and sparse retrieval"""
    search_start = time.time()
    logger.info(f"Starting hybrid search for query: '{query[:50]}...' in document {document_id}")

    # Dense retrieval (semantic similarity)
    dense_start = time.time()
    dense_results = dense_search(hybrid_index, query, document_id, top_k * 2)
    dense_time = time.time() - dense_start

    # Sparse retrieval (keyword matching)
    sparse_start = time.time()
    sparse_results = sparse_search(hybrid_index, query, document_id, top_k * 2)
    sparse_time = time.time() - sparse_start

    # Combine and rerank results
    rerank_start = time.time()
    combined_results = combine_and_rerank(dense_results, sparse_results, query, top_k)
    rerank_time = time.time() - rerank_start

    # Expand short chunks to include their long context
    expand_start = time.time()
    expanded_results = expand_with_context(hybrid_index, combined_results)
    expand_time = time.time() - expand_start

    total_search_time = time.time() - search_start
    logger.info(f"""
=== HYBRID SEARCH SUMMARY ===
Query: '{query[:50]}...'
Results: {len(expanded_results)} chunks
Total: {total_search_time:.3f}s
============================
""")
    return expanded_results

def dense_search(hybrid_index: HybridRAGIndex, query: str, document_id: str, top_k: int) -> List[Dict]:
    """Dense semantic search using Pinecone with namespace"""
    # generate synchronous embedding for the query
    query_embeddings = generate_embeddings([query])
    if not query_embeddings or not is_valid_embedding(query_embeddings[0]):
        logger.warning("Query embedding generation failed or invalid")
        return []
        
    # query_embeddings is a list of embedding lists — take first
    q_vec = np.array(query_embeddings[0], dtype=float)
    norm = np.linalg.norm(q_vec)
    if norm > 1e-6:
        q_vec = (q_vec / norm).tolist()
    else:
        logger.warning("Query embedding has zero norm")
        return []

    res = hybrid_index.pinecone.query(
        vector=q_vec,
        top_k=top_k * 3,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
        filter={"document_id": {"$eq": document_id}}
    )

    results: List[Dict] = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {}) or {}
        cid = meta.get("chunk_id")
        # Rehydrate full text from local cache
        chunk_local = hybrid_index.metadata.get(cid, {}) if cid else {}
        chunk_data = {
            "text": chunk_local.get("text", meta.get("short_text", "")),
            "chunk_id": cid,
            "document_id": meta.get("document_id"),
            "file_path": meta.get("file_path"),
            "title": meta.get("title"),
            "page_number": meta.get("page_number"),
            "type": meta.get("type"),
            "chunk_type": meta.get("chunk_type", "original"),
            "parent_id": meta.get("parent_id"),
        }
        results.append({
            "score": float(match.get("score", 0.0)),
            "chunk_data": chunk_data,
            "retrieval_type": "dense"
        })
        if len(results) >= top_k:
            break
    return results

def sparse_search(hybrid_index: HybridRAGIndex, query: str, document_id: str, top_k: int) -> List[Dict]:
    """Sparse keyword search using BM25"""
    if not hybrid_index.sparse_index:
        return []
    
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []
    bm25_scores = hybrid_index.sparse_index.get_scores(query_tokens)
    
    # Map bm25 index positions back to chunk ids
    chunk_ids_in_order = list(hybrid_index.metadata.keys())
    if len(bm25_scores) == 0:
        return []
    top_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]
    
    results: List[Dict] = []
    for idx in top_indices:
        if idx < len(chunk_ids_in_order):
            cid = chunk_ids_in_order[idx]
            chunk_data = hybrid_index.metadata.get(cid)
            if not chunk_data:
                continue
            if chunk_data.get('document_id') == document_id:
                results.append({
                    'score': float(bm25_scores[idx]),
                    'chunk_data': chunk_data,
                    'retrieval_type': 'sparse'
                })
            if len(results) >= top_k:
                break
    return results

def combine_and_rerank(dense_results: List[Dict], sparse_results: List[Dict], query: str, top_k: int) -> List[Dict]:
    """Combine dense and sparse results with reciprocal rank fusion"""
    # Normalize scores
    for result in dense_results:
        result['dense_score'] = result['score']
    for result in sparse_results:
        result['sparse_score'] = result['score']

    # Combine results by chunk_id
    combined: Dict[str, Dict] = {}
    
    # Add dense results
    for i, result in enumerate(dense_results):
        cid = result['chunk_data']['chunk_id']
        combined[cid] = result
        combined[cid]['dense_rank'] = i + 1
        combined[cid]['sparse_rank'] = len(sparse_results) + 1

    # Add sparse results
    for i, result in enumerate(sparse_results):
        cid = result['chunk_data']['chunk_id']
        if cid in combined:
            combined[cid]['sparse_rank'] = i + 1
            combined[cid]['sparse_score'] = result['score']
        else:
            combined[cid] = result
            combined[cid]['dense_rank'] = len(dense_results) + 1
            combined[cid]['sparse_rank'] = i + 1

    # Calculate reciprocal rank fusion score
    k = 60
    for cid, result in combined.items():
        dense_rank = result.get('dense_rank', 100)
        sparse_rank = result.get('sparse_rank', 100)
        result['rrf_score'] = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))

    # Sort by RRF score and return top_k
    sorted_results = sorted(combined.values(), key=lambda x: x['rrf_score'], reverse=True)
    return sorted_results[:top_k]

def expand_with_context(hybrid_index: HybridRAGIndex, results: List[Dict]) -> List[Dict]:
    """Expand short chunks with their corresponding long context"""
    expanded_results: List[Dict] = []
    
    for result in results:
        chunk_data = result['chunk_data']
        chunk_type = chunk_data.get('chunk_type', 'original')
        
        if chunk_type == 'short':
            parent_id = chunk_data.get('parent_id')
            document_id = chunk_data.get('document_id')
            
            # Look for long chunk with same parent
            long_context = None
            if document_id in hybrid_index.document_chunks:
                for long_chunk in hybrid_index.document_chunks[document_id].get('long', []):
                    if long_chunk.get('parent_id') == parent_id:
                        long_context = long_chunk.get('text')
                        break
            
            result['context'] = long_context or chunk_data.get('text', '')
            result['short_text'] = chunk_data.get('text', '')
        else:
            result['context'] = chunk_data.get('text', '')
            result['short_text'] = chunk_data.get('text', '')
        
        expanded_results.append(result)
    
    return expanded_results

def remove_document_from_hybrid_index(hybrid_index: HybridRAGIndex, document_id: str):
    """Remove a document's vectors from Pinecone and clean local caches."""
    # Delete from Pinecone with namespace
    try:
        hybrid_index.pinecone.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=PINECONE_NAMESPACE  # Add namespace parameter
        )
        logger.info(f"Deleted Pinecone vectors for document {document_id} in namespace {PINECONE_NAMESPACE}")
    except Exception as e:
        logger.error(f"Pinecone delete error: {e}")

    # Clean local caches
    to_del = [cid for cid, cdata in hybrid_index.metadata.items() if cdata.get('document_id') == document_id]
    for cid in to_del:
        try:
            del hybrid_index.metadata[cid]
        except KeyError:
            pass

    if document_id in hybrid_index.document_chunks:
        try:
            del hybrid_index.document_chunks[document_id]
        except KeyError:
            pass

    # Rebuild BM25 from remaining metadata
    try:
        texts = [c['text'] for _, c in hybrid_index.metadata.items() if c.get('text') and c['text'].strip()]
        tokenized = tokenize_texts_parallel(texts) if texts else []
        hybrid_index.bm25_corpus = tokenized
        hybrid_index.sparse_index = BM25Okapi(hybrid_index.bm25_corpus) if tokenized else None
    except Exception as e:
        logger.error(f"BM25 rebuild error after deletion: {e}")

def save_hybrid_index(hybrid_index: HybridRAGIndex):
    """Save local metadata and BM25 to disk."""
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadata': hybrid_index.metadata,
            'document_chunks': hybrid_index.document_chunks,
            'bm25_corpus': hybrid_index.bm25_corpus,
            'chunk_to_parent': hybrid_index.chunk_to_parent
        }, f)

    if hybrid_index.sparse_index:
        bm25_path = os.path.join(VECTOR_STORE_DIR, BM25_FILE)
        with open(bm25_path, 'wb') as f:
            pickle.dump(hybrid_index.sparse_index, f)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Synchronous embedding (used in dense_search query embedding)"""
    embeddings: List[List[float]] = []
    for text in texts:
        if not text or not text.strip():
            logger.warning("Skipping empty text for embedding generation")
            embeddings.append([0.0] * EMBED_DIM)  # Return zero vector for alignment
            continue
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text.strip(),
                task_type="retrieval_document"
            )
            emb = result.get('embedding')
            if not emb:
                raise ValueError("Received empty embedding")
                
            if not is_valid_embedding(emb):
                logger.warning(f"Generated invalid embedding for query text: '{text[:50]}'")
                embeddings.append([0.0] * EMBED_DIM)
                continue
                
            emb_arr = np.array(emb, dtype=float)
            norm = np.linalg.norm(emb_arr)
            if norm > 1e-6:
                emb_norm = (emb_arr / norm).tolist()
            else:
                # If query embedding is all zeros, keep it as-is (rare) but return as floats
                logger.warning(f"Query embedding has zero norm: '{text[:50]}'")
                emb_norm = [0.0] * EMBED_DIM
                
            embeddings.append(emb_norm)
        except Exception as e:
            logger.error(f"Error generating synchronous embedding for text: {e}")
            # Fallback: append a zero-vector for alignment
            embeddings.append([0.0] * EMBED_DIM)
    return embeddings

# Legacy wrappers for compatibility
def create_or_load_index():
    """Legacy function - returns HybridRAGIndex and empty metadata"""
    hybrid_index = HybridRAGIndex()
    return hybrid_index.create_or_load_index(), {}

def add_chunks_to_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    """Legacy function - calls async add on HybridRAGIndex"""
    if isinstance(index, HybridRAGIndex):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_hybrid_index(index, chunks, document_path))
    else:
        hybrid_index = HybridRAGIndex()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_hybrid_index(hybrid_index, chunks, document_path))

def search_document(index, metadata, query: str, document_id: str, top_k: int = 5):
    """Legacy function - uses hybrid search when index is HybridRAGIndex"""
    if isinstance(index, HybridRAGIndex):
        return hybrid_search(index, query, document_id, top_k)
    else:
        hybrid_index = HybridRAGIndex()
        hybrid_index.metadata = metadata
        return dense_search(hybrid_index, query, document_id, top_k)
