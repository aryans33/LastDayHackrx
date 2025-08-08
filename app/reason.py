import os
import json
import asyncio
from typing import List, Dict, Any
import httpx
from dotenv import load_dotenv
from app.models import ClauseReference  # CHANGED: absolute import
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# GROQ Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-8b-8192"  # Working model from test

# Fallback for testing without GROQ key
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Using local Mistral as fallback.")
    GROQ_API_BASE = os.getenv("MISTRAL_API_BASE", "http://localhost:11434/v1")
    GROQ_MODEL = os.getenv("MISTRAL_MODEL_NAME", "mistral-7b-instruct")

async def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using Mistral LLM based on retrieved chunks.
    """
    logger.info(f"Generating answer for question: {question[:50]}...")
    logger.info(f"Using {len(chunks)} chunks as context")
    
    # Prepare context from chunks with detailed formatting
    context_parts = []
    clause_refs = []
    
    for i, chunk in enumerate(chunks):
        # Add to context
        context_parts.append(f"[Source {i+1} - Page {chunk.get('page_number', 'N/A')}]\n{chunk['text']}\n")
        
        # Create clause reference
        clause_refs.append(ClauseReference(
            title=chunk.get('title', f"Section {i+1}"),
            page_number=chunk.get('page_number', 1),
            text_snippet=chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        ))
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Based on the following document excerpts, answer the question comprehensively and accurately.

Document Context:
{context}

Question: {question}

Please provide:
1. A direct answer to the question
2. Your reasoning based on the document
3. Your confidence level (High/Medium/Low)

Answer:"""

    try:
        # Call LLM API
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authorization header if API key exists
        if GROQ_API_KEY:
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert insurance policy analyst. Provide accurate, detailed answers based strictly on the provided document excerpts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GROQ_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                # ...rest of your function...
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Handle the error as appropriate
