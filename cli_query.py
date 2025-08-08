#!/usr/bin/env python3
"""
CLI script for testing the document Q&A system.
Usage: python cli_query.py --file path/to/doc.pdf --question "What's covered?"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app import parse, embed, retrieve, reason, utils
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_document_query(file_path: str, questions: list) -> dict:
    """
    Process a document and answer questions about it.
    """
    logger.info(f"Processing document: {file_path}")
    
    try:
        # 1. Parse and chunk document
        logger.info("Step 1: Parsing document...")
        parsed_doc = await parse.parse_document(file_path)
        
        logger.info("Step 2: Chunking document...")
        chunks = parse.chunk_document(parsed_doc)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # 3. Create embeddings and store in FAISS
        logger.info("Step 3: Creating/loading FAISS index...")
        index = embed.create_or_load_index()
        
        logger.info("Step 4: Adding chunks to index...")
        embed.add_chunks_to_index(index, chunks, file_path)
        
        # 4. Process each question
        responses = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Step 5.{i}: Processing question: {question}")
            
            # Retrieve relevant chunks
            logger.info("  - Retrieving relevant chunks...")
            relevant_chunks = retrieve.retrieve_chunks(index, question, file_path)
            logger.info(f"  - Retrieved {len(relevant_chunks)} chunks")
            
            # Generate answer using Mistral
            logger.info("  - Generating answer with Mistral...")
            answer_data = await reason.generate_answer(question, relevant_chunks)
            
            responses.append({
                "question": question,
                **answer_data
            })
        
        result = {
            "document_processed": os.path.basename(file_path),
            "chunk_count": len(chunks),
            "responses": responses
        }
        
        logger.info("Processing completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "error": str(e),
            "document_processed": os.path.basename(file_path),
            "responses": []
        }

def print_results(results: dict):
    """
    Pretty print the results.
    """
    print("\n" + "="*80)
    print(f"📄 DOCUMENT: {results.get('document_processed', 'Unknown')}")
    
    if 'error' in results:
        print(f"❌ ERROR: {results['error']}")
        return
    
    if 'chunk_count' in results:
        print(f"📝 CHUNKS GENERATED: {results['chunk_count']}")
    
    print("="*80)
    
    for i, response in enumerate(results.get('responses', []), 1):
        print(f"\n🤔 QUESTION {i}: {response['question']}")
        print(f"💡 ANSWER: {response['answer']}")
        print(f"🧠 RATIONALE: {response['rationale']}")
        
        if 'confidence' in response:
            print(f"📊 CONFIDENCE: {response['confidence']}")
        
        clauses = response.get('clauses_referenced', [])
        if clauses:
            print(f"📚 CLAUSES REFERENCED ({len(clauses)}):")
            for j, clause in enumerate(clauses, 1):
                print(f"   {j}. {clause['title']} (Page {clause['page_number']})")
                print(f"      \"{clause['text_snippet'][:100]}...\"")
        
        print("-" * 80)

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Query documents using LLM-powered Q&A system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_query.py --file sample.pdf --question "What is the coverage?"
  python cli_query.py --file policy.docx --question "What are exclusions?" --question "What is the premium?"
  python cli_query.py --file doc.pdf --questions-file questions.txt
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to the document file (PDF or DOCX)'
    )
    
    parser.add_argument(
        '--question', '-q',
        action='append',
        help='Question to ask about the document (can be used multiple times)'
    )
    
    parser.add_argument(
        '--questions-file',
        help='File containing questions (one per line)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file to save results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found.")
        sys.exit(1)
    
    # Collect questions
    questions = []
    
    if args.question:
        questions.extend(args.question)
    
    if args.questions_file:
        if os.path.exists(args.questions_file):
            with open(args.questions_file, 'r') as f:
                file_questions = [line.strip() for line in f if line.strip()]
                questions.extend(file_questions)
        else:
            print(f"❌ Error: Questions file '{args.questions_file}' not found.")
            sys.exit(1)
    
    if not questions:
        print("❌ Error: No questions provided. Use --question or --questions-file.")
        sys.exit(1)
    
    print(f"🚀 Starting document analysis...")
    print(f"📄 Document: {args.file}")
    print(f"❓ Questions: {len(questions)}")
    
    # Process the document
    results = await process_document_query(args.file, questions)
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
