import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from pathlib import Path
from models import DocumentRequest, RunResponse, HackRXRequest, HackRXResponse
import utils, parse, embed, retrieve, reason
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="LLM-Powered Document Query-Answering System with LlamaIndex")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "Empty body"
    except:
        body_str = "Could not decode body"
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {body_str}")
    logger.error(f"Content-Type: {request.headers.get('content-type')}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Invalid request format",
            "body": body_str,
            "content_type": request.headers.get('content-type')
        }
    )

security = HTTPBearer()
HACKRX_TOKEN = "3b3b7f8e0cb19ee38fcc3d4874a8df6dadcdbfec21b7bbe39a73407e2a7af8a0"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

@app.get("/")
def root():
    return {"message": "Welcome to the LLM-Powered Document Q&A API. Visit /docs for usage."}

@app.post("/run", response_model=RunResponse)
async def run_query(request: DocumentRequest):
    logger.info(f"Received request: documents={request.documents}, questions={len(request.questions)}")
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents field is required")
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")

        document_path = request.documents
        if os.path.exists(document_path):
            doc_path = document_path
            logger.info(f"Using local file: {doc_path}")
        elif document_path.startswith(('http://', 'https://')):
            doc_path = await utils.download_document(document_path)
            logger.info(f"Downloaded file to: {doc_path}")
        else:
            raise HTTPException(status_code=400, detail=f"File not found: {document_path}")

        logger.info("Starting document parsing...")
        parsed_doc = await parse.parse_document(doc_path)
        chunks = parse.chunk_document(parsed_doc, doc_path)

        logger.info("Creating hybrid embeddings and storing...")
        hybrid_index = embed.HybridRAGIndex().create_or_load_index()
        # IMPORTANT: await the async add
        await embed.add_chunks_to_hybrid_index(hybrid_index, chunks, doc_path)

        answers = []
        for i, question in enumerate(request.questions, 1):
            logger.info(f"Processing question {i}/{len(request.questions)}: {question[:50]}...")
            relevant_chunks = retrieve.retrieve_chunks(hybrid_index, question, doc_path)
            answer_data = await reason.generate_answer(question, relevant_chunks)
            answers.append(answer_data.get("answer", "Unable to determine answer"))

        logger.info(f"Successfully processed {len(answers)} questions using hybrid RAG")
        return RunResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    logger.info(f"Received HackRX request: documents={request.documents}, questions={len(request.questions)}")
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents field is required")
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")

        document_path = request.documents
        if os.path.exists(document_path):
            doc_path = document_path
        elif document_path.startswith(('http://', 'https://')):
            doc_path = await utils.download_document(document_path)
        else:
            raise HTTPException(status_code=400, detail=f"File not found: {document_path}")

        parsed_doc = await parse.parse_document(doc_path)
        chunks = parse.chunk_document(parsed_doc, doc_path)

        # Use HybridRAGIndex with Pinecone
        hybrid_index = embed.HybridRAGIndex().create_or_load_index()
        await embed.add_chunks_to_hybrid_index(hybrid_index, chunks, doc_path)

        answers = []
        for i, question in enumerate(request.questions):
            if i > 0:
                await asyncio.sleep(1)
            relevant_chunks = retrieve.retrieve_chunks(hybrid_index, question, doc_path)
            answer_data = await reason.generate_answer(question, relevant_chunks)
            answers.append(answer_data.get("answer", "Unable to determine answer"))

        return HackRXResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing HackRX request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document(document_path: str, question: str):
    """Helper for single document processing"""
    try:
        parsed_doc = await parse.parse_document(document_path)
        chunks = parse.chunk_document(parsed_doc, document_path)

        hybrid_index = embed.HybridRAGIndex().create_or_load_index()
        await embed.add_chunks_to_hybrid_index(hybrid_index, chunks, document_path)

        relevant_chunks = retrieve.retrieve_chunks(hybrid_index, question, document_path)
        answer_data = await reason.generate_answer(question, relevant_chunks)
        return answer_data
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
