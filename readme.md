# LLM-Powered Document Query-Answering System

A comprehensive FastAPI-based service for querying PDF and DOCX documents using advanced AI technologies including LlamaIndex, Mistral OPCR chunking, Gemini embeddings, and local FAISS vector search.

## 🔧 Tech Stack

- **Document Parsing**: LlamaParse for structure-aware PDF/DOCX parsing
- **Chunking**: Mistral OPCR-inspired chunking with tabular data awareness
- **Embeddings**: Google Gemini `embedding-001` model
- **Vector Store**: FAISS (local storage)
- **Retrieval**: LlamaIndex with semantic similarity search
- **LLM Reasoning**: Mistral 7B-Instruct (local or API)
- **Backend**: FastAPI with structured JSON responses

## 🚀 Features

- **Multi-format Support**: PDF and DOCX document processing
- **Intelligent Chunking**: Separate handling for text and tabular data
- **Document Isolation**: Prevents cross-document contamination in retrieval
- **Structured Responses**: JSON output with answers, rationale, and source references
- **Local Vector Storage**: FAISS index with persistence
- **Batch Question Processing**: Handle multiple questions per document

## 📁 Project Structure

```
llm_docqa_project/
├── app/
│   ├── main.py           # FastAPI application and endpoints
│   ├── models.py         # Pydantic data models
│   ├── parse.py          # LlamaParse integration and OPCR chunking
│   ├── embed.py          # Gemini embeddings and FAISS management
│   ├── retrieve.py       # LlamaIndex retrieval and reranking
│   ├── reason.py         # Mistral-based answer generation
│   └── utils.py          # File handling utilities
├── vector_store/         # FAISS index and metadata storage
├── .env                  # API keys and configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔐 Environment Setup

Create a `.env` file with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MISTRAL_API_BASE=http://localhost:11434/v1
MISTRAL_MODEL_NAME=mistral-7b-instruct
LLAMA_CLOUD_API_KEY=your_llama_parse_api_key_here
```

## 🛠 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   - Get a Gemini API key from Google AI Studio
   - Get a LlamaParse API key from LlamaIndex
   - Set up local Mistral (using Ollama) or configure remote API

3. **Start the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the API documentation:**
   Open `http://localhost:8000/docs`

## 📝 API Usage

### POST `/run`

Process a document and answer questions about it.

**Request Body:**
```json
{
  "document_url": "https://example.com/document.pdf",
  "questions": [
    "What is the waiting period for cataract surgery?",
    "Does this policy cover knee surgery?"
  ]
}
```

**Response:**
```json
{
  "responses": [
    {
      "question": "What is the waiting period for cataract surgery?",
      "answer": "The waiting period for cataract surgery is 24 months from the policy start date.",
      "rationale": "According to Clause 4.2 on page 8, cataract surgery has a specific waiting period of 24 months as stated in the exclusions section.",
      "clauses_referenced": [
        {
          "title": "Waiting Periods",
          "page_number": 8,
          "text_snippet": "Cataract surgery requires a waiting period of 24 months from policy inception..."
        }
      ]
    }
  ],
  "document_processed": "policy_document.pdf"
}
```

## 🔍 Key Features Explained

### Intelligent Document Parsing
- Uses LlamaParse for structure-aware extraction
- Maintains formatting and relationships
- Handles complex layouts and tables

### Advanced Chunking Strategy
- **Text Sections**: OPCR-inspired chunking with semantic boundaries
- **Tabular Data**: Row-based or logical group chunking
- **Header Detection**: Automatic section identification
- **Size Optimization**: Configurable chunk sizes for optimal retrieval

### Document Isolation
- Each document gets a unique ID
- Retrieval is filtered by document ID
- Prevents cross-contamination between documents
- Maintains context integrity

### Structured Reasoning
- Mistral LLM provides detailed explanations
- References specific clauses and page numbers
- Confidence scoring for answers
- Fallback handling for edge cases

## 🧪 Testing

### Sample curl request:
```bash
curl -X 'POST' \
  'http://localhost:8000/run' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_url": "https://example.com/sample.pdf",
    "questions": ["What are the key benefits?"]
  }'
```

## 🔧 Configuration

### Chunk Size Tuning
Modify `max_chunk_length` in `parse.py` to adjust chunk sizes based on your documents.

### Retrieval Parameters
Adjust `top_k` in `retrieve.py` to change the number of chunks retrieved per query.

### LLM Parameters
Modify temperature and max_tokens in `reason.py` for different response styles.

## 🚀 Production Deployment

1. **Use a production ASGI server:**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Configure environment variables for production APIs**

3. **Set up persistent storage for FAISS indices**

4. **Implement proper error handling and logging**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Note**: Make sure to configure all API keys properly before running the application. The system requires active internet connection for Gemini embeddings and document parsing.
