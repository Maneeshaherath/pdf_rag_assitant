# Production PDF RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system that lets users upload one or more PDF files and chat with them using an OpenAI model.

It combines:
- FastAPI backend for ingestion and query APIs
- Streamlit frontend with a modern chat experience
- Chroma persistent vector database for document memory
- Hybrid retrieval plus reranking for better answer quality

## What This Project Does

- Uploads and indexes PDF documents
- Splits documents into semantic chunks with metadata
- Stores embeddings in persistent Chroma DB
- Expands user queries with LLM rewriting
- Retrieves relevant chunks using:
  - Vector retrieval with MMR
  - Keyword retrieval with BM25
- Re-ranks retrieved chunks with a cross-encoder model
- Generates grounded answers from context only
- Streams answer tokens in UI
- Shows sources (file and page)
- Supports multi-document querying with document filters

## Why It Is Useful

- Reduces hallucination by enforcing context-grounded answers
- Improves retrieval quality using hybrid search plus reranking
- Works with local persistent indexing across restarts
- Provides both API and UI for flexible usage

## System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PDF RAG ASSISTANT SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────────┐
                            │   Streamlit UI      │
                            │   (app.py, ui.py)   │
                            │  - Chat Interface   │
                            │  - PDF Upload       │
                            │  - Source Display   │
                            └──────────┬──────────┘
                                       │
                                       │ HTTP
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      FastAPI Backend (api.py)        │
                    │  ┌─────────────────────────────────┤ |
                    │  │ ▪ POST /upload (PDF ingestion)  │ |
                    │  │ ▪ POST /query (stream answers)  │ |
                    │  │ ▪ GET /documents (list docs)    │ |
                    │  │ ▪ GET /health (status check)    │ |
                    │  └─────────────────────────────────┤ |
                    └──┬───────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
   │Ingestion│  │  Retriever   │  │Reranker  │  │   LLM    │
   │(PDF→    │  │  (Hybrid     │  │(Cross-   │  │(Answer   │
   │ Chunks) │  │   Search)    │  │encoder)  │  │Generate) │
   └────┬────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘
        │              │               │              │
        │     ┌────────┤               │              │
        │     │        │               │              │
        └─────┼────┬───┼───────────────┼──────────────┘
              │    │   │               │
              ▼    ▼   ▼               ▼
        ┌──────────────────────────────────────────────┐
        │        Embedding Models & LLM Services       │
        │  ┌──────────────────────────────────────────┤|
        │  │ • OpenAI Embeddings (text-embedding-3...)|│
        │  │ • OpenAI Chat Model (gpt-4o-mini)        │|
        │  │ • BM25 Keyword Retrieval                 |│
        │  │ • HuggingFace Reranker                   |│
        │  └──────────────────────────────────────────┤|
        └──────────┬───────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   ┌──────────────┐   ┌─────────────────┐
   │ Chroma DB    │   │  OpenAI API     │
   │(Vector Store)│   │  (External LLM) │
   │  - Index     │   │  - Embeddings   │
   │  - Persist   │   │  - Chat Model   │
   └──────────────┘   └─────────────────┘
```

### Architecture Components Explained

#### **1. Streamlit Frontend (UI Layer)**
- **Files**: `app.py`, `ui.py`
- **Purpose**: Provides an interactive chat interface for end users
- **Features**:
  - Real-time PDF document upload with progress tracking
  - Persistent chat history within session
  - Document filtering for targeted queries
  - Source attribution (file name + page number)
  - Token streaming for responsive answers
- **How it works**: Communicates with FastAPI backend via HTTP requests, displays LLM responses in real-time

#### **2. FastAPI Backend (API Layer)**
- **File**: `api.py`
- **Purpose**: Handles all business logic and orchestrates the RAG pipeline
- **Key Endpoints**:
  - `POST /upload`: Accepts PDF files and triggers ingestion pipeline
  - `POST /query`: Accepts user queries and returns generated answers
  - `GET /documents`: Lists all indexed documents in Chroma DB
  - `GET /health`: System health check
- **Responsibilities**: 
  - CORS configuration for frontend access
  - Request validation using Pydantic schemas
  - Error handling and logging
  - Pipeline orchestration

#### **3. Document Ingestion Pipeline**
- **File**: `ingestion.py`
- **Process Flow**:
  1. **Load**: Extract text from PDF using PyPDFLoader
  2. **Chunk**: Split documents using RecursiveCharacterTextSplitter
     - Configurable chunk size (default: 800 chars)
     - Overlap for context preservation (default: 100 chars)
  3. **Embed**: Generate embeddings using OpenAI's text-embedding-3-small
  4. **Store**: Persist embeddings and metadata in Chroma DB
  5. **Deduplicate**: Skip re-ingestion of already processed documents using SHA256 hash
- **Output**: Indexed document chunks with metadata (source, page, content)

#### **4. Retrieval System (Hybrid Search)**
- **File**: `retriever.py`
- **Retrieval Strategy** (from `retriever.py`):
  1. **Query Expansion**: LLM generates 3 query rewrites to improve coverage
  2. **Hybrid Retrieval**:
     - **Vector Search**: MMR (Maximum Marginal Relevance) with k=8, fetch_k=25
       - Balances relevance with diversity
       - Uses semantic similarity from embeddings
     - **Keyword Search**: BM25 algorithm with k=8
       - Catches exact term matches that embeddings might miss
     - **Fusion**: Combines both result sets for comprehensive coverage
  3. **Document Filtering**: Optional filter by source documents
- **Why Hybrid?**: Vector search excels at semantic matching, BM25 at exact terms. Combined = best of both worlds

#### **5. Reranking Module**
- **File**: `reranker.py`
- **Purpose**: Re-rank retrieved chunks by relevance to create higher-quality context
- **Model**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
  - Scores all retrieved chunks against the query
  - Selects top-4 chunks (configurable)
- **Benefit**: Filters out low-relevance results before going to LLM, improving answer quality and reducing token usage

#### **6. LLM & Answer Generation**
- **File**: `llm.py`
- **Components**:
  - **Chat Model**: OpenAI's gpt-4o-mini (configurable)
  - **System Prompt**: Strict prompt enforcing context-only answers
  - **Anti-Hallucination**: Returns "I don't know" if context is insufficient
  - **Streaming**: Token-by-token generation for responsive UI
- **Process**:
  1. Format retrieved chunks as context
  2. Inject user query into prompt
  3. Generate answer respecting context boundaries
  4. Stream tokens to UI for live feedback

#### **7. Vector Store (Chroma DB)**
- **Location**: `./chroma_db/` (persistent on disk)
- **Purpose**: Centralized storage for:
  - Document embeddings (vector representations)
  - Document chunks (raw text)
  - Metadata (file names, page numbers, hash IDs)
- **Benefits**:
  - Persistent storage survives application restarts
  - Fast similarity search using vector indices
  - Collection-based organization

#### **8. Configuration Manager**
- **File**: `config.py`
- **Purpose**: Centralized settings management via environment variables
- **Key Settings**:
  - API credentials (OpenAI key, model names)
  - Search parameters (chunk size, k values, rerank threshold)
  - Upload limits (max file size, query length)
  - Service endpoints (host, port, base URL)

### Data Flow Through the System

```
User Input (PDF)
       │
       ▼
┌──────────────┐
│   Upload     │ → FastAPI /upload endpoint
│   Endpoint   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  Ingestion Pipeline      │
│  • Load PDF              │
│  • Split into chunks     │
│  • Generate embeddings   │
│  • Store in Chroma       │
└──────┬───────────────────┘
       │
       ▼
Chroma Vector DB (indexed & persistent)
       │
       │
User Query
       │
       ▼
┌──────────────────────────┐
│  Retrieval Pipeline      │
│  • Expand query (3 ways) │
│  • Hybrid search         │
│    - Vector (MMR)        │
│    - Keyword (BM25)      │
│  • Combine results       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────┐
│   Reranker   │ → Cross-encoder scores
│              │ → Selects top-4 chunks
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│   LLM Generation         │
│   • Format context       │
│   • Insert query         │
│   • Generate answer      │
│   • Stream tokens        │
└──────┬───────────────────┘
       │
       ▼
Streamlit UI (displays answer + sources)
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive chat UI |
| **Backend** | FastAPI | REST API & orchestration |
| **Vector DB** | Chroma | Document embeddings & storage |
| **Embeddings** | OpenAI (text-embedding-3-small) | Semantic representation |
| **LLM** | OpenAI (gpt-4o-mini) | Answer generation |
| **Reranking** | HuggingFace Cross-encoder | Relevance scoring |
| **Keyword Search** | BM25 (rank-bm25) | Exact term matching |
| **PDF Processing** | PyPDF | Document extraction |
| **Validation** | Pydantic | Schema validation |

### Key Design Principles

1. **Modularity**: Each component (ingestion, retrieval, reranking, generation) is independently testable and replaceable
2. **Hybrid Retrieval**: Combines semantic and keyword search for comprehensive results
3. **Anti-Hallucination**: Strict prompting ensures LLM only uses provided context
4. **Persistence**: Chroma DB stores embeddings permanently, avoiding re-processing
5. **Streaming**: Token streaming provides real-time user feedback
6. **Configurability**: All parameters are environment-driven for easy deployment flexibility

## Project Structure

- app.py: Streamlit launcher
- ui.py: Streamlit chat UI
- api.py: FastAPI server and endpoints
- config.py: central settings
- schemas.py: request and response schemas
- ingestion.py: PDF load, chunk, metadata, dedup
- retriever.py: query expansion plus hybrid retrieval
- reranker.py: cross-encoder reranking
- llm.py: strict prompting and streaming generation

## Features

- MMR retriever with configurable k and fetch_k
- LLM query expansion (multiple rewrites)
- Hybrid retrieval (vector plus BM25)
- Cross-encoder reranking for top chunks
- Strict anti-hallucination prompt with fallback: I don't know
- Token streaming in UI
- Conversation memory in session
- File-size and query-length validation
- Structured logging and error handling
- Source display with file and page metadata

## Requirements

- Python 3.10 or newer
- OpenAI API key
- Windows, Linux, or macOS

## Clone and Run

1. Clone the repository

```bash
git clone https://github.com/Maneeshaherath/pdf_rag_assitant
cd your-repo-name
```

2. Create and activate virtual environment

Windows CMD:

```bash
python -m venv venv
venv\Scripts\activate
```

PowerShell:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure API key

Option A: environment variable (recommended for backend)

Windows CMD:

```bash
set OPENAI_API_KEY=YOUR_NEW_KEY
```

PowerShell:

```bash
$env:OPENAI_API_KEY="YOUR_NEW_KEY"
```

Option B: Streamlit secrets (UI)

Create `.streamlit/secrets.toml` with:

```toml
OPENAI_API_KEY = "YOUR_NEW_KEY"
```

5. Start backend

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

6. Start frontend in another terminal

```bash
streamlit run app.py
```

7. Open the UI

- Streamlit usually runs at http://localhost:8501
- Upload PDFs from sidebar
- Ask questions in the chat box at bottom

## API Endpoints

- GET /health
- GET /documents
- POST /upload
- POST /query
- POST /query/stream

## Configuration

All main settings are centralized in `config.py`, including:

- model names
- chunk size and overlap
- retrieval and reranking limits
- max query length and max upload size
- API host and port
- Chroma path

## Security Notes

- Never commit API keys to git
- Keep `.streamlit/secrets.toml` in .gitignore
- Rotate keys immediately if exposed
- Prefer environment variables or secret managers in deployment

## Common Troubleshooting

1. 401 Unauthorized from OpenAI
- Wrong key or stale terminal env variable
- Re-set OPENAI_API_KEY in the same terminal where Uvicorn starts

2. Long first response time
- First reranker load downloads model weights once
- Later queries are much faster due to cache

3. Empty document filter
- Ensure upload succeeded
- Check GET /documents endpoint

4. Streamlit noisy transformer warnings
- Optional workaround: disable watcher in terminal before launch

```bash
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
streamlit run app.py
```

## Deployment Notes

For production deployment:

- Run FastAPI behind a reverse proxy
- Use managed secrets
- Add authentication and rate limiting
- Add monitoring and structured logs
- Pin dependency versions for reproducibility

## License

This project is licensed under the MIT License. See the LICENSE file for details.
