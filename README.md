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

## Architecture

- Frontend: Streamlit chat interface
- Backend: FastAPI
- Vector Store: Chroma (persistent on disk)
- LLM: OpenAI chat model
- Embeddings: text-embedding-3-small
- Reranker: cross-encoder ms-marco-MiniLM-L-6-v2

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
git clone https://github.com/your-username/your-repo-name.git
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
