# DocChat AI — RAG based Document Question-Answering System .

DocChat AI is an enterprise-grade RAG (Retrieval-Augmented Generation) document intelligence assistant. It enables users to upload PDF documents, automatically parse and structure their contents (including scanned pages via Gemini Vision OCR), index them using hybrid vector-and-keyword search, and engage in context-grounded conversational chat.

All generative answers are backed by precise, page-stamped source citations with real-time response latency metrics.

---
<!--
## 🖥️ User Interface Preview

### 1. Welcome Screen (No Document Loaded)
<img src="assets/welcome_screen.png" width="100%" alt="DocChat AI Welcome Screen" />

### 2. Chat & Source Citations
<img src="assets/chat_screen_1.png" width="100%" alt="DocChat AI Chat & Citations" />

### 3. Continuous Conversational Flow
<img src="assets/chat_screen_2.png" width="100%" alt="DocChat AI Conversational Flow" />

---
-->

## 🚀 Architectural Highlights

### ⚡ Backend Architecture (FastAPI + PGVector + Hybrid RAG)
- **FastAPI Framework**: High-performance, asynchronous REST API with thin route handlers and threadpool execution for CPU/IO-heavy operations.
- **Structure-Aware Chunking (`chunking.py`)**: Hierarchical section tree building using PyMuPDF. Removes running headers/footers, preserves tables as row/column grids, and prepends section breadcrumbs (`Section: A > B | Page 4`) to eliminate context fragmentation.
- **Vision OCR Fallback (`ocr.py`)**: Automatic fallback to Gemini 2.0 Flash-Lite (`google-genai` SDK) for textless/scanned pages and chart/figure data extraction, guarded by a PostgreSQL-backed daily request budget manager.
- **Dual Database Storage**:
  - **PGVector (`docchat_chunks` collection)**: Vector embeddings generated via `gemini-embedding-001` for dense semantic similarity search.
  - **PostgreSQL (`session_chunks` & `sessions` tables)**: Stores full chunk text, page bounds, and section metadata for BM25 keyword search and context expansion via `psycopg` (v3).
- **Hybrid Search & Reciprocal Rank Fusion (`retrieval.py` & `bm25_index.py`)**: Merges dense vector hits with in-memory BM25 keyword scores using Reciprocal Rank Fusion (RRF) to excel at both semantic queries and exact term/code lookups.
- **Cross-Encoder Reranking (`reranker.py`)**: Optional Cohere Cross-Encoder (`rerank-english-v3.0`) re-scoring to optimize candidate ordering before LLM prompting.
- **Small-to-Big Context Window Expansion (`context.py`)**: Dynamically expands high-precision retrieved chunks with surrounding section neighbor text within character budgets.
- **ChatGroq LLM Generation (`generation.py`)**: High-speed inference using ChatGroq models (`openai/gpt-oss-120b`) with strictly grounded prompt constraints and page-level source attribution.

### 💻 Frontend Architecture (React + Vite)
- **Drag-and-Drop Ingestion**: Interactive drag-and-drop or file picker PDF upload interface.
- **Session Stats & Metadata**: Displays processed document name, page count, and generated chunk counts.
- **Rich Markdown Formatting**: Renders assistant responses with bolding, lists, code blocks, and formatted tables via `react-markdown` and `remark-gfm`.
- **Interactive Citations Accordion**: Inserts page-stamped context excerpts (`Page 3`, section path, chunk type) used by the LLM.
- **Performance Monitor**: Real-time response timer (`⚡ 0.85s response time`).
- **Session Purge**: One-click cleanup to terminate sessions and wipe associated vector and database records.

---

## 🏛️ Backend Internal Architecture

The backend is built around a clean, decoupled tier architecture separating entry points, REST routes, pipeline stages, database storage, and external AI service singletons.

```text
backend/
├── main.py                     # FastAPI application entry point, startup lifecycle & CORS
├── chunking.py                 # Structure-aware hierarchical PDF chunker (+ PyMuPDF)
├── ocr.py                      # Gemini Vision OCR provider with rate limits & daily budget tracker
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration file
└── app/
    ├── config.py               # Centralized environment settings single-source-of-truth
    ├── embeddings.py           # GeminiEmbeddings adapter using Google GenAI SDK (gemini-embedding-001)
    ├── api/
    │   ├── routes.py           # Thin FastAPI route handlers (/health, /upload, /chat, /session)
    │   └── schemas.py          # Pydantic request and response schemas
    ├── database/
    │   ├── connection.py       # Psycopg (v3) connection factory (`get_db_conn`)
    │   ├── sessions.py         # `sessions` table CRUD & DDL initialization
    │   └── chunk_store.py      # `session_chunks` table CRUD & section neighbor lookup
    ├── services/
    │   ├── vector_store.py     # Shared PGVector vector store singleton accessor
    │   ├── llm.py              # ChatGroq LLM engine singleton accessor
    │   └── reranker.py         # Cohere Cross-Encoder / Passthrough reranker accessor
    └── pipeline/
        ├── ingest.py           # Stage 1: Document Ingestion (OCR -> Chunk -> Embed -> PGVector/DB)
        ├── bm25_index.py       # Stage 2a: Session-isolated BM25 keyword index cache & search
        ├── retrieval.py        # Stage 2b: Hybrid Retrieval (Dense + Sparse BM25 + RRF + Rerank)
        ├── context.py          # Stage 3: Small-to-Big context window neighbor expansion
        └── generation.py       # Stage 4: Answer Generation (ChatGroq + Source Citation extraction)
```

---

## 📐 Pipeline Data Flow & System Architecture

### 1. Document Ingestion Pipeline (`POST /upload`)

```mermaid
flowchart TD
    User([User uploads PDF]) -->|POST /upload| API[FastAPI Handler: routes.py]
    API -->|Async Threadpool| Ingest[Pipeline: ingest.py]
    Ingest -->|Structure Parsing| Chunker[chunking.py + PyMuPDF]
    Chunker -->|Scanned / Image Pages| OCR[ocr.py: Gemini Vision OCR]
    OCR -->|Check & Log Quota| Budget[Postgres Daily Budget Tracker]
    Chunker -->|Section Breadcrumbs| Contextualizer[contextualize]
    Contextualizer -->|Gemini Embeddings| Embedder[embeddings.py: gemini-embedding-001]
    Embedder -->|Dense Vectors| PGVector[(PGVector: docchat_chunks)]
    Contextualizer -->|Raw Text & Metadata| ChunkDB[(Postgres: session_chunks)]
    Ingest -->|Record Session| SessionDB[(Postgres: sessions)]
    API -->|Return Session Metadata| Client[React Frontend]
```

### 2. 4-Stage Hybrid RAG Query Pipeline (`POST /chat`)

```mermaid
flowchart TD
    UserQuery([User asks a question]) -->|POST /chat| API[FastAPI Handler: routes.py]
    API -->|Validate Session| Sessions[database/sessions.py]
    API -->|Invoke Pipeline| Gen[pipeline/generation.py]
    Gen -->|Stage 2: Hybrid Retrieve| Retrieve[pipeline/retrieval.py]
    
    subgraph Stage 2: Hybrid Retrieval & Reranking
        Retrieve -->|Dense Search k=20| VStore[services/vector_store.py -> PGVector]
        Retrieve -->|Sparse Search k=20| BM25[pipeline/bm25_index.py -> BM25Okapi]
        VStore & BM25 -->|Merge Hits| RRF[Reciprocal Rank Fusion]
        RRF -->|Top Candidates| Reranker[services/reranker.py: Cohere Reranker]
    end

    Reranker -->|Top k=5 Docs| ContextStage[pipeline/context.py]
    
    subgraph Stage 3: Small-to-Big Context Expansion
        ContextStage -->|Fetch Section Neighbors| ChunkDB[(Postgres: session_chunks)]
        ChunkDB -->|Widen Text Windows| ExpandedDocs[Expanded Context Documents]
    end

    ExpandedDocs -->|Stage 4: Prompt & Generate| LLM[services/llm.py: ChatGroq LLM]
    LLM -->|Answer Markdown| Response[Format Answer & Retrieved Citations]
    Response -->|Return JSON| Client[React Frontend]
```

---

## 🗄️ Database Storage Architecture

DocChat AI uses **PostgreSQL** with the `pgvector` extension (deployed via Docker Compose).

| Table / Collection | Engine | Purpose |
| :--- | :--- | :--- |
| **`docchat_chunks`** | PGVector | Dense vector embeddings (`gemini-embedding-001`) with JSONB metadata (`session_id`, `chunk_id`, `section_id`, `seq`). |
| **`session_chunks`** | PostgreSQL | Text content, section breadcrumbs, page bounds, and sequence numbers for BM25 keyword indexing and context expansion. |
| **`sessions`** | PostgreSQL | Session lifecycle state, original document name, total page count, chunk count, and creation timestamps. |
| **`ocr_usage`** | PostgreSQL | Atomic daily request budget counter for Gemini Vision OCR calls. |

---

## 🔌 API Endpoints Reference

The FastAPI backend exposes the following RESTful routes:

| Method | Endpoint | Description | Request Payload | Success Response |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/health` | Server liveness probe | None | `{"status": "ok"}` |
| **POST** | `/upload` | Upload PDF, run ingestion pipeline, create session | `multipart/form-data` (`file`) | `UploadResponse` (`session_id`, `doc_name`, `pages`, `chunks`, `warnings`) |
| **POST** | `/chat` | Execute 4-stage RAG query pipeline | `ChatRequest` (`session_id`, `question`) | `ChatResponse` (`answer`, `sources`, `elapsed`, `session_id`) |
| **GET** | `/session/{session_id}` | Retrieve document session statistics | Path param `session_id` | `SessionInfoResponse` (`session_id`, `doc_name`, `pages`, `chunks`) |
| **DELETE** | `/session/{session_id}` | Wipe vectors, DB rows, and BM25 cache for a session | Path param `session_id` | `{"detail": "Session deleted."}` |

---

## 📂 Complete Project Directory Structure

```text
DocChat-AI/
├── docker-compose.yml              # PostgreSQL + PGVector container setup
├── Corpus.pdf                      # Sample PDF document
├── assets/                         # Application preview screenshots
│   ├── welcome_screen.png
│   ├── chat_screen_1.png
│   └── chat_screen_2.png
├── backend/
│   ├── .env                        # API keys and environment configuration
│   ├── requirements.txt            # Python dependencies
│   ├── main.py                     # FastAPI application entry point
│   ├── chunking.py                 # Hierarchical structure-aware PDF chunker
│   ├── ocr.py                      # Gemini Vision OCR provider with rate limits
│   └── app/
│       ├── config.py               # Centralized configuration values
│       ├── embeddings.py           # GeminiEmbeddings wrapper class
│       ├── api/
│       │   ├── routes.py           # REST route handlers
│       │   └── schemas.py          # Pydantic data schemas
│       ├── database/
│       │   ├── connection.py       # Psycopg 3 connection factory
│       │   ├── sessions.py         # Session management CRUD
│       │   └── chunk_store.py      # Chunk metadata & neighbor CRUD
│       ├── services/
│       │   ├── vector_store.py     # PGVector vector store manager
│       │   ├── llm.py              # ChatGroq LLM manager
│       │   └── reranker.py         # Cohere Reranker manager
│       └── pipeline/
│           ├── ingest.py           # Stage 1: Document Ingestion
│           ├── bm25_index.py       # Stage 2a: Session BM25 keyword index
│           ├── retrieval.py        # Stage 2b: Hybrid Retrieval & RRF
│           ├── context.py          # Stage 3: Small-to-big context expansion
│           └── generation.py       # Stage 4: Prompting & answer generation
└── frontend/
    ├── package.json                # Frontend dependencies & scripts
    ├── vite.config.js              # Vite build configuration
    ├── index.html                  # Main HTML entrypoint
    └── src/
        ├── main.jsx                # React root mount
        ├── App.jsx                 # Layout & application state
        ├── api.js                  # Axios REST API client
        ├── index.css               # Design system & custom CSS styles
        └── components/
            ├── Sidebar.jsx         # Upload dropzone, stats, & session reset
            ├── ChatWindow.jsx      # Message container & welcome state
            ├── MessageBubble.jsx   # Markdown response & citation accordion
            └── ChatInput.jsx       # Query input & submit controls
```

---

## 🛠️ Setup & Local Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Docker Desktop (for PostgreSQL + PGVector)

---

### Step 1: Start PostgreSQL with PGVector
Run the included `docker-compose.yml` to launch a PostgreSQL 16 instance with `pgvector` pre-installed:

```powershell
docker compose up -d
```
This starts PostgreSQL on `localhost:5432` with database `docchat`, user `docchat`, and password `docchat`.

---

### Step 2: Backend Setup

#### 1. Obtain API Keys
- **Google Gemini API Key** (Embeddings & Vision OCR): [Google AI Studio](https://aistudio.google.com/)
- **Groq API Key** (Fast LLM Inference): [Groq Console](https://console.groq.com/)
- **Cohere API Key** (Optional - Reranking): [Cohere Dashboard](https://dashboard.cohere.com/)

#### 2. Configure Environment
Create `backend/.env` with your settings:

```env
DATABASE_URL="postgresql+psycopg://docchat:docchat@localhost:5432/docchat"

GOOGLE_API_KEY="your_google_gemini_api_key"
EMBEDDING_MODEL="gemini-embedding-001"

GROQ_API_KEY="your_groq_api_key"
LLM_MODEL="openai/gpt-oss-120b"

COHERE_API_KEY="your_cohere_api_key_optional"
RERANK_MODEL="rerank-english-v3.0"

OCR_ENABLED=true
OCR_MAX_PAGES=24
```

#### 3. Install Dependencies & Start Server
```powershell
cd backend

# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate      # On Windows
# source env/bin/activate   # On macOS/Linux

# Install requirements
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```
The API documentation is accessible at `http://localhost:8000/docs`.

---

### Step 3: Frontend Setup

```powershell
cd ../frontend

# Install npm packages
npm install

# Start Vite dev server
npm run dev
```
Open your browser and navigate to `http://localhost:5173`.

---

## 🛠️ Technologies Used

### Backend & AI Pipelines
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Vector Database**: [PGVector](https://github.com/pgvector/pgvector) via `langchain-postgres`
- **Relational Database**: [PostgreSQL 16](https://www.postgresql.org/) via `psycopg` (v3)
- **Embeddings**: Google GenAI SDK (`google-genai`) with `gemini-embedding-001`
- **LLM Engine**: ChatGroq (`langchain-groq`)
- **Vision OCR**: Gemini 2.0 Flash-Lite with daily request budget tracking
- **Cross-Encoder Reranker**: Cohere Python Client (`cohere` v2)
- **Sparse Keyword Search**: `rank-bm25` (BM25Okapi)
- **PDF Extraction**: PyMuPDF (`pymupdf`)

### Frontend
- **Framework**: [React 18](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Markdown & Citations**: `react-markdown` + `remark-gfm`
- **Icons**: `lucide-react`
- **HTTP Client**: `axios`
