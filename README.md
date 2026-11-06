# DocChat AI — RAG based Document Question-Answering System

DocChat AI is an enterprise-grade RAG (Retrieval-Augmented Generation) document intelligence assistant. It enables users to upload PDF documents, automatically parse and structure their contents (including scanned pages via Gemini Vision OCR), index them using hybrid vector-and-keyword search, and engage in context-grounded conversational chat.

All generative answers are backed by precise, page-stamped source citations with real-time response latency metrics.

## Saved conversations

Each PDF upload creates a separate conversation owned by the signed-in account. The sidebar shows that account's history, ordered by recent activity. Selecting a conversation opens its own panel with its document, messages, citations, and response times without uploading the PDF again. Every sign-in opens a fresh new-chat screen; previous conversations open only when selected from history.

**New chat** leaves earlier conversations in history. **Delete conversation** permanently removes the selected conversation, its messages, and its indexed document after confirmation. Signing out preserves saved history.

Messages are stored in PostgreSQL in `chat_messages`; the existing document chunks and vectors support resumed questions after server restarts. The latest 12 messages (up to 12,000 characters) provide context for follow-up questions. Startup automatically creates the message table and adds session activity timestamps. Keep the Docker `pgdata` volume to retain data.

History is saved for new exchanges after this update. Older messages that existed only in browser memory cannot be recovered. Documents from before account ownership was introduced remain unassigned and are not exposed to any account.

### Chat workspace

1. Sign up or sign in. Every sign-in opens **New chat**, while saved conversations load into the sidebar.
2. Upload or drag a PDF into the **center upload card**. A fresh chat shows instructions for uploading, asking questions, and returning later.
3. Once processing finishes, the document opens in its own conversation panel. Start with a suggested question or use the composer below.
4. Select a saved conversation from the sidebar to resume it. Search by document name; chats are grouped into Today, Yesterday, Previous 7 days, and Earlier.
5. Choose **New chat** to upload another document without deleting earlier sessions. Use **Delete chat** inside a conversation to remove it after confirmation.

The workspace uses the dark blue theme, a dedicated history sidebar, and a centered message composer. Only the selected session's panel is shown. Its messages, loading state, errors, citations, and composer are isolated from other sessions; delayed responses from another session cannot appear in the current chat. Unsent drafts are cleared when switching panels. The layout adapts to smaller screens.

---
<!--
## 🖥️ User Interface Preview

These screenshots show an earlier UI version. The current workspace uses the centered upload flow and separate session panels described above.

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
  - **PostgreSQL**: Stores accounts, browser authentication sessions, owned document sessions, complete chat history, and chunk metadata via `psycopg` (v3).
- **Hybrid Search & Reciprocal Rank Fusion (`retrieval.py` & `bm25_index.py`)**: Merges dense vector hits with in-memory BM25 keyword scores using Reciprocal Rank Fusion (RRF) to excel at both semantic queries and exact term/code lookups.
- **Cross-Encoder Reranking (`reranker.py`)**: Optional Cohere Cross-Encoder (`rerank-english-v3.0`) re-scoring to optimize candidate ordering before LLM prompting.
- **Small-to-Big Context Window Expansion (`context.py`)**: Dynamically expands high-precision retrieved chunks with surrounding section neighbor text within character budgets.
- **Persistent Conversations**: Saves each successful question and answer, citations, and response time in one transaction. Recent saved messages help resolve follow-up questions before retrieval.
- **Account Isolation**: Authentication and ownership checks restrict session listings, messages, document queries, and deletion to the signed-in account.
- **ChatGroq LLM Generation (`generation.py`)**: High-speed inference using ChatGroq models (`openai/gpt-oss-120b`) with strictly grounded prompt constraints and page-level source attribution.

### 💻 Frontend Architecture (React + Vite)
- **Fresh Chat on Sign-In**: Loads history without automatically opening an earlier conversation.
- **Centered Upload & Instructions**: PDF picker, drag-and-drop upload, progress, and errors live in the new-chat screen.
- **Searchable History Sidebar**: Conversation entries are grouped by date and highlight the active session.
- **Separate Conversation Panels**: `ConversationPanel.jsx` is keyed by session ID and owns that session's messages, request state, errors, and composer.
- **Document Header**: Shows the selected filename, page count, and conversation deletion action.
- **Dark Blue Responsive Layout**: Dedicated sidebar, centered chat content, and composer with layouts for smaller screens.
- **Rich Markdown Formatting**: Renders assistant responses with bolding, lists, code blocks, and formatted tables via `react-markdown` and `remark-gfm`.
- **Interactive Citations Accordion**: Inserts page-stamped context excerpts (`Page 3`, section path, chunk type) used by the LLM.
- **Performance Monitor**: Real-time response timer (`⚡ 0.85s response time`).
- **Confirmed Deletion**: Removes a conversation, its messages, document vectors, chunk records, and BM25 cache after confirmation.

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
    │   ├── auth.py             # Signup, login, logout, cookie authentication & request protection
    │   ├── routes.py           # Document, chat, session history & message routes
    │   └── schemas.py          # Pydantic request and response schemas
    ├── database/
    │   ├── connection.py       # Psycopg (v3) connection factory (`get_db_conn`)
    │   ├── auth.py             # Accounts, hashed browser sessions & authentication rate limits
    │   ├── sessions.py         # Document sessions, chat_messages, activity timestamps & DDL
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
    API -->|Authenticate and Check Ownership| Sessions[database/sessions.py]
    Sessions -->|Load Recent Messages| History[(Postgres: chat_messages)]
    History -->|Conversation Context| Gen
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
    Response -->|Save Question, Answer and Citations| Messages[(Postgres: chat_messages)]
    Messages -->|Return JSON After Commit| Client[React Frontend]
```

---

## 🗄️ Database Storage Architecture

DocChat AI uses **PostgreSQL** with the `pgvector` extension (deployed via Docker Compose).

| Table / Collection | Engine | Purpose |
| :--- | :--- | :--- |
| **`docchat_chunks`** | PGVector | Dense vector embeddings (`gemini-embedding-001`) with JSONB metadata (`session_id`, `chunk_id`, `section_id`, `seq`). |
| **`session_chunks`** | PostgreSQL | Text content, section breadcrumbs, page bounds, and sequence numbers for BM25 keyword indexing and context expansion. |
| **`sessions`** | PostgreSQL | Document name, page/chunk counts, account ownership (`user_id`), creation time, and last activity (`updated_at`). |
| **`chat_messages`** | PostgreSQL | Ordered user/assistant messages, JSONB citations, response times, and timestamps; deleted automatically when their session is deleted. |
| **`users`** | PostgreSQL | Account names, unique email addresses, and salted password hashes. |
| **`auth_sessions`** | PostgreSQL | Hashed browser authentication tokens, account references, and expiration times. |
| **`auth_attempts`** | PostgreSQL | Per-client authentication attempt counters for rate limiting. |
| **`ocr_usage`** | PostgreSQL | Atomic daily request budget counter for Gemini Vision OCR calls. |

---

## 🔌 API Endpoints Reference

### Authentication

Create an account or sign in before uploading or chatting. Signup signs you in
automatically. Accounts and seven-day browser sessions persist in PostgreSQL;
passwords use salted PBKDF2-SHA256 hashes and session tokens are stored hashed.
Sign out revokes the browser session and clears the workspace.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/auth/signup` | Create account with `name`, `email`, `password` (12–128 characters) |
| POST | `/auth/login` | Sign in with `email`, `password` |
| GET | `/auth/me` | Get the signed-in user, or 401 |
| POST | `/auth/logout` | Revoke the session and clear its cookie |

Document upload, chat, session history, saved messages, metadata, and deletion require authentication. Documents
belong to their uploader; another account receives 404 when requesting them.
Existing documents without an owner remain inaccessible; upload them again after
signing in. Startup creates the account tables and adds the ownership column
automatically without assigning existing documents to an arbitrary account.

The frontend sends cookies using `credentials: 'include'`. API clients must keep
the session cookie and send `X-DocChat-Request: 1` on state-changing requests.
Browser origins are checked against `FRONTEND_ORIGINS`. Signup and login share a
database-backed limit of 20 attempts per client IP per 15 minutes; behind a proxy,
configure Uvicorn to trust forwarded client IPs only from your actual proxy.

Optional backend environment settings:

```env
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
AUTH_SESSION_SECONDS=604800
AUTH_COOKIE_SECURE=false
```

For production, use HTTPS, set `AUTH_COOKIE_SECURE=true`, and list your frontend
origin explicitly. Host the frontend and API on the same site (for example,
`app.example.com` and `api.example.com`) for the SameSite=Lax cookie. In local
development, use the same hostname for both services (both `localhost` or both
`127.0.0.1`). Password recovery and email verification are not included.

The FastAPI backend exposes the following RESTful routes:

| Method | Endpoint | Description | Request Payload | Success Response |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/health` | Server liveness probe | None | `{"status": "ok"}` |
| **POST** | `/upload` | Upload PDF, run ingestion pipeline, create session | `multipart/form-data` (`file`) | `UploadResponse` (`session_id`, `doc_name`, `pages`, `chunks`, `warnings`) |
| **POST** | `/chat` | Answer using the selected document and recent history; persist the successful exchange | `ChatRequest` (`session_id`, `question`) | `ChatResponse` (`answer`, `sources`, `elapsed`, `session_id`) |
| **GET** | `/sessions` | List only the signed-in account's sessions, ordered by recent activity | None | `SessionHistoryResponse[]` with metadata, `created_at`, and `updated_at` |
| **GET** | `/session/{session_id}/messages` | Open a session and restore its ordered messages | Path param `session_id` | `SessionDetailResponse` with document metadata and `messages` |
| **GET** | `/session/{session_id}` | Retrieve document session statistics | Path param `session_id` | `SessionInfoResponse` (`session_id`, `doc_name`, `pages`, `chunks`) |
| **DELETE** | `/session/{session_id}` | Delete the owned session, its messages, vectors, chunk rows, and BM25 cache | Path param `session_id` | `{"detail": "Session deleted."}` |

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
│   ├── tests/                      # Authentication, ownership, history & follow-up regressions
│   └── app/
│       ├── config.py               # Centralized configuration values
│       ├── embeddings.py           # GeminiEmbeddings wrapper class
│       ├── api/
│       │   ├── auth.py             # Authentication routes & dependencies
│       │   ├── routes.py           # Document, chat & history routes
│       │   └── schemas.py          # Pydantic data schemas
│       ├── database/
│       │   ├── connection.py       # Psycopg 3 connection factory
│       │   ├── auth.py             # Account & browser session persistence
│       │   ├── sessions.py         # Document sessions & durable message history
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
    ├── tests/
    │   └── sessions.test.mjs        # Login, panel isolation & delayed-response regressions
    └── src/
        ├── main.jsx                # React root mount
        ├── App.jsx                 # Authentication, history & selected-session navigation
        ├── api.js                  # Fetch API client with cookies & request protection
        ├── index.css               # Design system & custom CSS styles
        └── components/
            ├── AuthScreen.jsx      # Signup & sign-in forms
            ├── Sidebar.jsx         # New chat, search & grouped conversation history
            ├── NewChat.jsx         # Centered PDF upload, progress & instructions
            ├── ConversationPanel.jsx # Session-owned messages, loading, errors & composer
            ├── ChatWindow.jsx      # Selected conversation, ready state & suggested questions
            ├── MessageBubble.jsx   # Markdown response & citation accordion
            └── ChatInput.jsx       # Query input & submit controls
```

---

## 🛠️ Setup & Local Installation

### Prerequisites
- Python 3.10+
- Node.js 20.19+ (20.x) or 22.12+ and npm, as required by the installed Vite version
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
Open your browser and navigate to `http://localhost:5173`. Create an account or sign in to open the new-chat workspace.

The frontend defaults to `http://localhost:8000` for API calls. To change it, create `frontend/.env` with:

```env
VITE_API_URL=http://localhost:8000
```

Restart the frontend after changing this setting. Keep the frontend origin in the backend's `FRONTEND_ORIGINS` list.

---

## Validation

Run backend regressions from the repository root with your backend virtual environment activated:

```powershell
python -m unittest discover -s backend/tests -v
```

The backend suite covers authentication, document ownership, private session history, message restoration, successful-turn persistence, failure handling, and follow-up question context. Database and AI calls are mocked; these checks do not replace a live PostgreSQL and model-service integration test.

Run frontend checks from `frontend`:

```powershell
npm test
npm run build
```

The frontend suite checks that sign-in opens New chat even with saved history, each selected session gets an isolated panel, New chat resets the canvas, and delayed loads or answers cannot replace another session's content.

After restarting the backend to apply the automatic schema updates, verify the live flow: upload a PDF, exchange messages, sign out, sign back in, and select the saved conversation. Confirm the restored messages and citations, then continue asking questions without uploading again.

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
- **Icons**: Inline SVG and text icons
- **HTTP Client**: Browser Fetch API with cookie credentials
- **Regression Tests**: Node.js test runner + `react-test-renderer`
