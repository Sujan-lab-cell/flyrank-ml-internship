# KaStack-RAG — Study Buddy

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents.

---

## Project Structure

```
KaStack-RAG/
├── frontend/          React + Tailwind UI
└── backend/           FastAPI RAG pipeline
    ├── app.py         Entry point
    ├── api/           Route handlers (upload, chat, persona)
    ├── services/      Core logic (parser, retriever, answer generator…)
    ├── models/        Embedding + summarization models
    ├── vectorstore/   FAISS index + build script
    ├── data/          Conversations, personas, topic summaries
    └── utils/         Config + helpers
```

---

## Quick Start

### Backend

```bash
cd backend
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

API docs available at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload a PDF — parses, embeds, stores in FAISS |
| POST | `/chat` | Ask a question — retrieves chunks, generates answer |
| GET | `/persona` | List available AI personas |
| GET | `/persona/{id}` | Get a specific persona |

### POST /upload
```
Content-Type: multipart/form-data
Body: file (PDF)

Response: { id, name, chunks }
```

### POST /chat
```json
{ "message": "What is the main topic?", "document_id": "optional-uuid" }

Response: { "answer": "...", "sources": [{ "page": 1, "text": "..." }] }
```

---

## Connecting Frontend to Backend

In `frontend/src/services/chatService.js` and `uploadService.js`, uncomment the real axios calls and remove the mock blocks. The request/response shapes already match.

Set the API URL in a `.env` file at the frontend root:
```
VITE_API_URL=http://localhost:8000
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Tailwind CSS v4, Vite |
| Backend | FastAPI, Python 3.11+ |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (faiss-cpu) |
| PDF Parsing | pdfplumber |
| Summarization | HuggingFace Transformers |
