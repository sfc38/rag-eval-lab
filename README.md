# RAG Eval Lab

A free, local-first application for **quantitatively evaluating and debugging
Retrieval-Augmented Generation pipelines**. Upload documents, build a vector index, run
reproducible experiments across RAG configurations (chunking, embeddings, top_k, reranking,
LLM), and inspect retrieval failures case by case.

The differentiator is not the chat demo — it is the **evaluation harness, the measured
findings, and the failure-analysis tooling**. Everything runs locally and free
(Ollama + sentence-transformers + ChromaDB).

> **Status:** Phase 1 (RAG MVP) complete. Evaluation harness, benchmarks, and the retrieval
> debugger arrive in subsequent phases — see the roadmap below. Headline findings will be
> published here as each experiment is run.

---

## Architecture

```text
Upload → Extract (pypdfium2) → Clean → Chunk → Embed → ChromaDB
Question → Embed → Retrieve (top_k, threshold) → [Rerank*] → Prompt → LLM → Answer + Sources
```

`*` reranking is a planned evaluation condition (Phase 4).

The pipeline is hand-written (no LangChain/LlamaIndex) so the evaluation logic stays
transparent and inspectable. Retrieval (`/rag/retrieve`) is deliberately separate from
generation (`/rag/ask`) — this is what makes retrieval-only metrics and the debugger possible.

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| API framework | FastAPI | MIT |
| Server | Uvicorn | BSD |
| Vector store | ChromaDB | Apache 2.0 |
| Embeddings / reranker | sentence-transformers | Apache 2.0 |
| PDF extraction | pypdfium2 | Apache 2.0 |
| Default LLM | Qwen2.5 3B via Ollama | Apache 2.0 |
| Frontend | Streamlit | Apache 2.0 |
| Charts | Plotly / Matplotlib | MIT / BSD |

All dependencies are free for commercial and portfolio use. **PyMuPDF is intentionally not
used** — it is AGPL-3.0; `pypdfium2` (Apache 2.0) replaces it.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — install from [ollama.com](https://ollama.com), then pull the default model:
   ```bash
   ollama pull qwen2.5:3b
   ```

---

## Setup

```bash
git clone https://github.com/sfc38/rag-eval-lab.git
cd rag-eval-lab

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

**Backend** (from the `backend/` directory):

```bash
cd backend
uvicorn app.main:app --reload
```

API docs: <http://127.0.0.1:8000/docs>

**Frontend** (from the `frontend/` directory, in a second terminal):

```bash
cd frontend
streamlit run streamlit_app.py
```

The frontend reads `API_URL` (default `http://localhost:8000`) so it can point at a remote
backend when deployed.

---

## API (Phase 1)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status + Ollama reachability |
| POST | `/documents/upload` | Upload a PDF/TXT/MD; extract + register |
| POST | `/documents/index` | Build the index for a config (cached by index hash) |
| GET | `/documents` | List registered documents |
| DELETE | `/documents/{id}` | Delete a document and its indexes |
| POST | `/rag/retrieve` | Retrieval only — chunks + scores (powers the debugger) |
| POST | `/rag/ask` | Retrieve + prompt + generate — answer with sources |

Evaluation endpoints (`/eval/run`, `/eval/runs`) are added in Phase 2+.

---

## Configuration axes

Every field below is an experiment variable, captured by `PipelineConfig`:

| Axis | Values |
|---|---|
| Chunking strategy | fixed · recursive · sentence |
| Chunk size / overlap | size 300–1000 · overlap 0–200 |
| Embedding model | `all-MiniLM-L6-v2` · `bge-small-en-v1.5` |
| Retrieval | top_k · similarity threshold (cosine) |
| Reranking | off · cross-encoder (Phase 4) |
| LLM | Ollama `qwen2.5:3b` · Gemini (demo, Phase 8) |

Two hashes drive reproducibility and index caching: `index_hash` (chunking + embedding) keys
the ChromaDB collection so sweeps over retrieval/LLM settings reuse an existing index;
`config_hash` identifies a full experiment run.

---

## Roadmap

1. **RAG MVP** — upload → index → ask with sources ✅
2. Golden QA dataset + retrieval metrics (recall@k, MRR, nDCG)
3. Experiment runner + chunking sweep → **Finding #1**
4. Cross-encoder reranking → **Finding #2** (quality vs latency)
5. Answer-quality eval (LLM-as-judge: faithfulness, relevance, abstention) → **Finding #3**
6. Retrieval Debugger (failure taxonomy)
7. Results write-up + analysis notebook
8. Online demo (HF Spaces / Streamlit Cloud, Gemini free tier) — optional

---

## License

MIT — free to use, modify, and distribute.
