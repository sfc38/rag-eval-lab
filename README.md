# RAG Eval Lab

A free, local-first application for **quantitatively evaluating and debugging
Retrieval-Augmented Generation pipelines**. Upload documents, build a vector index, run
reproducible experiments across RAG configurations (chunking, embeddings, top_k, reranking,
LLM), and inspect retrieval failures case by case.

The differentiator is not the chat demo — it is the **evaluation harness, the measured
findings, and the failure-analysis tooling**. Everything runs locally and free
(Ollama + sentence-transformers + ChromaDB).

> **Status:** Phase 2 complete — the retrieval evaluation harness (recall@k, MRR, nDCG via
> evidence-span matching) and `/eval/*` endpoints are in. The experiment sweep, reranking,
> answer-quality judging, and the retrieval debugger arrive in subsequent phases. Headline
> findings will be published here as each experiment is run.

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
| Default LLM | Qwen2.5-VL 3B via Ollama | Apache 2.0 |
| Frontend | Streamlit | Apache 2.0 |
| Charts | Plotly / Matplotlib | MIT / BSD |

All dependencies are free for commercial and portfolio use. **PyMuPDF is intentionally not
used** — it is AGPL-3.0; `pypdfium2` (Apache 2.0) replaces it.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — install from [ollama.com](https://ollama.com), then pull the default model:
   ```bash
   ollama pull qwen2.5vl:3b
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
| POST | `/eval/run` | Retrieval-only evaluation against the golden dataset |
| GET | `/eval/runs` | List past evaluation runs (summaries) |
| GET | `/eval/runs/{run_id}` | Full results for one run (summary + per-question rows) |

### Evaluation methodology (Phase 2)

Retrieval is scored without manual chunk labeling. Each golden record carries verbatim
**evidence spans**; a chunk is **relevant** if it contains (or substantially overlaps) any span.
Per question we compute:

- **recall@k** (k = 1, 3, 5, 10) — fraction of relevant chunks found in the top k
- **MRR** — reciprocal rank of the first relevant chunk
- **nDCG@k** — rank-weighted relevance gain

Answerable and unanswerable questions are reported separately; unanswerable questions (≈10% of
the dataset) are an answer-side concern measured in Phase 5. Every run is persisted under
`results/runs/<run_id>/` as `config.json`, `results.csv`, and `summary.json` — reproducible and
diffable.

A small **dev sample** (`eval_data/sample/`) ships for development. The portfolio-grade golden
dataset (2–3 public documents, 50–100 manually verified records) is the next deliverable.

#### Run an evaluation
```bash
# 1. upload the sample doc (note the returned document_id)
curl -s -F "file=@eval_data/sample/sample_rag_overview.txt" \
  http://localhost:8000/documents/upload
# 2. evaluate (substitute the document_id)
curl -s -X POST http://localhost:8000/eval/run -H "Content-Type: application/json" -d '{
  "document_id": "<id>",
  "dataset_path": "../eval_data/sample/golden_qa.jsonl",
  "dataset_document_id": "sample_rag_overview"
}'
```

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
| LLM | Ollama `qwen2.5vl:3b` · Gemini (demo, Phase 8) |

Two hashes drive reproducibility and index caching: `index_hash` (chunking + embedding) keys
the ChromaDB collection so sweeps over retrieval/LLM settings reuse an existing index;
`config_hash` identifies a full experiment run.

---

## Roadmap

1. **RAG MVP** — upload → index → ask with sources ✅
2. **Retrieval metrics** (recall@k, MRR, nDCG) + `/eval/*` + dev sample dataset ✅
3. Experiment runner + chunking sweep → **Finding #1**
4. Cross-encoder reranking → **Finding #2** (quality vs latency)
5. Answer-quality eval (LLM-as-judge: faithfulness, relevance, abstention) → **Finding #3**
6. Retrieval Debugger (failure taxonomy)
7. Results write-up + analysis notebook
8. Online demo (HF Spaces / Streamlit Cloud, Gemini free tier) — optional

---

## License

MIT — free to use, modify, and distribute.
