# Appening Agentic AI RAG

A custom Python retrieval-augmented chatbot for the Agentic AI eBook. It uses a LangGraph workflow, a FastAPI endpoint, and a grounding grader. Answers are based only on retrieved eBook passages; weak retrieval and unsupported generated answers produce a refusal.

## Architecture

1. `ingestion.py` downloads the configured PDF when it is missing, extracts page text with PyPDF, and splits it into overlapping chunks. Each chunk retains its source and page metadata.
2. `vector_store.py` selects persistent local Chroma or hosted Pinecone. Local mode uses deterministic feature-hash embeddings and needs no provider credentials; this is intended for development, not production semantic search.
3. `graph.py` runs retrieve -> generate -> grade in LangGraph. The generation prompt limits claims to retrieved passages. Retrieval below `MIN_RELEVANCE_SCORE` skips generation. A structured LLM grader refuses answers that are not supported and bounds confidence by both grader confidence and retrieval relevance.
4. `api.py` exposes `POST /chat` and `GET /health`.

## Requirements

- Python 3.11 or newer
- For hosted Pinecone mode: a Pinecone index and API key, plus an OpenAI API key for embeddings and generation
- Network access for the initial PDF download

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The defaults select local Chroma and do not require API keys for indexing. OpenAI credentials are required for chat generation in either mode. Set `VECTOR_STORE_BACKEND=pinecone`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `OPENAI_API_KEY` to use hosted storage with OpenAI embeddings.

## Ingest the eBook

```powershell
appening-ingest
```

The configured PDF is downloaded to `data/Ebook-Agentic-AI.pdf`; chunks are indexed into the selected vector store. Re-run after updating the source document. To re-download the source explicitly:

```powershell
appening-ingest --force-download
```

For Pinecone, create an index compatible with the configured OpenAI embedding model before ingestion. Check the model's vector dimension in the OpenAI documentation and match the index dimension.

## Run the API

```powershell
uvicorn appening_rag.api:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/docs` for the interactive API schema.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType 'application/json' `
  -Body '{"query":"What is Agentic AI?"}'
```

Every successful chat response has this shape:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["..."],
  "confidence_score": 0.82
}
```

The out-of-scope question "What is the capital of France?" should be refused when its retrieved relevance is below the configured threshold. The API does not claim that a numeric confidence is a calibrated probability.

## Tests

```powershell
python -m pytest
```

The graph tests cover grounded answer flow, low-relevance refusal, and unsupported-answer rejection. API tests assert the required JSON response contract. Tests use deterministic fakes and do not make provider calls.

## Repository submission

Create a public GitHub repository and push this workspace to it before submitting the repository link. Do not commit `.env`, provider credentials, the downloaded PDF, or local Chroma data.
