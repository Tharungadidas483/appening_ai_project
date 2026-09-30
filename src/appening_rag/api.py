from fastapi import FastAPI, HTTPException

from appening_rag.schemas import ChatRequest, ChatResponse
from appening_rag.service import RAGService, get_service

app = FastAPI(title="Appening Agentic AI RAG", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        payload = get_service().ask(request.query.strip())
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"RAG service is unavailable: {error}") from error
    return ChatResponse.model_validate(payload)
