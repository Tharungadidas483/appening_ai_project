from functools import lru_cache
from typing import Any

from appening_rag.config import Settings, get_settings
from appening_rag.graph import build_rag_graph
from appening_rag.vector_store import create_vector_store


class RAGService:
    def __init__(self, settings: Settings, vector_store: Any | None = None, answer_model: Any | None = None, grader_model: Any | None = None):
        settings.validate_runtime()
        self.settings = settings
        self.vector_store = vector_store or create_vector_store(settings)
        if answer_model is None or grader_model is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for answer generation")
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=settings.openai_chat_model,
                temperature=0,
                api_key=settings.openai_api_key.get_secret_value(),
            )
            answer_model = answer_model or model
            grader_model = grader_model or model
        self.graph = build_rag_graph(
            self.vector_store,
            answer_model,
            grader_model,
            top_k=settings.retrieval_top_k,
            min_relevance_score=settings.min_relevance_score,
        )

    def ask(self, query: str) -> dict[str, Any]:
        result = self.graph.invoke({"query": query})
        return {
            "query": query,
            "final_answer": result["final_answer"],
            "retrieved_context_chunks": result.get("retrieved_context_chunks", []),
            "confidence_score": result.get("confidence_score", 0.0),
        }


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())
