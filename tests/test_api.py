from fastapi.testclient import TestClient

from appening_rag.api import app
from appening_rag.service import get_service


class FakeService:
    def ask(self, query: str):
        return {
            "query": query,
            "final_answer": "Information not available in the eBook.",
            "retrieved_context_chunks": [],
            "confidence_score": 0.95,
        }


def test_chat_returns_required_response_contract():
    app.dependency_overrides[get_service] = FakeService
    try:
        response = TestClient(app).post("/chat", json={"query": "What is Agentic AI?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert set(response.json()) == {
        "query",
        "final_answer",
        "retrieved_context_chunks",
        "confidence_score",
    }
    assert response.json()["confidence_score"] == 0.95
