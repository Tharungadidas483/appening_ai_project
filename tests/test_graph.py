from types import SimpleNamespace

from langchain_core.documents import Document

from appening_rag.graph import REFUSAL, build_rag_graph


class FakeVectorStore:
    def __init__(self, score: float):
        self.score = score

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        return [(Document(page_content="Agentic AI systems can plan and act toward goals."), self.score)]


class FakeAnswerModel:
    def invoke(self, messages):
        return SimpleNamespace(content="Agentic AI systems can plan and act toward goals.")


class FakeStructuredModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return {"supported": True, "confidence": 0.9}


class UnsupportedStructuredModel(FakeStructuredModel):
    def invoke(self, messages):
        return {"supported": False, "confidence": 0.1}


def test_grounded_answer_has_context_and_bounded_confidence():
    graph = build_rag_graph(FakeVectorStore(0.92), FakeAnswerModel(), FakeStructuredModel())

    result = graph.invoke({"query": "What can agentic systems do?"})

    assert result["final_answer"].startswith("Agentic AI systems")
    assert result["retrieved_context_chunks"] == ["Agentic AI systems can plan and act toward goals."]
    assert result["confidence_score"] == 0.9


def test_low_relevance_query_is_refused_without_answering():
    graph = build_rag_graph(FakeVectorStore(0.1), FakeAnswerModel(), FakeStructuredModel())

    result = graph.invoke({"query": "What is the capital of France?"})

    assert result["final_answer"] == REFUSAL
    assert result["confidence_score"] == 0.9


def test_unsupported_generated_answer_is_replaced_with_refusal():
    graph = build_rag_graph(FakeVectorStore(0.9), FakeAnswerModel(), UnsupportedStructuredModel())

    result = graph.invoke({"query": "What can agentic systems do?"})

    assert result["final_answer"] == REFUSAL
