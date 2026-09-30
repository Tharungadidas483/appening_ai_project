from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from appening_rag.schemas import GroundingAssessment

REFUSAL = "I could not find enough information about that in the Agentic AI eBook."


class RAGState(TypedDict, total=False):
    query: str
    documents: list[Document]
    relevance_scores: list[float]
    final_answer: str
    confidence_score: float
    retrieved_context_chunks: list[str]


def build_rag_graph(
    vector_store: VectorStore,
    answer_model: Any,
    grader_model: Any,
    top_k: int = 4,
    min_relevance_score: float = 0.35,
):
    def retrieve(state: RAGState) -> dict[str, Any]:
        results = vector_store.similarity_search_with_relevance_scores(state["query"], k=top_k)
        documents = [document for document, _ in results]
        scores = [max(0.0, min(1.0, float(score))) for _, score in results]
        return {
            "documents": documents,
            "relevance_scores": scores,
            "retrieved_context_chunks": [document.page_content for document in documents],
        }

    def generate(state: RAGState) -> dict[str, Any]:
        scores = state.get("relevance_scores", [])
        if not scores or max(scores) < min_relevance_score:
            return {"final_answer": REFUSAL}

        context = "\n\n---\n\n".join(state["retrieved_context_chunks"])
        response = answer_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Answer using only the supplied excerpts from the Agentic AI eBook. "
                        "Do not use outside knowledge or infer unsupported facts. If the excerpts "
                        "do not answer the question, say that the information is not available."
                    )
                ),
                HumanMessage(content=f"Question:\n{state['query']}\n\nEBook excerpts:\n{context}"),
            ]
        )
        answer = response.content if hasattr(response, "content") else str(response)
        return {"final_answer": answer.strip()}

    def grade(state: RAGState) -> dict[str, Any]:
        scores = state.get("relevance_scores", [])
        best_relevance = max(scores, default=0.0)
        answer = state.get("final_answer", REFUSAL)
        if answer == REFUSAL:
            return {"confidence_score": round(1.0 - best_relevance, 3)}

        context = "\n\n---\n\n".join(state.get("retrieved_context_chunks", []))
        assessor = grader_model.with_structured_output(GroundingAssessment)
        assessment = assessor.invoke(
            [
                SystemMessage(
                    content=(
                        "Check whether the answer's factual claims are fully supported by the "
                        "provided eBook excerpts. Return supported=false if any factual claim is "
                        "missing or contradicted. Set confidence from 0 to 1."
                    )
                ),
                HumanMessage(content=f"Excerpts:\n{context}\n\nAnswer:\n{answer}"),
            ]
        )
        if isinstance(assessment, dict):
            assessment = GroundingAssessment.model_validate(assessment)
        if not assessment.supported:
            return {"final_answer": REFUSAL, "confidence_score": round(1.0 - best_relevance, 3)}
        confidence = min(best_relevance, assessment.confidence)
        return {"confidence_score": round(max(0.0, min(1.0, confidence)), 3)}

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("grade", grade)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", END)
    return workflow.compile()
