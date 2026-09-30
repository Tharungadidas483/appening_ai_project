from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(ge=0, le=1)


class GroundingAssessment(BaseModel):
    supported: bool = Field(description="Whether every factual claim is supported by the supplied context")
    confidence: float = Field(ge=0, le=1, description="Confidence that the answer is supported by context")
