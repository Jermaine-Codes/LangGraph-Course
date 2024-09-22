from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class State(TypedDict):
    question: str
    original_answer: str
    diverse_outputs: List[str]
    observed_consistency: float
    self_reflection_certainty: float
    final_confidence: float

class BSDetectorState(TypedDict):
    messages: List[BaseMessage]
    state: State
