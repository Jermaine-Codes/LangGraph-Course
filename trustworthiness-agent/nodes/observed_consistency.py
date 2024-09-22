from models import BSDetectorState
from utils import calculate_similarity

def calculate_observed_consistency(state: BSDetectorState) -> BSDetectorState:
    original_answer = state["state"]["original_answer"]
    diverse_outputs = state["state"]["diverse_outputs"]
    
    similarities = [calculate_similarity(original_answer, output) for output in diverse_outputs]
    observed_consistency = sum(similarities) / len(similarities)
    
    state["state"]["observed_consistency"] = observed_consistency
    return state
