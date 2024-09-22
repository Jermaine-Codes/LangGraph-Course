from models import BSDetectorState

def aggregate_confidence(state: BSDetectorState) -> BSDetectorState:
    observed_consistency = state["state"]["observed_consistency"]
    self_reflection_certainty = state["state"]["self_reflection_certainty"]
    
    beta = 0.7  # Adjust this value as needed
    final_confidence = beta * observed_consistency + (1 - beta) * self_reflection_certainty
    
    state["state"]["final_confidence"] = final_confidence
    return state
