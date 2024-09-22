from transformers import pipeline

nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def calculate_similarity(text1: str, text2: str) -> float:
    result = nli_model(text1, candidate_labels=[text2, f"not {text2}"])
    similarity = result['scores'][0]  # Probability of entailment
    return similarity
