import re
import json
from langchain_openai import ChatOpenAI
from models import BSDetectorState

def perform_self_reflection(state: BSDetectorState) -> BSDetectorState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question = state["state"]["question"]
    original_answer = state["state"]["original_answer"]
    
    prompt = f"""1. Question: {question}, Proposed Answer: {original_answer}.
    Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template:
    explanation: [insert analysis], answer: [choose one letter from among choices A through C, write the letter only.]

    2. Question: {question}, Proposed Answer: {original_answer}.
    Are you really sure the proposed answer is correct? Choose again: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template:
    explanation: [insert analysis], answer: [choose one letter from among choices A through C, write the letter only.]

    Provide your answer in the following format:
    [
        {{
            "explanation": "[Insert analysis here]",
            "answer": "[Choose one letter from among choices A through C, write the letter only]"
        }},
        {{
            "explanation": "[Insert analysis here]",
            "answer": "[Choose one letter from among choices A through C, write the letter only]"
        }}
    ]
    """
    
    response = llm.invoke(prompt)
    
    # Use regex to extract the entire JSON-like structure
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, response.content, re.DOTALL)
    
    if not match:
        print("Failed to find JSON-like structure in the response.")
        print(response.content)
        raise ValueError("Failed to find JSON-like structure in the response.")
    
    try:
        json_data = json.loads(response.content)
        if len(json_data) != 2:
            raise ValueError(f"Expected two items in the JSON array, but found {len(json_data)}.")
        
        first_explanation = json_data[0]['explanation']
        first_rating = json_data[0]['answer']
        second_explanation = json_data[1]['explanation']
        second_rating = json_data[1]['answer']
    except json.JSONDecodeError:
        print("Failed to parse JSON-like structure.")
        print(response.content)
        raise ValueError("Failed to parse JSON-like structure.")
    
    # Calculate confidence based on both ratings
    if first_rating == 'A' and second_rating == 'A':
        confidence = 1.0
    elif first_rating == 'B' or second_rating == 'B':
        confidence = 0.0
    else:
        confidence = 0.5
    
    state["state"]["self_reflection_certainty"] = confidence
    state["state"]["self_reflection_explanations"] = [first_explanation.strip(), second_explanation.strip()]
    state["state"]["self_reflection_ratings"] = [first_rating, second_rating]
    return state
