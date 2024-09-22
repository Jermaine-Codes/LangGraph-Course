from langchain_openai import ChatOpenAI
from models import BSDetectorState
import re
def generate_diverse_outputs(state: BSDetectorState) -> BSDetectorState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
    question = state["state"]["question"]
    
    # Generate diverse outputs (you may want to customize this prompt)
    prompt = f"""Please strictly use the following template to provide your answer:
                Explanation: [insert step-by-step analysis], Answer: [provide your answer], Question: {question}"""
    
    diverse_outputs = []
    for _ in range(5):  # Generate 5 diverse outputs
        response = llm.invoke(prompt)
        diverse_outputs.append(response.content)
    answers = [re.search(r'Answer:\s*(.+)', output).group(1).strip() if re.search(r'Answer:\s*(.+)', output) else '' for output in diverse_outputs]
    state["state"]["diverse_outputs"] = answers
    state["state"]["original_answer"] = state["state"]["original_answer"] or diverse_outputs[0]  # Use the first output as the original answer if not set
    
    return state
