from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from models import BSDetectorState
from nodes.diverse_outputs import generate_diverse_outputs
from nodes.observed_consistency import calculate_observed_consistency
from nodes.self_reflection import perform_self_reflection
from nodes.confidence_aggregation import aggregate_confidence
from langchain_core.messages import HumanMessage

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create the graph
workflow = StateGraph(BSDetectorState)

# Add nodes
workflow.add_node("generate_diverse_outputs", generate_diverse_outputs)
workflow.add_node("calculate_observed_consistency", calculate_observed_consistency)
workflow.add_node("perform_self_reflection", perform_self_reflection)
workflow.add_node("aggregate_confidence", aggregate_confidence)

# Define the edges
workflow.set_entry_point("generate_diverse_outputs")
workflow.add_edge("generate_diverse_outputs", "calculate_observed_consistency")
workflow.add_edge("calculate_observed_consistency", "perform_self_reflection")
workflow.add_edge("perform_self_reflection", "aggregate_confidence")
workflow.add_edge("aggregate_confidence", END)

# Compile the graph
graph = workflow.compile()
question = "Explain three-body problem."
# Run the graph
result = graph.invoke({
    "messages": [HumanMessage(content=question)],
    "state": {
        "question": question,
        "original_answer": "",
        "diverse_outputs": [],
        "observed_consistency": 0.0,
        "self_reflection_certainty": 0.0,
        "final_confidence": 0.0
    }
})

print(result)
