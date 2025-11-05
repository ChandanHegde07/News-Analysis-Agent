from langgraph.graph import StateGraph, START, END
from .utils.state import NewsState
from .utils.nodes import (
    gather_news_sources,
    scrape_and_clean,
    evaluate_trustworthiness,
    extract_key_facts,
    create_final_report
)

def create_news_agent():
    """Construct the LangGraph workflow"""
    
    # Initialize graph with state
    workflow = StateGraph(NewsState)
    
    # Add nodes
    workflow.add_node("gather_sources", gather_news_sources)
    workflow.add_node("clean_content", scrape_and_clean)
    workflow.add_node("evaluate_trust", evaluate_trustworthiness)
    workflow.add_node("extract_facts", extract_key_facts)
    workflow.add_node("generate_report", create_final_report)
    
    # Define edges (linear flow)
    workflow.add_edge(START, "gather_sources")
    workflow.add_edge("gather_sources", "clean_content")
    workflow.add_edge("clean_content", "evaluate_trust")
    workflow.add_edge("evaluate_trust", "extract_facts")
    workflow.add_edge("extract_facts", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Compile the graph
    return workflow.compile()
