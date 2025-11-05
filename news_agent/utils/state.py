from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages

class NewsState(TypedDict):
    """Shared state that flows through all nodes"""
    query: str
    sources: List[str]
    raw_articles: List[Dict]
    cleaned_articles: List[Dict]
    trustworthiness_scores: List[Dict]
    extracted_facts: List[str]
    final_report: str
    messages: Annotated[list, add_messages]
