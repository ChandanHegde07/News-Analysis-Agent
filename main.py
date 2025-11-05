import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0.3
)

# Define State
class NewsState(TypedDict):
    urls: list
    scraped_content: dict
    analyses: list
    final_report: str

PROMPTS = {
    "analyze": """Analyze this article and provide:
1. Main topic
2. 3-5 key facts
3. Important entities
4. Summary (2-3 sentences)

Article: {content}""",

    "report": """Create a professional news analysis report:

Analysis 1: {analysis1}
Analysis 2: {analysis2}"""
}

# Node 1: Scrape Content
def scrape_node(state: NewsState) -> NewsState:
    print("[Node 1] Scraping content...")
    
    scraped = {}
    for url in state["urls"][:2]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(["script", "style"]):
                tag.decompose()
            
            text = soup.get_text()
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            scraped[url] = text[:3000]
            print(f"Scraped: {url}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            scraped[url] = ""
    
    return {**state, "scraped_content": scraped}

# Node 2: Analyze Content
def analyze_node(state: NewsState) -> NewsState:
    print("[Node 2] Analyzing content...")
    
    analyses = []
    for url, content in state["scraped_content"].items():
        if content:
            try:
                prompt = PROMPTS["analyze"].format(content=content)
                response = llm.invoke(prompt)
                analyses.append(response.content)
                print(f"Analyzed: {url}")
            except Exception as e:
                print(f"Error analyzing {url}: {e}")
    
    return {**state, "analyses": analyses}

# Node 3: Generate Report
def report_node(state: NewsState) -> NewsState:
    print("[Node 3] Generating report...")
    
    if len(state["analyses"]) < 2:
        report = "Not enough analyses to generate report"
    else:
        try:
            prompt = PROMPTS["report"].format(
                analysis1=state["analyses"][0],
                analysis2=state["analyses"][1]
            )
            response = llm.invoke(prompt)
            report = response.content
            print("Report generated")
        except Exception as e:
            print(f"Error generating report: {e}")
            report = f"Error: {str(e)}"
    
    return {**state, "final_report": report}

# Build the Graph
def create_news_graph():
    workflow = StateGraph(NewsState)
    
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)
    
    workflow.add_edge(START, "scrape")
    workflow.add_edge("scrape", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

# Main
def main():
    print("\n" + "="*60)
    print("NEWS ANALYZER (LangGraph)")
    print("="*60 + "\n")
    
    graph = create_news_graph()
    
    initial_state = {
        "urls": [
            "https://www.theverge.com/ai-artificial-intelligence",
            "https://www.techcrunch.com/tag/artificial-intelligence/",
        ],
        "scraped_content": {},
        "analyses": [],
        "final_report": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["final_report"])
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
