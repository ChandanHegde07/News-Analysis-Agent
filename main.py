import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
import requests

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

PROMPTS = {
    "analyze": """Analyze this article about AI/Technology and provide:
1. Main topic
2. 3-5 key facts
3. Important entities mentioned
4. Summary (2-3 sentences)

Article:
{content}

Provide a structured analysis.""",

    "report": """Create a comprehensive news analysis report:

Analysis 1:
{analysis1}

Analysis 2:
{analysis2}

Provide a professional report combining these analyses."""
}

def scrape_url(url: str) -> str:
    """Scrape content from a URL"""
    print(f"Scraping: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        print(f"Scraped {len(text)} characters")
        return text[:3000]  # Limit to 3000 chars
        
    except Exception as e:
        print(f"Error: {e}")
        return ""

def analyze_content(content: str, title: str) -> str:
    """Analyze article content with LLM"""
    print(f"Analyzing: {title}")
    
    try:
        prompt = PROMPTS["analyze"].format(content=content)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        return "Error analyzing content"

def generate_report(analysis1: str, analysis2: str) -> str:
    """Generate final report"""
    print("Generating report...")
    
    try:
        prompt = PROMPTS["report"].format(
            analysis1=analysis1,
            analysis2=analysis2
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        return "Error generating report"

def main():
    """Main function"""
    print("\n" + "="*60)
    print("SIMPLE NEWS ANALYZER")
    print("="*60 + "\n")
    
    # URLs to analyze
    urls = [
        "https://www.theverge.com/ai-artificial-intelligence",
        "https://www.techcrunch.com/tag/artificial-intelligence/",
    ]
    
    analyses = []
    
    # Scrape and analyze each URL
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] Processing...")
        content = scrape_url(url)
        
        if content:
            analysis = analyze_content(content, url)
            analyses.append(analysis)
            print("Analysis complete")
        else:
            print("Skipped (no content)")
    
    # Generate report
    if len(analyses) >= 2:
        print("\n" + "="*60)
        report = generate_report(analyses[0], analyses[1])
        print("FINAL REPORT")
        print("="*60)
        print(report)
    else:
        print("\nNot enough analyses to generate report")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
