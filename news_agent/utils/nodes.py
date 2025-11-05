"""
Node functions for the News Analysis Agent
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import fetch_rss_feeds, scrape_article_content
from .state import NewsState
from .prompts import (
    TRUSTWORTHINESS_PROMPT,
    FACT_EXTRACTION_PROMPT,
    REPORT_GENERATION_PROMPT,
    CONTENT_CLEANING_PROMPT
)
import json

# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_output_tokens=2048
)

def gather_news_sources(state: NewsState) -> NewsState:
    """Node 1: Gather news from RSS feeds"""
    print("📰 Gathering news sources...")
    
    default_sources = [
        "https://news.google.com/rss/search?q=technology",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
    ]
    
    sources = state.get("sources", default_sources)
    raw_articles = fetch_rss_feeds(sources)
    
    print(f"✅ Found {len(raw_articles)} articles")
    
    return {
        "sources": sources,
        "raw_articles": raw_articles
    }

def scrape_and_clean(state: NewsState) -> NewsState:
    """Node 2: Scrape full content and clean with LLM"""
    print("🧹 Scraping and cleaning content...")
    
    cleaned = []
    articles_to_process = state["raw_articles"][:8]  # Limit to 8 articles
    
    for idx, article in enumerate(articles_to_process, 1):
        print(f"  Processing article {idx}/{len(articles_to_process)}...")
        
        if 'link' in article:
            content = scrape_article_content(article['link'])
            
            if 'text' in content and len(content['text']) > 200:
                # Use LLM to clean content
                try:
                    prompt = CONTENT_CLEANING_PROMPT.format(
                        content=content['text'][:3000]  # Limit input size
                    )
                    cleaned_text = llm.invoke(prompt).content
                    
                    content['text'] = cleaned_text
                    cleaned.append(content)
                except Exception as e:
                    print(f"  ⚠️  Error cleaning article: {e}")
                    # Use original if cleaning fails
                    cleaned.append(content)
    
    print(f"✅ Cleaned {len(cleaned)} articles")
    
    return {"cleaned_articles": cleaned}

def evaluate_trustworthiness(state: NewsState) -> NewsState:
    """Node 3: Evaluate source credibility using Gemini"""
    print("🔍 Evaluating trustworthiness...")
    
    scores = []
    
    for idx, article in enumerate(state["cleaned_articles"], 1):
        print(f"  Analyzing article {idx}/{len(state['cleaned_articles'])}...")
        
        try:
            prompt = TRUSTWORTHINESS_PROMPT.format(
                title=article.get('title', 'Unknown'),
                authors=', '.join(article.get('authors', [])) or 'Unknown',
                publish_date=article.get('publish_date', 'Unknown'),
                content_preview=article['text'][:800]
            )
            
            response = llm.invoke(prompt)
            
            # Try to parse JSON response
            try:
                score_data = json.loads(response.content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create basic structure
                score_data = {
                    "score": 7.0,
                    "reasoning": response.content[:200],
                    "red_flags": [],
                    "strengths": []
                }
            
            scores.append({
                "article": article['title'],
                "url": article.get('url', ''),
                "evaluation": score_data
            })
            
        except Exception as e:
            print(f"  ⚠️  Error evaluating article: {e}")
            scores.append({
                "article": article['title'],
                "evaluation": {"score": 5.0, "reasoning": "Error during evaluation"}
            })
    
    print(f"✅ Evaluated {len(scores)} articles")
    
    return {"trustworthiness_scores": scores}

def extract_key_facts(state: NewsState) -> NewsState:
    """Node 4: Extract important facts using Gemini"""
    print("📊 Extracting key facts...")
    
    facts = []
    
    for idx, article in enumerate(state["cleaned_articles"], 1):
        print(f"  Extracting from article {idx}/{len(state['cleaned_articles'])}...")
        
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(
                title=article['title'],
                content=article['text'][:4000]  # Limit to stay within token limits
            )
            
            response = llm.invoke(prompt)
            
            facts.append({
                "source": article['title'],
                "url": article.get('url', ''),
                "facts": response.content
            })
            
        except Exception as e:
            print(f"  ⚠️  Error extracting facts: {e}")
    
    print(f"✅ Extracted facts from {len(facts)} articles")
    
    return {"extracted_facts": facts}

def create_final_report(state: NewsState) -> NewsState:
    """Node 5: Generate comprehensive report using Gemini"""
    print("📝 Generating final report...")
    
    # Format facts for the report
    facts_text = "\n\n".join([
        f"### {fact['source']}\n{fact['facts']}" 
        for fact in state["extracted_facts"]
    ])
    
    # Format trustworthiness scores
    scores_text = "\n".join([
        f"- {score['article']}: {score['evaluation'].get('score', 'N/A')}/10"
        for score in state["trustworthiness_scores"]
    ])
    
    try:
        prompt = REPORT_GENERATION_PROMPT.format(
            facts=facts_text,
            scores=scores_text
        )
        
        response = llm.invoke(prompt)
        final_report = response.content
        
        print("✅ Report generated successfully")
        
    except Exception as e:
        print(f"⚠️  Error generating report: {e}")
        final_report = f"Error generating report: {str(e)}\n\nRaw facts:\n{facts_text}"
    
    return {"final_report": final_report}
