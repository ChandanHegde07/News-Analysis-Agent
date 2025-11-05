import os
from typing import List, Dict
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

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key, 
    temperature=0.3,
    max_output_tokens=2048
)


def gather_news_sources(state: NewsState) -> NewsState:
    """Node 1: Gather news from RSS feeds"""
    print("Gathering news sources...")
    
    default_sources = [
        "https://feeds.theverge.com/rss/index.xml"
    ]
    
    sources = state.get("sources", default_sources)
    raw_articles = fetch_rss_feeds(sources)
    
    print(f"Found {len(raw_articles)} articles")
    
    return {
        "sources": sources,
        "raw_articles": raw_articles
    }


def scrape_and_clean(state: NewsState) -> NewsState:
    """Node 2: Scrape full content and clean with LLM"""
    print("Scraping and cleaning content...")
    
    cleaned = []
    articles_to_process = state["raw_articles"][:8]
    
    for idx, article in enumerate(articles_to_process, 1):
        print(f"  Processing article {idx}/{len(articles_to_process)}...")
        
        if 'link' not in article:
            continue
            
        try:
            content = scrape_article_content(article['link'])
            
            if 'text' in content and len(content['text']) > 200:
                prompt = CONTENT_CLEANING_PROMPT.format(
                    content=content['text'][:3000]
                )
                cleaned_text = llm.invoke(prompt).content
                content['text'] = cleaned_text
                cleaned.append(content)
                
        except Exception as e:
            print(f"Skipping article: {str(e)[:50]}")
    
    print(f"Cleaned {len(cleaned)} articles")
    return {"cleaned_articles": cleaned}


def evaluate_trustworthiness(state: NewsState) -> NewsState:
    """Node 3: Evaluate source credibility using Gemini"""
    print("Evaluating trustworthiness...")
    
    scores = []
    cleaned_articles = state.get("cleaned_articles", [])
    
    for idx, article in enumerate(cleaned_articles, 1):
        print(f"  Analyzing article {idx}/{len(cleaned_articles)}...")
        
        try:
            prompt = TRUSTWORTHINESS_PROMPT.format(
                title=article.get('title', 'Unknown'),
                authors=', '.join(article.get('authors', [])) or 'Unknown',
                publish_date=article.get('publish_date', 'Unknown'),
                content_preview=article.get('text', '')[:800]
            )
            
            response = llm.invoke(prompt)
            
            try:
                score_data = json.loads(response.content)
            except json.JSONDecodeError:
                score_data = {
                    "score": 7.0,
                    "reasoning": response.content[:200],
                    "red_flags": [],
                    "strengths": []
                }
            
            scores.append({
                "article": article.get('title', 'Unknown'),
                "url": article.get('url', ''),
                "evaluation": score_data
            })
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            scores.append({
                "article": article.get('title', 'Unknown'),
                "evaluation": {"score": 5.0, "reasoning": "Error during evaluation"}
            })
    
    print(f"Evaluated {len(scores)} articles")
    return {"trustworthiness_scores": scores}


def extract_key_facts(state: NewsState) -> NewsState:
    """Node 4: Extract important facts using Gemini"""
    print("Extracting key facts...")
    
    facts = []
    cleaned_articles = state.get("cleaned_articles", [])
    
    for idx, article in enumerate(cleaned_articles, 1):
        print(f"  Extracting from article {idx}/{len(cleaned_articles)}...")
        
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(
                title=article.get('title', 'Unknown'),
                content=article.get('text', '')[:4000]
            )
            
            response = llm.invoke(prompt)
            
            facts.append({
                "source": article.get('title', 'Unknown'),
                "url": article.get('url', ''),
                "facts": response.content
            })
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
    
    print(f"Extracted facts from {len(facts)} articles")
    return {"extracted_facts": facts}


def create_final_report(state: NewsState) -> NewsState:
    """Node 5: Generate comprehensive report using Gemini"""
    print("Generating final report...")
    
    facts = state.get("extracted_facts", [])
    scores = state.get("trustworthiness_scores", [])
    
    # Format facts for report
    facts_text = "\n\n".join([
        f"### {fact.get('source', 'Unknown')}\n{fact.get('facts', '')}" 
        for fact in facts
    ])
    
    # Format scores
    scores_text = "\n".join([
        f"- {score.get('article', 'Unknown')}: {score.get('evaluation', {}).get('score', 'N/A')}/10"
        for score in scores
    ])
    
    try:
        prompt = REPORT_GENERATION_PROMPT.format(
            facts=facts_text or "No facts extracted",
            scores=scores_text or "No scores available"
        )
        
        response = llm.invoke(prompt)
        final_report = response.content
        print("Report generated successfully")
        
    except Exception as e:
        print(f"Error generating report: {str(e)[:50]}")
        final_report = f"Error generating report.\n\nRaw facts:\n{facts_text}"
    
    return {"final_report": final_report}
