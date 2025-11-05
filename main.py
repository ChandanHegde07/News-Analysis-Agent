from news_agent.agent import create_news_agent
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create and run agent
    agent = create_news_agent()
    
    # Execute workflow
    result = agent.invoke({
        "query": "latest technology news",
        "sources": [
            "https://news.google.com/rss/search?q=AI+technology",
            "https://feeds.bbci.co.uk/news/technology/rss.xml"
        ]
    })
    
    # Print final report
    print("\n" + "="*50)
    print("NEWS ANALYSIS REPORT")
    print("="*50 + "\n")
    print(result["final_report"])

if __name__ == "__main__":
    main()
