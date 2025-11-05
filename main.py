import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv()

from news_agent.agent import create_news_agent


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Default news sources
DEFAULT_SOURCES = [
    "https://news.google.com/rss/search?q=AI+technology",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
]

CUSTOM_SOURCES = [
    "https://news.google.com/rss/search?q=machine+learning",
    "https://feeds.bbci.co.uk/news/world/rss.xml"
]


def run_analysis(sources, query):
    """Execute the news analysis workflow"""
    try:
        # Verify API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        
        logger.info(f"Starting analysis with {len(sources)} sources")
        
        # Create and run agent
        agent = create_news_agent()
        result = agent.invoke({"query": query, "sources": sources})
        
        # Display results
        print("\n" + "="*60)
        print("NEWS ANALYSIS REPORT")
        print("="*60 + "\n")
        print(result["final_report"])
        
        # Show statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Articles Gathered: {len(result.get('raw_articles', []))}")
        print(f"Articles Cleaned: {len(result.get('cleaned_articles', []))}")
        print(f"Trustworthiness Scores: {len(result.get('trustworthiness_scores', []))}")
        print(f"Facts Extracted: {len(result.get('extracted_facts', []))}")
        print("="*60 + "\n")
        
        logger.info("Analysis completed successfully")
        return 0
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}", exc_info=True)
        print(f"Unexpected error: {e}")
        print("Check news_agent.log for details")
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        logger.info("Running with custom sources")
        return run_analysis(CUSTOM_SOURCES, "machine learning and world news")
    else:
        logger.info("Running with default sources")
        return run_analysis(DEFAULT_SOURCES, "latest technology news")


if __name__ == "__main__":
    sys.exit(main())
