import feedparser
from newspaper import Article
import requests
from typing import List, Dict

def fetch_rss_feeds(feed_urls: List[str]) -> List[Dict]:
    """Fetch articles from RSS feeds"""
    articles = []
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:  # Limit to 5 per source
            articles.append({
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'source': feed_url
            })
    return articles

def scrape_article_content(url: str) -> Dict:
    """Scrape full article content"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': str(article.publish_date),
            'url': url
        }
    except Exception as e:
        return {'error': str(e), 'url': url}
