# services/news_fetcher.py
import requests, os, datetime, json
from services.embeddings import index_text

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

def fetch_news(query="supply chain disruption", limit=10):
    """
    Fetches live news articles using NewsAPI.org
    (requires API key) or fallback demo feed.
    """
    if not NEWS_API_KEY:
        # fallback sample
        return [{"title": f"{query} example headline", "description": "Demo article body"}]

    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    data = res.json()
    articles = data.get("articles", [])
    return [{"title": a["title"], "description": a["description"]} for a in articles]

def store_news(query="supply chain disruption"):
    news = fetch_news(query)
    docs = [{"id": f"news_{i}", "text": n["title"] + "\n" + n.get("description", "")} for i, n in enumerate(news)]
    index_text(docs, namespace="news_data")
    json.dump(news, open("data/json_store/news_cache.json","w"), indent=2)
    return len(news)
