import os
import logging
import requests
from pymongo import MongoClient

# ========== 🔹 Configure Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== 🔹 Load Environment Variables ==========
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    logging.error("❌ Missing News API Key. Set it as an environment variable.")
    exit(1)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

try:
    client = MongoClient(MONGO_URI)
    db = client["fake_news_detection"]
    collection = db["live_news"]
    collection.create_index("url", unique=True)  # Ensure unique URLs
    logging.info("✅ Connected to MongoDB.")
except ConnectionError as e:
    logging.error(f"❌ MongoDB Connection Failed: {e}")
    exit(1)

# ========== 🔹 News API Configuration ==========
QUERY = "fake news OR misinformation OR fact-check OR hoax"
URL = f"https://newsapi.org/v2/everything?q={QUERY}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

def fetch_trending_fake_news():
    """Fetch trending fake news articles from NewsAPI and store them in MongoDB."""
    try:
        # Fetch data from News API with timeout & error handling
        response = requests.get(URL, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()

        if "articles" not in data:
            logging.warning("⚠️ No articles found in API response.")
            return []

        articles = data["articles"]
        news_list = []

        for article in articles:
            news_item = {
                "title": article.get("title", "No Title"),
                "description": article.get("description", "No Description"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "source": article["source"].get("name", "Unknown Source"),
            }

            # Avoid duplicate news storage
            if news_item["url"]:
                collection.update_one({"url": news_item["url"]}, {"$set": news_item}, upsert=True)
                news_list.append(news_item)
                logging.info(f"✅ Saved: {news_item['title']}")

        return news_list

    except requests.exceptions.Timeout:
        logging.error("❌ Request Timed Out.")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error fetching data: {e}")
        return []

# Run the script
if __name__ == "__main__":
    fetch_trending_fake_news()
