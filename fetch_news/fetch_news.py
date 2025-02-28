import requests
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["fake_news_detection"]
collection = db["live_news"]

# News API Key (Replace with your actual key)
NEWS_API_KEY = "59593215cd46458c9214ba33b88c2831"

# Search query for fake news-related keywords
QUERY = "fake news OR misinformation OR fact-check OR hoax"

# News API URL
URL = f"https://newsapi.org/v2/everything?q={QUERY}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

def fetch_trending_fake_news():
    try:
        # Fetch data from News API
        response = requests.get(URL)
        response.raise_for_status()  # Raise an error for HTTP issues

        data = response.json()

        if "articles" in data:
            articles = data["articles"]
            news_list = []

            for article in articles:
                news_item = {
                    "title": article["title"],
                    "description": article["description"],
                    "url": article["url"],
                    "published_at": article["publishedAt"],
                    "source": article["source"]["name"],
                }

                # Store in MongoDB (Prevent duplicate links)
                collection.update_one(
                    {"url": news_item["url"]}, {"$set": news_item}, upsert=True
                )

                news_list.append(news_item)
                print(f"Saved: {news_item['title']}")

            return news_list
        else:
            print("No articles found.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

# Run the script
if __name__ == "__main__":
    fetch_trending_fake_news()
