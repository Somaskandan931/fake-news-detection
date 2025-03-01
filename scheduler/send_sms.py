from twilio.rest import Client
from pymongo import MongoClient
import os

# Load Twilio credentials from environment variables
TWILIO_SID = os.getenv("your sid here")
TWILIO_AUTH_TOKEN = os.getenv("your auth token here")
TWILIO_PHONE_NUMBER = os.getenv("your twilio phone number here")

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["fake_news_detection"]  # Replace with your database name
news_collection = db["live_news"]  # Collection for fake news
users_collection = db["subscribed_users"]  # Collection for storing user phone numbers

def send_twilio_sms():
    # Fetch the latest 3 flagged fake news items
    flagged_news = list(news_collection.find({"is_fake": True}, {"title": 1, "source": 1}).limit(3))

    if not flagged_news:
        print("No flagged news to send.")
        return

    # Format the message body
    message_body = "🚨 Fake News Alert 🚨\n"
    for news in flagged_news:
        message_body += f"- {news.get('title', 'Unknown Title')} ({news.get('source', 'Unknown Source')})\n"

    # Fetch user phone numbers
    phone_numbers = [user["phone_number"] for user in users_collection.find({}, {"phone_number": 1})]

    if not phone_numbers:
        print("No users subscribed to SMS alerts.")
        return

    # Twilio SMS sending
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

    for number in phone_numbers:
        try:
            message = client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=number
            )
            print(f"✅ Sent SMS to {number}, SID: {message.sid}")
        except Exception as e:
            print(f"❌ Failed to send SMS to {number}: {str(e)}")

if __name__ == "__main__":
    send_twilio_sms()
