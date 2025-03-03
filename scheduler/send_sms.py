import os
import logging
from twilio.rest import Client
from pymongo import MongoClient
from dotenv import load_dotenv

# ========== 🔹 Load Environment Variables ==========
load_dotenv()

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ========== 🔹 Configure Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== 🔹 Validate Twilio Credentials ==========
if not all([TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    logging.error("❌ Missing Twilio credentials. Please set environment variables correctly.")
    raise ValueError("Twilio credentials missing")

# ========== 🔹 MongoDB Connection ==========
try:
    client = MongoClient(MONGO_URI)
    db = client["fake_news_detection"]  # Your MongoDB database name
    news_collection = db["live_news"]  # Collection for flagged fake news
    users_collection = db["subscribed_users"]  # Collection for storing user phone numbers
    logging.info("✅ Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise e

def send_twilio_sms():
    try:
        # Fetch the latest 3 flagged fake news items, sorted by latest
        flagged_news = list(news_collection.find({"is_fake": True}, {"title": 1, "source": 1}).sort("_id", -1).limit(3))

        if not flagged_news:
            logging.info("No flagged fake news to send.")
            return

        # Format the message body (limit SMS to avoid truncation)
        message_body = "🚨 Fake News Alert 🚨\n"
        for news in flagged_news:
            message_body += f"- {news.get('title', 'Unknown Title')} ({news.get('source', 'Unknown Source')})\n"

        message_body = message_body[:1600]  # Truncate to Twilio's limit

        # Fetch user phone numbers
        phone_numbers = [user.get("phone_number") for user in users_collection.find({}, {"phone_number": 1})]

        if not phone_numbers:
            logging.info("No users subscribed to SMS alerts.")
            return

        # Initialize Twilio Client
        twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

        # Send SMS to each user
        for number in phone_numbers:
            try:
                message = twilio_client.messages.create(
                    body=message_body,
                    from_=TWILIO_PHONE_NUMBER,
                    to=number
                )
                logging.info(f"✅ Sent SMS to {number}, SID: {message.sid}")
            except Exception as e:
                logging.error(f"❌ Failed to send SMS to {number}: {e}")

    except Exception as e:
        logging.error(f"❌ Error in send_twilio_sms: {e}")
        raise e

if __name__ == "__main__":
    send_twilio_sms()
