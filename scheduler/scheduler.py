import schedule
import time
import threading
from send_sms import send_twilio_sms
from retrain_model import retrain_model
from fetch_news import fetch_news  # Import fetching function

# Schedule fetching of trending fake news every 3 hours
schedule.every(3).hours.do(fetch_news)

# Schedule SMS sending daily at 9 AM
schedule.every().day.at("09:00").do(send_twilio_sms)

# Schedule model retraining every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(retrain_model)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait for 1 min before checking again

# Run in a separate thread to keep the main program responsive
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Keep the main thread running
while True:
    time.sleep(1)
