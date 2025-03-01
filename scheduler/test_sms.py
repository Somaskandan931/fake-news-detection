from twilio.rest import Client

# Twilio credentials (replace these with your actual values)
TWILIO_ACCOUNT_SID = "your sid here"
TWILIO_AUTH_TOKEN = "your auth token here"
TWILIO_PHONE_NUMBER = "your twilio phone number here"
RECIPIENT_PHONE_NUMBER = "the recipient number here"


def send_test_sms () :
    try :
        # Initialize Twilio client
        client = Client( TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN )

        # Send SMS
        message = client.messages.create(
            body="Hello! This is a test message from Twilio.",
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )

        print( f"✅ SMS sent successfully! Message SID: {message.sid}" )

    except Exception as e :
        print( f"❌ Failed to send SMS: {e}" )


# Run the test function
send_test_sms()
