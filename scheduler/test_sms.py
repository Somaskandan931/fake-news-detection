from twilio.rest import Client

# Twilio credentials (replace these with your actual values)
TWILIO_ACCOUNT_SID = "ACff2a6eafe96c4ed11edebdc9d485bc51"
TWILIO_AUTH_TOKEN = "ea13afdab903ef5d98e064199009af6c"
TWILIO_PHONE_NUMBER = "+18576666048"
RECIPIENT_PHONE_NUMBER = "+917200288242"


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
