import streamlit as st
import requests
import matplotlib.pyplot as plt

# ========== 🔹 Configuration ==========
FASTAPI_URL = "https://fake-news-detection-tjo8.onrender.com"  # Change this if deploying backend

# ========== 🔹 Streamlit UI ==========
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection System")
st.markdown("**Verify news articles, subscribe for alerts, and provide feedback.**")

# ✅ News Verification Section
st.header("🔍 Verify News")
news_text = st.text_area("Enter a news article or statement:", height=150)

col1, col2 = st.columns([1, 1])
with col1:
    check_news_btn = st.button("Check News", key="check_news", disabled=not news_text.strip())
with col2:
    refresh_btn = st.button("🔄 Refresh", key="refresh")

if refresh_btn:
    st.experimental_rerun()

if check_news_btn:
    with st.spinner("Analyzing... ⏳"):
        try:
            response = requests.post(f"{FASTAPI_URL}/verify-news", json={"text": news_text})
            response.raise_for_status()  # Raise error for bad responses
            result = response.json()

            st.success(f"🟢 **Result: {result.get('label', 'Unknown')}** (Confidence: {result.get('confidence', 'N/A')})")

            # Display Explanation Graph
            if isinstance(result.get("explanation"), dict):  # Ensure explanation is in dict format
                shap_values = result["explanation"]
                words = list(shap_values.keys())
                values = list(shap_values.values())

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(words, values, color=['green' if v > 0 else 'red' for v in values])
                ax.set_xlabel("SHAP Value (Impact on Prediction)")
                ax.set_title("🔎 SHAP Explanation for News Classification")
                ax.invert_yaxis()  # Invert so highest impact appears at the top

                # Display the plot in Streamlit
                st.pyplot(fig)
            else:
                st.warning("⚠️ SHAP explanation not available for this text.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

st.markdown("---")

# ✅ Subscribe for SMS Alerts
st.header("📲 Subscribe for Fake News Alerts")
phone_number = st.text_input("Enter your phone number:")

if st.button("Subscribe", key="subscribe", disabled=not phone_number.strip()):
    with st.spinner("Subscribing... ⏳"):
        try:
            response = requests.post(f"{FASTAPI_URL}/subscribe", data={"phone_number": phone_number})
            response.raise_for_status()
            st.success("✅ Successfully subscribed for alerts!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Subscription failed: {e}")

st.markdown("---")

# ✅ Submit Feedback
st.header("💬 Submit Feedback")
feedback_text = st.text_area("Enter your feedback:", height=100)

if st.button("Submit Feedback", key="feedback", disabled=not (feedback_text.strip() and news_text.strip())):
    with st.spinner("Submitting feedback... ⏳"):
        try:
            response = requests.post(f"{FASTAPI_URL}/feedback", data={"news_text": news_text, "feedback": feedback_text})
            response.raise_for_status()
            st.success("✅ Feedback submitted successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error submitting feedback: {e}")

st.markdown("📢 *Thank you for using the Fake News Detection System!* 🚀")
