import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI setup
st.set_page_config(page_title="Sentiment Detector", layout="centered")
st.title("💬 Review Sentiment Classifier")
st.write("This app predicts whether a customer review is **Positive** or **Negative**.")

# Input text box
review = st.text_area("✍️ Enter customer review here:", height=150)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please type a review first.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        probability = model.predict_proba(review_vector)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Positive Sentiment ({probability*100:.2f}% confidence)")
        else:
            st.error(f"❌ Negative Sentiment ({probability*100:.2f}% confidence)")

sentiment-review-app/
│
├── app.py
├── sentiment_model.pkl        ✅ Trained logistic regression model
├── tfidf_vectorizer.pkl       ✅ Saved TfidfVectorizer
├── requirements.txt           ✅ Dependency list

streamlit
scikit-learn
pandas
textblob
joblib

