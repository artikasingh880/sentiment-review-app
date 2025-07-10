import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from bert_model1 import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis with BERT", layout="centered")

st.markdown("## BERT Sentiment Analysis App")
st.markdown("Enter a product review below. This app will analyze its sentiment.")

# Input box
user_input = st.text_area("Enter your review here:", height=150)

# Button row
col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("Analyze")
with col2:
    reset = st.button("Clear")

# On Analyze
if analyze:
    if user_input.strip():
        # Get predictions
        sentiment_probs = predict_sentiment(user_input)
        pos = sentiment_probs["Positive"]
        neg = sentiment_probs["Negative"]
        neu = sentiment_probs["Neutral"]  

#  Word Cloud 
st.markdown("###  Word Cloud")
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["product", "amazon", "buy", "purchase", "item"])
wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',
    stopwords=custom_stopwords,
    max_words=100,
    contour_color='steelblue',
    contour_width=1,
    random_state=42
).generate(user_input)
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wc, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

# 💬 Show the user's input under sentiment category
st.markdown("### 💬 Review Categorization Based on Sentiment")

if pos > neg and pos > neu:
    st.success("✅ **Positive Statement**")
    st.write(user_input)
elif neg > pos and neg > neu:
    st.error("❌ **Negative Statement**")
    st.write(user_input)
else:
    st.info("😐 **Neutral Statement**")
    st.write(user_input)

        # ✅ Sentiment Result
        if pos > neg and pos > neu:
            st.success("😊 Sentiment: Positive (100%)")
        elif neg > pos and neg > neu:
            st.error("😠 Sentiment: Negative (100%)")
        else:
            st.info("😐 Sentiment: Neutral (100%)")

        # 🥧 Pie Chart
        st.markdown("### 🥧 Sentiment Pie Chart")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [pos, neg, neu],
            labels=[" Positive", " Negative", " Neutral"],
            autopct='%1.1f%%',
            startangle=90,
            colors=["green", "red", "yellow"]
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # 📊 Bar Chart
        st.markdown("### Sentiment Bar Chart")
        fig2, ax2 = plt.subplots()
        ax2.bar(["Positive", "Negative", "Neutral"], [pos*100, neg*100, neu*100], color=["green", "red", "gray"])
        ax2.set_ylabel("Confidence (%)")
        ax2.set_ylim(0, 100)
        for i, val in enumerate([pos, neg, neu]):
            ax2.text(i, val * 100 + 1, f"{val*100:.1f}%", ha='center')
        st.pyplot(fig2)

        # 📝 Feedback Section
        st.markdown("---")
        st.markdown("### Leave Feedback (Optional)")
        name = st.text_input("Your Name:")
        comments = st.text_area("Your Feedback:")
        if st.button("Submit Feedback"):
            if comments.strip():
                feedback = pd.DataFrame([[name, comments]], columns=["Name", "Feedback"])
                try:
                    feedback.to_csv("feedback.csv", mode='a', index=False, header=False)
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error("Error saving feedback.")
            else:
                st.warning("Please enter some feedback before submitting.")
    else:
        st.warning("Please enter a review.")

# Reset the form
if reset:
    st.rerun()


