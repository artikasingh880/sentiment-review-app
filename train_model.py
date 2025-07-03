import pandas as pd
import joblib
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Dataset.csv")

from textblob import TextBlob

# Label reviews as positive (1) if polarity > 0, else negative (0)
df['sentiment_label'] = df['body'].apply(lambda x: 1 if TextBlob(str(x)).sentiment.polarity > 0 else 0)


# Classify into Positive or Negative
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 1 if x > 0.1 else 0)  # 1 = Positive, 0 = Negative

# Feature and label
X = df['body']
y = df['sentiment_label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model & vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
