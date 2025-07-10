import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    # Map star ratings to sentiment groups
    negative = probs[0].item() + probs[1].item()     # 1–2 stars
    neutral  = probs[2].item()                       # 3 stars
    positive = probs[3].item() + probs[4].item()     # 4–5 stars

    return {
        "Positive": positive,
        "Negative": negative,
        "Neutral": neutral
    }



