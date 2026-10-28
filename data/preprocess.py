import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import time
import spacy

from tqdm import tqdm


try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
 
analyzer = SentimentIntensityAnalyzer()
 
 
def create_sentiment(text):
 
    score = analyzer.polarity_scores(str(text))["compound"]
 
    if score >= 0.05:
        return "positive"
 
    elif score <= -0.05:
        return "negative"
 
    else:
        return "neutral"
    


nlp = spacy.load("en_core_web_sm")

# Negation words carry sentiment, so never remove them
KEEP = {"not", "no", "never", "nor", "none", "nt"}


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    doc = nlp(
        text,
        disable=["parser", "ner"]
    )

    words = []

    for token in doc:

        if token.lemma_ in KEEP or not token.is_stop:
            words.append(token.lemma_)

    return " ".join(words)


def clean_batch(texts):

    start_time = time.time()

    cleaned = []

    total_rows = len(texts)

    print(f"\nTotal reviews to clean: {total_rows:,}")
    print("Starting text cleaning...\n")

    docs = nlp.pipe(
        (str(text) for text in texts),
        disable=["parser", "ner"],
        batch_size=200
    )

    for doc in tqdm(
        docs,
        total=total_rows,
        desc="Cleaning Reviews",
        unit="rows"
    ):

        words = [
            t.lemma_ for t in doc
            if t.lemma_ in KEEP or not t.is_stop
        ]

        cleaned.append(" ".join(words))

    end_time = time.time()

    total_time = end_time - start_time

    print("\nText Cleaning Completed!")

    print(f"Total rows cleaned: {len(cleaned):,}")

    print(f"Total time: {total_time:.2f} seconds")

    print(f"Total time: {total_time / 60:.2f} minutes")

    return cleaned
