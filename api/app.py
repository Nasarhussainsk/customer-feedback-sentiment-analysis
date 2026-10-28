from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.data.preprocess import clean_text
 
app = FastAPI()
 
model = joblib.load(
    "models/sentiment_model.pkl"
)
 
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)
 
class Review(BaseModel):
    text: str
 
@app.post("/predict")
def predict(review: Review):
 
    X = vectorizer.transform(
        [clean_text(review.text)]
    )
 
    sentiment = model.predict(X)[0]
 
    return {
        "sentiment": sentiment
    }