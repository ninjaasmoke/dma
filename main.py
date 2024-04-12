import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
from io import BytesIO


import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Lemmatization
lemmatizer = WordNetLemmatizer()

def load_from_url(url):
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print("Error fetching data from URL:", e)
        return None


# Define a comprehensive text preprocessing function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    # Removal of non-word characters
    text = re.sub(r"\W", " ", text)
    # Removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    text = " ".join([word for word in words if word not in stop_words])
    words = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in words])
    return text


# Load the model
log_model_tfidf = joblib.load("logistic_regression_model.joblib")
# log_model_ngrams = load_from_url("https://github.com/ninjaasmoke/data-mining/releases/download/some-tag/logistic_regression_model_ngrams.joblib")

nb_model_tfidf = joblib.load("naive_bayes_model_tfidf.joblib")
# nb_model_ngrams = load_from_url("https://github.com/ninjaasmoke/data-mining/releases/download/some-tag/naive_bayes_model_ngrams.joblib")


# Vectorize text using TF-IDF
vectorizer_tfidf = joblib.load('tfidf_vectorizer.joblib')
# vectorizer_ngrams = load_from_url("https://github.com/ninjaasmoke/data-mining/releases/download/some-tag/vectorizer_ngram.joblib")


# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define request body model
class TextRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index():
    print("Index page")
    return FileResponse("static/index.html")


# Define endpoint
@app.post("/logisticClassifier")
async def classify_text(request: TextRequest):
    print("Classifying text...")
    print(request.text)
    # Preprocess text
    preprocessed_text = preprocess_text(request.text)
    # Vectorize text
    vectorized_text = vectorizer_tfidf.transform([preprocessed_text])
    # Predict
    prediction = log_model_tfidf.predict(vectorized_text)
    # Map prediction to class label
    class_label = "AI-generated" if prediction[0] == 1 else "Human-written"
    return {"prediction": class_label}


@app.post("/logisticClassifierNgrams")
async def classify_text(request: TextRequest):
    print("Classifying text...")
    print(request.text)
    # Preprocess text
    preprocessed_text = preprocess_text(request.text)
    # Vectorize text
    vectorized_text = vectorizer_ngrams.transform([preprocessed_text])
    # Predict
    prediction = log_model_ngrams.predict(vectorized_text)
    # Map prediction to class label
    class_label = "AI-generated" if prediction[0] == 1 else "Human-written"
    return {"prediction": class_label}

@app.post("/naiveBayesClassifier")
async def classify_text(request: TextRequest):
    print("Classifying text...")
    print(request.text)
    # Preprocess text
    preprocessed_text = preprocess_text(request.text)
    # Vectorize text
    vectorized_text = vectorizer_tfidf.transform([preprocessed_text])
    # Predict
    prediction = nb_model_tfidf.predict(vectorized_text)
    # Map prediction to class label
    class_label = "AI-generated" if prediction[0] == 1 else "Human-written"
    return {"prediction": class_label}

@app.post("/naiveBayesClassifierNgrams")
async def classify_text(request: TextRequest):
    print("Classifying text...")
    print(request.text)
    # Preprocess text
    preprocessed_text = preprocess_text(request.text)
    # Vectorize text
    vectorized_text = vectorizer_ngrams.transform([preprocessed_text])
    # Predict
    prediction = nb_model_ngrams.predict(vectorized_text)
    # Map prediction to class label
    class_label = "AI-generated" if prediction[0] == 1 else "Human-written"
    return {"prediction": class_label}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
