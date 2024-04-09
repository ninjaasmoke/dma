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
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in words])
    return text


# Load the model
model = joblib.load("logistic_regression_model.joblib")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()

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
    # Preprocess text
    preprocessed_text = preprocess_text(request.text)
    # Vectorize text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # Predict
    prediction = model.predict(vectorized_text)
    # Map prediction to class label
    class_label = "AI-generated" if prediction[0] == 1 else "Human-written"
    return {"prediction": class_label}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
