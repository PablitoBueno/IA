
# Installation and Required Downloads
!pip install fastapi uvicorn scikit-learn pandas transformers torch joblib rake-nltk nltk

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from rake_nltk import Rake

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Data models for FastAPI
class Comment(BaseModel):
    comment: str
    date: Optional[str] = None

class Product(BaseModel):
    product_name: str
    comments: List[Comment]

class AnalysisRequest(BaseModel):
    products: List[Product]

# Class for Sentiment Analysis
class ReviewAnalyzer:
    def __init__(self, data):
        """
        Initializes the pipelines and stores the data.
        :param data: List of products with comments (JSON format).
        """
        self.data = data
        self.results = []
        # Pipeline for detecting specific emotions
        self.emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        # Pipeline for detecting polarity (positive/negative)
        self.polarity_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def clean_text(self, text):
        """
        Removes stopwords using CountVectorizer and returns cleaned text.
        :param text: Original text.
        :return: Cleaned text with tokens.
        """
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        tokens = vectorizer.get_feature_names_out()
        return " ".join(tokens)

    def process_data(self):
        """
        Processes each product, extracting emotions, polarity, keywords, and calculating metrics.
        """
        for product in self.data:
            product_name = product["product_name"]
            comments = product["comments"]

            # Lists and counters for each product
            comment_results = []
            emotion_labels = []
            pos_count = 0
            neg_count = 0

            for comment_obj in comments:
                text = comment_obj["comment"]
                cleaned = self.clean_text(text)

                # Get specific emotion using Hartmann's model
                try:
                    emotion = self.emotion_pipeline(cleaned)
                    emotion_label = emotion[0]['label']
                except Exception as e:
                    logging.error(f"Error processing comment (emotion): {text}. Error: {e}")
                    emotion_label = "error"

                # Get polarity using the SST-2 model
                try:
                    polarity = self.polarity_pipeline(cleaned)
                    polarity_label = polarity[0]['label']  # "POSITIVE" or "NEGATIVE"
                except Exception as e:
                    logging.error(f"Error processing comment (polarity): {text}. Error: {e}")
                    polarity_label = "error"

                if polarity_label.upper() == "POSITIVE":
                    pos_count += 1
                elif polarity_label.upper() == "NEGATIVE":
                    neg_count += 1

                emotion_labels.append(emotion_label)
                comment_results.append({
                    "original": text,
                    "cleaned": cleaned,
                    "emotion": emotion_label,
                    "polarity": polarity_label,
                    "date": comment_obj.get("date", "")
                })

            total_comments = len(comments)
            emotion_counts = pd.Series(emotion_labels).value_counts().to_dict()
            ranking_score = (pos_count - neg_count) / total_comments if total_comments > 0 else 0

            # Keyword extraction using RAKE
            all_text = " ".join([c["comment"] for c in comments])
            rake_extractor = Rake()  # Uses RAKE default stopwords
            rake_extractor.extract_keywords_from_text(all_text)
            ranked_phrases = rake_extractor.get_ranked_phrases()
            top_keywords = ranked_phrases[:3] if ranked_phrases else []

            self.results.append({
                "product_name": product_name,
                "emotion_distribution": emotion_counts,
                "top_keywords": top_keywords,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "ranking_score": ranking_score,
                "comments": comment_results
            })

    def get_results(self):
        """
        Returns the analysis results.
        """
        return self.results

# FastAPI instance
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# Sentiment analysis endpoint
@app.post("/analyze")
def analyze_reviews(request: AnalysisRequest):
    try:
        # Convert request data to expected format (list of dictionaries)
        data = [product.dict() for product in request.products]
        analyzer = ReviewAnalyzer(data)
        analyzer.process_data()
        results = analyzer.get_results()
        return {"results": results}
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Error during comment analysis.")

# Root endpoint for testing
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API. Use the /analyze endpoint to submit data."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
    
