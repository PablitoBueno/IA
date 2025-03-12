# -*- coding: utf-8 -*-
"""sentimentalAnalysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ue84dve8x1MNrOe0u9td3xysZyx5n2Dm
"""

# Instalação e Downloads Necessários
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

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Modelos de dados para o FastAPI
class Comment(BaseModel):
    comentario: str
    data: Optional[str] = None

class Product(BaseModel):
    product_name: str
    comments: List[Comment]

class AnalysisRequest(BaseModel):
    products: List[Product]

# Classe para Análise de Comentários (Sentimentos)
class ReviewAnalyzer:
    def __init__(self, data):
        """
        Inicializa os pipelines e armazena os dados.
        :param data: Lista de produtos com comentários (formato JSON).
        """
        self.data = data
        self.results = []
        # Pipeline para identificar emoções específicas
        self.emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        # Pipeline para identificar a polaridade (positivo/negativo)
        self.polarity_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def limpar_texto(self, texto):
        """
        Remove stopwords usando CountVectorizer e retorna o texto limpo.
        :param texto: Texto original.
        :return: Texto limpo com tokens.
        """
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([texto])
        tokens = vectorizer.get_feature_names_out()
        return " ".join(tokens)

    def process_data(self):
        """
        Processa cada produto, extraindo emoções, polaridade, palavras-chave e calculando métricas.
        """
        for product in self.data:
            product_name = product["product_name"]
            comments = product["comments"]

            # Listas e contadores para cada produto
            comment_results = []
            emotion_labels = []
            pos_count = 0
            neg_count = 0

            for comment_obj in comments:
                text = comment_obj["comentario"]
                cleaned = self.limpar_texto(text)

                # Obter a emoção específica usando o modelo Hartmann
                try:
                    emotion = self.emotion_pipeline(cleaned)
                    emotion_label = emotion[0]['label']
                except Exception as e:
                    logging.error(f"Erro ao processar o comentário (emoção): {text}. Erro: {e}")
                    emotion_label = "error"

                # Obter a polaridade usando o modelo SST-2
                try:
                    polarity = self.polarity_pipeline(cleaned)
                    polarity_label = polarity[0]['label']  # "POSITIVE" ou "NEGATIVE"
                except Exception as e:
                    logging.error(f"Erro ao processar o comentário (polaridade): {text}. Erro: {e}")
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
                    "data": comment_obj.get("data", "")
                })

            total_comments = len(comments)
            emotion_counts = pd.Series(emotion_labels).value_counts().to_dict()
            ranking_score = (pos_count - neg_count) / total_comments if total_comments > 0 else 0

            # Extração de palavras-chave usando RAKE
            all_text = " ".join([c["comentario"] for c in comments])
            rake_extractor = Rake()  # Usa os stopwords padrão do RAKE
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
        Retorna os resultados da análise.
        """
        return self.results

# Instanciando o FastAPI
app = FastAPI(title="API de Análise de Comentários", version="1.0")

# Endpoint para análise de sentimentos
@app.post("/analyze")
def analyze_reviews(request: AnalysisRequest):
    try:
        # Converte os dados do request para o formato esperado (lista de dicionários)
        data = [product.dict() for product in request.products]
        analyzer = ReviewAnalyzer(data)
        analyzer.process_data()
        results = analyzer.get_results()
        return {"results": results}
    except Exception as e:
        logging.error(f"Erro durante a análise: {e}")
        raise HTTPException(status_code=500, detail="Erro durante a análise dos comentários.")

# Endpoint raiz para teste
@app.get("/")
def root():
    return {"message": "API de Análise de Comentários. Utilize o endpoint /analyze para enviar os dados."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)