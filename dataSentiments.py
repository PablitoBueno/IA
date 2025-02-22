# Instalação das bibliotecas necessárias (caso ainda não estejam instaladas)
!pip install scikit-learn pandas transformers torch matplotlib joblib rake-nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from google.colab import files
from rake_nltk import Rake  # Importa o RAKE

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# -------------------------------
# Upload do JSON com os comentários dos produtos
# -------------------------------
print("Por favor, faça o upload do arquivo JSON com os comentários dos produtos:")
uploaded = files.upload()

# Ler o arquivo JSON carregado
for file_name in uploaded.keys():
    logging.info(f"Arquivo carregado: {file_name}")
    with open(file_name, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

# ------------------------------------
# Configuração dos pipelines de sentimentos
# ------------------------------------
# Pipeline para identificar a emoção específica (ex: joy, anger, sadness, etc.)
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Pipeline para identificar a polaridade (positivo/negativo) usando um modelo pré-treinado
polarity_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Definindo os sentimentos positivos e negativos de forma mais abrangente (para referência)
positive_labels = {"joy", "admiration", "amusement", "approval", "caring", "excitement", "gratitude", "love", "optimism", "pride", "relief"}
negative_labels = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "fear", "grief", "remorse", "sadness"}

# Função de pré-processamento: remove stopwords e retorna os tokens limpos
def limpar_texto(texto):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([texto])
    tokens = vectorizer.get_feature_names_out()
    return " ".join(tokens)

# -------------------------------
# Processar cada produto e armazenar os resultados
# -------------------------------
results = []

for product in test_data:
    product_name = product["product_name"]
    comments = product["comments"]
    
    # Listas para armazenar os resultados de cada comentário
    comment_results = []       # Informações detalhadas de cada comentário
    emotion_labels = []        # Para a distribuição de emoções (modelo Hartmann)
    pos_count = 0              # Contador de comentários POSITIVE (modelo polaridade)
    neg_count = 0              # Contador de comentários NEGATIVE (modelo polaridade)
    
    for comment_obj in comments:
        text = comment_obj["comentario"]
        cleaned = limpar_texto(text)
        
        # Obter a emoção específica com o modelo Hartmann
        try:
            emotion = emotion_pipeline(cleaned)
            emotion_label = emotion[0]['label']
        except Exception as e:
            logging.error(f"Erro ao processar o comentário (emoção): {text}. Erro: {e}")
            emotion_label = "error"
        
        # Obter a polaridade com o modelo SST-2
        try:
            polarity = polarity_pipeline(cleaned)
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
            "polarity": polarity_label
        })
    
    total_comments = len(comments)
    # Distribuição de emoções (do modelo Hartmann)
    emotion_counts = pd.Series(emotion_labels).value_counts().to_dict()
    
    # Ranking score baseado na polaridade (diferença normalizada entre positivos e negativos)
    ranking_score = (pos_count - neg_count) / total_comments if total_comments > 0 else 0

    # Extração de palavras-chave usando RAKE
    # Concatena todos os comentários do produto em um único texto
    all_text = " ".join([comment_obj["comentario"] for comment_obj in comments])
    rake_extractor = Rake()  # Inicializa o RAKE (usa stopwords padrão em inglês)
    rake_extractor.extract_keywords_from_text(all_text)
    # Obtem as frases-chave ranqueadas
    ranked_phrases = rake_extractor.get_ranked_phrases()
    # Seleciona as 3 principais palavras/chaves
    top_keywords = ranked_phrases[:3] if ranked_phrases else []
    
    results.append({
        "product_name": product_name,
        "comments": comment_results,
        "emotion_distribution": emotion_counts,
        "top_keywords": top_keywords,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "ranking_score": ranking_score
    })

# Converter os resultados para um DataFrame (opcional, para visualização)
df_products = pd.DataFrame(results)
logging.info("Resumo dos produtos e suas análises:")
print(df_products[["product_name", "emotion_distribution", "top_keywords", "ranking_score", "pos_count", "neg_count"]])

# ------------------------------------
# Gerar gráficos de pizza para a distribuição de emoções de cada produto
# e incluir as palavras-chave (que descrevem o produto) no título do gráfico
# ------------------------------------
for res in results:
    labels = list(res["emotion_distribution"].keys())
    sizes = list(res["emotion_distribution"].values())
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    title_text = f"{res['product_name']} - Distribuição de Emoções\nPalavras-chave: {', '.join(res['top_keywords'])}"
    plt.title(title_text)
    plt.axis('equal')
    plt.show()

# ------------------------------------
# Gerar Rankings
# ------------------------------------
# Ranking geral: produtos ordenados do melhor para o pior (maior ranking_score primeiro)
ranking_geral = sorted(results, key=lambda x: x["ranking_score"], reverse=True)
print("\nRanking Geral dos Produtos (melhor avaliação):")
for idx, prod in enumerate(ranking_geral, 1):
    print(f"{idx}. {prod['product_name']} - Score: {prod['ranking_score']:.2f} (Positivos: {prod['pos_count']}, Negativos: {prod['neg_count']})")

# Ranking dos produtos com predominância de sentimentos negativos
ranking_negativos = [prod for prod in results if prod["neg_count"] > prod["pos_count"]]
ranking_negativos = sorted(ranking_negativos, key=lambda x: (x["neg_count"] - x["pos_count"]), reverse=True)
print("\nRanking dos Produtos com Predominância de Sentimentos Negativos:")
for idx, prod in enumerate(ranking_negativos, 1):
    diff = prod["neg_count"] - prod["pos_count"]
    print(f"{idx}. {prod['product_name']} - Diferença (Neg - Pos): {diff} (Positivos: {prod['pos_count']}, Negativos: {prod['neg_count']})")
