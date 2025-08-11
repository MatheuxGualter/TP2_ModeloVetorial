import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

def preprocess_text(doc, use_stemming=False):
    doc = doc.lower()
    doc = unidecode.unidecode(doc)
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = re.sub(r'\d+', '', doc)
    tokens = word_tokenize(doc)
    stop_words_list = stopwords.words('english')
    processed_tokens = [token for token in tokens if token not in stop_words_list]
    doc = ' '.join(processed_tokens)
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc

processed_docs = []
for music in musics:
    with open(f'./Documentos/{music}.txt', 'r', encoding='utf-8') as entrada:
        content = entrada.read()
        processed_content = preprocess_text(content)
        processed_docs.append(processed_content)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)

# 5 CONSULTAS (Ex2)
queries = [
    "Baby I'm losing control in this cruel summer",
    "Sweet love and beautiful things",
    "Million dollar baby, good luck babe",
    "Drinking and dancing in a bar song",
    "I remember everything about us"
]

results_data = []

print("Executando consultas (Sem Stemming)...")
for query_text in queries:
    # Pré-processa a consulta da mesma forma que os documentos
    processed_query = preprocess_text(query_text)

    # Vetoriza a consulta
    query_vector = vectorizer.transform([processed_query])

    # Calcula a similaridade
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Ordena e pega os índices do Top 5
    relevant_indices = np.argsort(similarities)[::-1]
    
    rank_count = 0
    for idx in relevant_indices:
        if similarities[idx] > 0 and rank_count < 5:
            rank_count += 1
            results_data.append({
                "Execução": "Sem Stemming",
                "Consulta": query_text,
                "Rank": rank_count,
                "Documento": musics[idx],
                "Similaridade": similarities[idx]
            })

df_results = pd.DataFrame(results_data)

# SALVANDO OS RESULTADOS NA PLANILHA
try:
    df_results.to_excel("./Ex2/top5.xlsx", sheet_name="Sem Stemming", index=False)
    print("\nArquivo 'top5.xlsx' criado com sucesso.")
except Exception as e:
    print(f"\nErro ao salvar o arquivo Excel: {e}")