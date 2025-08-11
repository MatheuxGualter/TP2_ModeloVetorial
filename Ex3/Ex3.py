import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os 

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

def preprocess_text(doc, use_stemming=False):
    """
    Realiza o pré-processamento de um texto com opção de aplicar stemming.
    """
    stemmer = PorterStemmer()
    
    doc = doc.lower()
    doc = unidecode.unidecode(doc)
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = re.sub(r'\d+', '', doc)
    tokens = word_tokenize(doc)
    stop_words_list = stopwords.words('english')
    
    processed_tokens = [token for token in tokens if token not in stop_words_list]

    if use_stemming: #etapa de stemming
        processed_tokens = [stemmer.stem(token) for token in processed_tokens]

    doc = ' '.join(processed_tokens)
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc

# PRÉ-PROCESSAMENTO DOS DOCUMENTOS COM STEMMING
print("Iniciando pré-processamento dos documentos com Stemming...")
processed_docs_stemmed = []
for music in musics:
    with open(f'./Documentos/{music}.txt', 'r', encoding='utf-8') as entrada:
        content = entrada.read()
        processed_content = preprocess_text(content, use_stemming=True)
        processed_docs_stemmed.append(processed_content)

vectorizer_stemmed = TfidfVectorizer()
tfidf_matrix_stemmed = vectorizer_stemmed.fit_transform(processed_docs_stemmed)

queries = [
    "Baby I'm losing control in this cruel summer",
    "Sweet love and beautiful things",
    "Million dollar baby, good luck babe",
    "Drinking and dancing in a bar song",
    "I remember everything about us"
]


results_data_stemmed = []

print("Executando consultas (Com Stemming)...")
for query_text in queries:
    # Pré-processa a consulta com stemming
    processed_query = preprocess_text(query_text, use_stemming=True)
    query_vector = vectorizer_stemmed.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix_stemmed).flatten()
    relevant_indices = np.argsort(similarities)[::-1]
    
    rank_count = 0
    for idx in relevant_indices:
        if similarities[idx] > 0 and rank_count < 5:
            rank_count += 1
            results_data_stemmed.append({
                "Execução": "Com Stemming",
                "Consulta": query_text,
                "Rank": rank_count,
                "Documento": musics[idx],
                "Similaridade": similarities[idx]
            })

# ADICIONANDO RESULTADOS À PLANILHA EXISTENTE (Ex3)
df_results_stemmed = pd.DataFrame(results_data_stemmed)
output_filename = "./Ex3/top5.xlsx"

try:
    # Verifica se o arquivo já existe para adicionar uma nova aba
    if os.path.exists(output_filename):
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl') as writer:
            df_results_stemmed.to_excel(writer, sheet_name="Com Stemming", index=False)
        print(f"\nResultados 'Com Stemming' adicionados à nova aba do arquivo '{output_filename}'.")
    else:
        # Se o arquivo não existir, cria um novo
        df_results_stemmed.to_excel(output_filename, sheet_name="Com Stemming", index=False)
        print(f"\nArquivo '{output_filename}' criado com os resultados 'Com Stemming'.")

except Exception as e:
    print(f"\nErro ao salvar o arquivo Excel: {e}")

    

