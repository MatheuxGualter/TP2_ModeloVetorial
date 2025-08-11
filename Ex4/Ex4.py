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
import time

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

def preprocess_text(doc, use_stemming=True):
    stemmer = PorterStemmer()
    doc = doc.lower()
    doc = unidecode.unidecode(doc)
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = re.sub(r'\d+', '', doc)
    tokens = word_tokenize(doc)
    stop_words_list = stopwords.words('english')
    processed_tokens = [token for token in tokens if token not in stop_words_list]
    if use_stemming:
        processed_tokens = [stemmer.stem(token) for token in processed_tokens]
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

queries = [
    "Baby I'm losing control in this cruel summer",
    "Sweet love and beautiful things",
    "Million dollar baby, good luck babe",
    "Drinking and dancing in a bar song",
    "I remember everything about us"
]

# EXECUÇÃO DAS CONSULTAS COM MEDIÇÃO DE TEMPO (Ex4)
query_times = []
print("\n--- Análise de Tempo de Execução das Consultas ---")

for query_text in queries:
    processed_query = preprocess_text(query_text)
    query_vector = vectorizer.transform([processed_query])

    # Medição de tempo
    start_time = time.time()
    
    # Etapa cujo tempo será medido: cálculo de similaridade e ordenação
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    query_times.append(elapsed_time)
    
    print(f"Consulta: '{query_text}'")
    print(f"Tempo para gerar o ranking: {elapsed_time:.6f} segundos.")
    
# CÁLCULO DO TEMPO MÉDIO
average_time = np.mean(query_times)
print(f"\nTempo médio por consulta: {average_time:.6f} segundos.")