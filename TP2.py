import string
import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# a) ---------------------------------

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

processed_docs = [] # Armazenará o conteúdo processado de cada música como uma string

for music in musics:
    with open(f'./Documentos/{music}.txt', 'r', encoding='utf-8') as entrada:
        doc = entrada.read()
        doc = doc.lower() # Usar minúsculas é uma prática comum
        doc = unidecode.unidecode(doc) # Remove acentuação

        # Remove caracteres de pontuação
        doc = doc.translate(str.maketrans('', '', string.punctuation))

        doc = re.sub(r'\d+', '', doc) # Remove números
        doc = re.sub(r'\s+', ' ', doc).strip() # Remove espaços extras
        processed_docs.append(doc)

# b) ---------------------------------
# Cria um vetorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Calcula as pontuações TF-IDF para os documentos
tfidf_matrix = vectorizer.fit_transform(processed_docs)

# Imprime o vocabulário (palavras)
print("Vocabulário:")
print(vectorizer.get_feature_names_out())
print("Tamanho do vocabulário:", len(vectorizer.get_feature_names_out()))
print("\nMatriz TF-IDF:")
print(tfidf_matrix.toarray())

# c) ---------------------------------
# Implementação da consulta e cálculo de similaridade

# Consulta de exemplo
query = "love and baby"

# Vetoriza a consulta usando o mesmo vocabulário
query_vector = vectorizer.transform([query])

# Calcula a similaridade cosseno entre a consulta e todos os documentos
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Ordena os documentos pela similaridade (do maior para o menor)
sorted_indices = np.argsort(similarities)[::-1]

print(f"\n--- Resultados da Similaridade para a Consulta: '{query}' ---")
for rank, idx in enumerate(sorted_indices, 1):
    # Mostra apenas os resultados com similaridade > 0
    if similarities[idx] > 0:
        print(f"{rank}º: '{musics[idx]}' (Similaridade: {similarities[idx]:.4f})")

# ------ DESCOBRINDO MAIOR PESO ENTRE OS DOCUMENTOS ------
max_idx = np.unravel_index(tfidf_matrix.argmax(), tfidf_matrix.shape)
doc_idx, term_idx = max_idx
max_value = tfidf_matrix[doc_idx, term_idx]
termo = vectorizer.get_feature_names_out()[term_idx]
documento = musics[doc_idx]

print(f"\nMaior peso TF-IDF: {max_value:.4f}")
print(f"Termo: '{termo}'")
print(f"Documento: '{documento}'")