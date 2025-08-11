import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
nltk.download('punkt_tab')

try:
    stopwords.words('english')
except LookupError:
    print("Baixando recursos do NLTK (stopwords, punkt)...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("Recursos baixados com sucesso.")

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

def preprocess_text(doc, use_stemming=False):
    """
    Realiza o pré-processamento de um texto: minúsculas, remoção de acentos,
    pontuação, números e stopwords (usando NLTK).
    """
    # 1. Minúsculas e remoção de acentos
    doc = doc.lower()
    doc = unidecode.unidecode(doc)

    # 2. Remoção de pontuação e números
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = re.sub(r'\d+', '', doc)

    # 3. Tokenização (dividir o texto em palavras)
    tokens = word_tokenize(doc)

    # 4. Remoção de stopwords do NLTK
    stop_words_list = stopwords.words('english')
    processed_tokens = [token for token in tokens if token not in stop_words_list]
    
    # 5. Juntar os tokens de volta em uma string
    doc = ' '.join(processed_tokens)
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc

processed_docs = []
for music in musics:
    with open(f'./Documentos/{music}.txt', 'r', encoding='utf-8') as entrada:
        content = entrada.read()
        processed_content = preprocess_text(content)
        processed_docs.append(processed_content)

print("Pré-processamento concluído.")