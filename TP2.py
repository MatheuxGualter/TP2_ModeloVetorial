import string
import unidecode
import re

musics = ['01 - Lose Control', '02 - A Bar Song (Tipsy)', '03 - Beautiful Things', '04 - I Had Some Help', '05 - Lovin on Me', '06 - Not Like Us', '07 - Espresso', '08 - Million Dollar Baby', '09 - I Remember Everything', '10 - Too Sweet', '11 - Stick Season', '12 - Cruel Summer', '13 - Greedy', '14 - Like That', '15 - Birds of a Feather', '16 - Please Please Please', '17 - Agora Hills', '18 - Good Luck, Babe!', '19 - Saturn', '20 - Snooze']

list_voc = [] # Armazenará o vocabulário processado de cada arquivo de música

for music in musics:
    with open(f'./Documentos/{music}.txt', 'r', encoding='utf-8') as entrada:
        doc = entrada.read()
        doc = doc.upper()
        doc = unidecode.unidecode(doc) # Remove acentuação

        # Remove caracteres de pontuação
        for pont in string.punctuation:
            doc = doc.replace(pont, "")

        doc = re.sub(r'\d+', '', doc) # Remove números dentre no texto da música
        spl = doc.split()
        doc = sorted(list(set(spl))) # Remove palavras duplicadas e depois converte para lista
        list_voc.append(doc)

for i, voc in enumerate(list_voc):
    print(f"Vocabulário para a música {i+1}:")
    print(voc)
    print("\n")