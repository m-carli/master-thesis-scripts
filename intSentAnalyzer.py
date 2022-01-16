#INTEGRATED SENTIMENT ANALYZER

import os
def cls():
    if os.name == 'nt': os.system("cls")
    else: os.system("clear")
    print("Integrated Sentiment Analyzer")

cls()
print("Loaded 0%   |          |") 
#Ho aggiunto una progressbar poiché questo programma potrebbe metterci un po' a caricarsi...
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
cls()
print("Loaded 10%  ||          |")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
cls()
print("Loaded 20%  |||        |")
import nltk
from nltk.corpus import wordnet as wn
cls()
print("Loaded 40%  |||||      |")
import pandas as pd
cls()
print("Loaded 60%  |||||||    |")
from matplotlib import pyplot
cls()
print("Loaded 80%  |||||||||  |")
import sys
try: import readline
except: print(' [ WARNING ] Library readline not installed. If you are using Linux, install readline for a smoother command-line experience.')
from classchar import CharModelTools

'''
Scopo del programma: ottenere una visualizzazione da cui sia possibile confrontare 
l'emotional arc del romanzo (ottenuto mediante sentiment analysis) con la
distribuzione dei personaggi nel romanzo (rilevata attraverso il loro nome proprio) e
con le porzioni del romanzo dove avviene l'interazione tra coppie di personaggi, a cui
vengono assegnate le parole chiave che caratterizzano l'interazione corrispondente.
'''

analyzer = SentimentIntensityAnalyzer()
cls()
print("Loaded 100%: Ready.")

book = input('Insert source book [.txt format, without extension]: ')
BOOK = f'{book}.txt'
window_value = input('Insert window width: ')
WITH_TFIDF = False
tfidf_choice = input('Use tfidf? (y/n) ')
if tfidf_choice == 'y': WITH_TFIDF = True
choice = input('Go directly to graph? y/n ')

CUSTOM_STOPWORDS = CharModelTools.custom_stopwords

#SEZIONE DISTRIBUZIONE PERSONAGGI
char_positions = {}
dict_interactions = {}
sentences = []
def add_char_distribution(book):
    '''
    Qui vengono aggiunte, sotto forma di punti ottenuti mediante la funzione scatter
    di matplotlib, la distribuzione dei nomi propri lungo il romanzo. Ad ogni personaggio
    (il cui nome viene collocato a lato del grafico) viene assegnata una riga, distanziata
    l'una dall'altra con un valore che consenta di non alterare troppo la visualizzazione
    dell'emotional arc.
    '''
    global interactions
    global char_positions
    global sentences

    '''
    Qui il programma chiede l'inserimento dei nomi dei personaggi da visualizzare sul 
    grafico, separati da una virgola, senza spazi. E' possibile anche inserire 
    definizioni perifrastiche (es. the wizard, the wise lord), purché denotino nel testo 
    un personaggio in maniera univoca, qualora un personaggio non abbia un nome proprio.
    Consiglio di non inserire più di 3 o 4 personaggi alla volta, per evitare grafici
    eccessivamente caotici.
    '''
    characters = input('Insert array of characters [capitalized, separated with comma]: ')
    char_array = characters.split(',')

    pyplot.xlabel('Distribution')
    pyplot.title('Characters distribution and sentiment tendency')

    y = []
    x = []

    with open(f'{book}.txt','r',encoding='utf-8') as f:
        total = f.read()
        #sentences = f.read().split('.')
        sentences = re.split('\n|\.', total)
        sentences = [sentence for sentence in sentences if sentence]
        if len(char_array) <= 3:
            nchar = -0.1
            space = 0.1
        else:
            nchar = -0.1
            space = 0.05

        #calcolo distribuzione per ogni personaggio inserito
        for name in char_array:
            tempx = []
            tempy = []
            index = 0
            for sentence in sentences:
                if name in sentence: #sentence.count(name) == 1:
                    tempx.append(index)
                    tempy.append(nchar)
            
                index += 1

            #Assegnazione label laterale per ogni personaggio
            pyplot.text(0, nchar-0.02, name, horizontalalignment='right', fontweight='semibold')

            x.append(tempx)
            y.append(tempy)
            char_positions[name] = nchar
            nchar += space

    print('Char positions:')
    print(char_positions)
    for n in range(len(x)):
        pyplot.scatter(x[n],y[n],s=10)

#SEZIONE TFIDF
'''
Il funzionamento di questa sezione (funzioni pre_process, sort_coo, extract_topn_from_vector,
main_tfidf) è uguale alla sezione dedicata alla TF-IDF nel programma networktfidf.py, 
leggermente riadattata alle esigenze di questo programma.
I commenti di quella porzione dunque valgono anche per questa.
'''
def pre_process(text, char1, char2):
    text = text.lower()
    text = re.sub(char1.lower(),'', text)
    text = re.sub(char2.lower(),'', text)
    for word in CUSTOM_STOPWORDS: text = re.sub(word,'', text)

    return text

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def extract_total_lines_tfidf(char1, char2):
    file = open(BOOK,'r',encoding='utf-8')
    content = file.read()
    content = re.sub('\n',' ', content)
    content = re.sub('</DIALOGUE>','</DIALOGUE>\n',content)
    content = re.sub('--',' ', content)

    total_lines_tolist = re.findall('".+?"', content)
    total_lines_tolist_processed = []
    for line in total_lines_tolist:
        total_lines_tolist_processed.append(pre_process(line, char1, char2))

    return total_lines_tolist_processed

def main_tfidf(x_start, x_end, char1, char2):
    global sentences

    #PARTE 1: LAVORAZIONE SINGOLA INTERAZIONE E CORPUS INTERO DELLE BATTUTE
    interaction_segment = sentences[x_start-1:x_end]
    interaction_segment_string = ' '.join(interaction_segment)
    interaction_segment_string = re.sub('\n',' ',interaction_segment_string)
    total_dialogue_string = ' '.join(re.findall(r'".+?"',interaction_segment_string))

    total_lines_tolist_processed = extract_total_lines_tfidf(char1, char2)

    #PARTE 2: VETTORIZZAZIONE CORPUS INTERO

    stopwords = set(nltk.corpus.stopwords.words('english'))
    cv = CountVectorizer(stop_words=stopwords)
    word_count_vector = cv.fit_transform(total_lines_tolist_processed)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names = cv.get_feature_names()
    doc = pre_process(total_dialogue_string, char1, char2)
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    print('\nKeywords:')
    keywords_formatted = ''
    index = 0
    for k in keywords:
        print(k, keywords[k])
        keywords_formatted += k + '\n'
        index += 1
        if index == 5: break

    return keywords_formatted

#SEZIONE ELABORAZIONE GRAFICO
def create_graph(file, win):
    global book
    global interactions
    global char_positions
    global dict_interactions
    global sentences
    global WITH_TFIDF

    #CREATE PANDAS DATASET FROM .TXT AND ROLLING
    data_book = pd.read_csv(file, sep="\n", header=None, index_col=None)

    print('data read successfully.')

    #applicazione rolling mean
    rolling = data_book.rolling(window=int(win))
    rolling_mean = rolling.mean()

    #PLOT DATA
    print('Making sentiment graph...')
    rolling_mean.plot()

    add_char_distribution(book)
    print("Adding character distribution to sentiment graph...")

    #Aggiunta di linee tratteggiate che separano i capitoli
    with open(f"Novels data/Catalogues {book}/{book}_numchapters.txt","r") as file:
        numbers = file.read().split('\n')
        nchap = 1
        del numbers[-1]
        for num in numbers:
            pyplot.axvline(x=int(num), color="black", linewidth=0.5, linestyle="--")
            nchap += 1

        file.close()

    '''
    Questa è la parte in cui vengono evidenziate le porzioni di grafico in cui vi è
    una interazione; vengono applicate come coordinate y la posizione y dei due
    personaggi coinvolti (assegnata nella funzione add_char_distribution), come
    coordinate x invece i numeri delle frasi iniziale e finale dell'interazione, sul
    totale delle frasi del romanzo (che vengono numerate in una parte del programma
    più in basso).
    '''
    keyerrors = 0
    for start in interactions:
        chars_in_this_interaction = dict_interactions[start].split(':')
        try:
            position_char1 = char_positions[chars_in_this_interaction[0]]
            position_char2 = char_positions[chars_in_this_interaction[1]]

            if position_char1 < position_char2: 
                pyplot.fill_betweenx([position_char1,position_char2],int(start),int(interactions[start]),color='red',alpha=0.2)
                better_text_position = position_char1
            else: 
                pyplot.fill_betweenx([position_char2,position_char1],int(start),int(interactions[start]),color='red',alpha=0.2)
                better_text_position = position_char2
        
            #prendere le coordinate x dell'interazione per sapere da quale a quale
            #frase va la singola interazione (poi estratta dall'array mediante
            #la formula array[3:6])
            '''
            Qui avviene l'estrazione delle keywords per l'interazione considerata.
            Vengono utilizzati il numero della frase di inizio e quello della fine
            dell'interazione per isolare la porzione di testo dell'interazione dal testo
            completo (il quale viene ripreso come un array di frasi numerate), e
            poi viene svolta una procedura di TF-IDF.
            '''
            if WITH_TFIDF:
                keywords = main_tfidf(int(start), int(interactions[start]), chars_in_this_interaction[0], chars_in_this_interaction[1])
                pyplot.text(int(start), better_text_position, keywords, fontsize=6)

        except KeyError:
            '''
            Al fine di evitare un grafico eccessivamente caotico, è meglio evitare di
            inserire tutti i personaggi di un romanzo nello stesso grafico. Quando
            nei tag viene trovato un personaggio che non rientra tra quelli inseriti,
            questa eccezione consente di saltarlo.
            '''
            print('Name not found, skipped interaction.')
            keyerrors += 1

    print(f'Number of skipped interactions: {keyerrors}')
    #print("Chapter separators added.")
    pyplot.show()

#SEZIONE SENTIMENT ANALYSIS
interactions = {}
dict_interactions = {}
if choice == 'n':
    #ESTRARRE POLARITA' E CREARE DATASET
    #creazione di directories dove immagazzinare i file prodotti dal programma
    try: os.mkdir('Novels data')
    except: pass
    try: os.mkdir(f'Novels data/Catalogues {book}')
    except: pass
    
    parameter = 0.8 #HARDCODED PARAMETER
    polarities = []
    positive_sent = {}
    negative_sent = {}
    all_sentences = []
    chapters_nums = []
    
    '''
    Qui il testo viene processato per la sentiment analysis. Il testo viene spezzato in
    frasi, e per ciascuna di esse viene calcolato e vettorizzato il valore compound.
    Vengono inoltre rilevate le posizioni del tag <CHAPTER> (per dividere i capitoli),
    e dei tag <DIALOGUE> per le interazioni dei personaggi, per le quali viene compilato
    un dizionario numerato, e uno con i punti di inizio e di fine.
    '''

    index = 0
    try:
        with open(BOOK,"r",encoding="utf-8") as f:
            start_dialogue = 0
            ninteractons = 0
            total = f.read() #.split('.')
            total = re.split('\n|\.', total)
            total = [e for e in total if e]
            print(f'TOTAL WITH NEW DIVIDING METHOD: {len(total)}')
            for sentence in total: #f.read().split()
                index += 1
                all_sentences.append(sentence)

                #Assegnazione valori compound
                vs = analyzer.polarity_scores(sentence)
                polarities.append(vs['compound'])

                if vs['compound'] >= float(parameter): positive_sent.update({index:vs['compound']})
                elif vs['compound'] <= -float(parameter): negative_sent.update({index:vs['compound']})

                #compilazione dizionari con numero capitoli e parametri delle interazioni
                if "<CHAPTER>" in sentence: chapters_nums.append(index)

                if '<DIALOGUE' in sentence:
                    start_dialogue = index
                    regex = re.search(r'<DIALOGUE .+?:.+>', sentence).group(0)
                    regex = re.sub('<DIALOGUE ','',regex)
                    regex = re.sub('>','',regex)
                    print(f'Interaction found: {regex}')
                    ninteractons += 1
                    dict_interactions[index] = regex

                if '</DIALOGUE>' in sentence:
                    interactions[start_dialogue] = index

            #print(f'TOTAL WITH NEW DIVIDING METHOD: {len(total)}')
            print('Number of sentences: ' + str(index) + '.\nTen percent: {}'.format(index/10))
            print('Number of interactions: ' + str(ninteractons))
            print('Dict interactions')
            print(dict_interactions)

    except Exception as e:
        sys.exit(e)

    #FILE WITH NUMBERS OF SENTENCES WITH CHAPTER NAME INSIDE
    '''
    Qui vengono creati due file: uno per immagazzinare le posizioni dei tag che separano
    i capitoli, l'altro con tutte le frasi del romanzo numerate.
    '''
    with open(f"Novels data/Catalogues {book}/{book}_numchapters.txt","w") as file:
        for num in chapters_nums: file.write(str(num) + '\n')
        f.close()

    len_pol = len(polarities)

    print(f'{book} data processed.')

    #CREATE FILE WITH NUMBERED SENTENCES
    index = 0
    with open(f"Novels data/Catalogues {book}/{book}_catalogue.txt","w") as file:
        for sentence in all_sentences:
            file.write(str(index) + ' > ' + sentence + '\n')
            index += 1

    print(f'{book} catalogue created')

    #CREATE FILE TO REGISTER POSITIONS OF PARTICULARLY LOW OR HIGH VALUES
    '''
    Qui vengono creati dei file dove vengono segnate le frasi con i valori del sentiment
    più alti o più bassi (sopra lo 0.8 o sotto il -0.8). L'idea è quella di aiutare poi
    a rintracciare nel grafico quelli che saranno i punti più alti o più bassi.
    '''
    with open(f"Novels data/Catalogues {book}/{book}_sentences_positive.txt","w") as f:
        for element in positive_sent: f.write(str(element) + '> ' + str(positive_sent[element]) + '\n')
        f.close()

    with open(f"Novels data/Catalogues {book}/{book}_sentences_negative.txt","w") as f:
        for element in negative_sent: f.write(str(element) + '> ' + str(negative_sent[element]) + '\n')
        f.close()

    print(f'{book}_sentences created')

    names = []
    for n in range(0, len_pol):
        names.append(n)

    #DATASET FILE
    with open(f"Novels data/Catalogues {book}/data_{book}.txt","w") as f:
        for n in names: f.write(str(polarities[n]) + '\n')
        f.close()

    print(f'data_{book} created.')
    create_graph(f'Novels data/Catalogues {book}/data_{book}.txt', window_value)

else:
    try: create_graph(f'Novels data/Catalogues {book}/data_{book}.txt', window_value)
    except Exception as e: print(e)
