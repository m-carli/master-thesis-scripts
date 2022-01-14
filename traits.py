#PARADIGM OF TRAITS ANALYZER

print(' Importing libraries...')
from sklearn.feature_extraction.text import CountVectorizer
print(' CountVectorizer OK')
from sklearn.feature_extraction.text import TfidfTransformer
print(' TfidfTransformer OK')
import nltk
from nltk.corpus import wordnet as wn
print(' nltk OK')
import re
try: import readline
except: print(' [ WARNING ] readline library missing. If you are using Linux, install readline for smoother command-line usage.')
import sys
from classchar import CharModelTools
print(' Libraries imported.')

'''
L'idea generale di questo programma è semplice: dalle unità sintattiche
(paragrafi, frasi) in cui compare il nome del personaggio scelto, ne estrae
le keywords mediante tf-idf rispetto all'insieme di paragrafi e frasi del
testo intero. Infine, consente di effettuare la medesima operazione su porzioni
specifiche del testo (ovvero range di paragrafi/frasi).
Vorrebbe essere un tentativo di operazionalizzare i modelli del personaggio 
di Barthes e Chatman, che ho spiegato nel dettaglio nel capitolo consegnato,
i quali fondamentalmente vedono il personaggio come un insieme di tratti
inferenziali (di natura semiologica o psicologica) modellati da indizi lessicali 
che il testo fornisce.
'''

#recupero liste di custom stopwords dalla classe

stopwords = set(nltk.corpus.stopwords.words('english'))
CONTRACTIONS = CharModelTools.contractions
CUSTOM_STOPWORDS = CharModelTools.custom_stopwords

CHARS = []

def pre_process(text):
	'''
	Filtrare il testo trasmesso come argomento togliendo i nomi dei personaggi,
	e le parole contenute nell'array custom stopwords.
	Ritorna un testo filtrato.
	'''
	global CHARS

	text = re.sub('<CHAPTER>','',text)
	text = re.sub('</DIALOGUE>','',text)
	text = re.sub('<DIALOGUE ','',text)
	text = re.sub('>','',text)
	text = re.sub('--','',text)
	text = text.lower()
	for contraction in CONTRACTIONS: text = re.sub(contraction, '', text)
	text_tokenized = nltk.word_tokenize(text)
	for index, word in enumerate(text_tokenized):
		if word in CUSTOM_STOPWORDS or word in CHARS or word in stopwords:
			text_tokenized.pop(index)

	text = ''
	for word in text_tokenized:
		text += word + ' '

	return text

#inserire il nome del file da analizzare
novel = input(' Insert file name of the novel [with extension]: ')

#qui per risparmiare tempo ho optato per l'hardcoding dei nomi dei personaggi 
#dei romanzi da analizzare
if novel == 'ignorance.txt': CHARS = ['josef', 'irena', 'milada', 'gustaf', 'martin', 'sylvie', 'n.']
elif novel == 'achebe things.txt': CHARS = ['okonkwo','nwoye','ikemefuna','ekwefi','unoka','ezinma','obierika','okoye','umuofia','uchendu','nwakibie']
elif novel == 'siddhartha.txt': CHARS = ['siddhartha','govinda','kamala','vasudeva','kamaswami','gotama']
elif novel == 'marquez love cholera.txt': CHARS = ['fermina','florentino','escolástica','lorenzo','juvenal','tránsito','urbino','daza','ariza']
elif novel == 'sarrasine.txt': CHARS = ['sarrasine','zambinella','bouchardon','clotilde','marianina','rochefide','filippo']

#suddivisione del romanzo in frasi e paragrafi
print(f' Reading novel "{novel}"...')
try: file = open(novel,'r', encoding='utf-8')
except FileNotFoundError: sys.exit(f' [ ERR ] No file named {novel} found.')
novel = file.read()
paragraphs = novel.split('\n')
sentences = re.split('\n|\.', novel)
file.close()

#eliminazione di frasi e paragrafi vuoti, con successivo controllo
paragraphs = [paragraph for paragraph in paragraphs if paragraph]
sentences = [sentence for sentence in sentences if sentence]
print(all(paragraphs))
print(all(sentences))

print(f' Sentences (total new dividing method, clean): {len(sentences)}')
print(f' Paragraphs (clean): {len(paragraphs)}')

#Inserire il nome con la lettera maiuscola
selected_char = input(' Insert name to analyze [capitalized]: ')
if len(selected_char) == 0: sys.exit(' [ LOG ] No input, program stopped.')
print(' Processing...')

#vettorizzazione per tf-idf, in modo analogo ai programmi già consegnati
cv = CountVectorizer(stop_words=stopwords) 
word_count_vector = cv.fit_transform(paragraphs)
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

def extract_char_data(first, last, object_):

	'''
	Questa funzione estrae i paragrafi e le frasi (viene infatti chiamata
	sotto due volte per estrarre entrambi questi oggetti) dove compare il
	nome del personaggio selezionato, dopo di che estrae una serie di keyword
	mediante tf-idf, rispetto al totale dei paragrafi/frasi.
	'''

	global selected_char, paragraphs, sentences
	char_processed = []
	char_not_processed = []

	if object_ == 'paragraphs': array_of_objects = paragraphs
	else: array_of_objects = sentences

	for e in array_of_objects[first:last]:
		if selected_char in e:
			char_processed.append(pre_process(e))
			char_not_processed.append(e)

	print(f' {object_} with "{selected_char}": {len(char_processed)}')

	total_char_string = ''
	total_char_string_not_processed = ''
	for e in char_processed: total_char_string += e + '\n'
	for e in char_not_processed: total_char_string_not_processed +=  e + '\n'

	feature_names = cv.get_feature_names_out()
	doc = pre_process(total_char_string)
	tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
	sorted_items = CharModelTools.sort_coo(tf_idf_vector.tocoo())
	keywords = CharModelTools.extract_topn_from_vector(feature_names, sorted_items, 20)

	return keywords, total_char_string

print('\n Keywords (paragraphs):')
print(extract_char_data(0, len(paragraphs)-1, 'paragraphs')[0])
print('\n Keywords (sentences):')
print(extract_char_data(0, len(sentences)-1, 'sentences')[0])

print('')

'''
Questa parte è pensata per essere utilizzata in coppia con il programma
intSentAnalyzer, poiché consente di effettuare la stessa analisi fatta sopra
per un range di frasi del testo. Ad esempio, se il programma intSentAnalyzer
sottolinea una concentrazione rilevante di un personaggio dalla frase x alla
frase y, qui è possibile inserire x e y per ricavare le keywords associate al
personaggio in quella porzione del testo.
'''

choice = input(' Perform analysis by sentence ranges? (y/n) ')
if choice == 'y':
	print(' [ INFO ] Leave input void to stop program.')
	while True:
		start_sample_num = input(' Number of first sentence of sample: ')
		if len(start_sample_num) == 0: break
		end_sample_num = input(' Number of last sentence of sample: ')
		if len(end_sample_num) == 0: break

		start_sample_sentence = sentences[int(start_sample_num)]
		end_sample_sentence = sentences[int(end_sample_num)]

		is_start_sample_too_short = len(start_sample_sentence.split(' ')) < 5
		is_end_sample_too_short = len(end_sample_sentence.split(' ')) < 5
		if is_start_sample_too_short or is_end_sample_too_short:
			choice = input(' [ LOG ] One of the sentences selected is very short, and therefore could occur more than once in a text, thus potentially leading to wrong results. The program will automatically look for a longer nearby sentence to be used as the sample limit. Is it ok? (y/n) ')
			if choice == 'y':
				if is_start_sample_too_short:
					for n in range(1,5):
						if len(sentences[int(start_sample_num)-n]) >= 5:
							start_sample_sentence = sentences[int(start_sample_num)-n]
							print(f' [ LOG ] Using start sentence n. {int(start_sample_num)-n}')
							break
				else:
					for n in range(1,5):
						if len(sentences[int(end_sample_num)-n]) >= 5:
							end_sample_sentence = sentences[int(end_sample_num)-n]
							print(f' [ LOG ] Using start sentence n. {int(end_sample_num)-n}')
							break

		start_sample_paragraph = ''
		end_sample_paragraph = ''
		for index, paragraph in enumerate(paragraphs):
			if start_sample_sentence in paragraph:
				start_sample_paragraph = paragraph
				start_sample_paragraph_num = index
			elif end_sample_sentence in paragraph:
				end_sample_paragraph = paragraph
				end_sample_paragraph_num = index

		print('\n Keywords for selected range (paragraphs):')
		print(extract_char_data(start_sample_paragraph_num, end_sample_paragraph_num, 'paragraphs')[0])

print(' [ LOG ] End of program.')