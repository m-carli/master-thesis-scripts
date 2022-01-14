#class for common tools for the analysis of the character

class CharModelTools:

	custom_stopwords = [
		'might', 'could', 'can', 'indeed',
		'would', 'among', 'still', 'surely',
		'shall', 'therefore', 'already',
		'yet', 'perhaps', 'towards', 'thing',
		'things', 'without', 'many', 'also',
		'like', 'able', 'well', 'upon', 'this',
		'finally', 'come', 'came','coming', 'one', 'two', 
		'three','much','less','must','go','going',
		'went','gone','though','say','said',
		'get','got','saying','getting','tell',
		'told','telling','ask','asked','reply',
		'replied','exclaim','exclaimed','maybe',
		'that','even','thus','anyway','really',
		'instead']

	contractions = ["can't","couldn't","mustn't","shouldn't",
			"i'll","you'll","he'll","she'll",
			"it'll","we'll","they'll","won't",
			"don't","didn't","needn't","i'm",
			"you're","she's","he's","it's",
			"we're","they're","we’re","they’re",
			"can’t","couldn’t","mustn’t","shouldn’t",
			"i’ll","you’ll","he’ll","she’ll",
			"it’ll","we’ll","they’ll","won’t",
			"don’t","didn’t","needn’t","i’m",
			"you’re","she’s","he’s","it’s",
			"'s","’s","ain't","ain’t"] #aggiornare

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

	def find_ngrams(input_list, n):
		return zip(*[input_list[i:] for i in range(n)])