import os
import sys
import nltk
import string
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

class WordEnvironmentVectors(object):
	"""
	Create feature vectors for each word in a corpus.
	Features are determined based on the words surrounding each corpus word.

	self.word_vectors is a mapping of each unique word in training to a list of 
	the feature vector created for each instance of that word in training

	To create vectors from files in a directory:
	wev = WordEnvironmentVectors.create_vectors(indir)
	wev.save_vecs(file)
	"""

	def __init__(self, vocab=None, window_size=5):
		#https://en.wiktionary.org/wiki/Appendix:Basic_English_word_list
		self.words = set(['wind', 'all', 'chain', 'rod', 'yellow', 'month', 'manager', 'sleep', 'skin', 'go', 'sponge', 'adjustment', 'hate', 'milk', 'protest', 'father', 'young', 'send', 'to', 'tail', 'rhythm', 'under', 'smile', 'division', 'woman', 'garden', 'song', 'far', 'fat', 'wave', 'spoon', 'every', 'fall', 'cook', 'word', 'trouble', 'feeble', 'condition', 'school', 'level', 'button', 'shock', 'list', 'brother', 'sand', 'married', 'knife', 'quick', 'special', 'round', 'blade', 'force', 'regret', 'tired', 'hanging', 'request', 'sign', 'jump', 'fold', 'rate', 'invention', 'design', 'acid', 'rough', 'theory', 'even', 'will', 'power', 'apparatus', 'clock', 'thunder', 'near', 'poison', 'current', 'public', 'waiting', 'new', 'net', 'ever', 'bird', 'body', 'full', 'degree', 'exchange', 'loose', 'here', 'automatic', 'behaviour', 'let', 'free', 'stem', 'change', 'box', 'boy', 'great', 'property', 'daughter', 'healthy', 'experience', 'credit', 'amount', 'library', 'smoke', 'opinion', 'narrow', 'punishment', 'beautiful', 'love', 'apple', 'humour', 'danger', 'private', 'throat', 'sugar', 'market', 'use', 'from', 'army', 'hospital', 'normal', 'doubt', 'music', 'memory', 'sort', 'pencil', 'door', 'comparison', 'company', 'trousers', 'sister', 'glass', 'flag', 'train', 'stick', 'impulse', 'teaching', 'baby', 'hole', 'fly', 'brown', 'account', 'wool', 'join', 'room', 'hour', 'shame', 'this', 'science', 'past', 'work', 'worm', 'roof', 'cat', 'soup', 'thin', 'island', 'male', 'root', 'example', 'history', 'control', 'heart', 'ready', 'bent', 'give', 'process', 'lock', 'tax', 'high', 'drawer', 'chin', 'slip', 'sense', 'sharp', 'voice', 'dress', 'simple', 'parcel', 'end', 'winter', 'turn', 'skirt', 'discussion', 'comfort', 'damage', 'machine', 'how', 'animal', 'answer', 'strong', 'goat', 'boot', 'map', 'plant', 'may', 'watch', 'after', 'horn', 'wrong', 'produce', 'blood', 'water', 'such', 'law', 'arch', 'parallel', 'man', 'a', 'short', 'attempt', 'neck', 'liquid', 'conscious', 'light', 'linen', 'chief', 'so', 'frequent', 'shade', 'basket', 'pleasure', 'egg', 'order', 'talk', 'wine', 'help', 'office', 'cheese', 'over', 'move', 'trade', 'brain', 'paper', 'through', 'committee', 'shake', 'existence', 'cold', 'still', 'ticket', 'tendency', 'before', 'chemical', 'group', 'thumb', 'till', 'writing', 'late', 'attraction', 'window', 'strange', 'minute', 'stiff', 'orange', 'then', 'good', 'food', 'material', 'nation', 'band', 'snake', 'kiss', 'front', 'now', 'day', 'bread', 'name', 'hook', 'servant', 'delicate', 'drop', 'tray', 'meal', 'bone', 'week', 'square', 'weight', 'house', 'fish', 'receipt', 'idea', 'gun', 'society', 'ball', 'measure', 'operation', 'event', 'amusement', 'out', 'living', 'canvas', 'flower', 'stitch', 'driving', 'space', 'profit', 'open', 'increase', 'farm', 'wide', 'print', 'surprise', 'cause', 'red', 'shut', 'umbrella', 'belief', 'story', 'cart', 'quite', 'small', 'reason', 'base', 'put', 'cough', 'card', 'care', 'fixed', 'language', 'rule', 'keep', 'motion', 'thing', 'plane', 'place', 'stone', 'view', 'loud', 'top', 'south', 'first', 'copper', 'sock', 'number', 'hearing', 'wash', 'owner', 'scissors', 'ring', 'mist', 'quality', 'tomorrow', 'size', 'sheep', 'little', 'monkey', 'bite', 'system', 'fiction', 'paint', 'attack', 'station', 'statement', 'white', 'angry', 'bitter', 'nut', 'friend', 'toe', 'stomach', 'that', 'shelf', 'needle', 'part', 'natural', 'copy', 'than', 'stocking', 'steel', 'distance', 'kind', 'tooth', 'wet', 'tree', 'grey', 'bed', 'bee', 'street', 'flame', 'store', 'iron', 'coat', 'feeling', 'and', 'bridge', 'false', 'fowl', 'mind', 'mine', 'sad', 'ant', 'medical', 'say', 'seed', 'have', 'need', 'seem', 'any', 'spade', 'ray', 'mountain', 'angle', 'coal', 'general', 'self', 'responsible', 'able', 'snow', 'note', 'chest', 'lip', 'take', 'green', 'soap', 'blue', 'play', 'pain', 'electric', 'though', 'price', 'who', 'boiling', 'regular', 'mouth', 'letter', 'position', 'knee', 'payment', 'muscle', 'observation', 'foot', 'nail', 'metal', 'disease', 'face', 'pipe', 'wind', 'clean', 'sun', 'sail', 'salt', 'fact', 'slope', 'selection', 'gold', 'ornament', 'cheap', 'bright', 'relation', 'shoe', 'earth', 'fear', 'crush', 'slow', 'knowledge', 'true', 'pump', 'crime', 'only', 'wood', 'black', 'disgust', 'circle', 'rice', 'hope', 'do', 'get', 'dependent', 'stop', 'pocket', 'whistle', 'cushion', 'silk', 'stamp', 'smash', 'leather', 'cry', 'morning', 'bag', 'bad', 'common', 'river', 'where', 'steam', 'secretary', 'art', 'oven', 'burst', 'smooth', 'frame', 'seat', 'trick', 'see', 'horse', 'powder', 'sea', 'hammer', 'arm', 'wire', 'probable', 'expert', 'leg', 'please', 'cruel', 'enough', 'future', 'finger', 'between', 'yesterday', 'reading', 'across', 'discovery', 'attention', 'cut', 'key', 'approval', 'debt', 'come', 'reaction', 'last', 'cow', 'country', 'ill', 'jelly', 'against', 'connection', 'grain', 'berry', 'among', 'point', 'wall', 'sweet', 'pot', 'harbour', 'walk', 'learning', 'news', 'vessel', 'respect', 'boat', 'comb', 'second', 'addition', 'west', 'political', 'mark', 'breath', 'secret', 'much', 'interest', 'certain', 'curtain', 'basin', 'waste', 'meeting', 'engine', 'dry', 'direction', 'flight', 'fire', 'prison', 'argument', 'lift', 'awake', 'representative', 'bulb', 'pig', 'present', 'sound', 'look', 'solid', 'straight', 'sex', 'rat', 'value', 'air', 'chalk', 'while', 'match', 'error', 'balance', 'guide', 'swim', 'tall', 'almost', 'equal', 'middle', 'sudden', 'brake', 'in', 'prose', 'if', 'different', 'shirt', 'make', 'wound', 'jewel', 'same', 'complex', 'wheel', 'harmony', 'feather', 'development', 'oil', 'suggestion', 'screw', 'I', 'cloud', 'drink', 'rail', 'effect', 'rain', 'hand', 'fruit', 'purpose', 'advertisement', 'military', 'dust', 'collar', 'destruction', 'butter', 'dark', 'safe', 'drain', 'off', 'colour', 'floor', 'well', 'thought', 'person', 'edge', 'bottle', 'mother', 'very', 'organization', 'the', 'reward', 'left', 'summer', 'knot', 'money', 'rest', 'not', 'violent', 'touch', 'polish', 'yes', 'blow', 'death', 'family', 'opposite', 'cup', 'thick', 'bell', 'sky', 'silver', 'instrument', 'verse', 'book', 'board', 'crack', 'east', 'hat', 'kick', 'grip', 'sticky', 'government', 'possible', 'early', 'birth', 'judge', 'bit', 'hollow', 'meat', 'desire', 'loss', 'necessary', 'like', 'plough', 'glove', 'nose', 'night', 'foolish', 'soft', 'page', 'because', 'old', 'laugh', 'spring', 'church', 'some', 'back', 'authority', 'hair', 'thread', 'growth', 'table', 'dear', 'transport', 'scale', 'leaf', 'for', 'substance', 'decision', 'kettle', 'ice', 'moon', 'religion', 'pen', 'unit', 'porter', 'cord', 'be', 'noise', 'run', 'business', 'rub', 'burn', 'agreement', 'expansion', 'cork', 'broken', 'step', 'eye', 'paste', 'by', 'dead', 'stage', 'on', 'about', 'of', 'industry', 'chance', 'side', 'stretch', 'range', 'important', 'act', 'mixed', 'tongue', 'or', 'road', 'bath', 'nerve', 'digestion', 'fertile', 'son', 'down', 'weather', 'brush', 'female', 'quiet', 'physical', 'elastic', 'support', 'there', 'question', 'long', 'fight', 'start', 'camera', 'low', 'way', 'wax', 'forward', 'war', 'happy', 'fork', 'tight', 'head', 'hard', 'north', 'complete', 'form', 'offer', 'but', 'cloth', 'heat', 'competition', 'line', 'ear', 'with', 'brick', 'he', 'pull', 'wise', 'bucket', 'up', 'record', 'carriage', 'limit', 'dirty', 'cake', 'distribution', 'grass', 'clear', 'pin', 'flat', 'taste', 'cover', 'year', 'deep', 'dog', 'twist', 'as', 'right', 'at', 'ink', 'ship', 'education', 'girl', 'cotton', 'again', 'no', 'peace', 'when', 'detail', 'insurance', 'field', 'tin', 'other', 'whip', 'branch', 'test', 'you', 'wing', 'smell', 'roll', 'poor', 'picture', 'star', 'plate', 'separate', 'why', 'town', 'insect', 'journey', 'brass', 'sneeze', 'structure', 'building', 'land', 'lead', 'potato', 'curve', 'together', 'warm', 'mass', 'time', 'push', 'serious'])
		self.vocab = vocab
		self.window_size = window_size
		self.word_vectors = defaultdict(list)
		self.punctuation = string.punctuation
		self.stopwords = stopwords.words('english')

	def clear_vectors(self):
		self.word_vectors = defaultdict(list)

	def get_file_text(self, f):
		words = []
		for line in f:
			if line.split():
				line = line.lower().split()
				for i, word in enumerate(line):
					try:
						word.decode('ascii')
						words.append(word)
					except UnicodeDecodeError:
						del line[i]

		return ' '.join(words)

	def tokenize(self, text):
		"""Split a block of text into sentences, and further into words"""
		sentences = []
		for s in sent_tokenize(text):
			sentences.append(word_tokenize(s))
		return sentences

	def get_word_vecs(self, sent):
		vecs = []
		#punctuation and stopwords may add too much noise to vectors
		#s = [w for w in sent if w not in self.punctuation and w not in self.stopwords]

		for i, w in enumerate(sent):
			#if w not in self.punctuation and w not in self.stopwords:
			if w in self.words and w not in self.stopwords:
				#instiantiate vector as the length of our feature set
				word_vec = [0]*len(self.vocab)
				window_words = []
				#handle the exception of the word being less than window_size
				#away from the beginning/end of the sentence
				for i in range(1,self.window_size+1) + range(-self.window_size,0):
					try:
						if sent[i] in self.vocab:
							window_words.append(sent[i])
					except (IndexError): pass

				for w in window_words:
					word_vec[self.vocab[w]] = 1
				vecs.append(word_vec)
			else:
				vecs.append(None)

		return vecs

	def create_vectors(self, f):
		f = open(f)
		vecs = []
		sents = self.tokenize(self.get_file_text(f))
		for s in sents:
			word_vecs = self.get_word_vecs(s)
			vecs.append(zip(s, word_vecs))

		return vecs

	def collect_word_vecs(self, word_vecs):
		for s in word_vecs:
			for word, vec in s:
				if vec is not None:
					self.word_vectors[word].append(vec)

	"""
	# Following functions needed before I preprocessed to get vocabulary
	def pad_vecs(self):
		#pad vectors so they have an index for each feature for clustering
			for word in self.word_vectors:
				for i, vec in enumerate(self.word_vectors[word]):
					self.word_vectors[word][i] = self.pad(vec, len(self.features), 0)

	def pad(self, l, size, padding):
		return l + [padding] * abs((len(l)-size))
	"""

	def create(self, filename):
		vecs = self.create_vectors(filename)
		self.collect_word_vecs(vecs)



"""
Hyperparameters:
	Should I remove stopwords and/or punctuation in get_word_vecs()?
	Size of window (self.window_size)
"""





