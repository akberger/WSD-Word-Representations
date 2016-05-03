#!/usr/bin/python

from WordEnvVecs import WordEnvironmentVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from itertools import islice
import gzip
import os
import sys

def main(indir, outdir, word_list_file, model_dir, vec_files_path, clustering='kmeans', iters=3):
	vecs_made = False
	files = iter_dir(indir)
	words = get_words(word_list_file)
	wev = WordEnvironmentVectors(words, vec_files_path)

	print 'Determining vocabulary {}'.format(datetime.now())
	vocab = determine_vocab(files, wev)
	wev.vocab = vocab
	
	"""Sequence of 3 commands for doing WSD then creating word2vec model with sense labeled words"""
	word_sense_clusters = create_word_sense_models(files, wev, clustering)
	write_relabeled_text(files, word_sense_clusters, outdir, wev)
	create_word2vec_model(outdir, model_dir)
	
	for i in range(iters):
		print "Performing iteration {} of model refinement on {} {}".format(i+1, model_dir, datetime.now())
		gold_word_senses = refine_senses(model_dir, words)
		word_sense_clusters = create_word_sense_models(files, wev, clustering, gold_senses=gold_word_senses, make_vec_files=False)
		write_relabeled_text(files, word_sense_clusters, outdir+str(i), wev)
		create_word2vec_model(outdir+str(i), model_dir+str(i))
		model_dir = model_dir+str(i)

def create_word_sense_models(files, wev, clustering, gold_senses=None, make_vec_files=True ,chunk_size=10):
	print "Creating word sense models from WEV {}".format(datetime.now())

	if make_vec_files:
		for file_chunk in [files[i:i+chunk_size] for i in xrange(0, len(files), 2*chunk_size)]:
			for fi in file_chunk:
				wev.create(fi)
	return make_clusters(wev, clustering, gold_senses)

def write_relabeled_text(files, word_sense_clusters, outdir, wev):
	print "Relabeling words with their sense {}".format(datetime.now())
	with gzip.open(outdir + '.gz', 'wb') as output:
		for fi in files:
			f = open(fi)
			sents = wev.tokenize(wev.get_file_text(f))
			for s in sents:
				word_vecs = wev.get_word_vecs(s)
				new_sent = determine_sense(s, word_vecs, word_sense_clusters)
				output.write(' '.join(new_sent) + '\n')

def create_word2vec_model(outdir, model_dir):
	print "Creating Word2Vec model {}".format(datetime.now())
	model = Word2Vec(LineSentence(outdir + '.gz'), size=200, window=5, min_count=1, workers=3, sg=1)
	model.init_sims(replace=True)
	model.save(model_dir)


def make_clusters(wev, clustering, gold_senses, n=25):
	print "Creating cluster models"
	word_sense_clusters = {}
	for word in wev.word_vectors:
		with open(wev.word_vectors[word], 'r') as vec_file:
			while True:
				vecs = list(islice(vec_file, n))
				#print vecs
				if not vecs: break

				if len(vecs) >= 2:
					try:
						vecs = np.array(([[float(x) for x in i.split()] for i in vecs]))
						#print vecs
						#sys.exit(0)
						if word in word_sense_clusters:
							word_sense_clusters[word].partial_fit(np.array(vecs))
						else:
							if gold_senses:
								#print 'gold_senses', gold_senses[word], word
								#word_sense_clusters[word] = Birch(n_clusters=gold_senses[word]).partial_fit(np.array(vecs))
								word_sense_clusters[word] = MiniBatchKMeans(n_clusters=gold_senses[word]).partial_fit(np.array(vecs))
							else:
								#word_sense_clusters[word] = Birch(n_clusters=10).partial_fit(vecs)
								word_sense_clusters[word] = MiniBatchKMeans(n_clusters=5).partial_fit(vecs)
					except (ValueError): pass
					
	return word_sense_clusters

def get_words(word_list_file):
	words = []
	f = open(word_list_file, 'r')
	for line in f:
		words.append(line.strip())

	return set(words)

def iter_dir(indir, filetype='.txt'):
	files = []
	for dirpath, dirname, filename in os.walk(indir):
		for f in filename:
			if f.endswith(filetype):
				files.append(os.path.join(dirpath, f))

	return files


def determine_vocab(files, wev, n=15):
	vocab = {}
	stops = wev.stopwords
	punc = wev.punctuation

	for fi in files:
		f = open(fi)
		sents = wev.tokenize(wev.get_file_text(f))
		for s in sents:
			for w in s:
				if w not in stops and w not in punc:
					vocab[w] = vocab.get(w, 0) + 1

	#remove items occuring less than n times
	cut_vocab = {k:v for k,v in vocab.items() if v >= n}
	print len(vocab), len(cut_vocab)
	#turn counts into feature indices 
	for i, item in enumerate(cut_vocab):
		cut_vocab[item] = i
	
	return cut_vocab 

def determine_sense(sent, word_vecs, word_sense_clusters):
	new_sent = []
	for i, vec in enumerate(word_vecs):
		word = sent[i]
		if vec is not None:
			if word in word_sense_clusters:
				vec = np.array(vec).reshape(1,-1).astype(float)
				#print len(word_sense_clusters[word].cluster_centers_)
				label = word_sense_clusters[word].predict(vec)
				new_sent.append(word+str(label[0])) #label word with its sense number
			else:
				new_sent.append(word)
		else:
			new_sent.append(word)

	return new_sent

def refine_senses(model, words):
	"""
	Determine a more accurate number of senses for each word based on the most_similar
	senses of each sense of each word in the model
	"""
	model = Word2Vec.load(model)
	gold_word_senses = {}
	
	for w in words:
		senses = get_senses(model, w)
		if len(senses) > 1:
			sense_overlaps = find_overlaps(senses)
			if sense_overlaps:
				gold_word_senses[w] = determine_num_senses(sense_overlaps)
				#print gold_word_senses[w], w
			else:
				gold_word_senses[w] = add_senses(senses)
				#print gold_word_senses[w], w
		elif len(senses) == 1:
			gold_word_senses[w] = add_senses(senses)
			#print gold_word_senses[w], w
		else:
			gold_word_senses[w] = 2

	with open('/Users/adamberger/Desktop/CLMasters/Word_Representation_WSD/gold_word_senses.txt', 'w') as f:
		for word in gold_word_senses:
			f.write(word + ' ' + str(gold_word_senses[word]) + '\n')


	return gold_word_senses

def get_senses(model, word, n=10):
	senses = {}
	
	for i in range(n):
		sense = word+str(i)
		try: senses[sense] = model.most_similar(sense)
		except (KeyError): pass

	try: senses[word] = model.most_similar(word)
	except (KeyError): pass

	return senses

def find_overlaps(senses):
	"""See if any of the senses have another of the senses in their most_similar list"""
	overlaps = {s : None for s in senses}
	sense_set = set(overlaps.keys())
	for s in senses:
		overlaps[s] = sense_set.intersection(set([w for w,sim in senses[s]]+[s]))

	for s in overlaps:
		if len(overlaps[s]) > 1: return overlaps
	return None

def determine_num_senses(sense_overlaps):
	"""
	Determine the number of senses the word should have. This is done by merging all
	senses that contain another of the senses in their most_similar list into a group. 
	Return the number of resulting groups.
	"""
	groups = [set(v) for v in sense_overlaps.values()]
	for x, g1 in enumerate(groups):
		for y, g2 in enumerate(groups):
			if x != y:
				if g1.intersection(g2):
					groups[x] = g1.union(g2)
					groups[y] = g1.union(g2)

	unique_groups = np.unique([list(g) for g in groups])
	#if there is just one group it is not a nested list, so the length of that
	#list is what will be returned instead of 1
	if unique_groups.dtype == 'S4':
		return 1
	else:
		return len(unique_groups)

def add_senses(senses, threshold = 0.7):
	count = len(senses)
	"""
	If there was no overlap between senses, maybe there weren't enough. In the
	next iteration, use one more sense for each sense that had an average most_similar
	value lower than the threshold.
	"""
	for s in senses:
		average_sim = sum([sim for w,sim in senses[s]])/float(len(senses[s]))
		if average_sim < threshold:
			count += 1

	return count

def _shape_repr(shape):
	"""taken from sklearn.utils.validation.py"""
	if len(shape) == 0:
		return "()"
	joined = ", ".join("%d" % e for e in shape)
	if len(shape) == 1:
		#special notation for singleton tuples
		joined += ","
	return "(%s)" % joined


if __name__ == '__main__':
	indir = sys.argv[1]
	outdir = sys.argv[2]
	word_list_file = sys.argv[3]
	model_dir = sys.argv[4]
	vec_files_path = sys.argv[5]

	main(indir, outdir, word_list_file, model_dir, vec_files_path, clustering='kmeans')

#TODO: run again on OANC, but change min_count for word2vec back to 5
#don't need to do first pass. maybe increase iterations by one (or two)
#not bad ones: 'road' (m1-m3), 'orange', 'apple'
