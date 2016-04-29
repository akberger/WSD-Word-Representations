#!/usr/bin/python

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from WordEnvVecs import WordEnvironmentVectors
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from operator import itemgetter
from nltk.corpus import stopwords
import numpy as np
import os
import sys

def main(indir, outdir, model_dir, clustering='kmeans', n=2):
	files = iter_dir(indir)
	wev = WordEnvironmentVectors()
	print 'Determining vocabulary...'
	print datetime.now()
	vocab, sent_count = determine_vocab(files, wev)
	wev.vocab = vocab

	print "Creating word sense models from word env vecs..."
	print datetime.now()
	word_sense_clusters = {}
	cluster_inertias = {}
	word_svds = None 
	for file_chunk in [files[i:i+n] for i in xrange(0, len(files), 100)]:
		for fi in file_chunk:
			wev.create(fi)
		word_sense_clusters = make_clusters(word_sense_clusters, wev, clustering, n)
		wev.clear_vectors()

	#print "Writing updated text file with words marked w/ their sense..."
	output = open(outdir, 'w')
	print "Relabeling words with their sense..."
	print datetime.now()
	model = None

	for fi in files:
		f = open(fi)
		sents = wev.tokenize(wev.get_file_text(f))
		for s in sents:
			word_vecs = wev.get_word_vecs(s)
			new_sent = determine_sense(s, word_vecs, word_sense_clusters, word_svds)

			"""
			if model:
				model.train(new_sent, total_examples=sent_count)
			else:
				model = Word2Vec(new_sent, size=200, window=5, min_count=5, workers=3, sg=1)
			"""
			output.write(' '.join(new_sent) + '\n')

	print "Creating Word2Vec model..."
	print datetime.now()
	model = Word2Vec(LineSentence(outdir), size=200, window=5, min_count=5, workers=3, sg=1)

	model.init_sims(replace=True)

	model.save(model_dir)
	print "Model saved"
	print datetime.now()

def iter_dir(indir, filetype='.txt'):
	files = []
	for dirpath, dirname, filename in os.walk(indir):
		for f in filename:
			if f.endswith(filetype):
				files.append(os.path.join(dirpath, f))

	return files


def determine_vocab(files, wev, n=2):
	total_sentences = 0
	vocab = {}
	stops = wev.stopwords
	punc = wev.punctuation

	for fi in files:
		f = open(fi)
		sents = wev.tokenize(wev.get_file_text(f))
		for s in sents:
			total_sentences += 1
			for w in s:
				if w not in stops and w not in punc:
					vocab[w] = vocab.get(w, 0) + 1

	#remove items occuring less than n times
	cut_vocab = {k:v for k,v in vocab.items() if v >= n}
	#turn counts into feature indices 
	for i, item in enumerate(cut_vocab):
		cut_vocab[item] = i
	
	return cut_vocab, total_sentences

def make_clusters(word_sense_clusters, wev, clustering, n=3):
	#print "Clustering word vectors..."
	for word in wev.word_vectors:
		if len(wev.word_vectors[word]) >= n:

			#svd = TruncatedSVD(n_components=50)
			vecs = np.array(wev.word_vectors[word])
			#dense_vecs = svd.fit_transform(np.array(vecs))

			#TODO: implement algorithm for determining best value of n_clusters
			if word in word_sense_clusters:
				word_sense_clusters[word].partial_fit(vecs)
			else:
				#word_sense_clusters[word] = MiniBatchKMeans(n_clusters=n, init='k-means++', verbose=False).partial_fit(vecs)
				word_sense_clusters[word] = Birch(n_clusters=None).partial_fit(vecs)

	return word_sense_clusters



def determine_sense(sent, word_vecs, word_sense_clusters, word_svds):
	new_sent = []
	for i, vec in enumerate(word_vecs):
		word = sent[i]
		if vec is not None:
			if word in word_sense_clusters:
				#new_vec = word_svds[word].transform(vec)
				#label = word_sense_clusters[word].predict(new_vec)
				vec = np.array(vec).reshape(1,-1)
				label =word_sense_clusters[word].predict(vec)
				new_sent.append(word+str(label[0])) #label word with its sense number
			else:
				new_sent.append(word)
		else:
			new_sent.append(word)

	return new_sent


if __name__ == '__main__':
	indir = sys.argv[1]
	outdir = sys.argv[2]
	model_dir = sys.argv[3]

	main(indir, outdir, model_dir, clustering='kmeans')

