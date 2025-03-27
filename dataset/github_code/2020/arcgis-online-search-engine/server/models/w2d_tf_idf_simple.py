# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import re
import json
import nltk
from nltk.stem import WordNetLemmatizer

import sys
import os

# from collections import defaultdict

import numpy as np

import gensim
# from gensim.models.doc2vec import TaggedDocument
# from gensim.models import Phrases
# from gensim import corpora

import warnings

def rewriteExistingFile(fileName):
	outfile = open(fileName, "w")
	outfile.write("")
	outfile.close()


def readJSONData(jsonFileName):
	json_data = open(jsonFileName, "r")
	content = json.load(json_data)
	print("Read %d Json Data successfully" % (len(content)))
	return content

def preprocessText(descrip):
	# get rid of the HTML tags
	soup = BeautifulSoup(descrip, 'html5lib')
	descrip = soup.get_text()

	

	# find and delete URL from the description
	descrip = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", descrip)
	

	# get rid of the other symbol and change all word ro lower case
	descrip = re.sub(r'[^a-zA-Z ]', "", descrip).lower()

	return descrip

def textLemmatization(descrip):
	# Do token based Lemmatization
	tokens = nltk.word_tokenize(descrip)
	lemmatizer = WordNetLemmatizer()
	lem = [lemmatizer.lemmatize(t) for t in tokens]
	return " ".join(lem)

def simpleSemanticSearch(searchText, w2vEmbeddingSize, w2vModelFileName, w2dModelFilePath, topn):
	search_doc = preprocessText(searchText)
	search_doc = textLemmatization(search_doc)

	w2v_model = gensim.models.Word2Vec.load(w2vModelFileName)

	cur_search_vec = np.zeros(w2vEmbeddingSize)
	valid_word_count = 0
	for token in search_doc.split():
		if token in w2v_model.wv.vocab:
			cur_search_vec += w2v_model.wv[token]
			valid_word_count += 1

	if valid_word_count == 0:
		valid_word_count = 1

	cur_search_vec = cur_search_vec/(valid_word_count*1.0)

	# print cur_search_vec

	w2d_model = gensim.models.KeyedVectors.load_word2vec_format(w2dModelFilePath)

	similar_vec = w2d_model.most_similar(positive=[cur_search_vec], topn=topn)

	return similar_vec

if __name__ == "__main__":
	cur_dir = os.path.abspath(os.curdir)

	w2vEmbeddingSize = 300

	searchText = sys.argv[1]
	topn = int(sys.argv[2])

	w2vModelFilePath = os.path.join( os.path.join(os.path.join(cur_dir, "server"), "data"), "arcgis_online_w2v.model")

	w2dModelFilePath = os.path.join( os.path.join(os.path.join(cur_dir, "server"), "data"), "arcgis_online_w2d_tf_idf.txt")

	results = simpleSemanticSearch(searchText, w2vEmbeddingSize, w2vModelFilePath, w2dModelFilePath, topn)

	print(json.dumps(results))

