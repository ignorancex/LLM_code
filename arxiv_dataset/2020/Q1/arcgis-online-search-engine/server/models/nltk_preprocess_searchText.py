from bs4 import BeautifulSoup
import re
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sys
import os

# from collections import defaultdict

import numpy as np

# import gensim
# from gensim.models.doc2vec import TaggedDocument
# from gensim.models import Phrases
# from gensim import corpora

import warnings

def preprocessText(descrip):
	# get rid of the HTML tags
	# soup = BeautifulSoup(descrip, 'html5lib')
	# descrip = soup.get_text()

	
	filtered_words = [word for word in descrip.split() if word not in stopwords.words('english')]
	descrip = " ".join(filtered_words)

	# find and delete URL from the description
	descrip = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", descrip)
	

	# get rid of the other symbol and change all word ro lower case
	descrip = re.sub(r'[^a-zA-Z_ ]', "", descrip)

	return descrip

def textLemmatization(descrip):
	# Do token based Lemmatization
	no_lem_list = descrip.split()
	descrip = descrip.lower()
	tokens = nltk.word_tokenize(descrip)
	lemmatizer = WordNetLemmatizer()
	
	jsonData = {}
	for i in range(len(tokens)):
		no_lem = no_lem_list[i]
		lem = lemmatizer.lemmatize(tokens[i])
		jsonData[no_lem] = lem

	# lem = [lemmatizer.lemmatize(t) for t in tokens]
	return jsonData


if __name__ == "__main__":

	searchText = sys.argv[1]
	# searchText = "disaster in california"
	
	search_doc = preprocessText(searchText)
	jsonData = textLemmatization(search_doc)

	print(json.dumps(jsonData))