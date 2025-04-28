from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import contingency_matrix

from collections import Counter
import archetypes as arch


categories = ['rec.autos', 'rec.sport.baseball', 'talk.politics.mideast']

newsgroups = fetch_20newsgroups(categories=categories)
data = newsgroups.data[:]
target = newsgroups.target[:]

target_names = newsgroups.target_names

vectorizer = TfidfVectorizer(stop_words='english', min_df=0.02, max_df=0.15)

X = vectorizer.fit_transform(data)
words = vectorizer.get_feature_names()

print(X.shape)

bicluster = arch.BiAA(n_archetypes=(3, 3), random_state=123)
bicluster.fit(X.toarray())
