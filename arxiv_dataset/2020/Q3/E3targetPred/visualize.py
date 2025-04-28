## Visualize Embeddings
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from keras.models import Model, load_model

from utils import *
from models import *
from perf_metrics import *
from feature_encodings import *

def get_PCA_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type):
	## PCA Embedding
	if feature_type == "input":
		target_pair_features =  np.concatenate((E3_features,Sub_features), axis=1)
	else:
		target_pair_features = Pair_features

	PCA_Embedding = PCA(n_components=2).fit_transform(target_pair_features)
	dt = pd.concat([pd.DataFrame(PCA_Embedding), 
		pd.DataFrame({'Label': target_label.argmax(axis=1)})], 
		axis=1).rename(columns={0:'x', 1:'y'})

	ax = sns.scatterplot(x='x', y='y', hue = 'Label', 
	                     palette = ['orange', 'blue'], alpha = 0.2, data=dt)

	sns.scatterplot(x="x", y="y", data=dt[dt.Label == 0.0], 
	                alpha=1.0, ax=ax)

	if feature_type == "input":
		plt.savefig('../figures/pca_feature_space_concatenated_features.png')
	else:
		plt.savefig('../figures/pca_feature_space_combined_features.png')


def get_TSNE_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type):
	## PCA Embedding
	if feature_type == "input":
		target_pair_features =  np.concatenate((E3_features,Sub_features), axis=1)
	else:
		target_pair_features = Pair_features

	TSNE_Embedding = TSNE(n_components=2).fit_transform(target_pair_features)
	dt = pd.concat([pd.DataFrame(TSNE_Embedding), 
		pd.DataFrame({'Label': target_label.argmax(axis=1)})], 
		axis=1).rename(columns={0:'x', 1:'y'})

	ax = sns.scatterplot(x='x', y='y', hue = 'Label', 
	                     palette = ['orange', 'blue'], alpha = 0.2, data=dt)

	sns.scatterplot(x="x", y="y", data=dt[dt.Label == 0.0], 
	                alpha=1.0, ax=ax)

	if feature_type == "input":
		plt.savefig('../figures/tsne_feature_space_concatenated_features.png')
	else:
		plt.savefig('../figures/tsne_feature_space_combined_features.png')

def get_UMAP_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type):
	## PCA Embedding
	if feature_type == "input":
		target_pair_features =  np.concatenate((E3_features,Sub_features), axis=1)
	else:
		target_pair_features = Pair_features

	UMAP_Embedding = umap.UMAP().fit_transform(target_pair_features)
	dt = pd.concat([pd.DataFrame(UMAP_Embedding), 
		pd.DataFrame({'Label': target_label.argmax(axis=1)})], 
		axis=1).rename(columns={0:'x', 1:'y'})

	ax = sns.scatterplot(x='x', y='y', hue = 'Label', 
	                     palette = ['orange', 'blue'], alpha = 0.2, data=dt)

	sns.scatterplot(x="x", y="y", data=dt[dt.Label == 0.0], 
	                alpha=1.0, ax=ax)

	if feature_type == "input":
		plt.savefig('../figures/umap_feature_space_concatenated_features.png')
	else:
		plt.savefig('../figures/umap_feature_space_combined_features.png')


def get_LSE_Embedding(E3_features, Sub_features, Pair_features, target_label, LV, Gap, lmd):
	Xin_train, y_train, Xin_test, y_test, model = Train_LSE(LV, Gap, lmd, E3_features, Sub_features, Pair_features, target_label)

	layer_outputs = [layer.output for layer in model.layers[1:14]]
	activation_model = Model(inputs = model.input, outputs= layer_outputs)

	train_embedding = activation_model.predict(Xin_train,batch_size=200, verbose=0)[12]
	test_embedding = activation_model.predict(Xin_test,batch_size=200, verbose=0)[12]
	train_label = y_train.argmax(axis=1)
	test_label = y_test.argmax(axis=1)

	dt_train = pd.concat([pd.DataFrame(train_embedding), 
		pd.DataFrame({'Label': train_label})], axis=1).rename(columns={0:'x', 1:'y'})
	dt_test = pd.concat([pd.DataFrame(test_embedding), 
		pd.DataFrame({'Label': test_label})], axis=1).rename(columns={0:'x', 1:'y'})
	dt = pd.concat([dt_train, dt_test], axis=0)

	if lmd == 1.0:
		## Train & Test Embedding
		fig1, ax = plt.subplots()
		ax = sns.scatterplot(x='x', y='y', hue = 'Label', palette = ['orange', 'blue'], alpha = 0.2, data=dt)
		sns.scatterplot(x="x", y="y", data=dt[dt.Label == 0.0], palette = ['orange'], alpha=1.0, ax=ax)
		plt.savefig('../figures/AE_Embedding.png')

	else:
		## Train Embedding
		fig1, train_ax = plt.subplots()
		train_ax = sns.scatterplot(x='x', y='y', hue = 'Label', palette = ['orange', 'blue'], alpha = 0.2, data=dt_train)
		sns.scatterplot(x="x", y="y", data=dt_train[dt_train.Label == 0.0], palette = ['orange'], alpha=1.0, ax=train_ax)
		plt.savefig('../figures/LSE_lambda_' + str(lmd) + '_Embedding_train.png')

		# Test Embedding
		fig2, test_ax = plt.subplots()
		sns.scatterplot(x='x', y='y', hue = 'Label', palette = ['orange', 'blue'], alpha = 0.2, data=dt_test)
		sns.scatterplot(x="x", y="y", data=dt_test[dt_test.Label == 0.0], palette = ['orange'], alpha=1.0, ax=test_ax)
		plt.savefig('../figures/LSE_lambda_' + str(lmd) + '_Embedding_test.png')


