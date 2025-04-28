## Visualize Embedding
from visualize import *
from Feature_encodings import * ## Feature encoding utils
from keras.utils.np_utils import to_categorical

## Loading and pre-processing E3-Target dataset
data_path = 'https://raw.githubusercontent.com/psychemistz/Colab_temp/master/E3target_pred/1metadata.csv'
dataset = pd.read_csv(data_path, index_col=None)

## Extract features
E3_features, Sub_features, Pair_features = Extract_Features(dataset, gap_size=6)
target_label = to_categorical(dataset['Label'])

## PCA Embedding
get_PCA_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'input') # Concatenated CKSAAP Feature
get_PCA_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'output') # Combined CKSAAP Feature

## TSNE Embedding
get_TSNE_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'input') # Concatenated CKSAAP Feature
get_TSNE_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'output') # Combined CKSAAP Feature

## UMAP Embedding
get_UMAP_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'input') # Concatenated CKSAAP Feature
get_UMAP_Embedding(E3_features, Sub_features, Pair_features, target_label, feature_type = 'output') # Combined CKSAAP Feature

## AE Embedding 
get_LSE_Embedding(E3_features, Sub_features, Pair_features, target_label, LV=2, Gap=6, lmd=1.0)

## LSE Embedding
get_LSE_Embedding(E3_features, Sub_features, Pair_features, target_label, LV=2, Gap=6, lmd=0.99)
get_LSE_Embedding(E3_features, Sub_features, Pair_features, target_label, LV=2, Gap=6, lmd=0.50)
get_LSE_Embedding(E3_features, Sub_features, Pair_features, target_label, LV=2, Gap=6, lmd=0.01)