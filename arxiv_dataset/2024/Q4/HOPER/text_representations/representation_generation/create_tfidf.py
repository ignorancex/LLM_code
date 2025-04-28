import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tqdm import tqdm

ufiles_path = ''
pfiles_path = ''

'''
create_reps(tp): This function takes a parameter tp which represents the type of data ("uniprot", "pubmed", or "uniprotpubmed"). It reads files from the specified paths (ufiles_path and pfiles_path), 
concatenates their contents based on the data type, and applies the TF-IDF vectorization technique using the TfidfVectorizer from scikit-learn. It creates a pandas DataFrame (df1) where 
each row represents a file, and each column represents a feature (word). The DataFrame is then saved as a CSV file. It also performs Principal Component Analysis (PCA) on the DataFrame 
using different numbers of components (256, 512, 1024, 2048), and saves the resulting PCA-transformed vectors as separate CSV files.
'''

def create_reps(tp):
    files = os.listdir(pfiles_path)
    file_list = []
    print('Fitting vectorizer...')
    for i in tqdm(range(len(files))):
        if tp == 'uniprot':
            contentu = open(ufiles_path + files[i])
            file_list.append(contentu.read())
        elif tp == 'pubmed':
            contentp = open(pfiles_path + files[i])
            file_list.append(contentp.read())
        elif tp == 'uniprotpubmed':
            contentu = open(ufiles_path + files[i])
            contentp = open(pfiles_path + files[i])
            file_list.append(contentu.read() + contentp.read())

    vectorizer = TfidfVectorizer(use_idf=True)
    fitted_vectorizer = vectorizer.fit(file_list)
    feature_names = fitted_vectorizer.get_feature_names()

    cols = feature_names
    cols.append('Entry')
    df1 = pd.DataFrame(columns=cols)
    del feature_names[-1]
    print('Creating ' + tp + ' vectors...')
    for i in tqdm(range(20)):
    #for i in tqdm(range(len(files))):
        file_content = []
        if tp == 'uniprot':
            contentu = open(ufiles_path + files[i])
            file_content.append(contentu.read())
        elif tp == 'pubmed':
            contentp = open(pfiles_path + files[i])
            file_content.append(contentp.read())
        elif tp == 'uniprotpubmed':
            contentu = open(ufiles_path + files[i])
            contentp = open(pfiles_path + files[i])
            file_content.append(contentu.read() + contentp.read())

        transformed_vector = fitted_vectorizer.transform(file_content)
        dense = transformed_vector.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        df['Entry'] = files[i][:-4]
        df1 = df1.append(df, ignore_index=True)

    path=os.getcwd()
    entry = df1.Entry
    df1.to_csv(os.path.join(path,'text_representations/representation_generation/tfidf_representations/' + tp + '_tfidf_vectors.csv'), index = False)
    
    '''
    df1.drop('Entry', inplace=True, axis=1)
    print('Performing principal component analysis...')
    
    pca = PCA(n_components=256)
    principalComponents = pca.fit_transform(df1)
    pca_vectors = pd.DataFrame(principalComponents)
    pca_vectors.insert(0, "Entry", entry, True)
    pca_vectors.to_csv(os.path.join(path,'text_representations/representation_generation/tfidf_representations/' + tp + '_tfidf_vectors_pca256_multi_col.csv'), index = False)

    pca = PCA(n_components=512)
    principalComponents = pca.fit_transform(df1)
    pca_vectors = pd.DataFrame(principalComponents)
    pca_vectors.insert(0, "Entry", entry, True)
    pca_vectors.to_csv(os.path.join(path,'text_representations/representation_generation/tfidf_representations/' + tp + '_tfidf_vectors_pca512_multi_col.csv'), index = False)

    pca = PCA(n_components=1024)
    principalComponents = pca.fit_transform(df1)
    pca_vectors = pd.DataFrame(principalComponents)
    pca_vectors.insert(0, "Entry", entry, True)
    pca_vectors.to_csv(os.path.join(path,'text_representations/representation_generation/tfidf_representations/' + tp + '_tfidf_vectors_pca1024_multi_col.csv'), index = False)

    pca = PCA(n_components=2048)
    principalComponents = pca.fit_transform(df1)
    pca_vectors = pd.DataFrame(principalComponents)
    pca_vectors.insert(0, "Entry", entry, True)
    pca_vectors.to_csv(os.path.join(path,'text_representations/representation_generation/tfidf_representations/' + tp + '_tfidf_vectors_pca2048_multi_col.csv'), index = False)
    '''
def main():
    create_reps("uniprot")
    create_reps("pubmed")
    create_reps("uniprotpubmed")


