
import os
import numpy as np
import pandas as pd 

datasets = {
    'sonar': 'sonar.data',
    'heart_disease': 'heart_disease.data',
    'ionosphere': 'ionosphere.data',
    'musk': 'musk1.data',
    'breast_cancer': 'breast-cancer-wisconsin.data',
    'diabetes': 'pima_indians_diabetes.csv',
    'spambase': 'spambase.data',
    'phoneme': 'phoneme.csv',
    'mammography': 'mammography.csv'
}

raw = 'data/classification/'
processed = os.path.join(raw, 'processed')

if __name__ == '__main__':
    for dataset in datasets:
        data = pd.read_csv(raw + datasets[dataset], header=None)
        ### Label encoding of target classes
        if dataset == 'sonar':
            target = data.iloc[:,-1].apply(lambda x: 1 if x=='M' else 0)
        elif dataset == 'ionosphere':
            target = data.iloc[:,-1].apply(lambda x: 1 if x=='g' else 0)
        elif dataset == 'breast_cancer':
            target = data.iloc[:,-1].apply(lambda x: 1 if x==2 else 0)
        elif dataset == 'mammography':
            target = data.iloc[:,-1].apply(lambda x: 1 if x=="'-1'" else 0)
        else:
            target = data.iloc[:,-1].apply(lambda x: 1 if x==0 else 0)
        # Convert target to numpy arrays
        target = target.values.astype('float64')[:,np.newaxis]

        ### Missing values inputing
        if dataset == 'breast_cancer':
            # Enconded as '?'. Replace with mean of columns.
            data.replace('?', np.nan, inplace=True)
            m6 = np.nanmean(data.iloc[:,6].values.astype('float'))
            data.iloc[:,6].replace(np.nan, m6, inplace=True)
        elif dataset == 'heart_disease':
            # Enconded as '?'. Replace with mean of columns.
            data.replace('?', np.nan, inplace=True)
            m11 = np.nanmean(data.iloc[:,11].values.astype('float'))
            m12 = np.nanmean(data.iloc[:,12].values.astype('float'))
            data.iloc[:,11].replace(np.nan, m11, inplace=True)
            data.iloc[:,12].replace(np.nan, m12, inplace=True)
        
        ### Feature positions
        if dataset == 'breast_cancer':
            features = data.iloc[:,1:-1].values.astype('float64')
        elif dataset == 'musk':
            features = data.iloc[:,2:-1].values.astype('float64')
        else:
            features = data.iloc[:,:-1].values.astype('float64')

        # Save processed dataset
        pd.DataFrame(np.concatenate((features, target), axis=1)).\
            to_csv(f'{processed}/{dataset}.csv', index=False, header=False)