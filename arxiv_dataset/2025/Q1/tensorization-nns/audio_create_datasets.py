import pandas as pd
import os

cwd = os.getcwd()


def balance_dataset(df, var, p, max_size=None):
    """Samples from dataset to keep values of `var` balanced with proportion `p`"""
    if max_size is None:
        max_size = len(df)
    
    gdf = df.groupby('client_id', group_keys=False)[df.columns]
    gdf = gdf.apply(lambda x: x.sample(n=1), include_groups=True)
        
    len_class0 = p / (1 - p) * len(gdf[gdf[var] == 1])
    len_class1 = len(gdf[gdf[var] == 1])
    
    if len_class0 > len(gdf[gdf[var] == 0]):
        q = 1 - p
        len_class1 = q / (1 - q) * len(gdf[gdf[var] == 0])
        len_class0 = len(gdf[gdf[var] == 0])
    
    class0_df = gdf[gdf[var] == 0].sample(n=int(len_class0))
    class1_df = gdf[gdf[var] == 1].sample(n=int(len_class1))
    
    balanced_df = pd.concat([class0_df, class1_df])
    balanced_df = df[df['client_id'].isin(balanced_df['client_id'].unique().tolist())]
    
    if len(balanced_df) > max_size:
        balanced_df = balanced_df.sample(frac=max_size / len(balanced_df))
    
    balanced_df = balanced_df.sample(frac=1.)
    return balanced_df


# Load train dataset (we will use just this as dataset from which we will
# create our custom train / val / test splits)
df = pd.read_csv(os.path.join(cwd, 'CommonVoice', 'train.tsv'),
                 sep='\t',
                 header=0)

# This is renamed as sex to make it more evident that we will treat this as a
# binary classification problem, distinsguishing voices as "male" or "female"
df.rename(columns={'gender': 'sex'}, inplace=True)

# Datasets with duration of audios in ms
duration = pd.read_csv(os.path.join(cwd, 'CommonVoice', 'clip_durations.tsv'),
                       sep='\t',
                       header=0)
duration.rename(columns={'clip': 'path', 'duration[ms]': 'duration'},
                inplace=True)

# Join dataset with duration
df = df.join(duration.set_index('path'), on='path')

# Keep audios that long at least 1000 ms
df = df[df['duration'] >= 1000]

# Clean sex and accents
df = df[~df['sex'].isna() & ~df['accents'].isna()]

# Keep at most some number of samples from each client
samples_per_client = 10
df = df.groupby('client_id', group_keys=False)[df.columns]
df = df.apply(lambda x: x.sample(n=min(samples_per_client, len(x))),
              include_groups=True)
# print(len(df))

# Turn sex into binary feature
df['sex'] = df['sex'].replace({'male': 1, 'female': 0})

# Add binary column for canadian and non-canadian english speakers
df['canadian'] = (df['accents'] == 'Canadian English').astype(int)
df['english'] = (df['accents'] == 'England English').astype(int)
# print(len(df))

# Remove elements that are not canadian or english
df = df[(df['canadian'] == 1) | (df['english'] == 1)]
# print(len(df))

# Keep relevant columns (only keep canadian as binary feature, if it's 0,
# it means it is english)
df = df.loc[:, ['client_id', 'path', 'sex', 'age', 'canadian']]

# Balance sex class in both groups canadian and english
df_canadian = df[df['canadian'] == 1]
df_canadian = balance_dataset(df_canadian, 'sex', 0.5)
# print(len(df_canadian))

df_english = df[df['canadian'] == 0]
df_english = balance_dataset(df_english, 'sex', 0.5)
# print(len(df_english))

df = pd.concat([df_canadian, df_english]).sample(frac=1.)
# print(len(df))


# Save dataset
os.makedirs(os.path.join(cwd, 'CommonVoice', 'datasets'), exist_ok=True)
df.to_csv(os.path.join(cwd, 'CommonVoice', 'datasets', 'full_df.tsv'),
          sep='\t',
          header=True,
          index=False)


# Create imbalanced datasets with different proportions of english accent
p_english_list = [0.005, 0.01, 0.05,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.95, 0.99, 0.995]

for p_english in p_english_list:
    imbalanced_df = balance_dataset(df, 'canadian', p_english, 20_000)
    
    # For each proportion of imbalance, we sample 10 datasets (sampling 80% of
    # the clients in the dataset)
    for idx in range(10):
        imbalanced_df = imbalanced_df.sample(frac=1.)
        
        client_list = imbalanced_df['client_id'].unique().tolist()
        aux_df = imbalanced_df[imbalanced_df['client_id'].isin(
            client_list[:int(0.8 * len(client_list))]
            )]
        
        os.makedirs(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx)),
            exist_ok=True)
        
        aux_test = imbalanced_df[~imbalanced_df.index.isin(aux_df.index)]
        aux_test.to_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'test_df.tsv'),
            sep='\t',
            header=True,
            index=False)
        
        # Each dataset is split in train/val (75%/25%) -> Total dataset is 80/20/20
        client_list = aux_df['client_id'].unique().tolist()
        aux_train = aux_df[aux_df['client_id'].isin(
            client_list[:int(0.75 * len(client_list))]
            )]
        aux_val = aux_df[~aux_df.index.isin(aux_train.index)]
        
        aux_train.to_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'train_df.tsv'),
            sep='\t',
            header=True,
            index=False)
        aux_val.to_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'val_df.tsv'),
            sep='\t',
            header=True,
            index=False)


# Split each test dataset into n_samples samples that will be used to tensorize
# the models trained with the corresponding datasets, and the rest which will be
# used to test the tensorized models
p_english_list = [0.005, 0.01, 0.05,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.95, 0.99, 0.995]

aux_df = df.groupby('client_id', group_keys=False)['client_id']
aux_df = aux_df.apply(lambda x: len(x), include_groups=True)
samples_per_client = aux_df.mean()

n_samples = 200
n_clients = int(n_samples // samples_per_client)

for p_english in p_english_list:
    for idx in range(10):
        aux_test = pd.read_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'test_df.tsv'),
            sep='\t',
            header=0)
        aux_test = aux_test.sample(frac=1)
        
        client_list = aux_test['client_id'].unique().tolist()
        aux_test_tensorize = aux_test[aux_test['client_id'].isin(
            client_list[:n_clients])]
        aux_test_unused = aux_test[~aux_test.index.isin(aux_test_tensorize.index)]
        
        # Check that we get enough samples in each dataset
        assert len(aux_test_tensorize) >= 100
        
        aux_test_tensorize.to_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'test_df_tensorize.tsv'),
            sep='\t',
            header=True,
            index=False)
        
        aux_test_unused.to_csv(
            os.path.join(cwd, 'CommonVoice', 'datasets',
                         str(p_english), str(idx), 'test_df_unused.tsv'),
            sep='\t',
            header=True,
            index=False)
