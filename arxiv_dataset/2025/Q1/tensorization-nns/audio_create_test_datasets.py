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


# Load validated dataset for the privacy attacks
df = pd.read_csv(os.path.join(cwd, 'CommonVoice', 'validated.tsv'),
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

# Load train dataset to keep in validated only clients not seen in training
df_train = pd.read_csv(os.path.join(cwd, 'CommonVoice', 'train.tsv'),
                       sep='\t',
                       header=0)
df_train.rename(columns={'gender': 'sex'}, inplace=True)

# Clean sex and accents
df_train = df_train[~df_train['sex'].isna() & ~df_train['accents'].isna()]

# Keep only 1 sample from each client
df_train = df_train.groupby('client_id', group_keys=False)[df_train.columns]
df_train = df_train.apply(lambda x: x.sample(n=1),
                          include_groups=True)

# Exclude clients in training datasets
# print(len(df))
df = df[~df['client_id'].isin(df_train['client_id'])]
# print(len(df))

# Turn sex into binary feature
df['sex'] = df['sex'].replace({'male': 1, 'female': 0})

# Add binary column for canadian and non-canadian english speakers
df['canadian'] = (df['accents'] == 'Canadian English').astype(int)
df['english'] = (df['accents'] == 'England English').astype(int)

# Remove elements that are not canadian or english
df = df[(df['canadian'] == 1) | (df['english'] == 1)]
print(len(df))

# Keep relevant columns (only keep canadian as binary feature, if it's 0,
# it means it is english)
df = df.loc[:, ['client_id', 'path', 'sex', 'age', 'canadian']]

# Balance sex class in both groups Canadian and english
df_canadian = df[df['canadian'] == 1]
df_canadian = balance_dataset(df_canadian, 'sex', 0.5)
# df_canadian = balance_dataset(df_canadian, 'age', 0.5)

df_english = df[df['canadian'] == 0]
df_english = balance_dataset(df_english, 'sex', 0.5)

df = pd.concat([df_canadian, df_english]).sample(frac=1.)
df = balance_dataset(df, 'canadian', 0.5)
print(len(df))

# Save dataset
os.makedirs(os.path.join(cwd, 'CommonVoice', 'datasets'), exist_ok=True)
df.to_csv(os.path.join(cwd, 'CommonVoice', 'datasets', 'full_df_test.tsv'),
          sep='\t',
          header=True,
          index=False)
