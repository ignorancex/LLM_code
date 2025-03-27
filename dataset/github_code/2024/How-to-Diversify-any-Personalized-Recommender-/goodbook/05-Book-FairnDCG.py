import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
######
from lenskit import batch, topn
import lenskit.crossfold as xf
import warnings
warnings.filterwarnings('ignore')
# !pip install lenskit_tf
from lenskit import topn, util
from lenskit.algorithms import Recommender, item_knn, user_knn as knn, als, tf
from lenskit.algorithms import basic


books = pd.read_csv('goodbook/books.csv')
ratings = pd.read_csv('goodbook/ratings.csv')
book_tags = pd.read_csv('goodbook/book_tags.csv')
tags = pd.read_csv('goodbook/tags.csv')


df_book = pd.read_csv ("RecSys_News/goodbook/processed_GB_Data.csv", sep=",", names= ["user", "item", "genre"])
# df_book = df_book[['item']].drop_duplicates()
unique_items_genres = df_book.reset_index(drop=True).drop_duplicates(subset=['item', 'genre'], keep='last')
unique_items_genres = unique_items_genres [['item', 'genre']]
unique_items_genres = unique_items_genres.reset_index(drop=True).drop_duplicates(subset=['item'], keep='last')

df_books = unique_items_genres.copy ()#.head (50)

ground_truth_F = pd.read_csv ("goodbook/test-book0.csv", sep=",", names= ["user", "item", "rating", "genre"])


def fair_dcg(predictions, k, relevant_categories):
    data = list(zip(predictions['rank'], predictions['genre']))

    sorted_data = sorted(data, key=lambda x: x[0])

    fair_dcg_score = 0.0
    total_weight = 0.0
    unique_categories = set()

    for i, (rank, genres) in enumerate(sorted_data, 1):
        if i > k:
            break  # Stop calculating after reaching the top k

        # Split the string into a list of genres
        genre_list = [genre.strip() for genre in genres.split(',')]

        # Convert the list of genres to a set
        genre_set = set(genre_list)

        print(genre_set)  # Print the genre set for debugging

        # Check if any relevant category is present in the genre set
        gain = 2 ** 1 - 1 if any(category in genre_set for category in relevant_categories) else 2 ** 0 - 1
        print (gain)
        discount = np.log2(i + 1)

        # Update total weight only if a relevant category is present
        if gain == 1:
            total_weight += 1.0

        # Update unique categories
        unique_categories.update(genre_set)
        print (unique_categories)
        fair_dcg_score += (gain / discount)

    # Normalize by the ideal DCG
    ideal_labels = [1] + [0] * (k - 1)  # Ideal ranking for binary relevance
    ideal_dcg = sum((2 ** label - 1) / np.log2(i + 1) for i, label in enumerate(ideal_labels, 1) if i <= k)
    fair_dcg_score /= ideal_dcg

    # Avoid division by zero
    if total_weight != 0:
        # Normalize by the total weight
        fair_dcg_score /= len(unique_categories) / total_weight
    else:
        fair_dcg_score = 0.0

    return fair_dcg_score

predictions_file_paths = [
    ("goodbook/results/Random/version_RecSys/RS-100_0.0_Original.csv", "data_1"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.01_Add.csv", "data_2"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.02_Add.csv", "data_3"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.05_Add.csv", "data_4"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.1_Add.csv", "data_5"),
    # ("RecSys_News/goodbook/results/RS-100_0.5_Add.csv", "data_6"),
]

# Create a dictionary to store DataFrames
datasets = {}

# Read the news DataFrame (assuming df_news is defined)
data_0 = ground_truth_F # df_news

# Iterate over prediction files
for file_path, dataset_name in predictions_file_paths:
    # Read the prediction file
    df = pd.read_csv(file_path)

    # Store the DataFrame in the dictionary
    datasets[dataset_name] = df

algorithm_to_focus = ['BPR', 'implicitMF', 'pop'][2]

for dataset_name, dataset_df in datasets.items():
    # Filter the dataset to include only rows where the Algorithm is 'BPR'
    filtered_df = dataset_df[dataset_df['Algorithm'] == algorithm_to_focus]

    # Merge the filtered dataset with df_book on the "item" column
    merged_df = pd.merge(filtered_df, df_books[['item', 'genre']], on='item', how='left')

    # Drop duplicates based on 'user', 'item', and 'Algorithm'
    merged_df = merged_df.drop_duplicates(subset=['user', 'item', 'Algorithm'])

    # Update the dataset in the datasets dictionary
    datasets[dataset_name] = merged_df

# Now, you have individual datasets for each dataset_name containing only the 'BPR' algorithm data
data_1 = datasets['data_1']
data_2 = datasets['data_2']
data_3 = datasets['data_3']
data_4 = datasets['data_4']
data_5 = datasets['data_5']
# data_6 = datasets['data_6']

predictions_F_org = data_1
predictions_F_1perc = data_2
predictions_F_2perc = data_3
predictions_F_5perc = data_4
predictions_F_10perc= data_5
predictions_F_10perc


ground_truth_F['rank'] = ground_truth_F.groupby('user').cumcount() + 1



#  'crime', 'memoir', 'religion', 'suspense', 'art', 'music', 'manga', 'cookbooks', 'psychology', 'travel', 'business', 'paranormal', 
relevant_categories = ['music', 'poetry', 'horror', 'spirituality', 'sports', 'christian', 'comics', 'manga', 'cookbooks', 'psychology', 'art']

datasets = [ground_truth_F, predictions_F_org, predictions_F_1perc, predictions_F_2perc, predictions_F_5perc, predictions_F_10perc]#, predictions_F_50perc]
dataset_names = ['Ground Truth', 'Org recs', 'Recs 1%', 'Recs 2%', 'Recs 5%', 'Recs 10%']#, 'Predictions 50%']

results = []

for i, dataset in enumerate(datasets):
    print (i, dataset)
    for k in range(1, 101):  # Adjust the range based on your needs
        print (k)
        score = fair_dcg(dataset, k, relevant_categories)
        results.append({'Dataset': dataset_names[i], 'k': k, 'fair_ndcg_score': score})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
# results_df.to_csv(f'RecSys_News/goodbook/results/fair_ndcg_results-{algorithm_to_focus}.csv', index=False)

results_df_norm = results_df.copy () #pd.read_csv ("RecSys_News/jobs_server/NRMS/nrms_results/fair_ndcg_results-nrms_2step.csv")
results_df_norm ["normalized"] = results_df_norm ["fair_ndcg_score"] / max (results_df_norm ["fair_ndcg_score"])#.max()

import seaborn as sns  # Import seaborn for color palettes

# Set the seaborn style and context for better aesthetics
sns.set(style="whitegrid")
sns.set_context("talk")

# Plot line plots for each dataset with a different color palette
datasets = results_df_norm['Dataset'].unique()

# Choose a color palette from seaborn, e.g., "pastel"
colors = sns.color_palette("pastel", len(datasets))

for i, dataset in enumerate(datasets):
    dataset_df = results_df_norm[results_df_norm['Dataset'] == dataset]
    plt.plot(dataset_df['k'], dataset_df['normalized'], label=dataset, color=colors[i])

# Set labels and title with increased text size and bold text
plt.xlabel('k', fontsize=14, fontweight='bold')
plt.ylabel('fair_ndcg_score', fontsize=14, fontweight='bold')
plt.title('Fair-nDCG vs k (1-step pre-process)', fontsize=16, fontweight='bold')

# Increase the size of tick labels and bolden them
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add legend with increased text size and bold text
plt.legend(fontsize=10, title='Legend', title_fontsize=10, prop={'weight': 'bold'})

# Save the plot to a PDF file
pdf_filename = f'goodbook/results/01-fair_ndcg_plot-{algorithm_to_focus}-1-step.pdf'
plt.savefig(pdf_filename, bbox_inches='tight')

# Show the plot
plt.show()


predictions_file_paths = [
    ("goodbook/results/Random/version_RecSys/RS-100_0.0_Original.csv", "data_1"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.01_Obf.csv", "data_2"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.02_Obf.csv", "data_3"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.05_Obf.csv", "data_4"),
    ("goodbook/results/Random/version_RecSys/RS-100_0.1_Obf.csv", "data_5"),
    # ("RecSys_News/goodbook/results/RS-100_0.5_Obf.csv", "data_6"),
]

# Create a dictionary to store DataFrames
datasets = {}

# Read the news DataFrame (assuming df_news is defined)
data_0 = ground_truth_F # df_news

# Iterate over prediction files
for file_path, dataset_name in predictions_file_paths:
    # Read the prediction file
    df = pd.read_csv(file_path)

    # Store the DataFrame in the dictionary
    datasets[dataset_name] = df

algorithm_to_focus = ['BPR', 'implicitMF', 'pop'][0]

for dataset_name, dataset_df in datasets.items():
    # Filter the dataset to include only rows where the Algorithm is 'BPR'
    filtered_df = dataset_df[dataset_df['Algorithm'] == algorithm_to_focus]

    # Merge the filtered dataset with df_book on the "item" column
    merged_df = pd.merge(filtered_df, df_books[['item', 'genre']], on='item', how='left')

    # Drop duplicates based on 'user', 'item', and 'Algorithm'
    merged_df = merged_df.drop_duplicates(subset=['user', 'item', 'Algorithm'])

    # Update the dataset in the datasets dictionary
    datasets[dataset_name] = merged_df

# Now, you have individual datasets for each dataset_name containing only the 'BPR' algorithm data
data_1 = datasets['data_1']
data_2 = datasets['data_2']
data_3 = datasets['data_3']
data_4 = datasets['data_4']
data_5 = datasets['data_5']
# data_6 = datasets['data_6']

predictions_F_org = data_1
predictions_F_1perc = data_2
predictions_F_2perc = data_3
predictions_F_5perc = data_4
predictions_F_10perc= data_5

ground_truth_F['rank'] = ground_truth_F.groupby('user').cumcount() + 1

#  'crime', 'memoir', 'religion', 'suspense', 'art', 'music', 'manga', 'cookbooks', 'psychology', 'travel', 'business', 'paranormal', 
relevant_categories = ['music', 'poetry', 'horror', 'spirituality', 'sports', 'christian', 'comics', 'manga', 'cookbooks', 'psychology', 'art']

datasets = [ground_truth_F, predictions_F_org, predictions_F_1perc, predictions_F_2perc, predictions_F_5perc, predictions_F_10perc]#, predictions_F_50perc]
dataset_names = ['Ground Truth', 'Org recs', 'Recs 1%', 'Recs 2%', 'Recs 5%', 'Recs 10%']#, 'Predictions 50%']

results = []

for i, dataset in enumerate(datasets):
    print (i, dataset)
    for k in range(1, 101):  # Adjust the range based on your needs
        print (k)
        score = fair_dcg(dataset, k, relevant_categories)
        results.append({'Dataset': dataset_names[i], 'k': k, 'fair_ndcg_score': score})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
# Save the DataFrame to a CSV file
# results_df.to_csv(f'RecSys_News/goodbook/results/fair_ndcg_results-{algorithm_to_focus}-2-step.csv', index=False)

results_df_norm = results_df.copy () #pd.read_csv ("RecSys_News/jobs_server/NRMS/nrms_results/fair_ndcg_results-nrms_2step.csv")
results_df_norm ["normalized"] = results_df_norm ["fair_ndcg_score"] / max (results_df_norm ["fair_ndcg_score"])#.max()
import seaborn as sns  # Import seaborn for color palettes

# Set the seaborn style and context for better aesthetics
sns.set(style="whitegrid")
sns.set_context("talk")

# Plot line plots for each dataset with a different color palette
datasets = results_df_norm['Dataset'].unique()

# Choose a color palette from seaborn, e.g., "pastel"
colors = sns.color_palette("pastel", len(datasets))

for i, dataset in enumerate(datasets):
    dataset_df = results_df_norm[results_df_norm['Dataset'] == dataset]
    plt.plot(dataset_df['k'], dataset_df['normalized'], label=dataset, color=colors[i])

# Set labels and title with increased text size and bold text
plt.xlabel('k', fontsize=14, fontweight='bold')
plt.ylabel('fair_ndcg_score', fontsize=14, fontweight='bold')
plt.title('Fair-nDCG vs k (2-step pre-process)', fontsize=16, fontweight='bold')

# Increase the size of tick labels and bolden them
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add legend with increased text size and bold text
plt.legend(fontsize=10, title='Legend', title_fontsize=10, prop={'weight': 'bold'})

# Save the plot to a PDF file
pdf_filename = f'goodbook/results/01-fair_ndcg_plot-{algorithm_to_focus}-2-step.pdf'
plt.savefig(pdf_filename, bbox_inches='tight')

# Show the plot
plt.show()
