import pandas as pd
import numpy as np
import json

filename = "../../data/final_normal_dataset.tsv"

data = pd.read_csv(filename, sep='\t')

# Assuming df_ground_truth and df_predictions are your dataframes containing ground truth and model predictions respectively

# # Sample dataframes
# ground_truth_data = {
#     'C1': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#     'C2': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     'C3': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
#     'C4': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
# }

# predictions_data = {
#     'C1': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],  # Example with errors in predictions
#     'C2': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     'C3': [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # Example with errors in predictions
#     'C5': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
# }

# # Creating dataframes
# df_ground_truth = pd.DataFrame(ground_truth_data)
# df_predictions = pd.DataFrame(predictions_data)

df_ground_truth = pd.read_csv("ground_truth.tsv", sep="\t")
df_predictions = pd.read_csv("predictions_chatGPT.tsv", sep="\t")


df_predictions.columns = df_ground_truth.columns

print(df_ground_truth.shape)
print(df_predictions.shape)

# Calculating error rate per label
error_rates = (df_ground_truth != df_predictions).mean(axis=1)

print("Error rates per label:")
print(error_rates)

id_wise_error_rate = pd.DataFrame(error_rates)

print(id_wise_error_rate.head())

objectwise_error_rate = {}

for idx, rows in id_wise_error_rate.iterrows():
    error = rows[0]
    object_name = data.iloc[idx]["Object"]

    if object_name.strip() in objectwise_error_rate.keys():
        objectwise_error_rate[object_name.strip()].append(error)
    else:
        objectwise_error_rate[object_name.strip()]= [error]
    # print(object_name)
    # print(error)
    # break
        
for key, values in objectwise_error_rate.items():
    objectwise_error_rate[key] = sum(values)/len(values)

# print(objectwise_error_rate)
print(len(objectwise_error_rate))

# with open("objectwise_error_rate.json", "w") as content:
#     json.dump(objectwise_error_rate, content)

# # Calculating error rate per row
# errors_per_row = (df_ground_truth != df_predictions).sum(axis=1)
# total_columns = df_ground_truth.shape[1]
# error_rates_per_row = errors_per_row / total_columns

# print("Error rates per row:")
# print(error_rates_per_row)

with open("objectwise_agreement.json", "r") as content:
    objectwise_agreement = json.load(content)

updated_objectwise_agreement = {}
for key, values in objectwise_error_rate.items():
    if key in objectwise_agreement.keys():
        updated_objectwise_agreement[key] = objectwise_agreement[key]


print(dict(list(updated_objectwise_agreement.items())[:5]))
print(dict(list(objectwise_error_rate.items())[:5]))
# Convert dictionaries to pandas Series
series1 = pd.Series(updated_objectwise_agreement)
series2 = pd.Series(objectwise_error_rate)

# Calculate correlation coefficient
correlation = series1.corr(series2)

print("Correlation between the values of the two dictionaries:", correlation)

with open("updated_objectwise_agreement.json", "w") as content:
    json.dump(updated_objectwise_agreement, content)