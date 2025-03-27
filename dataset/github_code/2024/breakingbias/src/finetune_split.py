# CURRENTLY USING A SMALL PROPORTION OF DATA FOR FINETUNING

# import pandas as pd
# import numpy as np

# path = "../data/finetune.csv"
# df = pd.read_csv(path)

# # first, shuffle the data
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # keep the first 10% of data as "train.csv", the next "5%" as valid.csv, the rest as "test.csv"
# train = df[:int(len(df)*0.1)]
# valid = df[int(len(df)*0.1):int(len(df)*0.15)]
# test = df[int(len(df)*0.15):]

# train.to_csv("../data/finetune/finetune_alldata/train.csv", index=False)
# valid.to_csv("../data/finetune/finetune_alldata/valid.csv", index=False)
# test.to_csv("../data/finetune/finetune_alldata/test.csv", index=False)

# FILTERING FOR TYPE

# import pandas as pd
# import numpy as np

# path = "../data/finetune.csv"
# df = pd.read_csv(path)

# # Filter initial training set where Instruction column contains 'Choose between Yes and No.'
# initial_train = df[df['Instruction'].str.contains('Choose between Mostly and Rarely.')]

# # Shuffle the initial training set
# initial_train = initial_train.sample(frac=1, random_state=42).reset_index(drop=True)

# # Select 10000 random items for 'train' without replacement
# train = initial_train.sample(n=10000, random_state=42)

# # Remove the 'train' items from 'initial_train' to ensure no overlap with 'val'
# initial_train_remaining = initial_train.drop(train.index)

# # Select 5000 random items for 'val' from the remaining items
# valid = initial_train_remaining.sample(n=5000, random_state=42)

# # Filter test set where Instruction column contains 'Choose between Likely and Unlikely.' or 'Choose between Mostly and Rarely.'
# conditions = [
#     df['Instruction'].str.contains('Choose between Yes and No.'),
#     df['Instruction'].str.contains('Choose between Likely and Unlikely.')
# ]
# test = df[np.logical_or.reduce(conditions)]

# # Write the datasets to CSV files
# train.to_csv("../data/finetune/finetune_type3/train.csv", index=False)
# valid.to_csv("../data/finetune/finetune_type3/valid.csv", index=False)
# test.to_csv("../data/finetune/finetune_type3/test.csv", index=False)


# FILTERING FOR SCENARIO

import pandas as pd
import numpy as np

path = "../data/finetune.csv"
df = pd.read_csv(path)

# Define scenarios for each set
# train_scenarios = ['Ability', 'Age', 'Body type', 'Characteristics', 'Cultural', 'Gender and sex']
# test_scenarios = ['Nationality', 'Nonce', 'Political Ideologies', 'Race and ethnicities', 'Religion', 'Sexual orientation', 'Socioeconomic class']
test_scenarios = ['Support of Authorities, Law or Custom', 'Extended Contact', 'Virtual Contact']

# Filter initial training set where 'scenario' column contains 'Education' or 'Workplace'
# initial_train_conditions = df['axis'].isin(train_scenarios)
# initial_train = df[initial_train_conditions]

# # Shuffle the initial training set
# initial_train = initial_train.sample(frac=1, random_state=42).reset_index(drop=True)

# # Assuming initial_train has enough samples, otherwise add a check here
# # Select 10000 random items for 'train' without replacement
# train = initial_train.sample(n=10000, random_state=42)

# # Remove the 'train' items from 'initial_train' to ensure no overlap with 'val'
# initial_train_remaining = initial_train.drop(train.index)

# # Assuming initial_train_remaining has enough samples, otherwise add a check here
# # Select 5000 random items for 'val' from the remaining items
# valid = initial_train_remaining.sample(n=5000, random_state=42)

# Filter test set where 'scenario' column contains 'Sports', 'Healthcare' or 'Community'
test_conditions = df['key_principle'].isin(test_scenarios)
test = df[test_conditions]

# Write the datasets to CSV files
# train.to_csv("../data/finetune/finetune_6biasdimensions/train.csv", index=False)
# valid.to_csv("../data/finetune/finetune_6biasdimensions/valid.csv", index=False)
test.to_csv("../data/finetune/finetune_3keyprinciples/test.csv", index=False)

#FILTERING FOR NEW DATA

# import pandas as pd
# import numpy as np

# # Load the new data file
# new_data_path = "../data2/finetune_newdata.csv"
# df_new = pd.read_csv(new_data_path)

# # Shuffle the new data
# df_new_shuffled = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

# # Check if there are enough samples for both train and val
# if len(df_new_shuffled) >= 15000:
#     # Select 10000 random items for 'train' without replacement
#     train = df_new_shuffled[:10000]

#     # Select the next 5000 items for 'val' ensuring no overlap with 'train'
#     valid = df_new_shuffled[10000:15000]

#     # Write the datasets to CSV files
#     train.to_csv("../data/finetune/finetune_newdata/train.csv", index=False)
#     valid.to_csv("../data/finetune/finetune_newdata/valid.csv", index=False)
# else:
#     raise ValueError("Not enough data to create train and val datasets with the desired number of items.")

# # Note: The test data creation step has been removed as per the instructions.

