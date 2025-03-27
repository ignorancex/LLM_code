# import pandas as pd
# import numpy as np

# np.random.seed(42) # Set a seed for reproducibility

# # Assuming you have a pandas DataFrame named 'data' with your data.
# INPUT_PATH = "../data/finetune/finetune_alldata/test.csv"
# OUTPUT_PATH = "../data/finetune/finetune_newdata/test_sampled.csv"

# data = pd.read_csv(INPUT_PATH)

# print(data.groupby(['key_principle', 'scenario', 'action_type']).size())

# # Define a function to process the data as per your requirements.
# def process_data(df):
#     # Filter by instruction type and keep a list of processed dataframes
#     processed_dfs = []
    
#     for instruction_type in df['Instruction'].unique():
#         # Filter the dataframe for the current instruction type
#         instruction_df = df[df['Instruction'] == instruction_type]
        
#         # Group by the (key_principle, scenario, action_type) tuple
#         grouped = instruction_df.groupby(['key_principle', 'scenario', 'action_type'])
        
#         # For each group, sample "n" rows from each bucket randomly
#         n = 200
#         sampled = grouped.apply(lambda x: x.loc[np.random.choice(x.index, size=n, replace=False)])
        
#         # Reset index to flatten the grouped dataframe
#         sampled = sampled.reset_index(drop=True)
        
#         # Append to the list of processed dataframes
#         processed_dfs.append(sampled)
    
#     # Concatenate all processed dataframes into one
#     return pd.concat(processed_dfs, ignore_index=True)

# # Apply the function to your dataframe
# processed_data = process_data(data)

# # show the distribution of the buckets
# print(processed_data.groupby(['key_principle', 'scenario', 'action_type']).size())

# processed_data.to_csv(OUTPUT_PATH, index=False)


import pandas as pd

# Load your dataset into a pandas DataFrame.
# Replace 'your_dataset.csv' with the actual path to your CSV file.
data = pd.read_csv('../data/finetune/finetune_type3/test.csv')

# Filter the data based on the 'Instruction' column.
test_type1 = data[data['Instruction'].str.contains('Choose between Yes and No.')]
test_type2 = data[data['Instruction'].str.contains('Choose between Likely and Unlikely.')]

# Save the subsets to CSV files.
test_type1.to_csv('../data/finetune/finetune_type3/test_type1.csv', index=False)
test_type2.to_csv('../data/finetune/finetune_type3/test_type2.csv', index=False)
