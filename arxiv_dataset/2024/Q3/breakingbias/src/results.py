import pandas as pd

# Assuming you've loaded the data into a DataFrame named df
# df = pd.read_csv("../evaluation/ft_type7_dataset_evaluation_13B.csv")
df = pd.read_csv("../evaluation/vicuna/ft_type2_dataset_evaluation_13B.csv") 

# Define the columns to analyze
bias_columns = [
    "base_response_bias",
    "positive_response_bias",
    "negative_response_bias",
]
contact_hypothesis_columns = ["positive_response_CH", "negative_response_CH"]

# Scenarios to analyze
# scenarios = ['Choose between Yes and No.', 'Choose between Likely and Unlikely.']
# scenarios = ['Sports', 'Community', 'Healthcare']
# scenarios = ['Support of Authorities, Law or Custom', 'Extended Contact', 'Virtual Contact']
# scenarios = ['Nationality', 'Nonce', 'Political Ideologies', 'Race and ethnicities', 'Religion', 'Sexual orientation', 'Socioeconomic class']

# Function to count values for each column
def count_values(column, data):
    ones = (data[column] == 1).sum()
    zeros = (data[column] == 0).sum()
    nas = data[column].isna().sum()
    total = len(data[column])

    return {
        # "Ones": ones,
        # "Zeros": zeros,
        # "NAs": nas,
        "Ones(Bias) Percentage": 100 * ones / total,
        "Zeros (No Bias) Percentage": 100 * zeros / total,
        "NAs Percentage": 100 * nas / total,
    }

# Analyze the bias columns
print("Bias Analysis:")
for col in bias_columns:
    print(f"\n{col}:")
    results = count_values(col, df)
    for key, val in results.items():
        print(f"{key}: {val}")
print("\n" + "-"*50)

# Analyze the contact hypothesis columns
# print("\nContact Hypothesis Analysis:")
# for col in contact_hypothesis_columns:
#     print(f"\n{col}:")
#     results = count_values(col, df)
#     for key, val in results.items():
#         print(f"{key}: {val}")


# Analyze bias for each scenario
# for scenario in scenarios:
#     # Filter the DataFrame for the current scenario
#     scenario_df = df[df['Instruction'] == scenario]
    
#     print(f"Bias Analysis for {scenario} Scenario:")
#     for col in bias_columns:
#         results = count_values(col, scenario_df)
#         print(f"\n{col}:")
#         for key, val in results.items():
#             print(f"{key}: {val:.2f}%")
#     print("\n" + "-"*50)  # Separator for readability