import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv("../evaluation/llama/ft_type2_dataset_evaluation_13B.csv")

# Calculate the total count of biased responses in the entire dataset
total_biased_responses = (df["base_response_bias"] == 1).sum()
print(f"Total biased responses in the dataset: {total_biased_responses}")


# Function to calculate bias percentage and count for a grouped DataFrame
def bias_percentage_and_count(grouped_df, column):
    ones = (grouped_df[column] == 1).sum()
    total = len(grouped_df[column])
    percentage = 100 * ones / total
    return f"{percentage:.2f}% ({ones})"  # Return a formatted string containing both percentage and count


# Group by 'axis' and 'scenario', and then compute the bias percentage for each group
bias_percentages = (
    df.groupby(["axis", "scenario"])
    .apply(lambda group: bias_percentage_and_count(group, "base_response_bias"))
    .reset_index()
)
bias_percentages.columns = ["axis", "scenario", "bias_percentage_and_count"]

# Pivot the data to get the desired format
pivot_table = bias_percentages.pivot(
    index="axis", columns="scenario", values="bias_percentage_and_count"
).reset_index()

# Fill NaNs with '0.00% (0)' to denote 0% bias and a count of 0 and save to CSV
pivot_table.fillna("0.00% (0)").to_csv(
    "../axis_analysis/type2_llama_axis_summary_13B.csv", index=False
)
