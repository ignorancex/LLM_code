import pandas as pd

# Load the CSV
df = pd.read_csv("results/kmodes_table1.csv")

# Abbreviate question columns
col_map = {
    "Q1_mode": "Q1",
    "Q2_mode": "Q2",
    "Q3_mode": "Q3",
    "Q4_mode": "Q4",
    "Q5_mode": "Q5",
    "Q6_mode": "Q6",
    "Q7_mode": "Q7",
    "Q8_mode": "Q8",
    "Q9_mode": "Q9",
    "Q10_mode": "Q10"
}
df = df.rename(columns=col_map)

# Set cluster as column headers, questions as rows
df_t = df.set_index("cluster").T
df_t = df_t.drop("size", errors="ignore")  # Optionally drop size row

# Output as Markdown
print(df_t.to_markdown())


print(df_t.to_latex(longtable=True, escape=False))