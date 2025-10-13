import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file into a DataFrame
df = pd.read_csv("./baseline_results.csv")

statistics_df = pd.DataFrame({
    "Mean": np.mean(df, axis=0),
    "Std": np.std(df, axis=0)
})
print(statistics_df)


