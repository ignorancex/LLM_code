# %%
import pandas as pd

df = pd.read_csv("./results/cifs_xmno_deeprelax.csv")
for col in df.columns[1:]:
    print(f"{col}: ", df[col].mean().round(3))

# %%