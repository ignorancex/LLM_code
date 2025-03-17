import pandas as pd
import sys
 
file1 = str(sys.argv[1])
file2 = str(sys.argv[2])
df1 = pd.read_csv(file1,  sep=',|\t', engine='python')
df2 = pd.read_csv(file2,  sep=',|\t', engine='python')

df1.columns = ["cell","type1"]
df2.columns = ["cell","type2"]

merged = df1
merged["type2"] = df2["type2"]
crosstab = pd.crosstab(merged.type1, merged.type2).T
crosstab.columns = [col.strip('\"') for col in crosstab.columns]
crosstab.index = [ind.strip('\"') for ind in crosstab.index]
crosstab.to_csv(str(sys.argv[3]))
