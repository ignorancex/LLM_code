import pandas as pd

path1 = "corrected/dollar_street_llava.csv"
path2 = "corrected/dollar_street_gpt.csv"
path3 = "corrected/dalle_street_gpt.csv"
path4 = "corrected/marvl_gpt.csv"
path5 = "corrected/dalle_street_llava.csv"
path6 = "corrected/marvl_llava.csv"

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)
df5 = pd.read_csv(path5)
df6 = pd.read_csv(path6)

df1['correct_prediction'] = df1['response'] == df1['true_sub_region']
df2['correct_prediction'] = df2['response'] == df2['true_sub_region']
df3['correct_prediction'] = df3['response'] == df3['true_sub_region']
df4['correct_prediction'] = df4['response'] == df4['true_country']
df5['correct_prediction'] = df5['response'] == df5['true_sub_region']
df6['correct_prediction'] = df6['response'] == df6['true_country']

print("Dollar Street llava:")
print(df1['correct_prediction'].mean())
print(df1['correct_prediction'].value_counts())
print(len(df1))

print("Dollar Street gpt:")
print(df2['correct_prediction'].mean())
print(df2['correct_prediction'].value_counts())
print(len(df2))

print("Dalle Street gpt:")
print(df3['correct_prediction'].mean())
print(df3['correct_prediction'].value_counts())
print(len(df3))

print("Dalle Street llava:")
print(df5['correct_prediction'].mean())
print(df5['correct_prediction'].value_counts())
print(len(df5))

print("Marvl gpt:")
print(df4['correct_prediction'].mean())
print(df4['correct_prediction'].value_counts())
print(len(df4))

print("Marvl llava:")
print(df6['correct_prediction'].mean())
print(df6['correct_prediction'].value_counts())
print(len(df6))
