import re

import pandas as pd


def clean_data(text):
    text = text.encode("utf-8", errors='ignore').decode("utf-8")
    text = re.sub("({.*})+", '', text)
    text = re.sub("(<.*>)+", "'", text)
    text = re.sub("[\"*]", "'", text)
    text = re.sub("[\'*]", '', text)
    return text.strip()

node = pd.read_csv('data/nodes/nodes.tsv', sep='\t')

# print(node.head(10))

print(node['text'][113])
print("\n")
print(clean_data(node['text'][113]))