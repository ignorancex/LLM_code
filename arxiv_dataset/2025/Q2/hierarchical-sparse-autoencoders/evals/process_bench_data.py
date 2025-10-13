import pickle

from torch import mode

data = pickle.load(open("translation_results.pkl", "rb"))
out_data = {("en", "fr"): [], ("en", "es"): [], ("en", "de"): [], ("fr", "es"): [], ("fr", "de"): [], ("es", "de"): []}

def avg_diff_size(lst1, lst2):
    diffs = [len(x ^ y) for x, y in zip(lst1, lst2)]
    return sum(diffs) / len(diffs)

for lang1, lang2 in out_data.keys():
    moe_top_k = [set(x[:7]) for x in data[lang1][0]]
    moe_top_k_2 = [set(x[:7]) for x in data[lang2][0]]

    out_data[(lang1, lang2)].append(avg_diff_size(moe_top_k, moe_top_k_2))

print(out_data)
