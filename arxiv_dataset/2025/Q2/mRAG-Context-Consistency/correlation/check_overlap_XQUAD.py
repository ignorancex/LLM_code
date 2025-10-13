import json
import os
import torch
from transformers import AutoTokenizer
from scipy import stats
import numpy as np

langs_raw = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']
langs = ['eng_Latn', 'arb_Arab', 'deu_Latn', 'ell_Grek', 'spa_Latn', 'hin_Deva', 'ron_Latn', 'rus_Cyrl', 'tha_Thai', 'tur_Latn', 'vie_Latn', 'zho_Hans']

data_aya = {
	'en': [84, 30, 32, 30, 29, 27, 21, 24, 14, 17, 15, 20],
	'es': [55, 74, 49, 47, 45, 43, 46, 41, 40, 42, 43, 31],
	'de': [53, 46, 69, 46, 45, 43, 45, 40, 38, 41, 43, 28],
	'ro': [51, 48, 47, 72, 43, 40, 42, 37, 38, 37, 40, 27],
	'vi': [48, 44, 45, 40, 75, 38, 39, 38, 38, 37, 39, 25],
	'tr': [43, 35, 35, 30, 30, 66, 31, 31, 31, 25, 32, 20],
	'el': [41, 36, 37, 32, 32, 31, 67, 30, 29, 28, 31, 20],
	'zh': [39, 37, 35, 34, 33, 33, 36, 75, 32, 26, 36, 25],
	'ar': [35, 29, 28, 29, 27, 24, 28, 27, 66, 29, 28, 19],
	'hi': [36, 29, 30, 28, 26, 25, 26, 27, 27, 62, 30, 17],
	'ru': [34, 30, 30, 30, 30, 30, 31, 30, 27, 30, 64, 22],
	'th': [11, 10, 11, 10, 10, 9, 10, 9, 10, 9, 10, 30],
}
data_llama = {
	'en': [77, 31, 34, 30, 28, 28, 20, 28, 15, 12, 22, 17],
	'es': [49, 66, 36, 36, 33, 32, 34, 24, 24, 28, 33, 26],
	'de': [46, 35, 59, 33, 31, 30, 31, 27, 24, 25, 30, 19],
	'ro': [40, 32, 31, 62, 25, 21, 23, 20, 18, 22, 25, 20],
	'vi': [42, 35, 33, 30, 68, 25, 26, 22, 22, 22, 26, 14],
	'tr': [31, 24, 24, 21, 20, 58, 19, 18, 16, 17, 19, 16],
	'el': [26, 23, 23, 20, 17, 18, 57, 16, 14, 15, 16, 12],
	'zh': [30, 24, 24, 22, 18, 19, 18, 63, 12, 12, 18, 12],
	'ar': [19, 14, 15, 13, 11, 11, 11, 10, 45, 8, 11, 8],
	'hi': [30, 23, 23, 23, 19, 20, 18, 15, 13, 55, 17, 15],
	'ru': [27, 20, 20, 18, 16, 15, 14, 12, 9, 8, 50, 10],
	'th': [22, 17, 18, 17, 13, 14, 13, 12, 11, 11, 14, 52],
}
data_gemma = {
	'en': [81, 65, 67, 64, 61, 62, 63, 57, 57, 63, 60, 57],
	'es': [55, 71, 48, 49, 47, 46, 49, 42, 42, 44, 44, 43],
	'de': [52, 46, 68, 46, 45, 44, 45, 40, 41, 43, 42, 42],
	'ro': [53, 48, 47, 72, 44, 40, 45, 38, 39, 42, 41, 41],
	'vi': [43, 40, 41, 40, 73, 37, 38, 33, 35, 36, 37, 34],
	'tr': [42, 36, 35, 35, 33, 65, 35, 29, 31, 33, 33, 31],
	'el': [34, 31, 31, 30, 28, 28, 65, 26, 24, 24, 29, 25],
	'zh': [39, 35, 34, 35, 33, 33, 36, 74, 33, 34, 35, 33],
	'ar': [28, 25, 24, 25, 24, 25, 25, 21, 61, 24, 24, 22],
	'hi': [39, 33, 32, 33, 31, 31, 31, 27, 28, 68, 30, 29],
	'ru': [33, 31, 31, 29, 27, 28, 30, 28, 27, 26, 62, 27],
	'th': [28, 26, 26, 26, 23, 23, 24, 24, 23, 24, 26, 71],
}
data_qwen = {
	'en': [83, 67, 68, 65, 63, 62, 64, 61, 61, 61, 62, 60],
	'es': [55, 74, 49, 50, 48, 45, 46, 46, 43, 43, 44, 44],
	'de': [51, 48, 72, 46, 45, 43, 46, 43, 41, 43, 43, 43],
	'ro': [46, 44, 42, 70, 41, 37, 38, 38, 36, 36, 37, 36],
	'vi': [51, 49, 48, 47, 79, 45, 45, 44, 45, 42, 44, 43],
	'tr': [40, 35, 35, 34, 34, 68, 32, 31, 30, 32, 30, 31],
	'el': [25, 23, 23, 22, 22, 19, 52, 19, 18, 19, 20, 20],
	'zh': [48, 46, 45, 44, 43, 41, 42, 83, 44, 43, 44, 44],
	'ar': [31, 29, 28, 29, 28, 26, 27, 26, 65, 27, 26, 27],
	'hi': [31, 27, 27, 27, 26, 24, 22, 24, 22, 66, 25, 25],
	'ru': [36, 36, 34, 33, 32, 33, 31, 30, 31, 30, 62, 32],
	'th': [32, 31, 29, 29, 29, 28, 30, 29, 28, 28, 30, 75],
}

out_aya = {
	'en': [0, 50, 44, 49, 53, 46, 55, 58, 58, 56, 54, 26],
	'es': [10, 0, 10, 7, 8, 7, 4, 4, 4, 6, 2, 6],
	'de': [8, 6, 0, 8, 7, 6, 3, 3, 3, 5, 2, 4],
	'ro': [11, 7, 10, 0, 5, 5, 4, 4, 5, 7, 2, 5],
	'vi': [17, 12, 17, 17, 0, 12, 9, 6, 6, 7, 6, 8],
	'tr': [14, 15, 21, 31, 26, 0, 15, 9, 9, 24, 5, 11],
	'el': [18, 12, 16, 13, 13, 11, 0, 4, 2, 3, 2, 3],
	'zh': [23, 19, 24, 24, 22, 19, 12, 0, 10, 24, 6, 8],
	'ar': [20, 15, 19, 17, 16, 16, 10, 5, 0, 4, 5, 5],
	'hi': [15, 13, 15, 14, 16, 12, 8, 4, 1, 0, 2, 3],
	'ru': [17, 15, 20, 16, 15, 11, 8, 5, 4, 4, 0, 5],
	'th': [18, 11, 14, 11, 16, 9, 7, 6, 3, 5, 4, 0],
}

out_llama = {
	'en': [0, 41, 34, 36, 43, 34, 43, 33, 36, 54, 35, 42],
	'es': [7, 0, 8, 8, 6, 5, 3, 10, 6, 6, 5, 6],
	'de': [5, 5, 0, 5, 6, 4, 4, 5, 3, 9, 3, 12],
	'ro': [6, 4, 6, 0, 8, 6, 4, 5, 1, 2, 3, 4],
	'vi': [10, 7, 11, 11, 0, 11, 5, 13, 3, 8, 5, 22],
	'tr': [13, 9, 11, 11, 17, 0, 7, 6, 3, 10, 3, 9],
	'el': [12, 9, 11, 10, 15, 8, 0, 3, 3, 5, 4, 13],
	'zh': [17, 13, 16, 15, 18, 13, 9, 0, 11, 19, 8, 18],
	'ar': [11, 10, 10, 11, 16, 8, 7, 9, 0, 14, 4, 17],
	'hi': [2, 2, 2, 1, 6, 2, 1, 2, 1, 0, 1, 4],
	'ru': [10, 7, 9, 8, 13, 7, 7, 7, 10, 21, 0, 9],
	'th': [17, 13, 15, 14, 19, 11, 7, 8, 2, 2, 3, 0],
}
out_gemma = {
	'en': [0, 6, 6, 7, 6, 5, 4, 4, 5, 2, 4, 4],
	'es': [9, 0, 12, 9, 9, 7, 6, 5, 4, 4, 2, 4],
	'de': [8, 6, 0, 9, 7, 6, 3, 2, 2, 2, 2, 2],
	'ro': [8, 7, 11, 0, 8, 9, 5, 6, 4, 2, 4, 3],
	'vi': [17, 14, 17, 13, 0, 14, 10, 11, 9, 6, 7, 8],
	'tr': [15, 14, 18, 16, 13, 0, 9, 11, 6, 4, 4, 6],
	'el': [16, 14, 17, 15, 15, 12, 0, 8, 4, 3, 3, 4],
	'zh': [18, 17, 21, 21, 18, 16, 13, 0, 7, 8, 5, 7],
	'ar': [13, 12, 14, 14, 12, 10, 9, 9, 0, 5, 4, 5],
	'hi': [8, 7, 10, 7, 7, 7, 6, 3, 1, 0, 2, 3],
	'ru': [17, 16, 19, 17, 17, 13, 11, 6, 5, 6, 0, 7],
	'th': [27, 24, 28, 24, 28, 20, 18, 20, 12, 11, 8, 0],
}
out_qwen = {
	'en': [0, 5, 4, 6, 4, 5, 2, 4, 4, 2, 3, 4],
	'es': [9, 0, 9, 8, 6, 7, 5, 6, 2, 4, 2, 4],
	'de': [8, 5, 0, 7, 5, 6, 3, 5, 2, 1, 1, 2],
	'ro': [10, 5, 10, 0, 4, 8, 5, 4, 2, 2, 2, 2],
	'vi': [14, 7, 9, 8, 0, 8, 2, 12, 1, 2, 2, 3],
	'tr': [14, 12, 15, 12, 8, 0, 7, 13, 4, 4, 4, 5],
	'el': [10, 7, 9, 9, 6, 7, 0, 2, 0, 2, 2, 1],
	'zh': [13, 12, 12, 13, 11, 12, 7, 0, 2, 3, 4, 4],
	'ar': [14, 11, 13, 12, 9, 10, 7, 11, 0, 4, 4, 4],
	'hi': [5, 4, 6, 5, 4, 4, 4, 4, 1, 0, 1, 2],
	'ru': [14, 9, 13, 10, 9, 10, 6, 13, 2, 3, 0, 3],
	'th': [23, 14, 19, 16, 15, 14, 7, 14, 2, 2, 3, 0],
}


# sum_aya = {k: list(np.array(v)+out_aya[k]) for k, v in data_aya.items()}
sum_aya = [list(np.array(v)+out_aya[k]) for k, v in data_aya.items()]
sum_llama=[list(np.array(v)) for k, v in data_llama.items()]
sum_gemma=[list(np.array(v)) for k, v in data_gemma.items()]
sum_qwen=[list(np.array(v)) for k, v in data_qwen.items()]

only_aya= [list(np.array(v)) for k, v in data_aya.items()]
only_llama=[list(np.array(v)) for k, v in data_llama.items()]
only_gemma=[list(np.array(v)) for k, v in data_gemma.items()]
only_qwen=[list(np.array(v)) for k, v in data_qwen.items()]

for model in ['CohereForAI/aya-expanse-8b']:
    tokenizer = AutoTokenizer.from_pretrained(model)

    overlap_vocab = dict()

    for lang in langs:
        tmp = dict()
        for folder in ['dev', 'devtest']:
            fname = folder + '/' + lang + '.' + folder
            with open(fname, 'r') as f:
                for line in f:
                    tokens = tokenizer.encode(line)
                    for token in tokens:
                        if token not in tmp.keys():
                            tmp[token] = 1
                        else:
                            tmp[token] += 1
        overlap_vocab[lang] = tmp

    overlap_list = []
    for lang1 in langs:
        tmp = []
        for lang2 in langs:
            tokens1 = set(overlap_vocab[lang1].keys())
            tokens2 = set(overlap_vocab[lang2].keys())
            overlap = tokens1.intersection(tokens2)
            #tmp.append(len(overlap)/(len(tokens1) + len(tokens2)))
            tmp.append(len(overlap)/len(tokens1.union(tokens2)))
        overlap_list.append(tmp)

    print(overlap_list)
    print()

    print(' & '.join(langs_raw) + '\\\\')
    # for res_model in [only_aya, only_llama, only_gemma, only_qwen]:
    for res_model in [sum_aya, sum_llama, sum_gemma, sum_qwen]:
        res_for_print = []
        for index, _overlap_list in enumerate(overlap_list):
            # print(stats.pearsonr(_overlap_list, sum_aya[index]))
            # x = _overlap_list
            # y = res_model[index]
            # print(x, y)
            if index != 11:
                x = _overlap_list[:index] + _overlap_list[(index+1):]
                y = res_model[index][:index] + res_model[index][(index+1):]
            else:
                x = _overlap_list[:index]
                y = res_model[index][:index]
            # print(len(x))
            # print(x, y)
            # print(abc)
            # corr = stats.spearmanr(x, y)
            corr = stats.pearsonr(x, y)
            # print(langs_raw[index])
            # print(_overlap_list)
            # print(res_model[index])
            # print(corr)
            # print()
            res_for_print.append(f"{round(corr.statistic,2)} ({round(corr.pvalue, 1)})")
        print(' & '.join(res_for_print) + '\\\\')