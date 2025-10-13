from evaluate import load
import json
import datasets
import sys
import pdb
from tqdm import tqdm
from collections import defaultdict

squad_v2_metric = load("squad_v2")
data = [json.loads(line) for line in open(sys.argv[1])]
index2f1 = defaultdict(list)
index2rouge = defaultdict(list)
index2em = defaultdict(list)

pred_all = [line["prediction"] for line in data]
ref_all = [line["reference"] for line in data]

avg_f1 = []
avg_em = []

for line in tqdm(data):
    index = line["index"] if line["index"] < 5 else 5
    pred = [line["prediction"]]
    ref = [line["reference"]]
    results = squad_v2_metric.compute(predictions=pred, references=ref)
    index2f1[index].append(results["f1"])
    index2em[index].append(results["exact"])
    avg_f1.append(results['f1'])
    avg_em.append(results['exact'])
    pred = line['prediction']['prediction_text']
    ref = line["reference"]['answers']['text'][0]

print(f"f1: {sum(avg_f1)/len(avg_f1)} || em: {sum(avg_em)/ len(avg_em)}")

index_list = sorted(list(index2f1.keys()))
for index in index_list:
    num = len(index2em[index])
    f1 = sum(index2f1[index]) / num
    em = sum(index2em[index]) / num
    print(f"Loc: {index} f1: {f1} em: {em} num: {num}")
