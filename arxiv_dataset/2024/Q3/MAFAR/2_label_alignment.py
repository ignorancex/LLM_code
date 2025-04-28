import os
import numpy as np
import json
import csv

num_labeller = 10

pos_threshold = 0.9
neg_threshold = 0.9
result_add = "predictions_10s"
true_label = "{}_label.json".format(result_add)
with open(true_label, 'r') as file:
    true_label = json.load(file)

text2label = {}
with open('class_labels_indices.csv', 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    index = 0
    for row in csvreader:
        text2label[row[2]] = index
        index += 1
print(text2label)
test_file = ['DJI_20231202102104_0001_D.npy', 'DJI_20231203012836_0002_D.npy', 'DJI_20231203015158_0003_D.npy', 'DJI_20231203015646_0004_D.npy', 'DJI_20231203020240_0005_D.npy', 'DJI_20231203022232_0006_D.npy', 'DJI_20231203025216_0007_D.npy', 'DJI_20231203051423_0008_D.npy', 'DJI_20231203051949_0009_D.npy', 'DJI_20231203052327_0010_D.npy', 'DJI_20231212164538_0038_D.npy', 'DJI_20231214154115_0040_D.npy', 'DJI_20231214175432_0041_D.npy', 'DJI_20231216095837_0042_D.npy', 'DJI_20231216100845_0043_D.npy', 'DJI_20231216164512_0044_D.npy', 'DJI_20231216165534_0045_D.npy', 'DJI_20231216174224_0046_D.npy', 'DJI_20231216175753_0047_D.npy', 'DJI_20231216180413_0048_D.npy', 'DJI_20231216181520_0049_D.npy', 'DJI_20231216184838_0050_D.npy', 'DJI_20231216200711_0052_D.npy', 'DJI_20231216201917_0053_D.npy', 'DJI_20231217123556_0054_D.npy']



model_list = os.listdir(result_add)
model_list.sort()
all_class = []
each_labeller_results = {}
each_model_results = {}
print(model_list)
num_model = len(model_list)
matched_num = 0
unmatched_num = 0

for file in test_file:
    file_name = file[:-4]
    each_model_results[file_name] = {}

    for model in model_list:
        model_add = os.path.join(result_add, model)
        model_result = np.load(os.path.join(model_add ,file), allow_pickle=True)
        if model not in list(each_model_results.keys()):
            each_model_results[file_name][model] = model_result

    result_shape = model_result.shape
    human_result = true_label[file_name]
    each_labeller_results[file_name] = {}
    for i in range(1,num_labeller+1):
        each_labeller_results[file_name][str(i)+".txt"] = np.zeros(result_shape)
    assert len(human_result) == result_shape[0]
    for x in range(len(human_result)):
        clip = human_result[x]
        y = clip[0]
        human_list = clip[1]
        for sample in range(len(y)):
            text = y[sample]
            human = human_list[sample]
            if text in list(text2label.keys()):
                label = text2label[text]
                each_labeller_results[file_name][human][x,label] = 1
                matched_num += 1
            else:
                unmatched_num += 1
                print("file：{} {} {}".format(file,human, text))
        all_class.append(text)
print(matched_num, unmatched_num)
np.save("each_labeller_results.npy",each_labeller_results)
np.save("each_model_results.npy",each_model_results)
print(len(set(all_class)))
from collections import Counter
item_counter = Counter(all_class)
print(item_counter)