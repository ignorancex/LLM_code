from datasets import load_from_disk
from icecream import ic
import numpy as np

np.random.seed(42)
dataset = load_from_disk('../yelp_review_full_split_train_dev')
ic(dataset)
testset = dataset['selected_train']
# Generate a test_tiny that have 100 datapoint, test_small have 1000 datapoint,
# with [20,20,20,20,20] and [200,200,200,200,200] in different class respectively
idx_list = np.arange(len(testset))
np.random.shuffle(idx_list)
ic(idx_list)
#label_num_tiny = [20, 20, 20, 20, 20]
label_num_small = [200, 200, 200, 200, 200]
#label_num_large = [2000, 2000, 2000, 2000, 2000]
tiny_set_dict = {
    "text": [],
    "label": [],
}
small_set_dict = {
    "text": [],
    "label": [],
}
large_set_dict = {
    "text": [],
    "label": [],
}
for i in range(len(testset)):
    datapoint = testset[int(idx_list[i])]
    if (label_num_small[datapoint['label']]>0):
        label_num_small[datapoint['label']]-=1
        small_set_dict['label'].append(datapoint['label'])
        small_set_dict['text'].append(datapoint['text'])
import datasets
from datasets import Dataset
#tiny_testset = Dataset.from_dict(tiny_set_dict)
small_testset = Dataset.from_dict(small_set_dict)
#large_testset = Dataset.from_dict(large_set_dict)
#ic(tiny_testset)
ic(small_testset)
#ic(large_testset)
#tiny_testset.save_to_disk("./yelp_tiny_test_new")
small_testset.save_to_disk("./yelp_small_train_new")
#large_testset.save_to_disk("./yelp_large_test_new")


