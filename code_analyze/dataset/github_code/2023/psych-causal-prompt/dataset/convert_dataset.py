from datasets import load_from_disk
dataset = load_from_disk('./yelp_large_test_new')
text_list = []
for i in dataset:
    text_list.append(i['text'])

import numpy as np
np.save("yelp_large_test_new.npy", np.array(text_list))