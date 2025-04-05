import glob
import numpy as np
import random
import os
import random

train_val_lt = glob.glob('./train/*/*.png')
#train_val_lt += glob.glob('./valid/*/*.png')

label2int = {}
for path in train_val_lt:
    class_name = os.path.basename(os.path.dirname(path))
    if class_name not in label2int:
        label2int[class_name] = len(label2int)

assert len(label2int) == 10

# output train validation list
fout = open('train_valid_lt.txt', 'w')
fout.write(f'path\tlabel\n')
for path in train_val_lt:
    class_name = os.path.basename(os.path.dirname(path))
    label = label2int[class_name]
    fout.write(f'{path}\t{label}\n')
fout.close()

test_lt = glob.glob('./test/*/*.png')
fout = open('test_lt.txt', 'w')
fout.write(f'path\tlabel\n')
for path in test_lt:
    class_name = os.path.basename(os.path.dirname(path))
    label = label2int[class_name]
    fout.write(f'{path}\t{label}\n')
fout.close()

index_lt = [i for i in range(len(train_val_lt))]
random.shuffle(index_lt)

def sep_start_list(index_lt, samples_each_class):
    class2lt = {}
    for index in index_lt:
        path = train_val_lt[index]
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class2lt:
            class2lt[class_name] = [index]
        else:
            class2lt[class_name].append(index)
    
    rest_lt = []
    start_lt = []

    for class_name in class2lt:
        s_lt = class2lt[class_name]
        random.shuffle(s_lt)
        t1_lt = s_lt[:samples_each_class]
        t2_lt = s_lt[samples_each_class:]
        start_lt += t1_lt
        rest_lt += t2_lt
    
    return rest_lt, start_lt

print(f'total number: {len(index_lt)}')

index_lt, start_index_lt = sep_start_list(index_lt, 500)
random.shuffle(index_lt)
train_index_lt = index_lt[:35000]
hamming_index_lt = index_lt[35000:40000]
val_index_lt = index_lt[40000:]

import json

with open('start_index_lt.json', 'w') as f:
    json.dump(start_index_lt, f)

with open('val_index_lt.json', 'w') as f:
    json.dump(val_index_lt, f)

with open('hamming_index_lt.json', 'w') as f:
    json.dump(hamming_index_lt, f)

with open('pool_index_lt.json', 'w') as f:
    json.dump(train_index_lt, f)


