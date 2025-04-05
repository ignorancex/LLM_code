import glob
import numpy as np
import random
import os
import random

train_val_lt = glob.glob('./train/*/*.png')
train_val_lt += glob.glob('./valid/*/*.png')

test_lt = glob.glob('./test/*/*.png')

random.shuffle(train_val_lt)

assert len(train_val_lt) == 180000
print(len(train_val_lt))

train_lt = train_val_lt[:120000]
hamming_lt = train_val_lt[120000:160000]
val_lt = train_val_lt[160000:]

def balance_list(data_lt):
    class2lt = {}
    for path in data_lt:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class2lt:
            class2lt[class_name] = [path]
        else:
            class2lt[class_name].append(path)
    class_num = [len(class2lt[class_name]) for class_name in class2lt]
    samples_each_class = min(class_num)
    
    target_lt = []
    for class_name in class2lt:
        s_lt = class2lt[class_name]
        t_lt = random.sample(s_lt, samples_each_class)
        target_lt += t_lt

    return target_lt

def sep_start_list(data_lt, samples_each_class):
    class2lt = {}
    for path in data_lt:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class2lt:
            class2lt[class_name] = [path]
        else:
            class2lt[class_name].append(path)
    
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

train_lt, start_lt = sep_start_list(train_lt, 20)

val_lt = balance_list(val_lt)
test_lt = balance_list(test_lt)

fout = open('train_lt.txt', 'w')
for i in range(len(train_lt)):
    name = train_lt[i]
    fout.write(name + '\n')
fout.close()

fout = open('start_lt.txt', 'w')
for i in range(len(start_lt)):
    name = start_lt[i]
    fout.write(name + '\n')
fout.close()

fout = open('val_lt.txt', 'w')
for i in range(len(val_lt)):
    name = val_lt[i]
    fout.write(name + '\n')
fout.close()

fout = open('test_lt.txt', 'w')
for i in range(len(test_lt)):
    name = test_lt[i]
    fout.write(name + '\n')
fout.close()

fout = open('hamming_lt.txt', 'w')
for i in range(len(hamming_lt)):
    name = hamming_lt[i]
    fout.write(name + '\n')
fout.close()


