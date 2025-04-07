import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import os

dir = ''
source = '0711_yt_seq_pn.csv'

name = '0711_yt_seq.csv'
mode = 0

source = os.path.join(dir, source)
all_lines = pd.read_csv(source, header=None)
all_lines = np.array(all_lines)

# read and write
if mode == 0:
    with open('val/val' + name, 'w', encoding='utf-8-sig') as f_val:
        pass
    with open('test/test' + name, 'w', encoding='utf-8-sig') as f_test:
        pass
    with open('train/train' + name, 'w', encoding='utf-8-sig') as f_train:
        pass

    for i in tqdm(range(len(all_lines))):
        if i % 10 == 1:
            with open('val/val' + name, 'a', encoding='utf-8-sig', newline='') as f_val:
                writer = csv.writer(f_val)
                writer.writerow(all_lines[i])
        elif i % 5 == 0:
            with open('test/test' + name, 'a', encoding='utf-8-sig', newline='') as f_test:
                writer = csv.writer(f_test)
                writer.writerow(all_lines[i])
        else:
            with open('train/train' + name, 'a', encoding='utf-8-sig', newline='') as f_train:
                writer = csv.writer(f_train)
                writer.writerow(all_lines[i])

if mode == 1:

    with open('/val/val' + name, 'w', encoding='utf-8-sig') as f_val:
        pass
    with open('/test/test' + name, 'w', encoding='utf-8-sig') as f_test:
        pass
    with open('/train/train' + name, 'w', encoding='utf-8-sig') as f_train:
        pass

    with open(source, 'r', encoding='utf-8', newline='') as f:
        for i in tqdm(len(all_lines)):
            if i % 5 == 0:
                with open('/test/test' + name, 'a', encoding='utf-8-sig', newline='') as f_test:
                    writer = csv.writer(f_test)
                    writer.writerow(all_lines[i])
            else:
                with open('/train/train' + name, 'a', encoding='utf-8-sig', newline='') as f_train:
                    writer = csv.writer(f_train)
                    writer.writerow(all_lines[i])

if mode == 2:
    with open('val/val' + name, 'w', encoding='utf-8-sig') as f_val:
        pass
    with open('test/test' + name, 'w', encoding='utf-8-sig') as f_test:
        pass
    with open('train/train' + name, 'w', encoding='utf-8-sig') as f_train:
        pass

    for i in tqdm(range(len(all_lines))):
        if (i % 10 == 0) or (i % 10 == 1):
            with open('val/val' + name, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(all_lines[i])
        elif (i % 10 == 2) or (i % 10 == 3):
            with open('test/test' + name, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(all_lines[i])
        else:
            with open('train/train' + name, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(all_lines[i])


