import os
import argparse
import pandas as pd

import numpy as np
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="")

    args, unknown = parser.parse_known_args()
    return args, unknown

def show_stat(path):
    train_data = os.path.join(path, 'train.json')
    val_data = os.path.join(path, 'val.json')
    test_data = os.path.join(path, 'test.json')

    data = pd.DataFrame()
    data = data.append(pd.read_json(train_data, lines=True), ignore_index=True)
    data = data.append(pd.read_json(val_data, lines=True), ignore_index=True)
    data = data.append(pd.read_json(test_data, lines=True), ignore_index=True)

    print(data.describe())
    print("Jumlah =0 ==> %d" % len(data[data['summary_len'] == 0]))
    print("Jumlah <10 ==> %d" % len(data[data['summary_len'] < 10]))
    print("Jumlah <20 ==> %d" % len(data[data['summary_len'] < 20]))
    print("Jumlah <40 ==> %d" % len(data[data['summary_len'] < 40]))
    print("Jumlah <50 ==> %d" % len(data[data['summary_len'] < 50]))
    print("Jumlah <60 ==> %d" % len(data[data['summary_len'] < 60]))

def main():
    args, unknown = read_args()

    print('Standard')
    show_stat(args.data_root)

    print('\nDancer')
    show_stat(os.path.join(args.data_root, 'dancer'))

if __name__ == "__main__":
    main()
