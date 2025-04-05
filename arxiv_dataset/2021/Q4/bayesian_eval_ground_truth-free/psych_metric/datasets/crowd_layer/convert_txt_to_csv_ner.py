"""Converts given data txt file into a more parsable formatted csv.
"""
import argparse

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Convert the given ner-mturk txt file into an easier to parse csv file.')

    parser.add_argument('input_file', help='Enter the file path to the csv of author names')

    parser.add_argument('output_file', help='Enter the file path to the desired output directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    txt = pd.read_csv(args.input_file, header=None, sep=' ', na_values='?', dtype=str, skip_blank_lines=False)
    # rename columns
    txt.columns = ['token'] + list(range(len(txt.columns) - 1))

    # add sequence column
    count = 0
    seq = np.empty(len(txt))
    for i in range(len(txt)):
        if txt.iloc[i].isna().all():
            seq[i] = np.nan
            count += 1
        else:
            seq[i] = count
    txt.insert(0, 'sequence', seq)

    # Remove all rows with only nas
    txt.dropna('index', 'all', inplace=True)

    # make sequence of dtype int
    txt['sequence'] = txt['sequence'].astype(int)

    # revert nas in token column to '?'
    txt['token'] = txt['token'].fillna('?')

    print(args.output_file)
    txt.to_csv(args.output_file, sep=' ', index=False, na_rep='NA')
