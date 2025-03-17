'''
Download subset of WikiMatrix data. Therer are 79 x-en (or en-x) pairs.
Train split: top 25 x-en pairs by num sents
Dev split: 4 languages curated by us
Test split: 4 languages of same sub-family of dev
'''
import argparse
import json

from pathlib import Path
from subprocess import call, Popen

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('split', nargs='?', choices=['train', 'dev', 'test'], default='train')
parser.add_argument('--dry-run', '-d', action='store_true')
parser.add_argument('--save-path', '-p', type=Path, default=None)


def extract_l2(fname):
    tup = fname.split('.')
    if len(tup) == 3:
        _, pair, _ = tup
    elif len(tup) == 2:
        _, pair = tup
    l1, l2 = pair.split('-')
    if l1 == 'en':
        return l2
    elif l2 == 'en':
        return l1
    return ''

def get_codes(df_sorted, split):
    # ISO639-P3 codes for each split
    all_codes = list(map(extract_l2, df_sorted['tsv']))
    train_codes = all_codes[:25]
    dev_codes = ['tt', 'fo', 'ml', 'fa']
    test_codes = ['kk', 'is', 'ta', 'tg']

    if split == 'train':
        return train_codes
    elif split == 'dev':
        assert all(code not in train_codes for code in dev_codes)
        return dev_codes
    elif split == 'test':
        assert all(code not in train_codes for code in test_codes)
        return test_codes


if __name__ == "__main__":
    args = parser.parse_args()
    bitexts = Path('./WikiMatrix/list_of_bitexts.txt')
    df = pd.read_csv(bitexts, sep='\t', names=['tsv', 'num_lines'])
    df = df[df['tsv'].str.contains('en')]
    df = df[~df['tsv'].str.contains('simple')]
    df_sorted = df.sort_values('num_lines', ascending=False)

    codes = get_codes(df_sorted, args.split)
    print(args.split, ':', codes)

    if not args.dry_run:
        save_path = args.save_path or Path(f'./wikimatrix-parts/{args.split}')
        save_path.mkdir(exist_ok=True, parents=True)
        threads = []

        for i, wiki_code in enumerate(codes, 1):
            pair = '-'.join(sorted(['en', wiki_code]))
            cmd = f'wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{pair}.tsv.gz' \
                  f' -P {save_path} -o /dev/null'
            print(f'running {cmd} ({i}/{len(codes)})')
            thread = Popen(cmd, shell=True)
            threads.append(thread)
        for i, p in enumerate(threads):
            print(f'waiting on thread {codes[i]}...')
            p.wait()
            print(f'{codes[i]} done!')
