import argparse
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

parser = argparse.ArgumentParser('measure_frequency')
parser.add_argument('--output', type=str, help='Path to output file where frequencies are to be stored')
parser.add_argument('--collection', type=str, help='Path to the raw collection')
args = parser.parse_args()

frequencies = {}
with open(args.collection) as f:
    for line in tqdm(f.readlines()):
        words = word_tokenize(line)
        
        words = line.strip().split(' ')
        for word in words:
            word = word.lower()
            if not word.isalpha():
                continue
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1

with open(args.output, 'wt') as f:
    for word, count in frequencies.items():
        f.write(f'{word},{count}\n')
