from tqdm import tqdm
import numpy as np
import argparse
import itertools
from pathlib import Path
from sentence_transformers import SentenceTransformer
from enum import Enum
import timeit
import math

class Model(Enum):
    AllMiniLM_L6 = 'allminilm_l6'
    AllMiniLM_L12 = 'allminilm_l12'
    GTE_BASE = 'gte_base'
    GTE_LARGE = 'gte_large'

    def __str__(self):
        return self.value

    def id(self):
        if self == self.AllMiniLM_L6:
            return 'sentence-transformers/all-MiniLM-L6-v2'
        elif self == self.AllMiniLM_L12:
            return 'sentence-transformers/all-MiniLM-L12-v2'
        elif self == self.GTE_BASE:
            return 'Alibaba-NLP/gte-base-en-v1.5'
        elif self == self.GTE_LARGE:
            return 'Alibaba-NLP/gte-large-en-v1.5'

    def dimensions(self):
        if self == self.AllMiniLM_L6:
            return 384
        elif self == self.AllMiniLM_L12:
            return 384
        elif self == self.GTE_BASE:
            return 768
        elif self == self.GTE_LARGE:
            return 1024


def chunks(iterable, batch_size):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


parser = argparse.ArgumentParser("embed")
parser.add_argument("--input", type=argparse.FileType('r'),
                    help=('Path to text file with one entry per line. '
                          'If file is tab-separated, this script takes the first column as input.'))
parser.add_argument("--output", type=str, help="Path to output file where embeddings are to be stored")
parser.add_argument("--model", type=Model, choices=list(Model))
parser.add_argument("--batch_size", help="Batch size", type=int, default=128, nargs="?")
parser.add_argument("--device", help="Device", type=str, default="mps", nargs="?")
args = parser.parse_args()

texts = [line.split('\t')[0] for line in args.input.readlines()]
print(f'Read {len(texts)} lines')

model = SentenceTransformer(args.model.id(), trust_remote_code=True, device=args.device)

embeddings = np.zeros((len(texts), args.model.dimensions()), dtype=np.float16)
begin = 0
start = timeit.default_timer()

for i, chunk in enumerate(tqdm(chunks(texts, batch_size=args.batch_size), total=int(math.ceil(len(texts) / args.batch_size)))):
    embeddings[begin:begin + args.batch_size] = model.encode(chunk).astype(np.float16)
    begin += args.batch_size

print(f'Done in {timeit.default_timer() - start:.2f}s')

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, embeddings)
