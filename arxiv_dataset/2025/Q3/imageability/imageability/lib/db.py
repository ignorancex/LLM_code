from imageability.lib import linalg

import faiss
import numpy as np
from pathlib import Path
from typing import List


def load_groundtruth(path: str):
    gt_words = []
    gt_scores = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            word, score = line.split('\t')
            gt_words.append(word.lower())
            gt_scores.append(float(score))

    return gt_words, np.array(gt_scores)


def load_collection(paths: List[str]):
    return np.concatenate([linalg.l2_normalize(np.load(path)) for path in paths], axis=0)


def index(collection: np.ndarray,
          index_path: Path,
          ncluster_multiplier: int = 8,
          ncodes: int = 64):
    if index_path.exists():
        print(r'Loading faiss index...')
        index = faiss.read_index(str(index_path))
    else:
        print(r'Building faiss index...')
        coarse_quantizer = faiss.IndexFlatL2(collection.shape[1])
        index = faiss.IndexIVFPQ(coarse_quantizer,
                                 collection.shape[1],
                                 ncluster_multiplier * int(np.sqrt(collection.shape[0])),
                                 ncodes, 8)
        index.train(collection)
        index.add(collection)
        faiss.write_index(index, str(index_path))

    return index
