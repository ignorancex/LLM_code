import os
import torch
import tqdm

import numpy as np
import pandas as pd

from SupervisedDataLoader import SupervisedDataLoader
from tests.hopkins import hopkins


def calculate_hopkins(_embs, _hbins):
    embs_hopkins = []
    for i in tqdm.tqdm(range(50)):
        embs_hopkins.append(hopkins(_embs, _hbins))
    return embs_hopkins, np.mean(embs_hopkins), np.std(embs_hopkins)


def load_embeddings(_emb_path):
    sent_emb = torch.load(_emb_path)
    train_path = os.path.join(wd, 'data/RESIDE/{d}_data/training_data.csv'.format(d=config['data_source']))
    valid_path = os.path.join(wd, 'data/RESIDE/{d}_data/validation_data.csv'.format(d=config['data_source']))
    data = SupervisedDataLoader(train_path, valid_path, 512)
    _valid_x, _valid_y = data.get_validation()

    valid_set = sent_emb[_valid_y].squeeze()
    batch = int(.001 * len(valid_set))
    eh, mu, st = calculate_hopkins(valid_set.numpy(), batch)
    print('Hopkins statistics:  Mu: {m}, sigma: {s}'.format(m=mu, s=st))
    return mu, st


if __name__ == "__main__":
    configs = [{'language_model': 'random',
                'data_source': 'riedel'},
                {'language_model': 'gem',
                'data_source': 'riedel'},
               {'language_model': 'glove',
                'data_source': 'riedel'},
               {'language_model': 'quickthought',
                'data_source': 'riedel'},
               {'language_model': 'skipthought',
                'data_source': 'riedel'},
               {'language_model': 'sentbert',
                'data_source': 'riedel'},
               {'language_model': 'laser',
                'data_source': 'riedel'},
               {'language_model': 'dct',
                'komp': 5,
                'data_source': 'riedel'},
               {"language_model": "infersent",
                "sentname": "infersent1glove",
                'data_source': 'riedel'},
               {"language_model": "infersent",
                "sentname": "infersent2ft",
                'data_source': 'riedel'}
               ]
    mus = []
    sds = []
    nms = []
    for config in configs:
        wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        if config['language_model'] == 'dct':
            sent_emb_path = os.path.join(wd,
                                         'sentence-embeddings/{lm}/{d}_{lm}_{k}_space.pt'.format(
                                             lm=config['language_model'],
                                             k=config['komp'],
                                             d=config['data_source']))
        elif config['language_model'] == 'infersent':
            sent_emb_path = os.path.join(wd,
                                         'sentence-embeddings/{lm}/{d}_{mn}_space.pt'.format(
                                             lm=config['language_model'],
                                             mn=config['sentname'],
                                             d=config['data_source']))
        else:
            sent_emb_path = os.path.join(wd,
                                         'sentence-embeddings/{lm}/{d}_{lm}_space.pt'.format(lm=config['language_model'],
                                                                                             d=config['data_source']))
        m, s = load_embeddings(sent_emb_path)
        mus.append(m)
        sds.append(s)
        nms.append(config['language_model'])
    out = pd.DataFrame({'names': nms, 'mus': mus, 'stdev': sds})
    print(out)
    with open('hopkins.tex', 'w') as tf:
        tf.write(out.to_latex(index=False))
