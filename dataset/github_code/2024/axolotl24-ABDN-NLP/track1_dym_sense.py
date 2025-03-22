import argparse
import logging
import random
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
import numpy as np
from tqdm import tqdm

import bertcloud as bc
import sensecluster as sc
import utils

NEW_PERIOD = "new"
OLD_PERIOD = "old"
SENSE_ID_COLUMN = "sense_id"
USAGE_ID_COLUMN = "usage_id"
PERIOD_COLUMN = "period"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--st", help="Similarity threshold", type=float, default=0.8)
    arg("--k", help="Number of neighboring words", type=int, default=5)
    arg("--lang", help="Langauge", type=str, default='ru')
    return parser.parse_args()


def gen_cloud(keyword, corpus, cache_path=None):
    if cache_path is not None:
        cloud = utils.load_from_disk(cache_path+'cloud.pkl')
        static_embeds = utils.load_from_disk(cache_path+'static_embeds.pkl')
        if cloud is not None and static_embeds is not None:
            return cloud, static_embeds
    # prbert_model = 'bert-base-multilingual-cased'
    prbert_model = 'E:/NLP-new/bert-base-multilingual-cased'
    cloud, stat_emb = bc.bert_cloud(keyword, corpus, prbert_model, sen_emb=True)
    static_embeds = utils.init_static_embedding()
    utils.update_static_embeds(static_embeds, stat_emb)
    if cache_path is not None:
        utils.save_to_disk(cache_path+'cloud.pkl', cloud)
        utils.save_to_disk(cache_path+'static_embeds.pkl', static_embeds)
        utils.save_sendata_as_txt(cache_path+'corpus.txt', corpus)

    return cloud, static_embeds


def main():
    args = parse_args()
    targets = pd.read_csv(args.test, sep="\t")
    print(" length usages ", len(targets.usage_id))
    print(" length targets.word ", len(targets.word), " len targets.word.unique() ", len(targets.word.unique()))
    for target_word in tqdm(targets.word.unique()):
        print("target word: ", target_word)
        this_word = targets[targets.word == target_word]
        new = this_word[this_word[PERIOD_COLUMN] == NEW_PERIOD]
        old = this_word[this_word[PERIOD_COLUMN] == OLD_PERIOD]
        new_examples = new.example.to_list()
        new_usage_ids = new[USAGE_ID_COLUMN]
        old_glosses = [
            f"{gl} {ex}".strip() if isinstance(ex, str) else gl
            for gl, ex in zip(old.gloss.to_list(), old.example.to_list())
        ]
        senses_old = old[SENSE_ID_COLUMN].to_list()
        latin_name = senses_old[0].split("_")[0]

        cloud1, static_embeds1 = gen_cloud(target_word, old_glosses, cache_path='./.cache/words/{}/1/sen_emb/'.format(target_word))
        cloud2, static_embeds2 = gen_cloud(target_word, new_examples, cache_path='./.cache/words/{}/2/sen_emb/'.format(target_word))
        static_embeds = utils.merge_static_embeds(static_embeds1, static_embeds2)

        sense_clu2, _ = \
                sc.cloud_cluster(cloud2, static_embeds, target_word, args.lang, k=args.k, t=args.st,
                                 cache_time='2')

        exs2senses = {}
        seen = set()
        for label, clu in enumerate(sense_clu2):
            found = ""
            examples_indices = clu.points
            examples = [new_examples[i] for i in examples_indices]
            for emb2, defs, sense_old in zip(cloud1, old_glosses, senses_old):
                if sense_old not in seen:
                    sim = utils.cosine_similarity(clu.center, emb2)
                    if sim >= args.st:
                        found = sense_old
                        seen.add(sense_old)
                        break
            if not found:
                found = f"{latin_name}_novel_{label}"
            for ex in examples:
                exs2senses[ex] = found
        assert len(new_examples) == new_usage_ids.shape[0]
        for usage_id, example in zip(new_usage_ids, new_examples):
            system_answer = exs2senses[example]
            row_number = targets[targets[USAGE_ID_COLUMN] == usage_id].index
            targets.loc[row_number, SENSE_ID_COLUMN] = system_answer

    logging.info(f"Writing the result to {args.pred}")
    targets.to_csv(args.pred, sep="\t", index=False)


if __name__ == "__main__":
    main()
