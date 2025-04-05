import json
from pathlib import Path

import fire
import nltk
import stanza
from tqdm import tqdm
from transformers import T5TokenizerFast

user_ent_type = {"EVENT", "FAC", "GPE", "LAW", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}

NLP = stanza.Pipeline(lang='en', processors='tokenize,ner', tokenize_no_ssplit=True)
TOKENIZER = T5TokenizerFast.from_pretrained("google/t5-v1_1-base")


def truncate_by_length(text, length=896):
    outputs = []
    num_tokens = 0
    sentences = nltk.sent_tokenize(text)
    for i, sent in enumerate(sentences):
        num_tokens += len(TOKENIZER.tokenize(sent))
        if num_tokens <= length:
            outputs.append(sent)
    if not outputs:
        return TOKENIZER.convert_tokens_to_string(TOKENIZER.tokenize(text)[:length])
    return " ".join(outputs)


def tldr(data_dir: Path, src_len: int = 896):
    for split in ("test", "val", "train"):
        with open(data_dir / f"{split}.jsonl", "w") as file:
            src = [truncate_by_length(x.strip(), length=src_len) for x in open(data_dir / f"{split}.source")]
            tgt = [x.strip() for x in open(data_dir / f"{split}.target")]
            assert len(src) == len(tgt)
            for s, t in tqdm(list(zip(src, tgt)), desc=split, dynamic_ncols=True):
                keywords = []
                src_template = "TL;DR:<extra_id_0>"
                tgt_template = "<extra_id_0>"
                i = 1
                prev_end = 0
                for entity in filter(lambda x: x.type in user_ent_type, NLP(t).ents):
                    keywords.append(entity.text)
                    src_template += f" {entity.text}<extra_id_{i}>"
                    tgt_template += t[prev_end:entity.start_char]
                    if t[entity.start_char - 1] == " ":
                        tgt_template = tgt_template[:-1]
                    tgt_template += f"<extra_id_{i}>"
                    prev_end = entity.end_char
                    i += 1
                tgt_template += t[prev_end:]
                tgt_template += f"<extra_id_{i}>"
                print(json.dumps(
                    {"src": " | ".join((src_template, s)), "tgt": tgt_template, "keywords": keywords, "ref": t}),
                      file=file)


def run(data_dir: str = "data/cnndm"):
    data_dir = Path(data_dir)
    tldr(data_dir)
    for seed in range(5):
        for s in (50, 100, 200, 500):
            tldr(data_dir / "sub" / f"train-{s}_seed-{seed}")


if __name__ == '__main__':
    fire.Fire(run)
