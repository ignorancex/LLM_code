import json
import re
from pathlib import Path

import fire
import nltk
import pandas as pd
import rouge
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

XSUM_KWARGS = dict(num_beams=6, length_penalty=1.0, max_length=256, min_length=10, no_repeat_ngram_size=3)
CNN_KWARGS = dict(num_beams=8, length_penalty=1.0, max_length=1024, min_length=55, no_repeat_ngram_size=3)
DEF_KWARGS = dict(num_beams=4, length_penalty=2.0, max_length=90, min_length=10, no_repeat_ngram_size=3)


def combine(src, tgt):
    gen = "".join(x + y for x, y in zip(src, tgt))
    gen = gen.replace("TL;DR:", "").strip()
    return gen


@torch.no_grad()
def run(data_path: str,
        checkpoint_path: str,
        output_path: str,
        do_sample: bool = False,
        do_greedy: bool = False,
        length_penalty: float = None,
        num_beams: int = None,
        min_length: int = None,
        max_length: int = None,
        no_tldr: bool = False,
        single_mask: bool = False,
        overwrite: bool = False,
        device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cnn" in data_path:
        beam_kwargs = CNN_KWARGS
    elif "xsum" in data_path:
        beam_kwargs = XSUM_KWARGS
    else:
        beam_kwargs = DEF_KWARGS
    if "ent" in data_path:
        beam_kwargs["max_length"] = 1024  # Generate more to cover all
    if length_penalty is not None:
        beam_kwargs["length_penalty"] = length_penalty
    if num_beams is not None:
        beam_kwargs["num_beams"] = num_beams
    if max_length is not None:
        beam_kwargs["max_length"] = max_length
    if min_length is not None:
        beam_kwargs["min_length"] = min_length
    print(beam_kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if output_path.exists() and not overwrite and len(list(open(output_path))) == len(list(open(data_path))):
        print("SKIP")
        return

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device)
    with open(output_path, "w") as file:
        hyp, ref = [], []
        cov = []
        for line in tqdm(list(open(data_path)), desc="Gen", dynamic_ncols=True):
            ins = json.loads(line)
            last_extra_id = re.findall(r"<extra_id_\d*>", ins["src"])[-1]
            m = re.match(r"<extra_id_(\d*)>", last_extra_id).group(1)
            eos_token = f"<extra_id_{int(m) + 1}>"
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

            src = ins["src"]
            if no_tldr:
                src = src.replace("TL;DR:", "")
            if single_mask:
                src = re.sub(r"<extra_id_\d*>", "<extra_id_0>", src)
                eos_token_id = tokenizer.convert_tokens_to_ids("<extra_id_1>")

            generated_ids = model.generate(**tokenizer(src, return_tensors="pt", truncation=True).to(device),
            # generated_ids = model.generate(**tokenizer(src, return_tensors="pt",).to(device),
                                           **beam_kwargs, eos_token_id=eos_token_id)
            generated = tokenizer.decode(generated_ids[0])
            src = re.split(r"<extra_id_\d*>", ins["src"].split("|")[0])
            gen = re.split(r"<extra_id_\d*>", generated)[1:]

            h = combine(src, gen)
            hyp.append(h)
            if "tgt" in ins:
                tgt = re.split(r"<extra_id_\d*>", ins["tgt"])[1:]
                ref.append(combine(src, tgt))
            else:
                ref.append(ins["ref"])
            if "entity" in ins:
                cov.append(ins["entity"] in h)
            else:
                cov.append(all(w.strip() in h for w in src[1:-1]))
                if not cov[-1]:
                    print([(w, w.strip() in h) for w in src[1:-1]])
                    print("OMG!!")

            print(h, file=file)

    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)

    scores = evaluator.get_scores(hyp, ref).items()
    scores = {"_".join((metric, k)): v for metric, vs in scores for k, v in vs.items()}
    scores["bleu"] = nltk.bleu_score.corpus_bleu([[nltk.word_tokenize(r)] for r in ref],
                                                 [nltk.word_tokenize(h) for h in hyp])
    if cov:
        scores["coverage"] = sum(cov) / len(cov)
    scores = pd.Series(scores)
    print(data_path, checkpoint_path)
    print(100 * scores)
    scores.to_json(str(output_path) + "_score.json")


if __name__ == '__main__':
    fire.Fire(run)
