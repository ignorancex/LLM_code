import argparse

from pathlib import Path

import sacrebleu
from sacremoses import MosesDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--output", '-o', required=True, nargs='+', type=Path)
parser.add_argument("--gold", '-g', required=True, nargs='+', type=Path)
parser.add_argument('--verbose', '-v', action='store_true', help='output text is verbose, process this')
parser.add_argument('--detokenize', '-d', action='store_true')
parser.add_argument('--no-tokenize', '-n', action='store_true')

detok = MosesDetokenizer(lang='en')

def get_fam_lines(lang_lines_path):
    fam_lines = []
    prev_fam = ''
    cnt = 0
    for line in lang_lines_path.open('r'):
        if line != prev_fam:
            if prev_fam:
                fam_lines.append((prev_fam, cnt))
            # cnt = 0
        prev_fam = line
        cnt += 1
    fam_lines.append((prev_fam, cnt))
    return fam_lines


def load_for_sacre_bleu(pred_name, gold_names=[], verbose=False, detokenize=False, no_tokenize=False):
    pred_name = Path(pred_name)
    with pred_name.open('r') as f:
        if verbose:
            output = [x.rsplit(' ||| ')[-1].strip() for x in f.readlines()]
        else:
            output = [x.strip() for x in f.readlines()]

    gold = []
    if gold_names:
        gold = []
        if isinstance(gold_names, Path):
            gold_names = [gold_names]
        elif isinstance(gold_names, str):
            gold_names = [Path(gold_names)]
        for g in gold_names:
            g = Path(g)
            with g.open('r') as f:
                gold.extend([x.strip() for x in f.readlines()])
        gold = [gold]
    if detokenize:
        output_detok = [detok.detokenize(x.split(' ')) for x in output]
        output = output_detok
    return output, gold

if __name__ == "__main__":
    args = parser.parse_args()
    bleu_list = []
    printed = False
    for out_fname in args.output:
        output, gold = load_for_sacre_bleu(out_fname, args.gold, args.verbose, args.detokenize)
        if not printed:
            print(f'For entire input set (num sents: {len(output)})')
            printed = True
        tokenizer = 'none' if args.no_tokenize else None
        bleu_list.append((out_fname, sacrebleu.corpus_bleu(output, gold, tokenize=tokenizer)))
    if len(bleu_list) == 1:
        bleu = bleu_list[0][1]
        print(bleu)
        print(bleu.score)
    else:
        print('from lowest to highest BLEU')
        bleu_list.sort(key=lambda x: x[1].score)
        for ent in bleu_list:
            print(ent[0])
            print(ent[1])
            print(ent[1].score)
            print('*' * 10)
