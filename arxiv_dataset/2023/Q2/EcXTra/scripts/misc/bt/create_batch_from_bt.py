'''
Makes a batch file from a set of input files. To be used in the back-translation setting, so it will
do the flips and foreign start tokens automatically.

For example, given files of form `xx_en`, will create batches for translating en->xx.

Also generates 2 helper JSON files:
    mapping for lang code -> foreign tok, foreign tok id
    list for the src langs of the dev set
'''

import argparse
import json
import sys
from collections import OrderedDict
from math import ceil
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle
from transformers import AutoTokenizer

from zsmt.create_mt_batches import write as write_batches
from zsmt.textprocessor import TextProcessor

xlm_name = 'xlm-roberta-large'

parser = argparse.ArgumentParser()
parser.add_argument('--bt-files', '-bf', type=Path, nargs='+')
parser.add_argument('--out-prefix', '-o', type=Path, required=True)
parser.add_argument('-num-lines', '-n', type=int, default=-1)
parser.add_argument('-dev-size', '-ds', type=int, default=-1)

parser.add_argument('--tokenizer-path', '-t', type=Path)

parser.add_argument('--wm-file', '-ef', type=Path, help='path to WikiMatrix en data')
parser.add_argument('--num-wm-lines', '-nwl', type=int, default=-1)

parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--lang-codes', '-lc',  help='list of langs', nargs='*')
parser.add_argument('--foreign-tok-d', '-ftd', type=Path, help='load foreign token dict')
parser.add_argument('--seed', type=int, default=2557)
parser.add_argument('--add-lang-code', '-alc', action='store_true')
parser.add_argument('--num-batches', '-nb', type=int, default=1)

def dev_train_split(data, dev_size):
    return data[:dev_size], data[dev_size:]

def get_foreign_tok_d(lang_codes, tokenizer, start_id=202201):
    ids = []
    curr_id = start_id
    while len(ids) < len(lang_codes):
        if tokenizer.convert_ids_to_tokens(curr_id).startswith('▁'):
            ids.append(curr_id)
        curr_id += 1
    foreign_toks = [(tok, id_) for tok, id_ in zip(tokenizer.convert_ids_to_tokens(ids), ids)]
    return OrderedDict(zip(lang_codes, foreign_toks))

def read_nlines(fname, nlines, strip=False):
    count = 0
    lines = []
    with fname.open('r') as f:
        for line in f:
            line = line.strip() if strip else line
            lines.append(line)
            count += 1
            if nlines != -1 and count >= nlines:
                break
    return lines

def infer_code(fname):
    if not fname.name.endswith('en'):
        return fname.suffix.lstrip('.')
    else:
        lang_pair = fname.suffixes[0].lstrip('.')
        return lang_pair.split('_')[0]

def is_src_en(fname):
    return not fname.name.endswith('en')

def write_nbatches(out_path, lines1, lines2, dst_tokenizer, nbatches):
    if not lines1 or not lines2:
        print('No lines passed in for this batch, skipping. Check --dev-size')
        return
    if nbatches == 1:
        out_name = str(out_path)
        write_batches(output_file=out_name, src_txt_file=lines1, dst_txt_file=lines2,
                  tp_dst=dst_tokenizer, quiet=False, sort=False)
    else:

        batch_size = ceil(len(lines1) / nbatches)

        quiet = False
        for i, start in enumerate(range(0, len(lines1), batch_size)):
            lines1_curr = lines1[start:start+batch_size]
            lines2_curr = lines2[start:start+batch_size]
            print(f'writing batch {i} of size {len(lines1_curr)}...')
            out_name = out_path.parent / f'{out_path.stem}.{i}{out_path.suffix}'
            write_batches(output_file=out_name, src_txt_file=lines1_curr, dst_txt_file=lines2_curr,
                    tp_dst=dst_tokenizer, quiet=quiet, sort=False)
            # quiet = True



if __name__ == "__main__":
    args = parser.parse_args()

    for fname in args.bt_files:
        if not fname.exists():
            print(f'could not find {fname}, aborting')
            sys.exit(-1)

    if args.lang_codes:
        lang_codes = args.lang_codes
    else:
        print('inferring lang names from --bt-files')
        lang_codes = [infer_code(x) for x in args.bt_files]
        print(lang_codes)

    out_dir = args.out_prefix.parent
    out_stem = args.out_prefix.name

    if args.tokenizer_path:
        print(f'loading from `{args.tokenizer_path}`')
        dst_tokenizer = TextProcessor(args.tokenizer_path)
    else:
        print(f'using pretrained tokenizer `{xlm_name}`')
        dst_tokenizer = TextProcessor(pretrained_name=xlm_name)

    if args.foreign_tok_d:
        with args.foreign_tok_d.open('r') as f:
            foreign_tok_d = json.load(f)
    else:
        src_tokenizer = AutoTokenizer.from_pretrained(xlm_name)
        foreign_tok_d = get_foreign_tok_d(lang_codes, src_tokenizer)
        del src_tokenizer
        foreign_tok_path = out_dir / f'{out_stem}.foreign_toks.json'
        with foreign_tok_path.open('w') as f:
            json.dump(foreign_tok_d, f)

    print('foreign tok dict is', foreign_tok_d)

    lines_src, lines_tgt = [], []

    dev_lc = []
    dev_src = []
    dev_tgt = []

    if args.wm_file:
        en_file = args.wm_file
        src_file = en_file.parent / f'{en_file.stem}.src'
        print(f'processing WikiMatrix files...')
        # In this script, we flip for BT, so we preemptively flip here
        #  so Wikimatrix data is the 'right way' after 2 flips
        lines_tgt_curr = read_nlines(src_file, args.num_wm_lines, strip=True)
        lines_src_curr = read_nlines(en_file, args.num_wm_lines, strip=True)
        if args.shuffle:
            print(' shuffling WikiMatrix...')
            lines_tgt_curr, lines_src_curr = shuffle(lines_tgt_curr, lines_src_curr, random_state=args.seed)

        # if args.add_lang_code:
        #     en_tok = foreign_tok_d['en'][0]
        #     lines_tgt_curr = [f'{en_tok} {line}' for line in lines_tgt_curr]

        if args.dev_size > 0:
            dev_size = args.dev_size * len(args.bt_files)
            dev_src_curr, lines_src_curr = dev_train_split(lines_src_curr, dev_size)
            dev_tgt_curr, lines_tgt_curr = dev_train_split(lines_tgt_curr, dev_size)
            dev_src.extend(dev_src_curr)
            dev_tgt.extend(dev_tgt_curr)
            dev_lc.extend([''] * len(dev_src_curr))

        lines_src.extend(lines_src_curr)
        lines_tgt.extend(lines_tgt_curr)

        print(f' read {len(lines_src)} lines')

    for code, fname in zip(lang_codes, args.bt_files):
        src_is_en = is_src_en(fname)

        print(f'processing {fname}...')
        print('src_is_en', src_is_en)
        lines_curr = read_nlines(fname, args.num_lines)
        tup = [x.split('|||') for x in lines_curr]
        lines_src_curr, lines_tgt_curr = [x[0].strip() for x in tup], [x[1].strip() for x in tup]

        # remove examples where src is duplicated
        print(f'deduplicating src...', end='')
        uniq_inds = sorted(np.unique(lines_src_curr, return_index=True)[1])
        print(f' kept {len(uniq_inds)}/{len(lines_src_curr)} deduped lines')
        lines_src_curr = [lines_src_curr[x] for x in uniq_inds]
        lines_tgt_curr = [lines_tgt_curr[x] for x in uniq_inds]

        # remove examples where either src or tgt is empty
        nz_inds1 = np.nonzero([len(x) for x in lines_src_curr])[0]
        nz_inds2 = np.nonzero([len(x) for x in lines_tgt_curr])[0]
        nz_inds = np.intersect1d(nz_inds1, nz_inds2)

        print(f' kept {len(nz_inds)}/{len(lines_src_curr)} non-zero lines')
        lines_src_curr = [lines_src_curr[x] for x in nz_inds]
        lines_tgt_curr = [lines_tgt_curr[x] for x in nz_inds]

        foreign_tok = foreign_tok_d[code][0]
        if args.add_lang_code and not src_is_en: # opposite to account for BT flip
            lines_tgt_curr = [f'{foreign_tok} {line}' for line in lines_tgt_curr]
        elif src_is_en:
            lines_src_curr = [line.replace(foreign_tok[1:], '', 1).lstrip() for line in lines_src_curr]

        if args.shuffle:
            print(' shuffling lines...')
            lines_tgt_curr, lines_src_curr = shuffle(lines_tgt_curr, lines_src_curr, random_state=args.seed)

        if args.dev_size > 0:
            dev_src_curr, lines_src_curr = dev_train_split(lines_src_curr, args.dev_size)
            dev_tgt_curr, lines_tgt_curr = dev_train_split(lines_tgt_curr, args.dev_size)
            dev_src.extend(dev_src_curr)
            dev_tgt.extend(dev_tgt_curr)
            to_write = code if src_is_en else 'en'
            dev_lc.extend([to_write] * len(dev_src_curr))

        lines_src.extend(lines_src_curr)
        lines_tgt.extend(lines_tgt_curr)

    if dev_src:
        dev_path = out_dir / f'{out_stem}.dev.batch'
        print('writing dev batches...')
        write_nbatches(dev_path, dev_tgt, dev_src, dst_tokenizer, 1)
        lc_path = out_dir / f'{out_stem}.dev_lang_codes.json'
        with lc_path.open('w') as f:
            json.dump(dev_lc, f)

    dev_path_src = out_dir / f'{out_stem}.dev_src.txt'
    with dev_path_src.open('w') as f:
        f.writelines('\n'.join(dev_tgt))

    dev_path_tgt = out_dir / f'{out_stem}.dev_tgt.txt'
    with dev_path_tgt.open('w') as f:
        f.writelines('\n'.join(dev_src))
    if args.shuffle:
        print(' shuffling lines...')
        lines_tgt, lines_src = shuffle(lines_tgt, lines_src, random_state=args.seed)

    train_path = out_dir / f'{out_stem}.batch'
    print('writing batches...')
    write_nbatches(train_path, lines_tgt, lines_src, dst_tokenizer, args.num_batches)
