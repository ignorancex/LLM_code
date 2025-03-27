from indicnlp.tokenize import indic_tokenize

import argparse
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('pred_name', type=Path)
parser.add_argument('ref_name', type=Path)

def write_tok(fname, do_tok=True):
    def tokenize_line(line, lang):
        return " ".join(indic_tokenize.trivial_tokenize(line, lang))

    with fname.open('r') as f:
        lines = [x.strip() for x in f.readlines()]

    if do_tok:
        lang = fname.suffix[1:3]
        lines = [tokenize_line(x, lang) for x in lines]

    tok_path = fname.parent / f'{fname.name}.tok'
    with tok_path.open('w') as fout:
        fout.writelines([line + '\n' for line in lines])

def main(pred_name, ref_name):
    write_tok(ref_name)
    write_tok(pred_name)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.pred_name, args.ref_name)
