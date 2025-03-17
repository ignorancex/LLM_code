# Takes in either a folder name, or individual .en and .[src] files.
# Performs the following:
# 1. normalizes punctuation
# 2. removes lines where the English text is non-ASCII
# 3. (optional) finds the longest common substring (LCS) of en and src sentences, and remove those
#    where LCS ratio to sentence length is too high.
# Outputs en_clean and src_clean files.


import argparse
from difflib import SequenceMatcher
from pathlib import Path
from string import punctuation

from download_wikimatrix import extract_l2
import normalize

PUNCT = set(punctuation)

parser = argparse.ArgumentParser()

parser.add_argument('--en', type=Path)
parser.add_argument('--src', type=Path)
parser.add_argument('--folder', type=Path, help='run on all .[src] & .en files in a folder')
parser.add_argument('--lang_fam', type=Path, default=None)
parser.add_argument('--lcs-thresh', type=float, default=0.0,
                    help='If specified, filters out those pairs with LCS > threshold. '
                         'For dev/test, recommended to set to 0.8')

def clean_en(en, src, en_clean, src_clean, lf=None, lf_clean=None, lcs_thresh=0.0):
    normalizer = normalize.MosesPunctNormalizer()

    num_lines_orig = 0
    num_lines_clean = 0
    num_bad_ascii = 0
    num_bad_lcs = 0

    if lf and lf_clean:
        flf = lf.open('r')
        flfclean = lf_clean.open('w')

    with en.open('r') as fen, src.open('r') as fsrc, \
         en_clean.open('w') as fenclean, src_clean.open('w') as fsrcclean:
        for line_en, line_src in zip(fen, fsrc):
            num_lines_orig += 1
            if (num_lines_orig % 1000) == 0:
                print(f'processing line {num_lines_orig}', end='\r')

            line_src = normalizer.normalize(line_src)
            line_en = normalizer.normalize(line_en)
            if lf:
                line_lf = flf.readline()

            if not line_en.isascii():
                num_bad_ascii += 1
                continue

            if lcs_thresh > 0:
                match = SequenceMatcher(None, line_en, line_src, autojunk=False).find_longest_match(
                            0, len(line_en), 0, len(line_src))
                lcs = line_src[match.b: match.b + match.size]
                lcs_ratio = len(lcs) / min(len(line_en), len(line_src))
                if lcs_ratio > lcs_thresh:
                    num_bad_lcs += 1
                    continue

            fenclean.write(line_en + '\n')
            fsrcclean.write(line_src + '\n')

            if lf:
                flfclean.write(line_lf)

            num_lines_clean += 1

    if lf and lf_clean:
        flf.close()
        flfclean.close()

    return num_lines_orig, num_lines_clean, num_bad_ascii, num_bad_lcs


if __name__ == "__main__":
    args = parser.parse_args()

    lf, lf_clean = args.lang_fam, None
    lcs_thresh = args.lcs_thresh

    if args.folder:
        contents = list(args.folder.glob('WikiMatrix.??-??.txt.??'))
        basenames = set(fname.name.rsplit('.', 2)[0] for fname in contents)
        srcs, ens = [], []
        for basename in basenames:
            l2 = extract_l2(basename)
            srcs.append(args.folder / f'{basename}.txt.{l2}')
            ens.append(args.folder / f'{basename}.txt.en')

        out_folder = args.folder.parent / f'{args.folder.name}_clean'
        out_folder.mkdir(exist_ok=True, parents=True)
    else:
        ens = [args.en]
        srcs = [args.src]

        if lf:
            lf_clean = lf.parent / lf.name.replace('Matrix.', 'Matrix.clean2.')

    for en, src in zip(ens, srcs):
        if args.folder:
            en_clean = out_folder / en.name
            src_clean = out_folder / src.name
        else:
            en_clean = en.parent / en.name.replace('Matrix.', 'Matrix.clean.')
            src_clean = src.parent / src.name.replace('Matrix.', 'Matrix.clean.')


        print('cleaning...')
        num_lines_orig, num_lines_clean, num_bad_ascii, num_bad_lcs = \
            clean_en(en, src, en_clean, src_clean, lf, lf_clean, lcs_thresh)
        print(f'num lines originally: {num_lines_orig}')
        print(f'num lines after cleaning: {num_lines_clean}')
        print(f'removed non-ASCII: {num_bad_ascii}')
        print(f'removed overlapping: {num_bad_lcs}')
        print(f'saved to {en_clean} , {src_clean}')
