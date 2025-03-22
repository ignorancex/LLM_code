'''
Download subset of WikiMatrix data based on allowed languages for LWLL.
'''
import argparse
import json

from pathlib import Path
from subprocess import call, Popen

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('split', nargs='?', choices=['train', 'dev', 'test'], default='train')
parser.add_argument('--dry-run', '-d', action='store_true')
parser.add_argument('--save-path', '-p', type=Path, default=None)

def get_iso_codes(split):
    # ISO639-P3 codes for each split
    if split == 'train':
        iso_codes = \
            set(["afr", "aka", "asm", "aze", "bam", "bre", "cat", "ces", "dan", "deu", "est", "eus",
            "fao", "fin", "fra", "glg", "hau", "heb", "hye", "isl", "ita", "jav", "kan", "kat", "kor", "lao",
            "lat", "ltz", "lug", "mkd", "mlg", "mlt", "msa", "nde", "nor", "nya", "oci", "ori", "orm", "pan",
            "pus", "rus", "srp", "tam", "tel", "tgl", "tir", "tsn", "tur", "ukr", "urd", "vie", "wol", "xho",
            "zho"])
    elif split == 'dev':
        iso_codes = set(['sin', 'pol', 'pes', 'kaz'])
    elif split == 'test':
        iso_codes = set(['mal', 'slv', 'arb', 'tgk'])
    return iso_codes

def get_allowed_codes(df, langs_d, split):
    iso_codes = get_iso_codes(split)
    to_download = []
    codes_found = set()
    excluded_codes = set()
    for fname in df['tsv']:
        _, pair, _ = fname.split('.')
        l1, l2 = pair.split('-')
        if l1 == 'en':
            en, oth = l1, l2
        elif l2 == 'en':
            oth, en = l1, l2
        else:
            continue
        lang_code = langs_d.get(oth, {}).get('ISO639P3code')
        lang_name = langs_d.get(oth, {}).get("Name")

        if oth == 'zh':
            lang_code = 'zho'

        if not lang_code:
            print(f'No ISO mapping found for Wiki [{oth}]')
            pass
        elif lang_code not in iso_codes:
            print(f'Wiki [{oth}] to ISO [{lang_code}] found ({lang_name}), not allowed!')
            excluded_codes.add(oth)
            pass
        else:
            to_download.append(oth)
            codes_found.add(lang_code)
            print(f'Wiki [{oth}] to ISO [{lang_code}] ({lang_name}) added')
    # print('excluded:', sorted(excluded_codes), len(excluded_codes))
    return to_download, codes_found

if __name__ == "__main__":
    args = parser.parse_args()
    bitexts = Path('./WikiMatrix/list_of_bitexts.txt')
    lang_path = Path('./lang_info.json')
    with lang_path.open('r') as f:
        langs = json.load(f)
    df = pd.read_csv(bitexts, sep='\t', names=['tsv', 'num_lines'])

    to_download, codes_found = get_allowed_codes(df, langs, args.split)
    print('to_download codes:', to_download, len(to_download))
    # print(sorted(codes_found))
    # print('not found', allowed_codes - codes_found)

    if not args.dry_run:
        save_path = args.save_path or Path(f'./wikimatrix-parts/{args.split}')
        save_path.mkdir(exist_ok=True, parents=True)
        threads = []

        for i, wiki_code in enumerate(to_download, 1):
            pair = '-'.join(sorted(['en', wiki_code]))
            cmd = f'wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{pair}.tsv.gz' \
                  f' -P {save_path} -o /dev/null'
            print(f'running {cmd} ({i}/{len(to_download)})')
            thread = Popen(cmd, shell=True)
            threads.append(thread)
        for i, p in enumerate(threads):
            print(f'waiting on thread {to_download[i]}...')
            p.wait()
            print(f'{to_download[i]} done!')
