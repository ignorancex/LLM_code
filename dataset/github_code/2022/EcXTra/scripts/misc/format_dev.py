'''
Before running, make sure you have the WMT19 data downloaded and untarred from:
    http://data.statmt.org/wmt19/translation-task/dev.tgz

Data prep code adapted from http://rtg.isi.edu/many-eng/data/v1/prep.tgz
'''

from pathlib import Path
import pdb
from format_wmt_data import handle_xml
import shutil
from iso639 import Lang
from process_m2e import CODES

from sacremoses import MosesTokenizer, MosesPunctNormalizer
from html import unescape
import logging

codes = [Lang(x).pt1 for x in CODES]
# code with dev sets:
# ['tr', 'fr', 'ru', 'zh', 'de', 'it', 'es', 'fi', 'hu', 'et', 'lt', 'lv', 'cs', 'ro', 'hi']
num_lines_per_lang = 200

###
wmt_dir = Path('wmt')
dev_dir = wmt_dir / 'dev'
output_dir = wmt_dir / 'dev_fmt'
output_dir.mkdir(exist_ok=True)

normr = MosesPunctNormalizer(
        lang='en',
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True,
    )
tok = MosesTokenizer(lang='en')

def tokenize_eng(text):
    try:
        text=unescape(text)
        text = normr.normalize(text)
        text = tok.tokenize(text, escape=False, return_str=True, aggressive_dash_splits=True,
            protected_patterns=tok.WEB_PROTECTED_PATTERNS)
        return text
    except:
        if text:
            logging.exception(f"error: {text}")
            return '<TOKERR> ' + text
        else:
            return ''
###

def make_dev_fmt_dir():
    num_missing = 0
    dev_codes = []
    for code in codes:
        matches = sorted(list(dev_dir.glob(f'newstest*.{code}')))
        if matches:
            src = matches[0]
            ref = dev_dir / (src.stem + '.en')
            shutil.copy(src, output_dir / src.name)
            shutil.copy(ref, output_dir / ref.name)
            dev_codes.append(code)
            continue

        matches = list(dev_dir.glob(f'*newsdev*.{code}.sgm')) or \
                list(dev_dir.glob(f'*newstest*.{code}.sgm'))
        matches = sorted(matches)
        if matches:
            src = matches[0]
            stem = src.name.split('.',1)[0].replace('-src', '', 1)
            ref = dev_dir / src.name.replace(f'-ref.{code}','-src.en')
            handle_xml(src, ref, output_dir, code)
            dev_codes.append(code)
            continue
        # else
        num_missing += 1
    print('num_missing', num_missing)
    print(f'found {len(dev_codes)}:', dev_codes)

def make_dev_fmt_files():
    src_name = wmt_dir / 'dev.src'
    en_name = wmt_dir / 'dev.en'
    lang_name = wmt_dir / 'dev.lang'
    with open(src_name, 'w') as f_src, open(en_name, 'w') as f_en, open(lang_name, 'w') as f_lang:
        for path in sorted(list(output_dir.glob('*'))):
            if path.name.endswith('.en'):
                continue
            code = path.suffix[1:]

            with path.open('r') as f:
                count = 0
                for line in f:
                    # line = tokenize_eng(line) + '\n'
                    f_src.write(line)
                    count += 1
                    if count >= num_lines_per_lang:
                        break

            path_en = path.with_suffix('.en')
            with path_en.open('r') as f:
                count = 0
                for line in f:
                    # line = tokenize_eng(line) + '\n'
                    f_en.write(line)
                    count += 1
                    if count >= num_lines_per_lang:
                        break
            f_lang.writelines([code + '\n' for _ in range(count)])
    print(src_name, en_name, lang_name)

if __name__ == "__main__":
    # make_dev_fmt_dir()
    make_dev_fmt_files()
