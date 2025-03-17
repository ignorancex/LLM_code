import icu

from argparse import ArgumentParser
from pathlib import Path
parser = ArgumentParser()
parser.add_argument('src_path')
parser.add_argument('out_path')
parser.add_argument('--handle-sinhala', action='store_true')

def contains_sinhala(s):
    # HACK: current ICU package requires special handling of Sinhala
    # detect if a string contains Sinhala characters automatically
    s = s.translate({ord(ch): None for ch in '0123456789'})
    try:
        chars_to_test = [max(s), *s[:5], * s[-5:]]
    except ValueError:
        print('WARNING: could not handle line:')
        print(s)
        return False
    return any('\u0d80' <= char <= '\u0dff' for char in chars_to_test)

def transliterate(srcs, out_path=None, handle_sinhala=False, translit_str='Any-Latin; Latin-ASCII'):
    tl = icu.Transliterator.createInstance(translit_str)

    # transliterator for Sinhala requires a different instance
    tl_sin = icu.Transliterator.createInstance('si-si_Latn; Latin-ASCII')

    translits = []
    if isinstance(srcs, str) or isinstance(srcs, Path):
        with open(srcs, "r") as r:
            srcs = [x.strip() for x in r.readlines()]
    if out_path:
        w = open(out_path, "w")
    for i, line in enumerate(srcs):
        if handle_sinhala and contains_sinhala(line):
            transliteration = tl_sin.transliterate(line)
        else:
            transliteration = tl.transliterate(line)

        translits.append(transliteration)
        if out_path:
            w.write(transliteration)
            w.write("\n")
        if i % 10000 == 0:
            print(f'processed {i} lines', end="\r")
    if out_path:
        w.close()
    print(f"Finished transliterating {i+1} lines")
    return translits

if __name__ == '__main__':
    args = parser.parse_args()
    transliterate(args.src_path, args.out_path, args.handle_sinhala)
