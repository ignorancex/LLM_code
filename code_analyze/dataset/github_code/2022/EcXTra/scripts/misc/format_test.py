from pathlib import Path

from format_dev import tokenize_eng

wmt_dir = Path('wmt')
test_dir = wmt_dir / 'test'
output_dir = wmt_dir / 'test_fmt'
output_dir.mkdir(exist_ok=True)

def make_test_fmt_files():
    for path in sorted(list(test_dir.glob('*'))):
        if path.name.endswith('.en') or path.name.endswith('doc_id'):
            continue

        code = path.suffix[1:]

        path_out = output_dir / f'{path.stem}.{code}'

        with path.open('r') as f, path_out.open('w') as f_out:
            for line in f:
                line = tokenize_eng(line) + '\n'
                f_out.write(line)

        path_en = path.with_suffix('.en')
        path_out_en = output_dir / f'{path.stem}.en'

        with path_en.open('r') as f, path_out_en.open('w') as f_out:
            for line in f:
                line = tokenize_eng(line) + '\n'
                f_out.write(line)
        print(path_out, path_out_en)

if __name__ == "__main__":
    make_test_fmt_files()
