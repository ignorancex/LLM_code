'''
Before running this script, download from http://rtg.isi.edu/many-eng/data/v1/ the files named train.v1*
'''

from pathlib import Path

# top 40 most common languages
CODES = ['tur','srp','fra','heb','rus','ara','zho','bos','nld','deu','por','nor','ita','spa','pol','fin','fas','swe','dan','ell','hun','slv','vie','est','slk', 'jpn','lit','lav','ukr','tha','ces','kor','ind','cat','mlt','ron','bul','hrv','hin', 'eus']
MAX = 2000000
ONLY_2M = True

data_dir = Path('/scratch/user/datasets')

eng = data_dir / 'train.v1.eng.tok'
src = data_dir / 'train.v1.src.tok'
lang_ids = data_dir / 'train.v1.lang'
prov = data_dir / 'train.v1.prov'

custom_id = 'custom' if not ONLY_2M else 'custom.2m'
out_eng = data_dir / f'train.{custom_id}.eng.tok'
out_src = data_dir / f'train.{custom_id}.src.tok'
out_lang = data_dir / f'train.{custom_id}.lang'
out_prov = data_dir / f'train.{custom_id}.prov'

count_d = {x: 0 for x in CODES}
num_lines = 0

if __name__ == "__main__":
    with eng.open() as f_eng, src.open() as f_src, lang_ids.open() as f_lang, prov.open() as f_prov, \
        out_eng.open('w') as f_out_eng, out_src.open('w') as f_out_src, \
        out_lang.open('w') as f_out_lang, out_prov.open('w') as f_out_prov:

        for sent_eng, sent_src, lang_id, prov_name in zip(f_eng, f_src, f_lang, f_prov):
            lang_id_n = lang_id
            lang_id = lang_id.strip()

            if lang_id not in count_d:
                continue
            if ONLY_2M and count_d[lang_id] >= MAX:
                continue

            f_out_eng.write(sent_eng)
            f_out_src.write(sent_src)
            f_out_lang.write(lang_id_n)
            f_out_prov.write(prov_name)
            count_d[lang_id] += 1
            num_lines +=1

            if num_lines % 250000 == 0:
                print(f'wrote {num_lines} lines ', end='\r')
                # break
            if ONLY_2M and count_d[lang_id] == MAX:
                print(f'wrote max lines for {lang_id}, not writing any more\n')

    print()
    with (data_dir / f'lang.{custom_id}.counts').open('w') as f_count:
        f_count.write(str(count_d), '\n')
