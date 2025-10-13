import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', default=False, action='store_true')
    args = parser.parse_args()

    simp_corpora = ['baidu-baike', 'map-cc', 'mcc4']
    trad_corpora = ['tw-wiki', 'cctw', 'ootc', 'twc4', 'twchat']
    mix_corpora = ['c4']

    all_corpora = simp_corpora + trad_corpora + mix_corpora

    for corpus in all_corpora:
        df_simp = pd.read_csv(f'source_data/corpus_count/regional_term/mc_{corpus}.csv')
        df_trad = pd.read_csv(f'source_data/corpus_count/regional_term/tw_{corpus}.csv')

        if args.once:
            df_simp = df_simp.rename(columns={'word':'word_simp', 'count':'count_simp'})
            df_trad = df_trad.rename(columns={'word':'word_trad', 'count':'count_trad'})

            df = pd.concat([df_simp, df_trad], axis=1)
            df = df[(df['count_simp']>=1) & (df['count_trad']>=1)]
            df_simp = df[['word_simp', 'count_simp']]
            df_trad = df[['word_trad', 'count_trad']]

            df_simp = df_simp.rename(columns={'word_simp':'word', 'count_simp':'count'})
            df_trad = df_trad.rename(columns={'word_trad':'word', 'count_trad':'count'})

        if len(df_simp) == 1:
            simp_std = 0
        else:
            simp_std = df_simp['count'].std()
        if len(df_trad) == 1:
            trad_std = 0
        else:
            trad_std = df_trad['count'].std()
            

        print(f"{{\\tt {corpus}}} & {df_simp['count'].mean():.2f} \\scriptsize{{$\\textcolor{{gray}}{{\\pm {simp_std:.2f}}}$}} & {df_trad['count'].mean():.2f} \\scriptsize{{$\\textcolor{{gray}}{{\\pm {trad_std:.2f}}}$}} \\\\")

