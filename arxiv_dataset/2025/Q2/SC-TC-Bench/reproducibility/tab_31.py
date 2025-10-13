import pandas as pd
import argparse
from utils import process_df
from collections import Counter

def identify_top_adjective(adj):
    new_adj = []
    for item in adj:
        if type(item) == str:
            item = item.replace('的', '')
            if ', ' in item:
                item = item.split(', ')
                for idx, k in enumerate(item):
                    if '资质' in k:
                        item[idx] = '最有资质'
                    elif '最好' in k or '最佳' in k:
                        item[idx] = '最好'
                    elif '最常见' in k:
                        item[idx] = '常见'
                    elif '最常見' in k or '普遍' in k:
                        item[idx] = '常見'
                new_adj += item
            elif '、' in item:
                item = item.split('、')
                for idx, k in enumerate(item):
                    if '资质' in k:
                        item[idx] = '最有资质'
                    elif '最好' in k or '最佳' in k:
                        item[idx] = '最好'
                    elif '最常见' in k:
                        item[idx] = '常见'
                    elif '最常見' in k or '普遍' in k:
                        item[idx] = '常見'
                new_adj += item
            elif '  \n' in item:
                item = item.split('  \n')
                for idx, k in enumerate(item):
                    if '资质' in k:
                        item[idx] = '最有资质'
                    elif '最好' in k or '最佳' in k:
                        item[idx] = '最好'
                    elif '最常见' in k:
                        item[idx] = '常见'
                    elif '最常見' in k or '普遍' in k:
                        item[idx] = '常見'
                new_adj += item
            else:
                if '资质' in item:
                    new_adj.append('最有资质')
                elif '最好' in item or '最佳' in item:
                    new_adj.append('最好')
                elif '最常见' in item:
                    new_adj.append('常见')
                elif '最常見' in item or '普遍' in item:
                    new_adj.append('常見')
                else:
                    new_adj.append(item)
    return new_adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default=None)
    parser.add_argument('--name_example', default=False, action='store_true')
    args = parser.parse_args()

    langs = ['simplified', 'traditional']
    regions = [1, 0]

    s = ''
    for lang in langs:
        for region in regions:
            df_raw = process_df(file_type='response', sub_file_type='raw', task='name', lang=lang, prompt_id=0, llm=args.llm, action='load')
            df_refine = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=0, llm=args.llm, action='load')
            df_refine = df_refine.dropna(subset=['extracted_name'])
            df_refine = df_refine[df_refine['region'] != -1].reset_index(drop=True).loc[:1999]
            
            df = pd.read_csv(f'response/regional_name/explanation/{args.llm}_{lang}.csv')
            adj = df[(df['region']==region) & (df['adjective']!= 'NA')]['adjective'].tolist()

            locals()[f'{lang}_{region}'] = Counter(identify_top_adjective(adj)).most_common(10)

            if args.name_example:
                if region == 0:
                    df = df[(df['region']==region) & (df['adjective']!= 'NA')]
                    df = df.dropna(subset=['adjective']).reset_index(drop=True)
                    names = []
                    for i in range(len(df)):
                        if '才华横溢' in df['adjective'][i] or '有才华' in df['adjective'][i] or '智慧' in df['adjective'][i] or '才智' in df['adjective'][i] or '才华出众' in df['adjective'][i]:
                            names.append(df['name'][i])
                    count = Counter(names).most_common(3)
                    l = []
                    for name, _ in count:
                        l.append(name)
                    s += ','.join(l)
                    s += ' & '

    if args.name_example:
        print(s[:-2])
    else:
        for i in range(10):
            print(f'{simplified_1[i][0]} & {traditional_1[i][0]} & {simplified_0[i][0]} & {traditional_0[i][0]}')

        