import pandas as pd
from collections import Counter
import ast
from utils import process_df

if __name__ == '__main__':

    models = ['qwen', 'baichuan2','chatglm2', 'breeze', 'taiwan-llm', 'ds','gpt4o','gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']
    langs = ['english', 'simplified', 'traditional']

    df_candidate = pd.read_csv('source_data/candidate_name_list/prompt_id_0.csv')
    candidate_list, mc_candidate_list, tw_candidate_list = [], [], []
    for i in range(len(df_candidate)):
        candidate_list += ast.literal_eval(df_candidate['candidate_name'][i])
        mc_candidate_list += ast.literal_eval(df_candidate['mainland_name'][i])
        tw_candidate_list += ast.literal_eval(df_candidate['taiwan_name'][i])
    name_counts_general = Counter(candidate_list)
    mc_candidate_list = set(mc_candidate_list)
    tw_candidate_list = set(tw_candidate_list)

    for model in models:
        for lang in langs:
            print(f'{model} {lang}')
            df = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=0, llm=model, action='load')
            df = df.dropna(subset=['extracted_name'])

            counts = Counter(df['extracted_name'].tolist())
            
            name_list, count_list, region_list = [], [], []

            for key in counts:
                if key in name_counts_general:
                    name_list.append(key)
                    count_list.append(counts[key]/name_counts_general[key])
                    if key in mc_candidate_list:
                        region_list.append('M')
                    elif key in tw_candidate_list:
                        region_list.append('T')
                    else:
                        region_list.append('NA')
            res_df = pd.DataFrame({'region':region_list, 'name':name_list, 'count':count_list})
            res_df = res_df.sort_values(by='count',ascending=False).reset_index(drop=True)
            print(res_df.loc[:4])
            
