import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from utils import ModelCardDict, process_df, sig_sign
import ast

def retrieve_name_count(df, col):
    name_arr = []
    for i in range(len(df)):
        if col == 'extracted_name':
            name_arr.append(df[col][i])
        else:
            name_arr += ast.literal_eval(df[col][i])
    return Counter(name_arr)


if __name__ == '__main__':
    model_latex_dict = ModelCardDict().model_latex_dict

    df_online_mc = pd.read_csv('source_data/corpus_count/regional_name/mc_c4.csv')
    df_online_tw = pd.read_csv('source_data/corpus_count/regional_name/tw_c4.csv')
    df_online = pd.concat([df_online_mc, df_online_tw]).reset_index(drop=True)

    df_source_name = pd.read_csv('source_data/regional_name_and_characteristics.csv')
    mc_name = df_source_name[df_source_name['region']=='Mainland China']['name'].tolist()
    tw_name = df_source_name[df_source_name['region']=='Taiwan']['name'].tolist()

    df_name = pd.read_csv('source_data/candidate_name_list/prompt_id_0.csv')
   
    # count name appearance
    name_count = retrieve_name_count(df=df_name, col='candidate_name')

    p_values_s = []
    corrs_s = []

    models = ["qwen", "baichuan2", "chatglm2",  "breeze", "taiwan-llm", "llama3-8b", "llama3-70b", "gpt4o", "gpt4", "gpt3.5"]
    for model in models:
        for lang in ['simplified', 'traditional', 'english']:
            df = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=0, llm=model, action='load')
            df = df.dropna(subset=['extracted_name']).reset_index(drop=True)

            # count name selection
            llm_name_count = retrieve_name_count(df=df, col='extracted_name')
    
            name_list = []
            normalized_count_list = []
            region_list = []
            for key in llm_name_count:
                if key in name_count and (key in mc_name or key in tw_name):
                    name_list.append(key)
                    normalized_count_list.append(llm_name_count[key] / name_count[key])
                    if key in mc_name:
                        region_list.append(1)
                    else:
                        region_list.append(0)
            res = pd.DataFrame({'name':name_list, 'count':normalized_count_list, 'region': region_list})
            res = res.sort_values(by='count', ascending=False).reset_index(drop=True)

            llm_freq_arr = []
            corpus_freq_arr = []

            for i in range(len(res)):
                name = res['name'][i]
                llm_freq = res['count'][i]
                corpus_freq = df_online[df_online['word']==name]['count'].values[0]
                llm_freq_arr.append(llm_freq)
                corpus_freq_arr.append(corpus_freq)
                
            correlation, p_value = spearmanr(llm_freq_arr, corpus_freq_arr)
            p_values_s.append(p_value)
            corrs_s.append(correlation)
    

    p_values_s = np.array(p_values_s)
    rejected, corrected_p_values_s, _, _ = multipletests(p_values_s, alpha=0.05, method='fdr_bh')

    for idx, model in enumerate(models):
        s = f'{model_latex_dict[model]} & '
        s += f"{corrs_s[idx*3]:.2f} & {sig_sign(corrected_p_values_s[idx*3])} & {corrs_s[idx*3+1]:.2f} & {sig_sign(corrected_p_values_s[idx*3+1])} &  {corrs_s[idx*3+2]:.2f} & {sig_sign(corrected_p_values_s[idx*3+2])} \\\\" 
        print(s) 
