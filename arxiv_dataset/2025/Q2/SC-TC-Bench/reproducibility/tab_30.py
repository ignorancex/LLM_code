import pandas as pd
from utils import ModelCardDict, process_df, sig_sign
from statsmodels.stats.proportion import proportions_ztest

if __name__ == '__main__':

    model_latex_dict = ModelCardDict().model_latex_dict

    df_source_name = pd.read_csv('source_data/regional_name_and_characteristics.csv')
    m_pool = df_source_name[df_source_name['gender']=='male']['name'].tolist()
    
    models = ["qwen", "baichuan2", "chatglm2", "breeze", "taiwan-llm", "ds", "gpt4o", "gpt4", "gpt3.5", "llama3-70b", "llama3-8b"]

    langs = ['simplified', 'traditional', 'english']

    for model in models:
        s = f'{model_latex_dict[model]} '
        for lang in langs:
            n_males, n_valid = 0,0
            df = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=3, llm=model, action='load')
            valid_names = df[df['region']!=-1]['extracted_name'].tolist()
            n_valid += len(valid_names)
            for name in valid_names:
                if name in m_pool:
                    n_males += 1
        
            _, pval = proportions_ztest(n_males, n_valid, value=0.5, alternative='larger')
                    
            s += f'& {n_males/n_valid*100:.2f} & {sig_sign(pval)} & {n_valid} '
        
        s += '\\\\'

        print(s)