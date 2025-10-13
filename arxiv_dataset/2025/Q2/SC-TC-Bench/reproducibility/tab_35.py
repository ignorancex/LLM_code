import pandas as pd
from utils import process_df, ModelCardDict


if __name__ == '__main__':
    model_latex_dict = ModelCardDict().model_latex_dict
    models = [ "qwen", "baichuan2", "chatglm2", "breeze", "taiwan-llm", "llama3-70b", "llama3-8b"]
    langs = ['simplified', 'traditional', 'english']
    for model in models:
        df_raw = pd.read_csv(f'response/regional_name/lastname_prob/{model}.csv')
        res = dict()
    
        n = 0
        n_prompt = 0
        for lang in langs:
            df = process_df(file_type='response', sub_file_type='raw', task='name', lang=lang, prompt_id=4, llm=model, action='load')

            for i in range(0, len(df), 2):
                n_prompt += 1
                first_name = df['lastname'][i] + df['firstname'][i]
                second_name = df['lastname'][i+1] + df['firstname'][i+1]
                first_prob = df['prob'][i]
                second_prob = df['prob'][i+1]

                raw_first_prob = df_raw[df_raw['name']==first_name]['prob'].values[0]
                raw_second_prob = df_raw[df_raw['name']==second_name]['prob'].values[0]

                if (first_prob >= second_prob and raw_first_prob >= raw_second_prob) or (first_prob < second_prob and raw_first_prob < raw_second_prob):
                    n += 1
        print(f"{model_latex_dict[model]} & {n / n_prompt * 100:.2f}\% \\\\")