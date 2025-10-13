import pandas as pd
from utils import ModelCardDict

if __name__ == '__main__':
    model_latex_dict = ModelCardDict().model_latex_dict
    models = ['qwen', 'baichuan2','chatglm2', 'breeze', 'taiwan-llm', 'ds','gpt4o','gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']
    langs = ['simplified', 'traditional']

    for model in models:
        df_simp = pd.read_csv(f'response/regional_term/incorrect_response_type_annotated_by_gpt/{model}_simplified.csv')
        df_trad = pd.read_csv(f'response/regional_term/incorrect_response_type_annotated_by_gpt/{model}_traditional.csv')
        s = f"{model_latex_dict[model]} & {len(df_simp[df_simp['label']==1])/len(df_simp)*100:.2f}\% & {len(df_simp[df_simp['label']==2])/len(df_simp)*100:.2f}\% & {len(df_trad[df_trad['label']==1])/len(df_trad)*100:.2f}\% & {len(df_trad[df_trad['label']==2])/len(df_trad)*100:.2f}\% \\\\"
        print(s)