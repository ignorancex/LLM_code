import pandas as pd
from utils import ModelCardDict, sig_sign
from scipy.stats import ttest_ind

if __name__ == '__main__':
    model_latex_dict = ModelCardDict().model_latex_dict
    regional_names = pd.read_csv('source_data/regional_name_and_characteristics.csv')

    models = [ "qwen", "baichuan2", "chatglm2", "breeze", "taiwan-llm", "llama3-70b", "llama3-8b"]

    for model in models:
        df = pd.read_csv(f'response/regional_name/lastname_prob/{model}.csv')
        new_df = pd.merge(df, regional_names, on='name')
        mc_prob = new_df[new_df['region']=='Mainland China']['prob']
        tw_prob = new_df[new_df['region']=='Taiwan']['prob']

        t_stat, p_two_sided = ttest_ind(mc_prob.tolist(), tw_prob.tolist(), equal_var=False)
        if t_stat < 0:
            pval = p_two_sided / 2
        else:
            pval = 1 - p_two_sided / 2
        
        print(f'{model_latex_dict[model]} & {mc_prob.mean():.2f} \\scriptsize{{$\\textcolor{{gray}}{{\\pm {mc_prob.std():.2f}}}$}} & {tw_prob.mean():.2f}  \\scriptsize{{$\\textcolor{{gray}}{{\\pm {tw_prob.std():.2f}}}$}} & {sig_sign(pval)} \\\\')