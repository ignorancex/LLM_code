import pandas as pd
from opencc import OpenCC
from transformers import AutoTokenizer
from utils import ModelCardDict, sig_sign
import numpy as np
from scipy import stats

def tokenize_name_list(arr, tokenizer, openai_model=False):
    res = []
    for name in arr:
        if openai_model:
            tokens = tokenizer.encode(name)
        else:
            tokens = tokenizer(name, return_tensors="pt")["input_ids"][0]
        res.append(len(tokens))
    return res

if __name__ == '__main__':

    df_name = pd.read_csv('source_data/regional_name_and_characteristics.csv')
    
    mc_original = df_name[df_name['region']=='Mainland China']['name'].tolist()
    tw_original = df_name[df_name['region']=='Taiwan']['name'].tolist()

    s2t_converter = OpenCC('s2t')
    t2s_converter = OpenCC('t2s')

    mc_converted = [s2t_converter.convert(x) for x in mc_original]
    tw_converted = [t2s_converter.convert(x) for x in tw_original]

    models = ['qwen', 'baichuan2', 'breeze', 'taiwan-llm', 'ds', 'gpt4o','gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']
    model_card_dict = ModelCardDict().model_card_dict
    model_latex_dict = ModelCardDict().model_latex_dict

    for model in models:
        if model in ['gpt4o','gpt4', 'gpt3.5']:
            openai_model = True
            import tiktoken
            tokenizer = tiktoken.encoding_for_model(model_card_dict[model])
        else:
            openai_model = False
            tokenizer = AutoTokenizer.from_pretrained(model_card_dict[model], trust_remote_code=True)

        token_count_mc_original = tokenize_name_list(arr=mc_original, tokenizer=tokenizer, openai_model=openai_model)
        token_count_mc_converted = tokenize_name_list(arr=mc_converted, tokenizer=tokenizer, openai_model=openai_model)
        token_count_tw_original = tokenize_name_list(arr=tw_original, tokenizer=tokenizer, openai_model=openai_model)
        token_count_tw_converted = tokenize_name_list(arr=tw_converted, tokenizer=tokenizer, openai_model=openai_model)

        _, p_value_mc = stats.ttest_ind(token_count_mc_original, token_count_mc_converted)
        _, p_value_tw = stats.ttest_ind(token_count_tw_original, token_count_tw_converted)

        s = f'{model_latex_dict[model]} & '
        s += f'{np.mean(token_count_mc_original):.2f} \\small{{$\\textcolor{{gray}}{{\\pm {np.std(token_count_mc_original):.2f}}}$}} & '
        s += f'{np.mean(token_count_mc_converted):.2f} \\small{{$\\textcolor{{gray}}{{\\pm {np.std(token_count_mc_converted):.2f}}}$}} & '
        s += f'{sig_sign(p_value_mc)} & '
        s += f'{np.mean(token_count_tw_original):.2f} \\small{{$\\textcolor{{gray}}{{\\pm {np.std(token_count_tw_original):.2f}}}$}} & '
        s += f'{np.mean(token_count_tw_converted):.2f} \\small{{$\\textcolor{{gray}}{{\\pm {np.std(token_count_tw_converted):.2f}}}$}} & '
        s += f'{sig_sign(p_value_tw)} \\\\'
        print(s)


    """
    Note that the output is not directly a table. Instead, the output is the source code for generating the table in latex. 
    To generate the table, please replace the placeholders in the following latex code with the printed s.

    Latex source code:

    \begin{table*}[t]
    \centering
    \adjustbox{max width=\linewidth}{
    \begin{tabular}{lcclccl}
    \toprule[1.1pt]
    Model      & Mainland Chinese Names & \begin{tabular}[c]{@{}l@{}} Mainland Chinese Names \\ (converted into \\ Traditional Chinese)\end{tabular}   & Significance    & Taiwanese Names & \begin{tabular}[c]{@{}l@{}} Taiwanese Names \\ (converted into \\ Simplified Chinese)\end{tabular} & Significance     \\ \midrule
    {{s_placeholder}}
    \bottomrule[1.1pt]
    \end{tabular}}
    \end{table*}
    """