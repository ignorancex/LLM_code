import pandas as pd
from utils import process_df

if __name__ == '__main__':
    target_names = ['王建国', '王俊凱']
    langs = ['simplified', 'traditional', 'english']

    df_candidate = pd.read_csv('source_data/candidate_name_list/prompt_id_0.csv')
   
    s1, s2 = '', ''
    for lang in langs:
        df = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=0, llm='baichuan2', action='load')
        n_total_1, n_select_1, n_total_2, n_select_2  = 0, 0, 0, 0
        for i in range(len(df_candidate)):
            extracted_name = df['extracted_name'][i]
            candidate_list = df_candidate['candidate_name'][i]

            if target_names[0] in candidate_list:
                n_total_1 += 1
            if target_names[1] in candidate_list:
                n_total_2 += 1

            if type(extracted_name) != float:
                if target_names[0] in extracted_name:
                    n_select_1 += 1
                if target_names[1] in extracted_name:
                    n_select_2 += 1
            
        s1 += f'{n_select_1/n_total_1*100:.2f}\\% & '
        s2 += f'{n_select_2/n_total_2*100:.2f}\\% & '
        
    print(s1)
    print(s2)

    """
    Note that the output is not directly a table. Instead, the output is the source code for generating the table in latex. 
    To generate the table, please replace the placeholders in the following latex code with the printed s.

    Latex source code:

    \begin{table*}[ht]
    \centering
    \begin{tabular}{cccc}
    \toprule[1.1pt]
    Name     & Simplified Chinese & Traditional Chinese & English\\\midrule
    {{s1_placeholder}}
    {{s2_placeholder}}
    \bottomrule[1.1pt]
    \end{tabular}
    \caption{\revise{Selection rates of {\baichuan} for \includegraphics[trim=10pt 14pt 10pt 9pt, clip=true, height=0.8em]{plots/chinese_characters/wang_jian_guo.pdf} and \includegraphics[trim=10pt 14pt 10pt 9pt, clip=true, height=0.8em]{plots/chinese_characters/wang_jun_kai.pdf} under prompts in Simplified Chinese, Traditional Chinese, and English.}}
    \label{tab:popular_name_selection_rate_examples}
    \end{table*}
    """

