import pandas as pd
import ast
import argparse
from utils import ModelCardDict, process_df, sig_sign
from statsmodels.stats.proportion import proportions_ztest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--gender', default=False, action='store_true')
    parser.add_argument('--overall', default=False, action='store_true')
    args = parser.parse_args()

    model_latex_dict = ModelCardDict().model_latex_dict

    df_source_name = pd.read_csv('source_data/regional_name_and_characteristics.csv')

    df_candidate = pd.read_csv('source_data/candidate_name_list/prompt_id_1.csv')

    mc_m_pool = set(df_source_name[(df_source_name['gender']=='male') & (df_source_name['region']=='Mainland China')]['name'].tolist())
    mc_f_pool = set(df_source_name[(df_source_name['gender']=='female') & (df_source_name['region']=='Mainland China')]['name'].tolist())

    tw_m_pool = set(df_source_name[(df_source_name['gender']=='male') & (df_source_name['region']=='Taiwan')]['name'].tolist())
    tw_f_pool = set(df_source_name[(df_source_name['gender']=='female') & (df_source_name['region']=='Taiwan')]['name'].tolist())


    gender_male_mc, gender_male_tw = [],[]
    for i in range(len(df_candidate)):
        candidate_list = set(ast.literal_eval(df_candidate['candidate_name'][i]))
        n_mc_m = len(candidate_list.intersection(mc_m_pool))
        n_mc_f = len(candidate_list.intersection(mc_f_pool))
        n_tw_m = len(candidate_list.intersection(tw_m_pool))
        n_tw_f = len(candidate_list.intersection(tw_f_pool))

        assert n_mc_m + n_mc_f + n_tw_m + n_tw_f == 20
        gender_male_mc.append(n_mc_m)
        gender_male_tw.append(n_tw_m)
        
    
    df_candidate.loc[:, 'gender_dist_mc'] = gender_male_mc
    df_candidate.loc[:, 'gender_dist_tw'] = gender_male_tw
    
    models = ['qwen', 'baichuan2','chatglm2', 'breeze', 'taiwan-llm', 'ds','gpt4o','gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']

    c_total_sig, c_total= 0,0
    for model in models:
        s = model_latex_dict[model] + ' & '
        df = process_df(file_type='response', sub_file_type='final', task='name', lang=args.lang, prompt_id=1, llm=model, action='load')
        for n_males in range(1,4):
            
            target_index =df_candidate[(df_candidate['gender_dist_mc']==n_males) & (df_candidate['gender_dist_tw']==n_males)].index
            tem_df = df.loc[target_index]

            if args.overall:
                mc_count = len(tem_df[tem_df['region']==1])
                tw_count = len(tem_df[tem_df['region']==0])                
                ratio = mc_count / (mc_count + tw_count)
                
                _, pval = proportions_ztest(mc_count, mc_count + tw_count, value=0.5, alternative='smaller')

                if n_males < 3:
                    s += f"{ratio*100:.2f} & {sig_sign(pval)} & {mc_count + tw_count} & "
                else:
                    s += f"{ratio*100:.2f} & {sig_sign(pval)} & {mc_count + tw_count} \\\\ "
                
                c_total += mc_count + tw_count
                if sig_sign(pval) in ['*', '**', '***']:
                    c_total_sig += mc_count + tw_count

            if args.gender:
                target_ratio = n_males / 10

                mc_tem_df = tem_df[tem_df['region']==1]['extracted_name'].tolist()
                
                n_mc_m, n_mc_f = 0,0
                for name in mc_tem_df:
                    if name in mc_m_pool:
                        n_mc_m += 1
                    elif name in mc_f_pool:
                        n_mc_f += 1

                tw_tem_df = tem_df[tem_df['region']==0]['extracted_name'].tolist()
                
                n_tw_m, n_tw_f = 0,0
                for name in tw_tem_df:
                    if name in tw_m_pool:
                        n_tw_m += 1
                    elif name in tw_f_pool:
                        n_tw_f += 1
                
                if n_mc_m + n_mc_f == 0:
                    pass
                else:
                    ratio_mc = n_mc_m/(n_mc_m + n_mc_f)

                    _, pval_mc = proportions_ztest(n_mc_m, n_mc_m + n_mc_f, value=target_ratio, alternative='larger')

                    sign_mc = sig_sign(pval_mc, no_ns=True)
                    
                if n_tw_m+n_tw_f == 0:
                    pass
                else:
                    ratio_tw = n_tw_m/(n_tw_m+n_tw_f)
                    _, pval_tw = proportions_ztest(n_tw_m, n_tw_m+n_tw_f, value=target_ratio, alternative='larger')
                    sign_tw = sig_sign(pval_tw, no_ns=True)

                if n_tw_m+n_tw_f != 0 and n_mc_m + n_mc_f != 0:
                    if n_males < 3:
                        s += f"{ratio_mc*100:.2f}{sign_mc} & {ratio_tw*100:.2f}{sign_tw} & "
                    else:
                        s += f"{ratio_mc*100:.2f}{sign_mc} & {ratio_tw*100:.2f}{sign_tw} \\\\ "
                elif n_tw_m+n_tw_f != 0 and n_mc_m + n_mc_f == 0:
                    if n_males < 3:
                        s += f"- & {ratio_tw*100:.2f}{sign_tw} & "
                    else:
                        s += f"- & {ratio_tw*100:.2f}{sign_tw} \\\\ "
                elif n_tw_m+n_tw_f == 0 and n_mc_m + n_mc_f != 0:
                    if n_males < 3:
                        s += f"{ratio_mc*100:.2f}{sign_mc} & - & "
                    else:
                        s += f"{ratio_mc*100:.2f}{sign_mc} & - \\\\ "

        print(s)
