import argparse
import pandas as pd
from utils import process_df, postprocess_response, get_oppo, label_response, collect_response_details

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_id', type=int, default=None)
    parser.add_argument('--no_gpt', default=False, action='store_true') # validation experiment when excluding items whose definition is sourced from gpt

    args = parser.parse_args()

    cs = []
    d = dict()
    models = ['qwen', 'baichuan2','chatglm2', 'breeze', 'taiwan-llm', 'ds','gpt4o','gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']
    s0 = ''
    s1 = ''
    s2 = ''

    if args.no_gpt:
        term = pd.read_csv('source_data/regional_term_and_definition.csv')
        terms =[]
        for i in range(15):
            terms += term['definition_source'].tolist()
    
    overview_dict_s, overview_dict_t = dict(), dict()

    for model_idx, model in enumerate(models):
        df_s = process_df(file_type='response', sub_file_type='correctness_annotated_by_gpt', task='term', lang='simplified', prompt_id=args.prompt_id, llm=model, action='load')
        df_t = process_df(file_type='response', sub_file_type='correctness_annotated_by_gpt', task='term', lang='traditional', prompt_id=args.prompt_id, llm=model, action='load')

        postprocess_response(df=df_s, model=model, lang='simplified', ver=args.prompt_id)
        postprocess_response(df=df_t, model=model, lang='traditional', ver=args.prompt_id)

        if model in ['ds']:
            df_s_response = process_df(file_type='response', sub_file_type='raw', task='term', lang='simplified', prompt_id=args.prompt_id, llm=model, action='load', no_think=True)
            df_t_response = process_df(file_type='response', sub_file_type='raw', task='term', lang='traditional', prompt_id=args.prompt_id, llm=model, action='load', no_think=True)
        else:
            df_s_response = process_df(file_type='response', sub_file_type='raw', task='term', lang='simplified', prompt_id=args.prompt_id, llm=model, action='load')
            df_t_response = process_df(file_type='response', sub_file_type='raw', task='term', lang='traditional', prompt_id=args.prompt_id, llm=model, action='load')

        df_s_prompt = process_df(file_type='prompt', task='term', lang='simplified', prompt_id=args.prompt_id)
        df_t_prompt = process_df(file_type='prompt', task='term', lang='traditional', prompt_id=args.prompt_id)
        
        assert len(df_s) == len(df_s_response)
        assert len(df_s) == len(df_s_prompt)

        if args.no_gpt:
            df_s.loc[:, 'gpt'] = terms
            df_s = df_s[df_s['gpt']!='gpt'].reset_index(drop=True)

            df_t.loc[:, 'gpt'] = terms
            df_t = df_t[df_t['gpt']!='gpt'].reset_index(drop=True)

            df_s_response.loc[:, 'gpt'] = terms
            df_s_response = df_s_response[df_s_response['gpt']!='gpt'].reset_index(drop=True)
            df_t_response.loc[:, 'gpt'] = terms
            df_t_response = df_t_response[df_t_response['gpt']!='gpt'].reset_index(drop=True)

            df_s_prompt.loc[:, 'gpt'] = terms
            df_s_prompt = df_s_prompt[df_s_prompt['gpt']!='gpt'].reset_index(drop=True)
            df_t_prompt.loc[:, 'gpt'] = terms
            df_t_prompt = df_t_prompt[df_t_prompt['gpt']!='gpt'].reset_index(drop=True)

        oppo_res_s = get_oppo(gt=df_t_prompt['gt'].tolist(), response=df_s_response['response'].tolist(), mode='ts')
        oppo_res_t = get_oppo(gt=df_s_prompt['gt'].tolist(), response=df_t_response['response'].tolist(), mode='st')

        res_df_s = df_s_prompt[['gt']]
        res_df_t = df_t_prompt[['gt']]

        res_s = label_response(df_gpt_correct=df_s, oppo=oppo_res_s)
        res_df_s.loc[:, 'label'] = res_s
        res_t = label_response(df_gpt_correct=df_t, oppo=oppo_res_t)
        res_df_t.loc[:, 'label'] = res_t

        for i in [1,2,0]:

            val1 = len(res_df_s[res_df_s['label']==i]) / len(res_df_s)*100
            val2 = len(res_df_t[res_df_t['label']==i]) / len(res_df_t)*100

            locals()[f's{i}'] += f'({(model_idx *0.2 + 0.1):.2f}, {val1:.2f}) ({(model_idx *0.2 + 0.1+0.05):.2f}, {val2:.2f}) '

        res_df_s = collect_response_details(res_df_s[['gt', 'label']], df_s_prompt)
        res_df_t = collect_response_details(res_df_t[['gt', 'label']], df_t_prompt)
        overview_dict_s['gt'] = df_s_prompt.loc[:109]['gt'].tolist()
        overview_dict_t['gt'] = df_t_prompt.loc[:109]['gt'].tolist()
        overview_dict_s[model] = res_df_s['label_counts'].tolist()
        overview_dict_t[model] = res_df_t['label_counts'].tolist()

    print(s1)
    print(s2)
    print(s0)

    pd.DataFrame(overview_dict_s).to_csv(f'supplementary_data/regional_term/response_details_simplified_{args.prompt_id}.csv', index=False)
    pd.DataFrame(overview_dict_t).to_csv(f'supplementary_data/regional_term/response_details_traditional_{args.prompt_id}.csv', index=False)

    """
    Note that the output is not directly a figure. Instead, the output is the source code for generating the figure in latex. 
    To generate the figure, please replace the placeholders in the following latex code with the printed s1, s2, and s0.

    Latex source code:

    \begin{figure}[t]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=\linewidth,
        ytick style={draw=none},
        height=5cm,
        ybar stacked,  
        xtick={0.1,0.15, 0.3,0.35, 0.5, 0.55, 0.7, 0.75, 0.9, 0.95, 1.1, 1.15,1.3, 1.35, 1.5, 1.55, 1.7, 1.75, 1.9, 1.95, 2.10, 2.15},
        xticklabels={\scriptsize{S}, \scriptsize{T},\scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T},\scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T},\scriptsize{S}, \scriptsize{T}, \scriptsize{S}, \scriptsize{T}},
        ymin=0,
        ymax=100,
        ylabel={\% Responses by Correctness},
        bar width=8.9pt,
        xmin=0.07,
        xmax=2.18,
        enlarge x limits=0.01, % Increase to space out the x coordinates more
        legend style={at={(0.5,1.35)}, anchor=north, legend columns=-1, draw=none, nodes={inner sep=10pt}}
      ]
      \addplot+[ybar,fill=babyblue, draw=black,postaction={pattern=north west lines}] coordinates {{s1_placeholder}};
      \addplot+[ybar, fill=yellow, draw=black] coordinates {{s2_placeholder}};
      \addplot+[ybar,fill=babyred, draw=black, postaction={pattern=north east lines}] coordinates {{s0_placeholder}};
    \draw [dashed, thick] (axis cs:0.625,0) -- (axis cs:0.625,100);
    \draw [dashed, thick] (axis cs:1.025,0) -- (axis cs:1.025,100);
      \legend{Correct response, Misaligned response, Incorrect response}
    \end{axis}
    \end{tikzpicture}
    \end{figure}
    """


