import argparse
import pandas as pd
from utils import process_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--lang', type=str, default='simplified')
    args = parser.parse_args()

    if args.model:
        df_response = process_df(file_type='response', sub_file_type='raw', task='name', lang=args.lang, prompt_id=0, llm=args.model, action='load')
        df_extraction = process_df(file_type='response', sub_file_type='final', task='name', lang=args.lang, prompt_id=0, llm=args.model, action='load')
        df = pd.concat([df_response, df_extraction[['region']]], axis=1)
        df = df[df['region']==-1]
        sampled_df = df.sample(n=100, random_state=2025).reset_index(drop=True)
        sampled_df[['response']].to_csv(f'supplementary_data/regional_name/sampled_invalid_response/{args.model}_{args.lang}.csv', index=False)
    
    else:
        models = ['chatglm2','breeze', 'gpt4o']
        types = ['insufficient','multiple','out-of-list']
        for invalid_type in types:
            s = f'{invalid_type} '
            for model in models:
                df = pd.read_csv(f'supplementary_data/regional_name/invalid_response_type_annotated_by_human/{model}_{args.lang}_labeled.csv')
                s += f"& {len(df[df['label']==invalid_type])/len(df)*100:.1f}\% "
            print(s)