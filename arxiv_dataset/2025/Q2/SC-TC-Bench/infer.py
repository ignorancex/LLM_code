import argparse
import pandas as pd
from utils import set_random_seed, LLMEngine, process_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--llm', type=str, default=None)
    parser.add_argument('--prompt_id', type=int, default=None)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()
    set_random_seed(seed=args.seed)

    llm_engine = LLMEngine(llm=args.llm)

    df_prompt = process_df(file_type='prompt', task=args.task, lang=args.lang, prompt_id=args.prompt_id)

    if args.resume:
        df_response = process_df(file_type='response', sub_file_type='raw', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='load')
        response_list = df_response['response'].tolist()
    else:
        response_list = []
    
    for idx in range(len(response_list), len(df_prompt)):
        prompt = df_prompt['prompt'][idx]
        response_str = llm_engine.generate_response(prompt)
        response_list.append(response_str)

        if idx % 10 == 0:
            print(f"{idx} done")
            print(response_str)
            df_response = pd.DataFrame({'response':response_list})
            process_df(file_type='response', sub_file_type='raw', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)
            
    df_response = pd.DataFrame({'response':response_list})
    process_df(file_type='response', sub_file_type='raw', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)
    print(args, 'done')
