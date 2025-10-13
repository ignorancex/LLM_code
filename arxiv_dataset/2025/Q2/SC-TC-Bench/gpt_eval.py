import argparse
import pandas as pd
from utils import prompt_with_openai, process_df
import re
import ast

def remove_english_characters_and_spaces(text):
    return re.sub(r'[a-zA-Z\s]', '', text)

def annotate_term_task(response_list, question, gt, res):
    prompt = f"""
    The question is {question}.
    The correct answer is {gt}.

    The model's answer is {res}. If the model's answer is correct, respond with 1. Otherwise, respond with 0. 
    """
    
    full_prompt=[{"role":"user","content":prompt}]
    response_str = prompt_with_openai(full_prompt=full_prompt, chat_model='gpt-4o-mini')
    response_list.append(response_str)

def annotate_name_task(res, dict_name, response_list, refine_flag_list, region, candidate_list_str, combined_list, mc_list, tw_list):
    if type(res) == float:
        refine_flag_list.append(False)
        name = 'nan'
    else:
        found_elements = [elem for elem in combined_list if elem in res]

        if len(found_elements) == 1:
            refine_flag_list.append(False)
            name = found_elements[0]
        else:
            refine_flag_list.append(True)

            prompt = f"""
            Here is the name list: {candidate_list_str}.

            #######

            The response of the LLM is {res}.

            #######

            Which name does LLM choose? If LLM chose a name, then only give me the name. If LLM did not choose a name or choose multiple names from the prompt, then respond 'NA'.
            """
            full_prompt=[{"role":"user","content":prompt}]
            name = prompt_with_openai(full_prompt=full_prompt, chat_model='gpt-4o-mini')
            
    response_list.append(name)
    if name not in dict_name:
        dict_name[name] = 1
    else:
        dict_name[name] += 1

    if name in mc_list:
        region.append(1)
    elif name in tw_list:
        region.append(0)
    else:
        region.append(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--prompt_id', type=int, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--llm', type=str, default=None)
    args = parser.parse_args()

    df_prompt = process_df(file_type='prompt', task=args.task, lang=args.lang, prompt_id=args.prompt_id)
    df = process_df(file_type='response', sub_file_type='raw', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='load')

    if args.task == 'name':
        df_name = pd.read_csv(f'source_data/candidate_name_list/prompt_id_{args.prompt_id}.csv')
        refine_flag_list,region = [],[]
        dict_name = dict()

    response_list = []
    for idx in range(len(df)):
        res = df['response'][idx]
        if args.task == 'term':
            question = df_prompt['prompt'][idx]
            gt = df_prompt['gt'][idx]
            if args.llm in ["llama3-8b", "llama3-70b"]:
                res = remove_english_characters_and_spaces(res)
            annotate_term_task(response_list, question, gt, res)
        elif args.task == 'name':
            mc_list = ast.literal_eval(df_name['mainland_name'][idx])
            tw_list = ast.literal_eval(df_name['taiwan_name'][idx])
            candidate_list = df_name['candidate_name'][idx]
            candidate_list_str = ', '.join(candidate_list)
            combined_list = mc_list + tw_list
            annotate_name_task(res, dict_name, response_list, refine_flag_list, region, candidate_list_str, combined_list, mc_list, tw_list)


        if idx % 10 == 0:
            print(f"{idx} done")

            if args.task == 'term':
                df_response = pd.DataFrame({'correct':response_list})
                process_df(file_type='response', sub_file_type='correctness_annotated_by_gpt', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)
            elif args.task == 'name':
                df_response = pd.DataFrame({"extracted_name":response_list,
                                        "refine_flag":refine_flag_list,
                                        'region': region})
                process_df(file_type='response', sub_file_type='final', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)

    if args.task == 'term':
        df_response = pd.DataFrame({'correct':response_list})
        process_df(file_type='response', sub_file_type='correctness_annotated_by_gpt', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)
    elif args.task == 'name':
        df_response = pd.DataFrame({"extracted_name":response_list,
                                "refine_flag":refine_flag_list,
                                'region': region})
        process_df(file_type='response', sub_file_type='final', task=args.task, lang=args.lang, prompt_id=args.prompt_id, llm=args.llm, action='save', df_to_save=df_response)
    print(args)
