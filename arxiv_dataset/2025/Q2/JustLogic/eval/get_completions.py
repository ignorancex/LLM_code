import os, requests, json
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
### FOR OPENAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
### FOR DEEPSEEK
# client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

def inference_openai(
        full_prompt, model, prompt_mode=None, stop=None, max_tokens=None
    ):

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    )
    result = completion.choices[0].message.content

    return completion, result

def inference_deepseek(
        full_prompt, model, prompt_mode=None, stop=None, max_tokens=None
    ):

    completion = client.chat.completions.create(
        model='deepseek-reasoner',
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        stream=False
    )
    result = completion.choices[0].message.content

    return completion, result

def inference_openrouter(
    full_prompt, model, temperature=0.6, max_tokens=None,
    quantizations=['fp16', 'bf16'], providers=None
):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer "+os.environ['OPENROUTER_API_KEY'],
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": model,
            "messages": [{
                "role": "user", "content": full_prompt
            }],
            'provider':{
                'order':providers, 'quantizations':quantizations, 'sort':'price'
            },
            'temperature': temperature,
            'max_completion_tokens':max_tokens,
            'max_tokens':max_tokens
        }),
    )
    if response.status_code >= 400:
        # print(dir(response))
        raise RuntimeError(f"Error: Received status code {response.status_code}. {response.text}")
    
    try:
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract model response: {e}\n\n{response.json()}")
        

def get_dataset(filename, samples_per_depth=None, max_depth=None):
    df = pd.read_csv(filename)
    if samples_per_depth:
        df = df.groupby("depth").head(samples_per_depth)
    if max_depth:
        df = df[df.depth <= max_depth]
    
    dataset = df.to_dict('records')
    return dataset

def get_prompt(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        prompt = fr.read()
    return prompt

def get_existing_results(filename):
    if 'csv' in filename:
        try:
            df = pd.read_csv(filename)
            dataset = df.to_dict('records')
            print(f'{len(dataset)} existing results found.')
        except:
            print('no existing results found.')
            dataset = []
    elif 'jsonl' in filename:
        try:
            dataset = []
            with open(filename, "r") as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            print(f'{len(dataset)} existing results found.')
        except:
            dataset = []
            print('no existing results found.')
    return dataset

### Convert csv to jsonl
# df = pd.read_csv("./eval/3_shot_cot_w_depth_deepseek-r1_results.csv")
# dataset = df.to_dict('records')
# with open("./eval/3_shot_cot_w_depth_deepseek-r1_results.jsonl", "a") as f:
#     for line in dataset:
#         f.write(json.dumps(line) + "\n")

def main():
    provider = 'openrouter' # openai or openrouter or deepseek
    model_name = 'meta-llama/llama-3.3-70b-instruct'
    tot = False
    results_file = f"./eval/3_shot_cot_w_depth_{model_name.split('/')[-1]}_results2.jsonl"
    dataset = get_dataset(
        './dataset/test_dataset.csv', samples_per_depth=50
    )
    prompt = get_prompt('./eval/prompt_3_shot_cot_w_depth.txt')
    prompt_tot_steps = get_prompt('./eval/prompt_tot_nextstep.txt')
    prompt_tot_eval = get_prompt('./eval/prompt_tot_eval.txt')
    results = get_existing_results(results_file)

    def generation(instance):
        full_prompt = prompt.format(
            PARAGRAPH = instance['paragraph'],
            STATEMENT = instance['question']
        )
        if provider == 'openai':
            _, completion = inference_openai(full_prompt, model_name)
        elif provider == 'openrouter':
            completion = inference_openrouter(
                full_prompt, model_name, quantizations=['bf16'],
                providers=['SambaNova']
            )
        elif provider == 'deepseek':
            _, completion = inference_deepseek(full_prompt, model_name)

        *front, last_word = completion.split()
        if 'TRUE' in last_word or 'True' in last_word:
            completion_ans = True
        elif 'FALSE' in last_word or 'False' in last_word:
            completion_ans = False
        else:
            completion_ans = 'Uncertain'
        result = instance
        result['predicted'] = completion_ans
        result['full_completion'] = completion

        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    
    def tot_generation(instance, steps=3):
        answer_so_far = ""
        steps_taken = steps
        completion_ans = 'Uncertain'
        for step in range(instance['depth']+2):
            full_prompt = prompt_tot_steps.format(
                PARAGRAPH = instance['paragraph'],
                STATEMENT = instance['question'],
                ANSWER_SO_FAR = ("None" if answer_so_far == "" else answer_so_far)
            )
            if provider == 'openai':
                _, completion = inference_openai(full_prompt, model_name)
            elif provider == 'openrouter':
                completion = inference_openrouter(
                    full_prompt, model_name, quantizations=['bf16', 'fp16'],
                    providers=['SambaNova']
                )
            elif provider == 'deepseek':
                _, completion = inference_deepseek(full_prompt, model_name)
            
            full_eval_prompt = prompt_tot_eval.format(
                PARAGRAPH = instance['paragraph'],
                STATEMENT = instance['question'],
                ANSWER_SO_FAR = answer_so_far,
                POSSIBLE_STEPS = completion
            )
            if provider == 'openai':
                _, nextstep_completion = inference_openai(full_eval_prompt, model_name)
            elif provider == 'openrouter':
                nextstep_completion = inference_openrouter(
                    full_eval_prompt, model_name, quantizations=['bf16', 'fp16'],
                    providers=['SambaNova']
                )
            elif provider == 'deepseek':
                _, nextstep_completion = inference_deepseek(full_eval_prompt, model_name)

            answer_so_far += nextstep_completion

            *front, last_word = answer_so_far.split()
            # print('last word:', last_word)
            if 'TRUE' in last_word or 'True' in last_word:
                completion_ans = True
                steps_taken = step+1
                break
            elif 'FALSE' in last_word or 'False' in last_word:
                completion_ans = False
                steps_taken = step+1
                break
            elif 'UNCERTAIN' in last_word or 'UNCERTAIN' in last_word:
                completion_ans = 'Uncertain'
                steps_taken = step+1
                break
        
        result = instance
        result['predicted'] = completion_ans
        result['steps_taken'] = steps_taken
        result['full_completion'] = answer_so_far
        # print(answer_so_far)
        # print('steps taken:', steps_taken)

        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        
    ### Sequential ###
    for i, instance in enumerate(tqdm(dataset)):
        if instance['id'] not in [r['id'] for r in results]:
            if tot:
                tot_generation(instance)
            else:
                generation(instance)

    ### Concurrent ###
    # with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    #     futures = []
    #     for i, instance in enumerate(dataset):
    #         if instance['id'] not in [r['id'] for r in results]:
    #             future = executor.submit(
    #                 generation, instance
    #             )
    #             futures.append(future)
        
    #     for future in tqdm(
    #         concurrent.futures.as_completed(futures), total=len(futures)
    #     ):
    #         future.result()

    # for i, instance in enumerate(tqdm(dataset[:21])):
    #     if instance['id'] in [r['id'] for r in results]:
    #         continue

    #     full_prompt = prompt.format(
    #         PARAGRAPH = instance['paragraph'],
    #         STATEMENT = instance['question']
    #     )
    #     if provider == 'openai':
    #         _, completion = inference_openai(full_prompt, model_name)
    #     elif provider == 'openrouter':
    #         completion = inference_openrouter(
    #             full_prompt, model_name, quantizations=['fp8'],
    #             providers=['DeepSeek', '']
    #         )
    #     elif provider == 'deepseek':
    #         _, completion = inference_deepseek(full_prompt, model_name)

    #     *front, last_word = completion.split()
    #     if 'TRUE' in last_word or 'True' in last_word:
    #         completion_ans = True
    #     elif 'FALSE' in last_word or 'False' in last_word:
    #         completion_ans = False
    #     else:
    #         completion_ans = 'Uncertain'
    #     result = instance
    #     result['predicted'] = completion_ans
    #     result['full_completion'] = completion
    #     results.append(result)

    #     if i % 10 == 0:
    #         results_df = pd.DataFrame.from_records(results)
    #         results_df.to_csv(results_file, index=False)

    #     print('{i}: {ans}'.format(i=i, ans=completion_ans))

    # results_df = pd.DataFrame.from_records(results)
    # results_df.to_csv(results_file, index=False)

    # return results

results = main()