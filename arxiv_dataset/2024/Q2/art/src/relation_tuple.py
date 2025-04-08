import json
import os
import random
import time
from datetime import datetime
import argparse
from typing import Union
from collections import OrderedDict, Counter
import regex
from tqdm import tqdm
from openai import OpenAI
from src.prompts import eightshot_prompt_0603 as eightshot_prompt
from src.tool import extract_relation_triples, extract_code_from_str, jsonlines_load
from src.interpreter.jupyter_backend import JupyterKernel


CACHE_DIR = ".cache"

backend = JupyterKernel(CACHE_DIR)


def execute_program(code):
    if "input(" in code or "while " in code:
        # input function is not allowed
        return None, None
    msg = backend.execute_code(code)
    backend.restart_jupyter_kernel()
    # backend.shutdown()
    
    return msg


def extract_num_turbo(solution: str):
    try:
        ans = solution.strip().split('\n')[-1].replace('The final answer:', '')
        prd = [x[0] for x in regex.finditer(
            r'[-\d\.,]+', ans) if regex.search(r'\d', x[0])]
        if len(prd) > 2:
            prd = prd[-1]
        elif len(prd):
            prd = prd[0]
        else:
            prd = None
    
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
    except:
        prd = None
    return prd


def get_user_assistant_messages(system_message: str, user_message: str, assistant_message: str):
    """Convert the given prompt into a list of messages for the OpenAI API."""
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split("\n\n\n\n")
    split_assistant_messages = assistant_message.split("\n\n\n\n")
    for i in range(len(split_user_messages)):
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages


def get_cot_prompt(data: dict, backbone: str):
    """Get the COT prompt for the given data and backbone."""
    if backbone == "gpt-3.5-turbo-0301":
        system_message = eightshot_prompt.GPT_3_5_TURBO_COT_SYSTEM
        user_message = eightshot_prompt.GPT_3_5_TURBO_COT_USER
        assistant_message = eightshot_prompt.GPT_3_5_TURBO_COT_ASSISTANT
    
    messages = get_user_assistant_messages(system_message, user_message, assistant_message)
    question_message = data["question"]
    messages += [{"role": "user", "content": f"Question: {question_message}"}]
    
    return messages


def get_relation_tuple_prompt(data: dict, backbone: str, need_feedback: bool = False, previous_response: Union[str, None] = None):
    """Get the COT relation triple prompt for the given data and backbone."""
    if backbone == "gpt-3.5-turbo-0301":
        system_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_SYSTEM
        user_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_USER
        assistant_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_ASSISTANT
        
    messages = get_user_assistant_messages(system_message, user_message, assistant_message)
    question_message = data["question"]
    if not need_feedback:
        messages += [{"role": "user", "content": f"Question: {question_message}"}]
    else:
        messages += [{"role": "user", "content": f"Question: {question_message}\nYour previous solution is: '{previous_response}'. Please rethink the question based on the previous solution."}]
    return messages


def get_relation_triple_program_verification_prompt(data: dict, backbone: str, need_feedback: bool = False, previous_response: Union[str, None] = None):
    relation_triples = data["relation_triples"]
    relation_triples = "\n".join(relation_triples)
    
    if backbone == "gpt-3.5-turbo-0301":
        system_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_PROGRAM_VERIFICATION_SYSTEM
        user_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_PROGRAM_VERIFICATION_USER
        assistant_message = eightshot_prompt.GPT_3_5_TURBO_RELATION_TRIPLE_PROGRAM_VERIFICATION_ASSISTANT
        
    messages = get_user_assistant_messages(system_message, user_message, assistant_message)
    question_message = data["question"]
    # if not need_feedback:
        # messages += [{"role": "user", "content": f"Question: {question_message}\nThinking process in relation triple format:{relation_triples}"}]
    # else:
        # messages += [{"role": "user", "content": f"Question: {question_message}\nThinking process in relation triple format:{relation_triples}\n{math_prompt.GPT_3_5_TURBO_FEEDBACK_PROMPT}\nPrevious reponse: {previous_response}"}]
        
        # messages += [{"role": "user", "content": f"Question: {question_message}\nThinking process in relation triple format:{relation_triples}\n\nYour previous solution is: {previous_response}. Please rethink the question based on the previous solution."}] # new feedback prompt to be tested
        
    messages += [{"role": "user", "content": f"Question: {question_message}\nThinking process in relation triple format:{relation_triples}"}]
    return messages


def query_cot(
    data: dict,
    client: OpenAI,
    cot_temperature: float,
    backbone: str,
    ):
    """CoT."""
    
    query_message = get_cot_prompt(data=data, backbone=backbone)
    if backbone == "gpt-3.5-turbo-0301":
        model_name = "gpt-3.5-turbo-0301"
        
    start_time = time.time()
    completions = []
    while True:
        try:
            cot_solution = client.chat.completions.create(
                model=model_name,
                # max_tokens=500,
                # stop='\n\n\n',
                messages=query_message,
                temperature=cot_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            cot_solution = None

        if cot_solution is not None:
            completions.extend([choice.message.content
                                for choice in cot_solution.choices])
            completions = completions[:1]
            return completions, query_message
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None, query_message


def query_relation_tuple(data: dict, client: OpenAI, relation_triple_temperature: float, backbone: str, need_feedback: bool = False, previous_response: Union[str, None] = None, max_tokens=4096):
    """Relation Triple."""
    
    query_message = get_relation_tuple_prompt(data=data, backbone=backbone, need_feedback=need_feedback, previous_response=previous_response)
    if backbone == "gpt-3.5-turbo-0301":
        model_name = "gpt-3.5-turbo-0301"
        
    start_time = time.time()
    completions = []
    while True:
        try:
            relation_tuple_solution = client.chat.completions.create(
                model=model_name,
                # max_tokens=max_tokens,
                # stop='\n\n\n',
                messages=query_message,
                temperature=relation_triple_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            relation_tuple_solution = None

        if relation_tuple_solution is not None:
            completions.extend([choice.message.content 
                                for choice in relation_tuple_solution.choices])
            completions = completions[:1]
            return completions, query_message
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None, query_message


def query_program_verification(data: dict, client: OpenAI, program_verification_temperature: float, backbone: str, need_feedback: bool = False, previous_response: Union[str, None] = None, max_tokens=4096):
    """Program Verification."""
    
    query_message = get_relation_triple_program_verification_prompt(data=data, backbone=backbone, need_feedback=need_feedback, previous_response=previous_response)

    if backbone == "gpt-3.5-turbo-0301":
        model_name = "gpt-3.5-turbo-0301"
    
    start_time = time.time()
    completions = []
    while True:
        try:
            program_verification_solution = client.chat.completions.create(
                model=model_name,
                # max_tokens=max_tokens,
                # stop='\n\n\n',
                messages=query_message,
                temperature=program_verification_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            program_verification_solution = None

        if program_verification_solution is not None:
            completions.extend([choice.message.content
                                for choice in program_verification_solution.choices])
            completions = completions[:1]
            return completions, query_message
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None, query_message


def query_math_relation_tuple(
    data: dict,
    client: OpenAI,
    relation_tuple_temperature: float,
    backbone: str,
    sc_num: int = 1,
    dataset_name=""):
    
    final_answers = []
    for i in range(sc_num):
        final_answer = None
        relation_tuple_solution, relation_triple_messages = query_relation_tuple(data=data, client=client, relation_triple_temperature=relation_tuple_temperature, backbone=backbone)
        if relation_tuple_solution is None:
            print("Time out")
            return None
        else:
            relation_tuple_solution = relation_tuple_solution[0]
            final_answer = extract_num_turbo(relation_tuple_solution)
            final_answers.append(final_answer)
    
    count = Counter(final_answers)
    majority_ans = count.most_common(1)[0][0]
    
    to_dump_data = OrderedDict(
        {
            "index": data["index"],
            "question": data["question"],
            "answer": data["answer"],
            "majority_answer": majority_ans,
            "final_answers": final_answers,
            "solution": {"solution": relation_tuple_solution, "messages": relation_triple_messages}
        })
    return to_dump_data


def query_math_cot(
    data: dict,
    client: OpenAI,
    cot_temperature: float,
    backbone: str,
    sc_num: int = 1,
    dataset_name=""):
    
    final_answers = []
    for i in range(sc_num):
        final_answer = None
        cot_solution, cot_messages = query_cot(data=data, client=client, cot_temperature=cot_temperature, backbone=backbone)
        if cot_solution is None:
            print("Time out")
            return None
        else:
            cot_solution = cot_solution[0]
            final_answer = extract_num_turbo(cot_solution)
            final_answers.append(final_answer)
    
    
    count = Counter(final_answers)
    majority_ans = count.most_common(1)[0][0]
    
    to_dump_data = OrderedDict(
        {
            "index": data["index"],
            "question": data["question"],
            "answer": data["answer"],
            "majority_answer": majority_ans,
            "final_answers": final_answers,
            "solution": {"solution": cot_solution, "messages": cot_messages}
        }
        )
    return to_dump_data


def query_relation_tuple_with_verification(
    data: dict,
    relation_tuple_temperature: float,
    program_verification_temperature: float,
    client: OpenAI,
    backbone: str,
    sc_num: int,
    max_num_tries: int = 3,
    max_tokens=4096,
    dataset_name="",
    ):
    """Query the OpenAI API for the math problem."""
    relation_tuple_solutions = []
    program_verification_solutions = []
    
    relation_triple_answers = []
    program_verification_answers = []
    
    final_answers = []
    
    for i in range(sc_num):
        cur_num_tries = 0
        previous_response = None
        need_feedback = False
        
        relation_triple_ans = None
        program_verification_ans = None
        final_answer = None
        
        relation_tuple_solution, relation_triple_messages = query_relation_tuple(
            data=data,
            client=client,
            relation_triple_temperature=relation_tuple_temperature,
            backbone=backbone,
            need_feedback=False,
            previous_response=None,
            max_tokens=max_tokens)
        
        if relation_tuple_solution is None:
            print("No relation tuple solution found.")
            return None
        else:
            relation_tuple_solution = relation_tuple_solution[0]
            print("Relation triple solution: ", relation_tuple_solution)
            relation_triple_ans = extract_num_turbo(relation_tuple_solution)
            relation_triple_answers.append(relation_triple_ans)
            relation_tuple_solutions.append({
                "solution": relation_tuple_solution, "first_time": True, "messages": relation_triple_messages})
        
        relation_triples = extract_relation_triples(relation_tuple_solution)
        
        if relation_triples is None or len(relation_triples) == 0:
            print("No relation tuples found.")
            program_verification_solution = None
            program_verification_ans = None
        else:
            data["relation_triples"] = relation_triples
            program_verification_solution, program_verification_messages = query_program_verification(
                data=data,
                client=client,
                program_verification_temperature=program_verification_temperature,
                backbone=backbone,
                need_feedback=False,
                previous_response=None,
                max_tokens=max_tokens)
        
        if program_verification_solution is None:
            print("No program verification solution")
            program_verification_ans = None
        else:
            program_verification_solution = program_verification_solution[0]
            generated_program = extract_code_from_str(program_verification_solution)
            if generated_program is None:
                program_verification_ans = None
            else:
                try:
                    exec_result, _ = execute_program(generated_program)
                    print("Executed result: ", exec_result)
                    program_verification_ans = extract_num_turbo(exec_result)
                except Exception as e:
                    print(e)
                    program_verification_ans = None
            
            program_verification_answers.append(program_verification_ans)
            program_verification_solutions.append({
                "solution": program_verification_solution,
                "first_time": True, 
                "messages": program_verification_messages})
        
        if relation_triple_ans is not None and program_verification_ans is not None:
            if abs(relation_triple_ans - program_verification_ans) < 1e-3:
                final_answers.append(relation_triple_ans)
                need_feedback = False
            else:
                # feedback to the model
                need_feedback = True
                final_answers.append(relation_triple_ans)
                final_answers.append(program_verification_ans)
            
        elif relation_triple_ans is not None and program_verification_ans is None:
            final_answers.append(relation_triple_ans)
            need_feedback = True
        elif relation_triple_ans is None and program_verification_ans is not None:
            final_answers.append(program_verification_ans)
            need_feedback = True
        else:
            # final_answers.append(None)
            need_feedback = True
        
        relation_triple_previous_response = None
        program_verification_previous_response = None
        while need_feedback and cur_num_tries < max_num_tries:
            # Previous response is not consistent or incorrect. Need feedback.
            cur_num_tries += 1
            
            relation_triple_previous_response = relation_tuple_solution if relation_triple_previous_response is None else relation_triple_previous_response
            program_verification_previous_response = program_verification_solution if program_verification_previous_response is None else program_verification_previous_response
            
            new_relation_triple_ans = None
            new_program_verification_ans = None
            new_final_answer = None
            
            new_relation_tuple_solution, new_relation_triple_messages = query_relation_tuple(
                data=data,
                client=client,
                relation_triple_temperature=relation_tuple_temperature,
                backbone=backbone,
                need_feedback=True,
                previous_response=relation_triple_previous_response,
                max_tokens=max_tokens)
            
            if new_relation_tuple_solution is None:
                print("No new relation tuple solution")
                # return None 
                new_relation_triple_ans = None
            else:
                new_relation_tuple_solution = new_relation_tuple_solution[0]
                new_relation_triple_ans = extract_num_turbo(new_relation_tuple_solution)
                relation_triple_answers.append(new_relation_triple_ans)
                relation_tuple_solutions.append({
                    "solution": new_relation_tuple_solution, 
                    "first_time": False, 
                    "messages": new_relation_triple_messages})
                relation_triple_previous_response = new_relation_tuple_solution
                
            new_relation_triples = extract_relation_triples(new_relation_tuple_solution)
            if len(new_relation_triples) == 0:
                print("No new relation triples found.")
                new_program_verification_solution = None
                new_program_verification_ans = None            
            else:
                data["relation_triples"] = new_relation_triples
                new_program_verification_solution, new_program_verification_messages = query_program_verification(
                    data=data,
                    client=client,
                    program_verification_temperature=program_verification_temperature,
                    backbone=backbone,
                    need_feedback=True,
                    previous_response=program_verification_previous_response,
                    max_tokens=max_tokens)
            
            if new_program_verification_solution is None:
                print("No new program verification solution")
                new_program_verification_ans = None 
                # return None
            else:
                new_program_verification_solution = new_program_verification_solution[0]
                new_generated_program = extract_code_from_str(new_program_verification_solution)
                if new_generated_program is None:
                    new_program_verification_ans = None
                else:
                    try:
                        new_exec_result, _ = execute_program(new_generated_program)
                        new_program_verification_ans = extract_num_turbo(new_exec_result)
                    except Exception as e:
                        print(e)
                        new_program_verification_ans = None
                program_verification_answers.append(new_program_verification_ans)
                program_verification_solutions.append(
                    {
                    "solution": new_program_verification_solution,
                    "first_time": False,
                    "messages": new_program_verification_messages
                    })
                program_verification_previous_response = new_program_verification_solution
                
            if new_relation_triple_ans is not None and new_program_verification_ans is not None:
                if abs(new_relation_triple_ans - new_program_verification_ans) < 1e-3:
                    # new_final_answer = new_relation_triple_ans
                    final_answers.append(new_relation_triple_ans)
                    need_feedback = False
                else:
                    # feedback to the model
                    need_feedback = True
                    final_answers.append(new_relation_triple_ans)
                    final_answers.append(new_program_verification_ans)
                    
            elif new_relation_triple_ans is not None and new_program_verification_ans is None:
                need_feedback = True
                final_answers.append(new_relation_triple_ans)
            elif new_relation_triple_ans is None and new_program_verification_ans is not None:
                need_feedback = True
                final_answers.append(new_program_verification_ans)
            else:
                # final_answers.append(None)
                need_feedback = True
        
        # the weight of sc in feedback loop should be higher.
        
        if need_feedback:
            pass
    
    if len(final_answers) == 0:
        majority_ans = None
    else:
        counter = Counter(final_answers)
        majority_ans = counter.most_common(1)[0][0]
    
    to_dump_data = OrderedDict(
        {
            "index": data["index"],
            "question": data["question"],
            "answer": data["answer"],
            "majority_answer": majority_ans,
            "final_answers": final_answers,
            "relation_triple_answers": relation_triple_answers,
            "program_verification_answers": program_verification_answers,
            "relation_tuple_solutions": relation_tuple_solutions,
            "program_verification_solutions": program_verification_solutions,
        }
    )
    
    return to_dump_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, choices=[
                        'gsm8k', 'svamp', 'asdiv', 'singleeq', 'singleop',
                        'singleaddsub', 'multiarith'], default='gsm8k')
    parser.add_argument("--backbone", type=str, choices=["gpt-3.5-turbo-0301"], default="gpt-3.5-turbo-0301")
    parser.add_argument("--relation_triple_temperature", type=float, default=0.)
    parser.add_argument("--program_verification_temperature", type=float, default=0.)
    parser.add_argument("--sc_num", type=int, default=1,
                        help="Self-consistency samples. 1 indicates greedy decoding")
    parser.add_argument("--cot_temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument(
        '--api_key', type=str, default='sk-', required=True)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--method", type=str, choices=["cot", "relation_tuple", "relation_tuple_with_verification"])
    args = parser.parse_args()
    
    start_index = args.start
    end_index = args.end
    
    dataset_name = args.dataset
    if args.method == "relation_tuple_with_verification":
        relation_tuple_temperature = args.relation_triple_temperature
        program_verification_temperature = args.program_verification_temperature
    elif args.method == "cot":
        cot_temperature = args.cot_temperature
    backbone = args.backbone
    sc_num = args.sc_num
    output_dir = args.output_dir
    api_key = args.api_key
    
    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    
    if dataset_name == "gsm8k":
        dataset = jsonlines_load("dataset/gsm8K_test.jsonl")
    elif dataset_name == "svamp":
        dataset = jsonlines_load("dataset/svamp.jsonl")
    elif dataset_name == "asdiv":
        dataset = jsonlines_load("dataset/asdiv.jsonl")
    elif dataset_name == "singleeq":
        dataset = jsonlines_load("dataset/single_eq.jsonl")
    elif dataset_name == "singleop":
        dataset = jsonlines_load("dataset/single_op.jsonl")
    elif dataset_name == "singleaddsub":
        dataset = jsonlines_load("dataset/single_addsub.jsonl")
    elif dataset_name == "multiarith":
        dataset = jsonlines_load("dataset/multiarith.jsonl")


    # slice data based on start and end
    total_num = len(dataset)
    print("total data: ", total_num)
    if end_index == -1:
        end_index = total_num
        
    if end_index > total_num:
        end_index = total_num

    tasks = dataset[start_index:end_index]
    task_num = len(tasks)
    print('Current total tasks: ', task_num)

    unfinished_tasks = []
    output_path = os.path.join(output_dir, f'{backbone}/')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if args.method == "relation_tuple_with_verification":
        save_path = os.path.join(
            output_path,
            f'{dataset_name}_sc{sc_num}_s{start_index}_e{end_index}_relationtuple_temp_{relation_tuple_temperature}_program_verification_temp{program_verification_temperature}_{dt_string}.jsonl')
    elif args.method == "cot":
        save_path = os.path.join(
            output_path,
            f'{dataset_name}_sc{sc_num}_s{start_index}_e{end_index}_cot_temp_{cot_temperature}_{dt_string}.jsonl')
    
    client = OpenAI(api_key=api_key)
    
    # run experiment
    progress_bar = tqdm(range(task_num))
    answer_list = []
    failure_answer_list = []
    for i in range(task_num):
        task = tasks[i]
        wait_time = min(sc_num * 100, 360)
        start_time = time.time()
        while True:
            try:
                if args.method == "relation_tuple_with_verification":
                    ans = query_relation_tuple_with_verification(
                        data=task,
                        relation_tuple_temperature=relation_tuple_temperature,
                        program_verification_temperature=program_verification_temperature,client=client,
                        backbone=backbone,
                        sc_num=sc_num, 
                        max_num_tries=3,
                        max_tokens=args.max_tokens,
                        dataset_name=dataset_name)
                elif args.method == "cot":
                    ans = query_math_cot(
                        data=task,
                        client=client,
                        cot_temperature=cot_temperature,
                        backbone=backbone,
                        sc_num=sc_num,
                        dataset_name=dataset_name)
            except Exception as e:
                print(e)
                ans = None

            if ans is not None:
                answer_list.append(ans)
                with open(save_path, "w") as fout:
                    json.dump(answer_list, fout, ensure_ascii=False, indent=4)
                    progress_bar.update(1)
                break
            
                
            if time.time() - start_time > wait_time:
                print('Time out')
                print('Current Task: ', i)
                answer_list.append(ans)
                with open(save_path, "w") as fout:
                    json.dump(answer_list, fout, ensure_ascii=False, indent=4)
                    progress_bar.update(1)
                unfinished_tasks.append(task)
                break
            
        sleep_time = random.uniform(3, 5)
        time.sleep(sleep_time)
        
    end_time_0 = time.time()
    print('Finish at time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f'Time used: {end_time_0 - start_time_0} seconds')
    if len(unfinished_tasks) > 0:
        print('Unfinished tasks: ')
        for task in unfinished_tasks:
            print(task)

    print('Done')
