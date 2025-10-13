import os
import re
import io
import sys
import ast
import json
import time
import random
import typing
import argparse
import requests
import numpy as np
from openai import OpenAI
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

import evaluation_config




os.environ["TOKENIZERS_PARALLELISM"] = "false" 







#============== vllm inference ===============

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        **kwargs
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=40,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task
        outputs = llm.generate(prompts, sampling_params)
        result_texts = [out.outputs[0].text for out in outputs]
        result_queue.put((task_id, result_texts))


def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Submit tasks
def submit_prompt_set(task_queues, prompt_sets):
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

# Collect results
def collect_results(result_queues, num_sets):
    results = [None] * num_sets
    for q in result_queues:
        task_id, result = q.get()
        results[task_id] = result
    return results

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# get token length
def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference
def generate_results(all_prompts, gpu_groups,task_queues, result_queues):
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    submit_prompt_set(task_queues, prompt_sets)
    results = collect_results(result_queues, len(prompt_sets))
    result_list = []
    for result_set in results:
        for r in result_set:
            result_list.append(r)
    return result_list















#============== API inference ===============

def fetch_completion(user_prompt: str) -> str:

    headers  = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": api_model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": api_temperature
    }
    r = requests.post(base_url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



def generate_results_api(prompts):
    total   = len(prompts)
    results = ["No outputs"] * total

    for batch_start in range(0, total, rpm_limit):
        batch_end   = min(batch_start + rpm_limit, total)
        batch_slice = range(batch_start, batch_end)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_idx = {
                pool.submit(fetch_completion, prompts[i]): i
                for i in batch_slice
            }

            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results[idx] = fut.result()

        elapsed   = time.time() - t0
        leftover  = max(0, 60.0 - elapsed)
        if batch_end < total and leftover:
            time.sleep(leftover)

        print(f"Processed {batch_end}/{total} prompts")

    return results


def save_prompts_to_jsonl(prompts,
                         filename,
                         system_content,
                         model,
                         max_tokens,
                         url):
    with open(filename, "w", encoding="utf-8") as fout:
        for i, user_prompt in enumerate(prompts, start=1):
            obj = {
                "custom_id": f"request-{i}",
                "method":    "POST",
                "url":       url,
                "body": {
                    "model":      model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user",   "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": api_temperature
                }
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} requests to {filename!r}")

def extract_completions(raw):
    lines = raw.strip().split("\n")
    records = [json.loads(line) for line in lines]
    bodies = [rec["response"]["body"] for rec in records]
    assistant_texts = [
        body["choices"][0]["message"]["content"]
        for body in bodies
    ]
    return assistant_texts

def generate_by_openai_batch(prompts):

    save_prompts_to_jsonl(
        prompts,
        filename=api_batch_filename,
        system_content="You are a helpful assistant.",
        model=api_model_name,
        max_tokens=max_tokens,
        url="/v1/chat/completions"
    )

    client = OpenAI(
        api_key=api_key,
    )

    batch_input_file = client.files.create(
        file=open(api_batch_filename, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    cprint(f"file id: {batch_input_file_id}", color = "green")
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    batch_id = batch.id
    cprint(f"batch id: {batch_id}", color = "green")
    cprint("You can check https://platform.openai.com/docs/guides/batch?lang=python to learn how to monitor and cancel this batch job with batch id and file id.", color = "green")

    import time
    start_time = time.time()
    last_index = 0
    min_interval = 2
    while True:
        time.sleep(5)
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            file_id = batch.output_file_id
            break
        if batch.status == "failed" or batch.status == "expired" or batch.status == "cancelled":
            cprint(batch.status, color = "green")
            return None
        elapsed = time.time() - start_time
        idx = int(elapsed // (60 * min_interval))
        if idx > last_index:
            last_index = idx
            num_completed = batch.request_counts.completed
            total_num = batch.request_counts.total
            failed_num = batch.request_counts.failed
            print(f"{idx * min_interval} minutes passed, {num_completed}/{total_num} completed, {failed_num} failed, {batch.status}")

    cprint(f"takes {time.time() - start_time}s to complete!", color = "green")
    file_response = client.files.content(file_id)
    return extract_completions(file_response.text)

# extract the code from response
def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


# read the configuration
def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=evaluation_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=evaluation_config.dataset)
    parser.add_argument("--use_api", type=str2bool, default=evaluation_config.use_api)
    parser.add_argument("--k_code", type=int, default=evaluation_config.k_code)
    parser.add_argument("--k_case", type=int, default=evaluation_config.k_case)
    parser.add_argument("--max_model_len", type=int, default=evaluation_config.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=evaluation_config.max_generation_token)
    parser.add_argument("--temp", type=float, default=evaluation_config.temp)
    parser.add_argument("--num_chunks", type=int, default=evaluation_config.num_chunks)
    parser.add_argument("--no_example", type=str2bool, default=evaluation_config.no_example)
    parser.add_argument("--max_test", type=int, default=evaluation_config.max_test)
    parser.add_argument("--trust_remote_code", type=str2bool, default=evaluation_config.trust_remote_code)
    parser.add_argument("--exe_verbose", type=str2bool, default=evaluation_config.exe_verbose)
    parser.add_argument("--is_final_eval", type=str2bool, default=evaluation_config.is_final_eval)
    parser.add_argument("--single_eval", type=str2bool, default=evaluation_config.single_eval)
    parser.add_argument("--scale_tuple_list", type=ast.literal_eval, default=evaluation_config.scale_tuple_list)
    parser.add_argument("--api_model_name", type=str, default=evaluation_config.api_model_name)
    parser.add_argument("--api_key", type=str, default=evaluation_config.api_key)
    parser.add_argument("--base_url", type=str, default=evaluation_config.base_url)
    parser.add_argument("--api_temperature", type=float, default=evaluation_config.api_temperature)
    parser.add_argument("--max_workers", type=int, default=evaluation_config.max_workers)
    parser.add_argument("--use_openai_batch_api", type=str2bool, default=evaluation_config.use_openai_batch_api)
    parser.add_argument("--max_tokens", type=int, default=evaluation_config.max_tokens)
    parser.add_argument("--rpm_limit", type=int, default=evaluation_config.rpm_limit)
    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=evaluation_config.gpu_groups)
    parser.add_argument("--system_prompts", type=str, default=evaluation_config.system_prompts)
    parser.add_argument("--system_case_prompts", type=str, default=evaluation_config.system_case_prompts)
    parser.add_argument("--special_requirements", type=str, default=evaluation_config.special_requirements)
    return parser.parse_args()


# convert read configuration to global variable
args = parse_args()
globals().update(vars(args))


# restriction
if single_eval:
    scale_tuple_list = []
    k_case = 0


# read dataset
with open("../data/" + dataset + ".json", 'r') as f:
    data = json.load(f)
#data = [data[i] for i in range(10)]
num = len(data)


# load model, tokenizer, build vllm engines...
if use_api == False:
    task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token)
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
else:
    outputs_name = "eval-" + api_model_name.replace("/", ".") + "-" + dataset
    if use_openai_batch_api:
        api_batch_filename = api_model_name.replace("/", ".") + "-" + dataset + ".jsonl"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")




# obtain prompt
def get_scaling_prompt(data_i, method):
    problem = data_i["question"]
    if method == "sample":
        return Template(system_prompts).render(language = "python", special_requirements = special_requirements, problem = problem)
    if method == "case":
        n_example = min(k_case, len(data_i["example_input"]))
        example_input = ", ".join([repr(item) for item in data_i['example_input']])
        example_output = ", ".join([repr(item) for item in data_i['example_output']])
        if n_example == 0:
            example_intro = """ """
        if n_example == 1:
            example_intro = """We already have one test sample:\n Its input is {{example_input}}. Its output is {{example_output}}.\n"""
            example_intro = Template(example_intro).render(example_input = example_input, example_output = example_output)
        if n_example > 1:
            example_intro = """We already have {{n_sample}} test samples:\n The inputs are, respectively, {{example_input}}. The corresponding outputs are {{example_output}}.\n"""
            example_intro = Template(example_intro).render(n_sample = n_example, example_input = example_input, example_output = example_output)
        return Template(system_case_prompts).render(problem = problem, example_intro = example_intro)

# formalize the input and output
def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c

# extract the unit tests from responses
def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = []
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = []
    
    # Also extract from the last occurrence of **Test Input:** to the end
    index = full_output.rfind("**Test Input:**")
    if index != -1:
        example_text = [full_output[index:]]
    else:
        example_text = []
    
    # If any essential piece is missing, return empties
    if example_text == [] or test_input == [] or test_output == []:
        return [], [], []
    
    return test_input, test_output, example_text






# initialization
code_generation_prompts = []
code_index = []
case_generation_prompts = []
case_index = []
for i in range(num):
    # preprocess
    data[i]["full_code_generation"] = []
    data[i]["full_case_generation"] = []
    data[i]["generated_code"] = []
    data[i]["case_input"] = []
    data[i]["case_output"] = []
    data[i]["case_text"] = []

    data_i = data[i].copy()
    # get code generation prompts
    prompt_i = get_scaling_prompt(data_i, "sample")
    data[i]["code_generation_prompt"] = prompt_i
    code_generation_prompts = code_generation_prompts + [prompt_i] * k_code
    code_index = code_index + [i] * k_code
    # get case generation prompts
    if no_example:
        data_i["example_input"] = []
        data_i["example_output"] = [] # no example provided
    prompt_i = get_scaling_prompt(data_i, "case")
    data[i]["case_generation_prompt"] = prompt_i
    case_generation_prompts = case_generation_prompts + [prompt_i] * k_case
    case_index = case_index + [i] * k_case








# sampling process

cprint("start generation...", "green")

# shuffle first, to achieve efficiency
all_prompts = code_generation_prompts + case_generation_prompts
N = len(all_prompts)
indices = list(range(N))
shuffled_idx = indices[:]      
random.shuffle(shuffled_idx)
shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
# generate
if use_api == False:
    shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues)
else:
    if use_openai_batch_api:
        shuffled_outputs = generate_by_openai_batch(shuffled_prompts)
    else:
        shuffled_outputs = generate_results_api(shuffled_prompts)
restored_outputs = [None] * N
for out, idx in zip(shuffled_outputs, shuffled_idx):
    restored_outputs[idx] = out
code_generation_result = restored_outputs[:len(code_generation_prompts)]
case_generation_result = restored_outputs[len(code_generation_prompts):]

cprint("generation job done!", "green")










# calculate response length

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

code_response_length = get_token_lengths(code_generation_result, tokenizer)
case_response_length = get_token_lengths(case_generation_result, tokenizer)
mean_code = sum(code_response_length)/len(code_response_length)
if len(case_response_length) == 0:
    mean_case = 0
else:
    mean_case = sum(case_response_length)/len(case_response_length)





# process generated codes
i = 0
for full_output in code_generation_result:
    code_output = extract_code(full_output)
    index_i = code_index[i]
    data[index_i]["full_code_generation"] = data[index_i]["full_code_generation"] + [full_output]
    data[index_i]["generated_code"] = data[index_i]["generated_code"] + [code_output]
    i += 1

# process generated unit tests
i = 0
for full_output in case_generation_result:
    test_input, test_output, example_text = extract_test_cases(full_output)
    index_i = case_index[i]
    data[index_i]["full_case_generation"] = data[index_i]["full_case_generation"] + [full_output]
    data[index_i]["case_input"] = data[index_i]["case_input"] + test_input
    data[index_i]["case_output"] = data[index_i]["case_output"] + test_output
    data[index_i]["case_text"] = data[index_i]["case_text"] + example_text
    i += 1

# output the data
os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


if use_api == False:
    stop_workers(task_queues, processes)















# execute the scripts


def worker(script, input_val, output_queue):
    
    input_lines = iter(input_val.splitlines())
    
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)

    context = {
        "__name__": "__main__",
        "input": fake_input,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Optional": typing.Optional,
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x, y):
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    chunk_size = n // num_chunks   
    remainder = n % num_chunks 
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        indices.append((start, end))
        start = end
    return indices

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks):

    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    i = 0
    for start, end in chunks:
        if exe_verbose:
            print(f"process {i}/{len(chunks)}")
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(sub_code_list, sub_test_input_list, sub_time_limit_list, worker)
        exe_results = exe_results + sub_exe_results
        i += 1
    return exe_results

def execute_scripts(outputs_name, num_chunks):

    with open("./temp_data/outputs-" + outputs_name + '.json', 'r') as f:
        data = json.load(f)
    
    # process cases

    # get input lists
    case_index_list = []
    case_position_list = []
    case_code_list = []
    case_input_list = []
    case_output_list = []
    case_time_limit_list = []
    test_index_list = []
    test_position_list = []
    test_code_list = []
    test_input_list = []
    test_output_list = []
    test_time_limit_list = []
    for i in range(len(data)):
        # case
        if len(data[i]["case_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["case_exe_results"] = None
            data[i]["case_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["case_input"])
            data[i]["case_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["case_bool_table"] = np.full((n_row, n_col), False, dtype=bool)
        
        # test
        if len(data[i]["test_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["test_exe_results"] = None
            data[i]["test_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = min(len(data[i]["test_input"]), max_test)
            data[i]["test_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["test_bool_table"] = np.full((n_row, n_col), False, dtype=bool)

        data_i = data[i].copy()

        for j in range(len(data_i["generated_code"])):
            for k in range(len(data_i["case_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["case_input"][k]
                case_output = data_i["case_output"][k]
                case_code_list.append(code)
                case_input_list.append(case_input)
                case_output_list.append(case_output)
                if "test_time_limit" in data_i.keys():
                    case_time_limit_list.append(data_i["test_time_limit"])
                else:
                    case_time_limit_list.append(1)
                    print("No time limit provided!")
                case_index_list.append(i)
                case_position_list.append((j, k))
        
        max_k = min(len(data_i["test_input"]), max_test)
        for j in range(len(data_i["generated_code"])):
            for k in range(max_k):
                code = data_i["generated_code"][j]
                test_input = data_i["test_input"][k]
                test_output = data_i["test_output"][k]
                test_code_list.append(code)
                test_input_list.append(test_input)
                test_output_list.append(test_output)
                if "test_time_limit" in data_i.keys():
                    test_time_limit_list.append(data_i["test_time_limit"])
                else:
                    test_time_limit_list.append(1)
                    print("No time limit provided!")
                test_index_list.append(i)
                test_position_list.append((j, k))
    
    # execute
    if single_eval == False:
        cprint("start execution for generated unit tests", "green")
        case_exe_results = run_scripts_with_chunk(case_code_list, case_input_list, case_time_limit_list, worker, num_chunks)
        cprint("execution job done!", "green")
    else:
        case_exe_results = []
    
    cprint("start execution for ground-truth unit tests", "green")
    test_exe_results = run_scripts_with_chunk(test_code_list, test_input_list, test_time_limit_list, worker, num_chunks)
    cprint("execution job done!", "green")

    for i in range(len(case_index_list)):
        index_i = case_index_list[i]
        j, k = case_position_list[i]
        data[index_i]["case_exe_results"][j][k] = case_exe_results[i]
        data[index_i]["case_bool_table"][j][k] = test_if_eq(case_exe_results[i], case_output_list[i])
    
    for i in range(len(test_index_list)):
        index_i = test_index_list[i]
        j, k = test_position_list[i]
        data[index_i]["test_exe_results"][j][k] = test_exe_results[i]
        data[index_i]["test_bool_table"][j][k] = test_if_eq(test_exe_results[i], test_output_list[i])
    
    stats_single = {
        "BoN_score": 0,
        "BoN_num": 0,
        "BoN_acc_score": 0,
        "BoN_acc_num": 0
    }
    stats = []
    for i in range(len(scale_tuple_list)):
        stats_i = stats_single.copy()
        stats_i["tuple"] = scale_tuple_list[i]
        stats.append(stats_i)
    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0
    case_score = 0
    case_num = 0
    case_acc_score = 0
    case_acc_num = 0
    p_01_score = 0
    p_01_num = 0
    p_00_score = 0
    p_00_num = 0
    for i in range(len(data)):
        if single_eval:
            all_test_table_i = data[i]["test_bool_table"].copy()
            correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
            code_score += len(correct_code_list)
            code_num += all_test_table_i.shape[0]
            code_acc_score += np.sum(all_test_table_i).item()
            code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]

        
        if data[i]["case_exe_results"] is None or data[i]["test_exe_results"] is None:
            continue
        all_case_table_i = data[i]["case_bool_table"].copy()
        all_test_table_i = data[i]["test_bool_table"].copy()
        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]
        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]
        sub_case_table_i = all_case_table_i[correct_code_list, :].copy()
        correct_case_list = np.where(sub_case_table_i.all(axis=0))[0].tolist()
        if len(correct_code_list) > 0:
            case_score += len(correct_case_list)
            case_num += sub_case_table_i.shape[1]
            case_acc_score += np.sum(sub_case_table_i).item()
            case_acc_num += sub_case_table_i.shape[0] * sub_case_table_i.shape[1]
            # get ps
            wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
            wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
            if len(wrong_code_list) > 0:
                if len(correct_case_list) > 0:
                    wrong_code_correct_case_table_i = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                    p_01_score += np.sum(~wrong_code_correct_case_table_i).item()
                    p_01_num += wrong_code_correct_case_table_i.shape[0] * wrong_code_correct_case_table_i.shape[1]
                if len(wrong_case_list) > 0:
                    wrong_code_wrong_case_table_i = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                    p_00_score += np.sum(wrong_code_wrong_case_table_i).item()
                    p_00_num += wrong_code_wrong_case_table_i.shape[0] * wrong_code_wrong_case_table_i.shape[1]
                
        index_id = 0
        for scale_num_code, scale_num_case in scale_tuple_list:
            case_table_i = all_case_table_i[:scale_num_code, :scale_num_case].copy()
            test_table_i = all_test_table_i[:scale_num_code, :].copy()
            best_code_index = np.sum(case_table_i, 1).argmax()
            sub_test_table_i = test_table_i[best_code_index, :].copy()
            stats[index_id]["BoN_score"] = stats[index_id]["BoN_score"] + int(all(sub_test_table_i))
            stats[index_id]["BoN_num"] = stats[index_id]["BoN_num"] + 1
            stats[index_id]["BoN_acc_score"] = stats[index_id]["BoN_acc_score"] + np.sum(sub_test_table_i).item()
            stats[index_id]["BoN_acc_num"] = stats[index_id]["BoN_acc_num"] + len(sub_test_table_i)
            assert int(all(sub_test_table_i)) / 1 <= np.sum(sub_test_table_i).item() / len(sub_test_table_i), "error"
            index_id += 1
    
    if is_final_eval:
        if use_api == False:
            outputs_result_name = "./results/results-eval-" + pretrained_model.replace("/", ".") + "-final_eval.txt"
        else:
            outputs_result_name = "./results/results-eval-" + api_model_name.replace("/", ".") + "-final_eval.txt"
    else:
        outputs_result_name = "./results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        # Your values
        def safe_divide(d1, d2):
            if d2 == 0:
                return 0
            return d1/d2
        code_acc = safe_divide(code_score, code_num)
        code_acc_acc = safe_divide(code_acc_score, code_acc_num)
        case_acc = safe_divide(case_score, case_num)
        case_acc_acc = safe_divide(case_acc_score, case_acc_num)
        p_01 = safe_divide(p_01_score, p_01_num)
        p_00 = safe_divide(p_00_score, p_00_num)

        save_and_print(f"code acc (average proportion of tasks the generated code can pass): {code_acc}\ncode accumulate acc (average proportion of unit tests the generated code can pass): {code_acc_acc}")

        if single_eval == False:
            save_and_print(f"estimated unit test acc (average proportion of tasks that the generated unit test can pass all correct code): {case_acc}\nestimated unit test accumulate acc (average proportion of correct code that the generated unit test can pass): {case_acc_acc}")


            save_and_print(f"estimated p_01: {1 - p_01}")
            save_and_print(f"estimated p_00: {p_00}")

            for i in range(len(stats)):
                tuple_name = stats[i]["tuple"]
                save_and_print(f"BoN setting {tuple_name}:")
                acc = stats[i]["BoN_score"] / stats[i]["BoN_num"]
                acc_acc = stats[i]["BoN_acc_score"] / stats[i]["BoN_acc_num"]
                save_and_print(f"acc: {acc}, accumulate acc: {acc_acc}")
        
            save_and_print(f"code average response length: {mean_code}, unit test average response length: {mean_case}")
        else:
            save_and_print(f"code average response length: {mean_code}")

    # convert np to list
    for i in range(len(data)):
        if data[i]["case_exe_results"] is None:
            continue
        data[i]["case_bool_table"] = data[i]["case_bool_table"].tolist()
    
    for i in range(len(data)):
        if data[i]["test_exe_results"] is None:
            continue
        data[i]["test_bool_table"] = data[i]["test_bool_table"].tolist()

    # output the data
    with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




execute_scripts(outputs_name, num_chunks)









