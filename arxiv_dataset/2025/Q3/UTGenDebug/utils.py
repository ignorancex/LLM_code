import os
import re
import time
import openai
import argparse
import traceback
import fire
import random
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset, load_dataset
from termcolor import colored
import sys
import ast
import timeout_decorator
import numpy as np
import json
import subprocess
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
from multiprocessing import Array, Value, Process, Queue 
from collections import defaultdict, OrderedDict
import multiprocessing  
import time  
from typing import List, Tuple, Any

import signal  
import resource  
from contextlib import contextmanager

HF_DATASET = {
    'mbpp_plus_fix': 'datasets/MBPP+Fix_test.jsonl',
    'mbpp_plus_fix_hard': 'datasets/MBPP+Fix_hard_test.jsonl',
    'humaneval_plus_fix': 'datasets/HE+Fix_test.jsonl',
    }


STOP_WORDS = {
    "llama3": ["<|eot_id|>", "<|end_of_text|>"],
    "llama3.1": ["<|eot_id|>", "<|end_of_text|>"],
    "tulu3": ["<|user|>", "<|assistant|>", "<|end_of_text|>"],
    "llama3_70b": ["<|eot_id|>", "<|end_of_text|>"],
    "gemma": ["<end_of_turn>"],
    "eurus": [],
    "qwen": ["<|endoftext|>", "<|im_end|>"],
    "big_qwen": ["<|endoftext|>", "<|im_end|>"],
    "deepseek": ["<｜end▁of▁sentence｜>", "### Response", "### Instruction:"]
}

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

HF_MODEL_NAMES = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-7b-it",
    "deepseek": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "tulu3": "allenai/Llama-3.1-Tulu-3-8B-SFT",
}


import time
from copy import deepcopy

import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile
from typing import Optional

class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def trusted_exec(code, inputs, entry_point, record_time=False, output_not_none=False):
    """Execute trusted code in place."""
    exec_globals = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]
    # print(fn)
    rtime = []
    ret = []
    for inp in inputs:
        inp = deepcopy(inp)
        # print(inp)
        if record_time:
            start = time.time()
            # print(type(fn))
            ret.append(fn(*inp))
            rtime.append(time.time() - start)
        else:
            ret.append(fn(*inp))

    if output_not_none:
        ret = [i is not None for i in ret]

    if record_time:
        return ret, rtime
    else:
        return ret


def trusted_check_exec(code, inputs, entry_point):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=1.0):
            trusted_exec(code, inputs, entry_point)
    except Exception:
        return False
    return True



def extract_arguments(response, func_name, return_str=False):
    if 'rgument' in response: 
        response = response.split('rgument', 1)[-1]
        arguments_str = response.split(f'{func_name}(', 1)[-1].split('\n', 1)[0].rsplit(')', 1)[0]
    else: arguments_str = response.rsplit(f'{func_name}(', 1)[-1].split('\n', 1)[0].rsplit(')', 1)[0]
    try:  
        # Evaluate the arguments string to get the actual arguments  
        # WARNING: Using eval() can be a security risk if the input is not trusted.  
        # In a production environment, it's better to parse the string manually  
        # or use a safe evaluation method.  
        arguments = eval(arguments_str)  
        if isinstance(arguments, tuple):
            if return_str: return list(arguments), arguments_str
            return list(arguments)
        if return_str: return [arguments], arguments_str
        return [arguments]
    except Exception as e:  
        if return_str: return None, None
        return None    


def he_extract_description(example):
    description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
    return description



def extract_output(text):
    if 'utput' in text:
        text = text.rsplit('utput', 1)[-1]
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, text, re.DOTALL)
    if len(sol) > 0:
        return sol[-1].strip('\n')
    pattern = r'`(.*?)`'
    sol = re.findall(pattern, text, re.DOTALL)
    if len(sol) > 0:
        return sol[-1].strip('\n').lstrip(' ')
    return None


input_completion_prompt = '''
Given a Python function `{signature}` to solve the following task:
{description}

Write a valid input based on this task description, i.e., an acceptable input consistent with task description that a correct program should be able to execute.
Provide a reasoning for your answer and present your response in the format below:
```
<reasoning>

Arguments: {entry_point}(<all arguments>)
```
Note that you MUST directly write ALL input arguments of the function in the correct order. Skip writing any names of arguments.
'''

joint_completion_prompt = '''
You are given a Python function `{signature}` to solve the following task:

## Task
{description}

Based on the task description above, write a unit test that 
1. Is **valid** input based on the task description, i.e., an acceptable input consistent with task description that a correct program should be able to execute.
2. The output enclosed in ```.``` and is **faithful** to the task description, i.e., the output of the unit test is consistent with what a correct program would yield.

Note:
- that you MUST directly write ALL input arguments of the function in the correct order. Skip writing any names of arguments.
- you MUST enclose the unit test inputs and outputs in ```.```

Respond in the format below:

## Unit Test

### Input Arguments
< step-by-step reasoning for constructing a unit test that is valid as per the task description>
Arguments: ```{entry_point}(<all arguments>)```

### Output
< step-by-step reasoning for what a **correct** {entry_point} would execute to based on the task description and your input above. Make sure your data type of the final answer matches the expected output type of the function. >
Output: ```<your final answer>```
'''

separate_unit_output_prompt = '''
Question: 

Assume there exists a python function `{signature}` to solve the task: {description}

A user calls this functions with the input: {entry_point}({unit_input}).
Based on the task objective of the function and the user's input, what is the output of the function if implemented **correctly**?

Make sure your data type of the final answer matches the expected output type of the function.
First explain your reasoning and in the end format your final answer as:

Output: ```<your final answer>```
'''

def make_signature(code, test_case):   
    # Parse the code to get an Abstract Syntax Tree (AST)  
    parsed_code = ast.parse(code)  
      
    # A dictionary to hold function names and their signatures  
    function_signatures = {}  
      
    # Walk through the AST to find all function definitions  
    for node in ast.walk(parsed_code):  
        if isinstance(node, ast.FunctionDef):  
            # Construct the function signature  
            function_name = node.name  
            args = []  
            for arg in node.args.args:  
                # Check if the argument has a type hint  
                if arg.annotation:  
                    # The annotation can be a Name (e.g., int) or more complex (e.g., List[int]).  
                    # We use ast.get_source_segment to extract the original source code for the annotation.  
                    arg_type = ': ' + ast.get_source_segment(code, arg.annotation)  
                else:  
                    arg_type = ''  
                args.append(f"{arg.arg}{arg_type}")  
            signature = f"{function_name}({', '.join(args)})"  
            function_signatures[function_name] = signature  
      
    # Use regex to extract the function name from the test case
    for sign in function_signatures:
        if sign in test_case:
            break
    return  function_signatures.get(sign)


import numpy as np
from math import inf

def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def eval_exact_match(out, exp, atol=0, use_set = False):
    if isinstance(out, str) and isinstance(exp, str):
        if 'error' in out.lower() and 'error' in exp.lower: return True
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if use_set:
        try:
            out = set(out)
            exp = set(exp)
        except: pass
    if isinstance(out, str) and isinstance(exp, str) and 'Error' in out and "Error" in exp: return True
    elif out != exp and atol != 0:
        return np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        return out == exp


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
        "from functools import *"
    ]}



@contextmanager  
def memory_limit(max_mem):  
    # Save the original memory limits  
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)  
    # Set the new memory limit  
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))  
    try:  
        yield  
    finally:  
        # Restore the original memory limits  
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))  

def unsafe_execute(
        dataset: str,
        entry_point: str,
        code: str,
        inputs: List,
        expected: List,
        verbose: bool = False,
        use_set: bool = False,
):
        
    exec_globals = {}
    pass_rate = []
    exec_out = []
    try:
        with time_limit(2.0), memory_limit(256 * 1024 * 1024):
            with swallow_io():
                exec(code, exec_globals)
                if len(entry_point):
                    fn = exec_globals[entry_point]
    except Exception as e:
        if not len(inputs) or not len(entry_point):
            if verbose: return _mapping[_TIMEOUT], 'Error: ' +  str(e)
            return _mapping[_TIMEOUT]
        # time out there is some error in the code itself
        if verbose: return [_mapping[_TIMEOUT]]*len(inputs), ['Error: ' +  str(e)]*len(inputs)
        return [_mapping[_TIMEOUT]]*len(inputs)
    if not len(inputs) or not len(entry_point):
        if verbose: return _mapping[_SUCCESS], _mapping[_SUCCESS]
        return _mapping[_SUCCESS]

    for i, inp in enumerate(inputs):
        try:
            with time_limit(2.0), memory_limit(256 * 1024 * 1024):
                with swallow_io():
                    out = fn(*inp)
            exp = expected[i]
            exact_match = eval_exact_match(out, exp, use_set=use_set)
            exec_out.append(out)
        except Exception as e:
            pass_rate.append(_mapping[_TIMEOUT])
            exec_out.append('Error: ' +  str(e))
            continue
        if not isinstance(exact_match, bool):
            try:
                exact_match = exact_match.all()
            except:
                pass_rate.append(_mapping[_FAILED])
                continue
        if exact_match:
            pass_rate.append(_mapping[_SUCCESS])
        else:
            pass_rate.append(_mapping[_FAILED])
    if verbose: 
        assert len(exec_out) == len(pass_rate), print('Look: ', len(pass_rate), len(exec_out))  
        return pass_rate, exec_out
    return pass_rate 

he_code_generation_prompt = '''
Write Python code to solve the task.
Write a Python function `{signature}` to solve the following problem: Present code in ```python```
```python
{prompt}
```
'''

code_generation_prompt = '''
Write Python code to solve the task.
Write a Python function `{signature}` to solve the following problem: Present code in ```python```
```python
"""
{prompt}
"""
```
'''
def extract_code(text):
    pattern = r'```python(.*?)```'
    sol = re.findall(pattern, text, re.DOTALL)
    if len(sol) > 0:
        return sol[0]
    
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, text, re.DOTALL)
    if len(sol) > 0:
        return sol[0]
    
    return text.split('```')[0]


joint_adv_prompt = '''
You are given a Python function `{signature}` to solve the following task:

## Task
{description}

## Code Solution:
```
{code}
```
The code solution I have provided to you is **incorrect**. Your job is to give feedback by generating a unit test that 
1. Is **valid** input based on the task description, i.e., an acceptable input consistent with task description that a correct program should be able to execute.
2. The output enclosed in ```.``` and is **faithful** to the task description, i.e., the output of the unit test is consistent with what a correct program would return.
3. **Breaks** the given code, i.e., does **not** execute to the **correct** output and brings out its mistakes and vulnerabilities.

Provide a reasoning for your answer and identify a general hypothesis or rationale identifying the cause of error. Then provide input and output of the unit test consistent with the pattern (hypotheis) you have identified.
Note:
- that you MUST directly write ALL input arguments of the function in the correct order. Skip writing any names of arguments.
- you MUST enclose the unit test inputs and outputs in ```.```

Respond in the format below:

## Hypothesis
< step-by-step reasoning >

Error Pattern: <an identified pattern of inputs that yields erroneous or incorrect outputs>

## Unit Test

### Input Arguments
< step-by-step reasoning for constructing a unit test that fits the error pattern identified above and is valid as per the task description>
Arguments: ```{entry_point}(<all arguments>)```

### Output
< step-by-step reasoning for what a **correct** {entry_point} would execute to based on the task description and your input above. Make sure your data type of the final answer matches the expected output type of the function. >
Output: ```<your final answer>```
'''

adv_input_completion_prompt = '''
Given a Python function `{signature}` to solve the following task:
{description}

Code Solution:
```
{code}
```

The code solution I have provided to you is incorrect. Your job is to give feedback by generating a unit test input that 
1. Valid input based on the task description, i.e., an acceptable input consistent with task description that a correct program should be able to execute.
2. Given code will NOT be able to solve and brings out its mistakes and vulnerabilities.

Provide a reasoning for your answer and present your response in the format below:
```
<reasoning>

Arguments: {entry_point}(<all arguments>)
```
Note that you MUST directly write ALL input arguments of the function in the correct order. Skip writing any names of arguments.
'''

def input_acceptance(code, test_input, func_name):
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    snippet = test_setup + '\n'  + code 
    return trusted_check_exec(snippet, test_input, func_name)

def check_attack_success(inst, code, unit_input):
    task_setup = "\n".join(IMPORT_HELPER["python"])
    decision = input_acceptance(task_setup + '\n' + inst['canonical_solution'], [unit_input], inst['entry_point'])
    if decision:
        unit_out = trusted_exec(task_setup + '\n' + inst["canonical_solution"], [unit_input], inst['entry_point'])
        [attack_succ], [func_out] = unsafe_execute('mbpp',  inst['entry_point'], task_setup + "\n" + code, [unit_input], unit_out, use_set=' set(' in inst['test'],  verbose=True )
        return attack_succ != "pass", unit_out[0], func_out
    else:
        return False, None, None
    

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(15)) # retry=retry_if_not_exception_type(InvalidRequestError)
def deepseek_call(prompt):
    client = OpenAI(api_key="DEEP SEEK KEY", base_url="https://api.deepseek.com")
    if args.use_temp:
        temp = 0.8
        top_p = 0.9
    else:
        temp = 0
        top_p = 1

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": prompt}
        ]  
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temp,
        max_tokens=2000,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout=40,
        stream=False
    )
    return response.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(25)) # retry=retry_if_not_exception_type(InvalidRequestError)
def llm_call(model, prompt):
    if args.use_temp:
        temp = 0.8
        top_p = 0.9
    else:
        temp = 0
        top_p = 1

    openai.api_key = "YOUR OPENAI KEY"
    ## If Using OpenAI via Azure
    openai.api_base = "" 
    openai.api_type = "azure"
    openai.api_version = ""
    deployment_name = model

    messages = [
        {"role": "user", "content": prompt}
        ]  
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        temperature=temp,
        max_tokens=2000,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout=40,
    )
    choices = response["choices"]
    completion_objs = [choice.message for choice in choices]
    completions = [completion.content for completion in completion_objs]
    return completions[0]