import tqdm
import json
import requests
import sys
import multiprocessing
import io
import re
import ray
import ast
import numpy as np
import threading
from datasets import load_dataset, Dataset

# 初始化 Ray
ray.init(ignore_reinit_error=True)

# 超时终止器
def timeout_handler(process, result_container):
    process.terminate()  # 强制终止进程
    result_container["error"] = "Timeout"

@ray.remote
def execute_code_with_timeout(test_code, input_data, timeout=3):
    """
    使用 Ray 实现的远程函数，用于在限定时间内执行代码。
    """
    manager = multiprocessing.Manager()  # 使用 Python 标准库中的 multiprocessing.Manager()
    result_container = manager.dict()

    def target_func():
        try:
            sys.stdin = io.StringIO(input_data)
            sys.stdout = io.StringIO()
            env_test = {}
            exec(test_code, env_test)
            result_container["output"] = sys.stdout.getvalue().strip()
            result_container["error"] = None
        except Exception as e:
            result_container["error"] = str(e)

    process = multiprocessing.Process(target=target_func)  # 创建进程执行代码
    process.start()

    timer = threading.Timer(timeout, timeout_handler, args=(process, result_container))  # 设置超时定时器
    timer.start()

    process.join(timeout)  # 等待进程完成
    timer.cancel()  # 取消定时器（如果进程已完成）

    return result_container.get("output", None), result_container.get("error", None)

@ray.remote
def worker(case, test_code, timeout):
    """
    Worker 函数，用于执行单个测试用例。
    """
    input_data = case['input'].strip()
    expected_output = case['output'].strip()
    actual_output, error = ray.get(execute_code_with_timeout.remote(test_code, input_data, timeout))
    
    if error:
        return {
            'input': input_data,
            'expected_output': expected_output,
            'actual_output': None,
            'error': error,
            'status': 'failed'
        }
    if actual_output == expected_output:
        return {
            'input': input_data,
            'expected_output': expected_output,
            'actual_output': actual_output,
            'error': None,
            'status': 'passed'
        }
    else:
        return {
            'input': input_data,
            'expected_output': expected_output,
            'actual_output': actual_output,
            'error': None,
            'status': 'failed'
        }

def test_cases_with_limit(output_cases, test_right_code, timeout=3):
    """
    使用 Ray 实现的并行化测试用例函数。
    """
    futures = [worker.remote(case, test_right_code, timeout) for case in output_cases]
    results = ray.get(futures)
    
    passed_case = [res for res in results if res['status'] == 'passed']
    unpassed_case = [res for res in results if res['status'] == 'failed']
    
    return passed_case, unpassed_case


if __name__ == "__main__":
    can_use_case = []
    can_use_case_all_pass = []
    pass_rate = []
    ds = load_dataset("/data/FastSSD/LLM_Models/TACO")
    Pbar = tqdm.tqdm(ds['train'])    
    for item in Pbar:
        if eval(item['solutions']) == []:
            continue
        try:
            express_code = json.loads(item['solutions'])[0]
            example_case = json.loads(item['input_output'])
        except:
            continue

        all_example_case = [{'input': str(x), 'output': str(y)} for x, y in zip(example_case['inputs'], example_case['outputs'])]
        passed_case, unpassed_case = test_cases_with_limit(all_example_case, express_code, timeout=10)

        # 计算通过率
        if (len(passed_case) + len(unpassed_case)) == 0:
            current_pass_rate = 0
        else:
            current_pass_rate = len(passed_case) / (len(passed_case) + len(unpassed_case))
        pass_rate.append(current_pass_rate)
        if current_pass_rate == 1.0:
            can_use_case_all_pass.append(item)
        if current_pass_rate > 0.0:
            can_use_case.append(item)
        Pbar.set_description(f"Avg Pass: {np.mean(pass_rate)}, All Pass: {np.mean(np.array(pass_rate) == 1.)}, Current: {current_pass_rate}")

json.dump(can_use_case, open("/home/xukaiyuan/Project/TreeSearch_Code/wash_code/select_data_train_can_use.json", 'w'))
json.dump(can_use_case_all_pass, open("/home/xukaiyuan/Project/TreeSearch_Code/wash_code/select_data_train_all_pass.json", 'w'))
