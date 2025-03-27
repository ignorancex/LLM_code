import re
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import multiprocessing
from tqdm import tqdm
import textgrad
from textgrad.engine import get_engine
from textgrad import Variable, TextualGradientDescent_v2, TextualGradientDescent, TextualGradientDescentwithMomentum
from textgrad.tasks import load_instance_task
from prompts import CodeTestTimewithTests, SYSTEM_PROMPT_FOR_FIRST_CODE, CODE_INSTANCE_ROLE_DESCRIPTION
from evaluators.lt_eval import LeetCodeEvaluator
from evaluators.py_eval import PythonEvaluator
import torch
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def config():
    parser = argparse.ArgumentParser(description="Code optimization.")
    parser.add_argument("--engine", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="The API to use.")
    parser.add_argument("--optimizer_version", type=str, default="v1", help="The optimizer to use.")
    parser.add_argument("--size", type=int, default=39, help="The number of total questions")
    return parser.parse_args()

args = config()

internal_evaluator = LeetCodeEvaluator()

optimizer_version = args.optimizer_version

def extract_model_size(engine_name):
    match = re.search(r'(\d+B)', engine_name)
    if match:
        return match.group(1)
    else:
        return "Unknown"
            
def optimization_one_iteration(optimizer, instance_var, prompt, tests):
    pt = PythonEvaluator()
    tests = tests.split("\n")

    state, test_string = pt.evaluate(instance_var.value, tests)

    optimizer.zero_grad()
    loss_fn = CodeTestTimewithTests(engine=ENGINE_API)

    test_time_loss = loss_fn(prompt, instance_var, test_string)

    test_time_loss.backward()
    optimizer.step()

    return state, test_string

def generate_starting_solution(prompt):
    print(prompt)
    llm_first_code = ENGINE_API.generate(prompt, system_prompt=SYSTEM_PROMPT_FOR_FIRST_CODE)
    if "```python" in llm_first_code and "```" in llm_first_code.split("```python")[1]:
        llm_first_code = llm_first_code.split("```python")[1].split("```")[0]

    print(llm_first_code)
    return llm_first_code

def evaluation_and_optimization_pipeline(args):
    task_id, prompt, index, tests, MAX_ITERS = args
    print("Start of optimization pipeline", index)

    generated_programs = []
    gpt_4_first_code = generate_starting_solution(prompt)
    n_iter = 0
    generated_programs.append({"code": gpt_4_first_code,
                               "gradients": None,
                               "task_id": task_id,
                               "state": None,
                               "test_string": None,
                               "iteration": n_iter,
                               })

    instance_var = Variable(gpt_4_first_code, requires_grad=True,
                            role_description=CODE_INSTANCE_ROLE_DESCRIPTION)

    if optimizer_version == "v1":
        # print(instance_var)
        optimizer = TextualGradientDescent(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"])
    elif optimizer_version == "v1_momentum":
        optimizer = TextualGradientDescentwithMomentum(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"],
                                       momentum_window=5)
    elif optimizer_version == "v2":
        optimizer = TextualGradientDescent_v2(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"])
    else:
        raise ValueError(f"Invalid optimizer version: {optimizer_version}")

    pt = PythonEvaluator()
    state, test_string = pt.evaluate(gpt_4_first_code, tests.split("\n"))

    generated_programs[-1]["state"] = state
    generated_programs[-1]["test_string"] = test_string

    for iter in range(1, MAX_ITERS + 1):
        state, test_string = optimization_one_iteration(optimizer, instance_var, prompt, tests)
        generated_programs[-1]["state"] = state
        generated_programs[-1]["test_string"] = test_string
        if all(state):
            break

        n_iter += 1
        generated_programs.append({"code": instance_var.value,
                                   "gradients": str(instance_var.gradients),
                                   "task_id": task_id,
                                   "state": None,
                                   "test_string": None,
                                   "iteration": n_iter,
                                   })

    print("End of optimization pipeline", index)
    return generated_programs

def evaluate_or_except(program):
    try:
        res = internal_evaluator.check_if_in_cache_or_submit(program["task_id"], program["code"])
        return res[0], res[1], res[2], res[3]
    except Exception as e:
        print(e)
        return False, -1, -1, -1

if __name__ == "__main__":

    SEEDS = [55]

    collection = {
        "task_id": [],
        "problem": [],
        "seed": [],
        "success": [],
        "test_cases_passed": [],
        "test_cases_total": [],
        "runtime": [],
        "local_tests": [],
        "local_test_state": [],
        "iteration_idx": []
    }

    for seed in SEEDS:
        instance_task = "LeetCodeHardEval"

        # TEST_ENGINE = "llama-3_1"
        TEST_ENGINE = args.engine
        ENGINE_API = get_engine(engine_name=TEST_ENGINE)

        MAX_PROGRAMS_TO_OPTIMIZE = 39
        MULTIPROC_POOLS = min([20, MAX_PROGRAMS_TO_OPTIMIZE, multiprocessing.cpu_count()])
        MULTIPROC_POOLS = 2
        MAX_ITERS = 4

        textgrad.set_backward_engine(ENGINE_API, override=True)

        dataset = load_instance_task(instance_task, ENGINE_API)

        code_dataset_examples = [(task_id, prompt, index, tests, MAX_ITERS)
                                 for index, (task_id, prompt, tests)
                                 in enumerate(dataset)][:MAX_PROGRAMS_TO_OPTIMIZE]
        programs_and_gradients = []
        for example in code_dataset_examples:
            result = evaluation_and_optimization_pipeline(example)
            programs_and_gradients.append(result)
            print(result)
        # with multiprocessing.Pool(MULTIPROC_POOLS) as pool:
        #     programs_and_gradients = pool.map(evaluation_and_optimization_pipeline, code_dataset_examples)

        all_programs = []
        zero_programs = []

        for list_of_programs in tqdm(programs_and_gradients):
            list_of_programs = [list_of_programs[0], list_of_programs[-1]]
            for iter, program in enumerate(list_of_programs):
                res = evaluate_or_except(program)
                program["evaluation"] = res[0]
                program["total_correct"] = res[1]
                program["total_tests"] = res[2]
                program["runtime"] = res[3]
                program["iteration_idx"] = iter
                all_programs.append(program)
                if program['iteration_idx'] == 0:
                    zero_programs.append(program)

        print("CounterZero", len(zero_programs))
        print("CounterAll", len([program for program in all_programs if program["iteration_idx"] == 0]))
        print(all_programs)
        for program in all_programs:
            collection["task_id"].append(program["task_id"])
            collection["problem"].append(program["code"])
            collection["seed"].append(seed)
            collection["iteration_idx"].append(program['iteration_idx'])
            collection["success"].append(program["evaluation"])
            collection["test_cases_passed"].append(program["total_correct"])
            collection["test_cases_total"].append(program["total_tests"])
            collection["runtime"].append(program["runtime"])
            collection["local_tests"].append(program["test_string"])
            collection["local_test_state"].append(program["state"])


        program_df = pd.DataFrame(collection)

        program_df.to_csv(f"../results_haibo/leetcode_supervised_{extract_model_size(TEST_ENGINE)}_{optimizer_version}.csv", index=False)
        print("Saved~!")

