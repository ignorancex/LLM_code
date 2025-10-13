import numpy as np
from soar.grid_formatting import grid_formatting
import copy
from soar.llm_utils import get_dic_only_correct
from soar.preprocess import get_dataset

#TODO: clean unused prompts and functions
# =================  prompt preprocessing  =================

def format_task(example: dict,max_example: int =-1,grid_display_mode="ascii", include_test=True,show_output_test=False,randomize_pair_order=False) -> str:
    """
    take a task and format the input output to be used in the prompt
    """
    if max_example==-1:
        max_example = len(example['train'])
    else:
        assert max_example>0, "max_example should be positive"
        max_example = min(max_example,len(example['train']))
    prompt = ""
    if randomize_pair_order:
        list_idx = np.random.permutation(max_example)
    else:
        list_idx = range(max_example)
    for i in list_idx:
        input_i=example['train'][i]["input"]
        x_shape,y_shape = np.array(input_i).shape
        prompt += f"## Input {i+1} (grid shape: {x_shape} by {y_shape}):\n"+grid_formatting(input_i,mode=grid_display_mode)+"\n\n"
        outptut_i=example['train'][i]["output"]
        x_shape,y_shape = np.array(outptut_i).shape
        prompt += f"## Output {i+1} (grid shape: {x_shape} by {y_shape}):\n"+grid_formatting(outptut_i,mode=grid_display_mode)+"\n\n"
    
    if include_test: #   /!\ show all test input ? it is ok for now but need to think about it
        max_example = min(max_example,len(example['test']))
        for i in range(max_example):
            input_i=example['test'][i]["input"]
            x_shape,y_shape = np.array(input_i).shape
            prompt += f"## Test Input {i+1} (grid shape: {x_shape} by {y_shape}):\n"+grid_formatting(input_i,mode=grid_display_mode)+"\n\n"
            if show_output_test:
                outptut_i=example['test'][i]["output"]
                x_shape,y_shape = np.array(outptut_i).shape
                prompt += f"## Test Output {i+1} (grid shape: {x_shape} by {y_shape}):\n"+grid_formatting(outptut_i,mode=grid_display_mode)+"\n\n"
    return prompt

def format_fewshot_tasks(examples: list[dict],max_example: int =-1,grid_display_mode="ascii",randomize_pair_order=False,show_output_test=False) -> str:
    """format_task for multiple task"""
    prompt=""
    for id,example in enumerate(examples):
        prompt += f"# Task {id+1}:\n"
        prompt += format_task(example,max_example=max_example,grid_display_mode=grid_display_mode,include_test=True,randomize_pair_order=randomize_pair_order,show_output_test=show_output_test)
        assert "solution" in example, "example should have a solution"
        prompt += "## Solution:\n"+example["solution"]+"\n"
    return prompt

def get_list_fewshot_example(examples_fewshot: list[dict], solution:str) -> list[dict]:
    """return a list of fewshot example where solution/reasoning is added in examples_fewshot[i]["solution"]"""
    fewshot_example = []
    for i,example in enumerate(examples_fewshot):
        ex=copy.deepcopy(example)
        ex["solution"] = solution[i]
        fewshot_example.append(ex)
    return fewshot_example


def get_solver_prompt(  task2solve: dict,
                        fewshot_examples: list[dict],
                        prompt_solver: str,
                        grid_display_mode="numpy",
                        alt_colors = True,
                        prompt_colors = True,
                        max_example: int = -1,
                        randomize_pair_order=False,
                        show_output_test=False #for her
                    ) -> str:
    """get a task and return the prompt for the solver"""
    additional_info = get_additional_info_prompt(alt_colors=alt_colors,grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    fewshot_example = format_fewshot_tasks(fewshot_examples,max_example=max_example,grid_display_mode=grid_display_mode,randomize_pair_order=randomize_pair_order)
    task2solve_formated = "# Task to solve:\n" + format_task(task2solve,grid_display_mode=grid_display_mode,max_example=max_example,randomize_pair_order=randomize_pair_order,show_output_test=show_output_test,include_test=True)
    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    assert "fewshot_example" in prompt_solver, "prompt_solver should have {fewshot_example}"
    assert "task" in prompt_solver, "prompt_solver should have {task}" 
    prompt_formated = prompt_solver.format(additional_info = additional_info, fewshot_example = fewshot_example, task = task2solve_formated)
    return prompt_formated.strip()




def get_solver_prompt_her_cot(  task2solve: dict,
                        fewshot_examples: list[dict],
                        prompt_solver: str,
                        grid_display_mode="ascii",
                        alt_colors = False,
                        prompt_colors = True,
                        max_example: int = -1,
                        randomize_pair_order=False,
                        show_output_test=False #for her
                    ) -> str:
    """get a task and return the prompt for the solver"""
    additional_info = get_additional_info_prompt(alt_colors=alt_colors,grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    fewshot_example = format_fewshot_tasks(fewshot_examples,max_example=max_example,grid_display_mode=grid_display_mode,randomize_pair_order=randomize_pair_order,show_output_test=True)
    task2solve_formated = "# Task to solve:\n" + format_task(task2solve,grid_display_mode=grid_display_mode,max_example=max_example,randomize_pair_order=randomize_pair_order,show_output_test=show_output_test,include_test=True)
    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    assert "fewshot_example" in prompt_solver, "prompt_solver should have {fewshot_example}"
    assert "task" in prompt_solver, "prompt_solver should have {task}" 
    prompt_formated = prompt_solver.format(additional_info = additional_info, fewshot_example = fewshot_example, task = task2solve_formated)
    return prompt_formated.strip()

# should remove that
def get_solver_prompt_finetuning(  task2solve: dict,
                        fewshot_examples: list[dict],
                        prompt_solver: str, 
                        grid_display_mode="ascii",
                        alt_colors = False,
                        prompt_colors = True,
                        max_example: int = -1,
                    ) -> str:
    """
    get a task and return the prompt for the solver (instruction for finetuning LLM + inference after finetuning)  
    """
    additional_info = get_additional_info_prompt(alt_colors=alt_colors,grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    fewshot_example = format_fewshot_tasks(fewshot_examples,max_example=max_example,grid_display_mode=grid_display_mode)
    task2solve_formated = "# Task to solve:\n" + format_task(task2solve,grid_display_mode=grid_display_mode,max_example=max_example)
    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    assert "task" in prompt_solver, "prompt_solver should have {task}" 
    prompt_formated = prompt_solver.format(additional_info = additional_info, fewshot_example = fewshot_example, task = task2solve_formated)
    return prompt_formated.strip()


def get_judge_prompt_finetuning(  task2solve: dict,
                        prompt_solver: str, 
                        grid_display_mode="ascii",
                        alt_colors = True,
                        prompt_colors = True,
                        max_example: int = -1,
                    ) -> str:
    """
    get a task and return the prompt for the solver (instruction for finetuning LLM + inference after finetuning)  
    """
    additional_info = get_additional_info_prompt(alt_colors=alt_colors,grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    task2solve_formated = "# Task:\n" + format_task(task2solve,grid_display_mode=grid_display_mode,max_example=max_example,show_output_test=True)
    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    assert "task" in prompt_solver, "prompt_solver should have {task}" 
    prompt_formated = prompt_solver.format(additional_info = additional_info, task = task2solve_formated)
    return prompt_formated.strip()



def get_additional_info_prompt(alt_colors=False,grid_shape="ascii", prompt_colors=True):
    """
    alt_colors; use alternative colors map
    grid_shape: change grid style
    prompt_colors: give colors mapping in the prompt

    """
    map_colors= """The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Purple; 3:Green; 4:Yellow; 5:Grey; 6:Fuschia; 7:Orange; 8:Teal; 9:Brown\n"""
    map_colors_alt= """The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Red; 3:Green; 4:Yellow; 5:Grey; 6:Pink; 7:Orange; 8:Purple; 9:Brown\n"""
    additional_info=""
    grid_shape_explanation=""
    # add or not prompt colors explanation
    colors_mapping = map_colors
    if alt_colors:
        colors_mapping = map_colors_alt
    if prompt_colors:
      additional_info += colors_mapping
    
    # grid shape style explanation
    if grid_shape in ["ascii","spreadsheet"]:
        grid_shape_explanation += "Grid are represented by elements separated by '|' and each each rows are separated by a Newline ('\n').\n"
    if grid_shape in ["spreadsheet"]:
        grid_shape_explanation += "Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc., and rows are denoted with 1, 2, 3, etc. So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.\n"
    

    additional_info += grid_shape_explanation
    return additional_info
    


## for repair prompt

def get_repair_prompt( task2solve: dict,
                        list_previous_response: list[dict],
                        prompt_solver: str,
                        grid_display_mode="numpy",
                        alt_colors = True,
                        prompt_colors = True,
                        max_example: int = -1,
                        show_output_test = False
                    ) -> str:
    """get a task and return the prompt for the solver
        previous_response should at least have the following keys: response (str)
        Optional keys (if you want to include them in the prompt): 
        - correct_train_input list[bool]: list of boolean indicating if the train input-output pairs are correct
        - predicted_train_output list[list]: list of predicted output for the train input-output pairs
    """
    additional_info = get_additional_info_prompt(alt_colors=alt_colors,grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    # if len(fewshot_examples) != 0:
    #     fewshot_example = format_fewshot_tasks(fewshot_examples,max_example=max_example,grid_display_mode=grid_display_mode)
    task2solve_formated = "# Task to solve:\n" + format_task(task2solve,grid_display_mode=grid_display_mode,max_example=max_example,include_test=True,show_output_test=show_output_test)
    previous_implementation = format_repair_previous_response(list_previous_response,grid_display_mode=grid_display_mode,show_output_test=True)
    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    # assert "fewshot_example" in prompt_solver, "prompt_solver should have {fewshot_example}"
    assert "task" in prompt_solver, "prompt_solver should have {task}" 
    prompt_formated = prompt_solver.format(additional_info = additional_info.strip(), task = task2solve_formated.strip(),previous_implementation=previous_implementation.strip())
    return prompt_formated.strip()

def format_repair_previous_response(previous_response: list[dict],grid_display_mode="ascii",show_output_test=False) -> str:
    """format the previous response to be included in the prompt
        previous_response should at least have the following keys: response (str)
        Optional keys (if you want to include them in the prompt): 
        - correct_train_input list[bool]: list of boolean indicating if the train input-output pairs are correct
        - predicted_train_output list[list]: list of predicted output for the train input-output pairs
    """

    previous_response_formated = ""

    len_pre_resp = len(previous_response)
    if len_pre_resp == 0:
        raise ValueError("The previous response should have at least 1 responses")
    elif len_pre_resp ==1:
        previous_response_formated = format_repair_one_previous_response(previous_response,0,grid_display_mode=grid_display_mode,show_output_test=show_output_test)
    else:
        for idx in range(len_pre_resp):
            previous_response_formated += f"\n# Implementation {idx+1}:\n"
            previous_response_formated += format_repair_one_previous_response(previous_response,idx,grid_display_mode=grid_display_mode,show_output_test=show_output_test)
    return previous_response_formated

def format_repair_one_previous_response(previous_response: dict[str],idx=0,grid_display_mode="ascii",show_output_test=False) -> str:
    """format the previous response to be included in the prompt"""
    max_size_display=50
    list_incorrect=""
    shape_grid=""
    if "code" in previous_response[idx]:
        response_formated = '```python\n'+previous_response[idx]["code"] +"\n```\n"
    elif 'response' in previous_response[idx]:
        response_formated = previous_response[idx]["response"] +"\n"
    elif 'text' in previous_response[idx]:
        response_formated = previous_response[idx]["text"] +"\n"
    else:
        raise ValueError("The previous response should have at least a code, response or text field")
    if 'correct_train_input' in previous_response[idx]:#,'predicted_train_output'
        list_correct_task = previous_response[idx]['correct_train_input']
        N_task = len(list_correct_task)
        N_correct_task = sum(list_correct_task)
        response_formated += "This implementation of transform function correctly worked on {}/{} train input-output pairs.\n".format(N_correct_task,N_task)
        response_formated += "Detailed results:\n"
        for i in range(N_task):
            response_formated += f"## Output {i+1} computed by `transform` is "
            if list_correct_task[i]:
                response_formated += "correct.\n"
            else:
                response_formated += "incorrect.\n"
                list_incorrect+= f"Output {i+1}, "
            if 'predicted_train_output' in previous_response[idx] and not(list_correct_task[i]):
                # show output grid only if it is not correct
                output2display = previous_response[idx]['predicted_train_output'][i]
                if isinstance(previous_response[idx]['predicted_train_output'][i],list):
                    try:
                        x,y=np.array(output2display).shape
                        
                        if x<max_size_display and y<max_size_display:
                            output2display = grid_formatting(output2display,mode=grid_display_mode)
                        else: 
                            output2display = f"Output is too large to display ({x}x{y}).\n\n"
                        shape_grid="grid shape: {} by {}".format(x,y)
                    except:
                        shape_grid=""
                        pass
                response_formated += f"The execution gave the following results ({shape_grid}):\n{output2display}\n"
    if show_output_test :
        list_correct_task = previous_response[idx]['predicted_test_output']
        N_task = len(list_correct_task)
        # N_correct_task = sum(list_correct_task)
        # response_formated += "This implementation of transform function correctly worked on {}/{} test input-output pairs.\n".format(N_correct_task,N_task)
        # response_formated += "Detailed results:\n"
        for i in range(N_task):
            response_formated += f"## Output Test {i+1} computed by `transform` (we don't know if it is correct or not) "
            # if list_correct_task[i]:
            #     response_formated += "correct.\n"
            # else:
            #     response_formated += "incorrect.\n"

                # list_incorrect+= f"Output Test {i+1} , "
            if 'predicted_test_output' in previous_response[idx]:
                # show output grid only if it is not correct
                output2display = previous_response[idx]['predicted_test_output'][i]
                if isinstance(previous_response[idx]['predicted_test_output'][i],list):
                    try:
                        x,y=np.array(output2display).shape
                        
                        if x<max_size_display and y<max_size_display:
                            output2display = grid_formatting(output2display,mode=grid_display_mode)
                        else: 
                            output2display = f"Output is too large to display ({x}x{y}).\n\n"
                        shape_grid="grid shape: {} by {}".format(x,y)
                    except:
                        shape_grid=""
                        pass
                response_formated += f"The execution gave the following results ({shape_grid}):\n{output2display}\n"
    list_incorrect=list_incorrect.strip()
    if len(list_incorrect)==0:
        response_formated += "The previous code give correct output for all Train input.\n"
    else:
        if list_incorrect[-1]==",":
            list_incorrect=list_incorrect[:-1].strip()
    response_formated += f"\nThe previous code give incorrect output for: {list_incorrect}. Now, you need to fix the code to produce correct output for all inputs."
    return response_formated


def format_fewshot_examples(dic_solution,base_path):
    train_data, _, _ = get_dataset(data_path=base_path)
    data_correct = get_dic_only_correct(dic_solution)
    list_k_fewshot = list(data_correct.keys())
    np.random.shuffle(list_k_fewshot)
    list_solution = ["```python\n" + np.random.choice(data_correct[key])["code"] + "\n```" for key in list_k_fewshot]
    list_fewshot_examples = get_list_fewshot_example([train_data[i] for i in list_k_fewshot], list_solution)
    return list_k_fewshot,list_fewshot_examples


system_prompt="""You are an advanced AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks. Your primary functions are:

1. Analytical Reasoning: You excel at breaking down complex problems into manageable components, identifying patterns, and applying logical deduction.

2. Python Code Generation: You can create efficient, readable, and effective Python code to solve ARC-AGI tasks.

3. Explanation and Documentation: You provide clear, step-by-step explanations of your reasoning process and code implementation.

4. Adaptability: You can adjust your approach based on the specific requirements of each ARC-AGI task, considering various problem-solving strategies.

5. Error Handling: You anticipate potential issues in your code and implement appropriate error handling mechanisms.

6. Visual Interpretation: You can interpret and reason about visual patterns and spatial relationships often present in ARC-AGI tasks.

When addressing ARC-AGI tasks, you should:
- Carefully analyze the problem statement
- Explain your reasoning process
- Generate Python code to solve the task
- Provide comments within the code for clarity

Your goal is to demonstrate high-level abstract reasoning skills while providing practical, executable solutions in Python."""


prompt_cot_fewshot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Infer the transformation rule, and implement it as a Python function.
You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation which works in general (at least for all given input-output pairs).
{additional_info}

Here are examples of ARC-AGI tasks, the reasoning step to find the solution, and the Python solution (`transform`):

{fewshot_example}

Now, solve the following ARC-AGI task:

{task}"""

prompt_fewshot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation that works in general (at least for all given input-output pairs and test input pairs).
{additional_info}
Here are examples of ARC-AGI tasks, the reasoning step to find the solution, and the Python solution (`transform`):
{fewshot_example}
Now, solve the following ARC-AGI task:

{task}"""


prompt_wo_fewshot_v1_ = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation that works in general (at least for all given input-output pairs and test input pairs).
{additional_info}{fewshot_example}
Now, solve the following ARC-AGI task:

{task}"""


prompt_judge = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs.
You'll be given a new Test input and an its associated Test Output grid (Test Input 1 and Test Output 1). Your task is to determine if Test Output grid was correctly generated from the Test Input grid using the transformation rule.
You need to pay special attention to the patterns and relationships between the input and output grids to make an accurate judgment.
Then respond with your decision, whether the output grid was correctly generated from the input grid or not.
Only respond with "Yes" if it is correct or "No".
{additional_info}
Now, solve the following ARC-AGI task:

{task}
Now, Look closely at the Test Input 1 and Test Output 1 to determine if the output was correctly generated from the input
Remember, you should provide your answer in the form of "Yes" or "No"."""

# For training the model 
prompt_cot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Infer the transformation rule, and implement it as a Python function.
You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation which works in general (at least for all given input-output pairs).
{additional_info}

Now, solve the following ARC-AGI task:

{task}"""

prompt_repair_cot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating and repairing Python code.
Your goal is to analyze input-output grid pairs and repair a previously implemented transformation function that wasn't working correctly for all cases.

You will be given information about which inputs were correctly transformed to give the correct output.

Start your response by carefully reasoning about the errors in the previous implementation and how to fix them. Use <reasoning></reasoning> tags for this analysis.

After your reasoning, write the corrected code in triple backticks (```python and then ```). Implement a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Ensure that your implementation works for all given input-output pairs, especially focusing on the cases that were previously incorrect.

{additional_info}

# Now, repair the following ARC-AGI task implementation:

{task}

Previous implementation:
{previous_implementation}
"""
prompt_repair_cot_fewshot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating and repairing Python code.
Your goal is to analyze input-output grid pairs and repair a previously implemented transformation function that weren't working correctly for all cases.

You will be given information about which inputs were correctly transformed to give the correct output.

Start your response by carefully reasoning about the errors in the previous implementation and how to fix them. Use <reasoning></reasoning> tags for this analysis.

After your reasoning, write the corrected code in triple backticks (```python and then ```). Implement a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Ensure that your implementation works for all given Input-Output pairs and it must also generalize to Test Input, especially focusing on the cases that were previously incorrect.

{additional_info}

# Repairing Example:
Here is an example of repairing, the incorrect answer, and the step to fix the solution:

## Task:

## Input 1 (grid shape: 3 by 3):
[[8 5 0]
 [8 5 3]
 [0 3 2]]

## Output 1 (grid shape: 6 by 6):
[[8 5 0 0 8 8]
 [8 5 3 3 5 5]
 [0 3 2 2 3 0]
 [0 3 2 2 3 0]
 [5 5 3 3 5 8]
 [8 8 0 0 5 8]]

## Input 2 (grid shape: 3 by 3):
[[3 8 2]
 [3 2 2]
 [8 5 2]]

## Output 2 (grid shape: 6 by 6):
[[3 8 2 8 3 3]
 [3 2 2 5 2 8]
 [8 5 2 2 2 2]
 [2 2 2 2 5 8]
 [8 2 5 2 2 3]
 [3 3 8 2 8 3]]

## Input 3 (grid shape: 3 by 3):
[[0 3 0]
 [6 6 6]
 [0 3 0]]

## Output 3 (grid shape: 6 by 6):
[[0 3 0 0 6 0]
 [6 6 6 3 6 3]
 [0 3 0 0 6 0]
 [0 6 0 0 3 0]
 [3 6 3 6 6 6]
 [0 6 0 0 3 0]]

## Test Input (grid shape: 3 by 3):
[[2 5 0]
 [2 5 1]
 [3 1 1]]

## incorrect answer:

<reasoning>
The outputs consist of a larger grid (6x6) compared to the inputs (3x3), and the expanded grids have several patterned properties.

Let's analyze each of the examples to understand the transformation:

### Example 1:

1. The original cells are split and duplicated in a specific mirrored manner along rows and columns.
2. The left and top rows include cells primarily from the original grid's left and top boundary.
3. The transformation involves mirroring both horizontally and vertically.

### Example 2:

- The similar mirroring pattern continues horizontally and vertically.

### Example 3:

- The same arrangement and expansion pattern are observed.

### Reasoning:

1. **Identify cells along the rows and columns.**
2. **Create an expanded (6x6) grid mirroring the arrangement:**

   - Top-left area is reflected horizontally (left edge) from the 3x3 grid.
   - Top-right area is mirrored horizontally (right edge) from the 3x3 grid.
   - Bottom-left area is mirrored vertically (bottom edge).
   - Bottom-right area is reflected horizontally and vertically.

Apply the transformation preserving the mirror symmetry along both axes.

### Code to Implement This:
Let's write code that expands 3x3 grid (mirroring top-left, top-right, bottom-left and bottom-right areas correctly) to a 6x6 pattern matching observations.

</reasoning>

```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Convert input 3x3 grid into the required 6x6 grid
    grid = np.array(grid_lst)
    output = np.zeros((6, 6), dtype=int)

    output[0:3, 0:3] = grid            # Original top-left corner
    output[0:3, 3:6] = grid[:, ::-1]   # Top-right mirrored horizontally
    output[3:6, 0:3] = grid[::-1, :]   # Bottom-left mirrored vertically
    output[3:6, 3:6] = grid[::-1, ::-1]# Bottom-right mirrored both horizontally and vertically

    return output.tolist()
```
This implementation of transform function correctly worked on 0/3 train input-output pairs.
Detailed results:
## Output 1 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[8 5 0 0 5 8]
 [8 5 3 3 5 8]
 [0 3 2 2 3 0]
 [0 3 2 2 3 0]
 [8 5 3 3 5 8]
 [8 5 0 0 5 8]]

## Output 2 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[3 8 2 2 8 3]
 [3 2 2 2 2 3]
 [8 5 2 2 5 8]
 [8 5 2 2 5 8]
 [3 2 2 2 2 3]
 [3 8 2 2 8 3]]

## Output 3 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[0 3 0 0 3 0]
 [6 6 6 6 6 6]
 [0 3 0 0 3 0]
 [0 3 0 0 3 0]
 [6 6 6 6 6 6]
 [0 3 0 0 3 0]]


## Answer fixed:

<reasoning>
The outputs look reasonably close, but the top right, bottom left, and bottom right areas are not correct. It seems like rather than mirroring, the correct transformation is actually rotation. In particular, the top left isn't rotated, the top right is rotated 90 degrees clockwise, the bottom right is rotated 180 degrees, and the bottom left is rotated 270 degrees.

So, the correct transformation rule is:

- Top left: No change
- Top right: Rotate 90 degrees clockwise
- Bottom right: Rotate 180 degrees clockwise
- Bottom left: Rotate 270 degrees clockwise

I'll use np.rot90 to implement this transformation.

I'll now revise the code to implement this corrected transformation.
</reasoning>

```python
import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Convert input 3x3 grid into the required 6x6 grid
    grid = np.array(grid_lst)
    output = np.zeros((6, 6), dtype=int)

    output[0:3, 0:3] = grid
    output[0:3, 3:6] = np.rot90(grid, -1)
    output[3:6, 3:6] = np.rot90(grid, -2)
    output[3:6, 0:3] = np.rot90(grid, -3)

    return output.tolist()
```


**Now, repair the following ARC-AGI task implementation:**
Remember, your task is not to simply copy-paste the previous implementation, but to provide a refined, accurate, and improved solution based on your critical analysis.

{task}

Previous implementation:
{previous_implementation}

## Answer fixed:
"""

    
prompt_repair_cot_fewshot_crossover_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating and repairing Python code.
Your goal is to analyze input-output grid pairs and repair previously implemented transformation functions that weren't working correctly for all cases.

You will be given information about which inputs were correctly transformed to give the correct output for two different implementations.

Start your response by carefully reasoning about the errors in both previous implementations and how to combine their strengths to create a more robust solution. Use <reasoning></reasoning> tags for this analysis.

After your reasoning, write the corrected code in triple backticks (```python and then ```). Implement a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Ensure that your implementation works for all given Input-Output pairs and it must also generalize to Test Input, especially focusing on the cases that were previously incorrect in both implementations.
{additional_info}

# Repairing Example:
Here is an example of repairing, the incorrect answer, and the step to fix the solution:

## Task:

## Input 1 (grid shape: 3 by 3):
[[8 5 0]
 [8 5 3]
 [0 3 2]]

## Output 1 (grid shape: 6 by 6):
[[8 5 0 0 8 8]
 [8 5 3 3 5 5]
 [0 3 2 2 3 0]
 [0 3 2 2 3 0]
 [5 5 3 3 5 8]
 [8 8 0 0 5 8]]

## Input 2 (grid shape: 3 by 3):
[[3 8 2]
 [3 2 2]
 [8 5 2]]

## Output 2 (grid shape: 6 by 6):
[[3 8 2 8 3 3]
 [3 2 2 5 2 8]
 [8 5 2 2 2 2]
 [2 2 2 2 5 8]
 [8 2 5 2 2 3]
 [3 3 8 2 8 3]]

## Input 3 (grid shape: 3 by 3):
[[0 3 0]
 [6 6 6]
 [0 3 0]]

## Output 3 (grid shape: 6 by 6):
[[0 3 0 0 6 0]
 [6 6 6 3 6 3]
 [0 3 0 0 6 0]
 [0 6 0 0 3 0]
 [3 6 3 6 6 6]
 [0 6 0 0 3 0]]

## Test Input (grid shape: 3 by 3):
[[2 5 0]
 [2 5 1]
 [3 1 1]]

## incorrect answer:

<reasoning>
The outputs consist of a larger grid (6x6) compared to the inputs (3x3), and the expanded grids have several patterned properties.

Let's analyze each of the examples to understand the transformation:

### Example 1:

1. The original cells are split and duplicated in a specific mirrored manner along rows and columns.
2. The left and top rows include cells primarily from the original grid's left and top boundary.
3. The transformation involves mirroring both horizontally and vertically.

### Example 2:

- The similar mirroring pattern continues horizontally and vertically.

### Example 3:

- The same arrangement and expansion pattern are observed.

### Reasoning:

1. **Identify cells along the rows and columns.**
2. **Create an expanded (6x6) grid mirroring the arrangement:**

   - Top-left area is reflected horizontally (left edge) from the 3x3 grid.
   - Top-right area is mirrored horizontally (right edge) from the 3x3 grid.
   - Bottom-left area is mirrored vertically (bottom edge).
   - Bottom-right area is reflected horizontally and vertically.

Apply the transformation preserving the mirror symmetry along both axes.

### Code to Implement This:
Let's write code that expands 3x3 grid (mirroring top-left, top-right, bottom-left and bottom-right areas correctly) to a 6x6 pattern matching observations.

</reasoning>

```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Convert input 3x3 grid into the required 6x6 grid
    grid = np.array(grid_lst)
    output = np.zeros((6, 6), dtype=int)

    output[0:3, 0:3] = grid            # Original top-left corner
    output[0:3, 3:6] = grid[:, ::-1]   # Top-right mirrored horizontally
    output[3:6, 0:3] = grid[::-1, :]   # Bottom-left mirrored vertically
    output[3:6, 3:6] = grid[::-1, ::-1]# Bottom-right mirrored both horizontally and vertically

    return output.tolist()
```
This implementation of transform function correctly worked on 0/3 train input-output pairs.
Detailed results:
## Output 1 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[8 5 0 0 5 8]
 [8 5 3 3 5 8]
 [0 3 2 2 3 0]
 [0 3 2 2 3 0]
 [8 5 3 3 5 8]
 [8 5 0 0 5 8]]

## Output 2 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[3 8 2 2 8 3]
 [3 2 2 2 2 3]
 [8 5 2 2 5 8]
 [8 5 2 2 5 8]
 [3 2 2 2 2 3]
 [3 8 2 2 8 3]]

## Output 3 is incorrect.
The execution gave the following results (grid shape: 6 by 6):
[[0 3 0 0 3 0]
 [6 6 6 6 6 6]
 [0 3 0 0 3 0]
 [0 3 0 0 3 0]
 [6 6 6 6 6 6]
 [0 3 0 0 3 0]]


## Answer fixed:

<reasoning>
The outputs look reasonably close, but the top right, bottom left, and bottom right areas are not correct. It seems like rather than mirroring, the correct transformation is actually rotation. In particular, the top left isn't rotated, the top right is rotated 90 degrees clockwise, the bottom right is rotated 180 degrees, and the bottom left is rotated 270 degrees.

So, the correct transformation rule is:

- Top left: No change
- Top right: Rotate 90 degrees clockwise
- Bottom right: Rotate 180 degrees clockwise
- Bottom left: Rotate 270 degrees clockwise

I'll use np.rot90 to implement this transformation.

I'll now revise the code to implement this corrected transformation.
</reasoning>

```python
import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Convert input 3x3 grid into the required 6x6 grid
    grid = np.array(grid_lst)
    output = np.zeros((6, 6), dtype=int)

    output[0:3, 0:3] = grid
    output[0:3, 3:6] = np.rot90(grid, -1)
    output[3:6, 3:6] = np.rot90(grid, -2)
    output[3:6, 0:3] = np.rot90(grid, -3)

    return output.tolist()
```


**Now, repair the following ARC-AGI task implementation:**
Remember, your task is not to simply copy-paste the previous implementation, but to provide a refined, accurate, and improved solution based on your critical analysis.

{task}

Previous implementation:
{previous_implementation}

## Answer fixed:
"""




prompt_repair_cot_fewshot_sol_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating and repairing Python code.
Your goal is to analyze input-output grid pairs and repair a previously implemented transformation function that wasn't working correctly for all cases.

You will be given information about which inputs were correctly transformed to give the correct output.

Start your response by carefully reasoning about the errors in the previous implementation and how to fix them. Use <reasoning></reasoning> tags for this analysis.

After your reasoning, write the corrected code in triple backticks (```python and then ```). Implement a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Ensure that your implementation works for all given input-output pairs, especially focusing on the cases that were previously incorrect.

{additional_info}

# General Example:
Here are an example on how to solve a task that can help you understand the process of reasoning and can be usefull for fixing an solution:

# Task 1:
## Input 1 (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 0 0 8 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 1 (grid shape: 6 by 9):
[[8 0 8 8 0 8 0 0 0]
 [8 8 0 0 8 8 0 0 0]
 [8 0 0 0 0 8 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Input 2 (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 8 8 8 0 0 0]
 [0 0 0 8 8 0 0 0 0]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 2 (grid shape: 6 by 9):
[[0 0 0 8 0 8 8 0 8]
 [0 0 0 8 8 8 8 8 8]
 [0 0 0 8 8 0 0 8 8]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Input 3 (grid shape: 6 by 9):
[[0 0 0 8 0 0 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 8 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 3 (grid shape: 6 by 9):
[[0 0 8 8 0 0 0 0 0]
 [8 8 0 0 8 8 0 0 0]
 [0 0 8 8 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Test Input (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 8 0 0 0 0 0]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Solution:
<reasoning>
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the change from the input to the output in detail for each of the examples. I'll also describe the input for each example as seems relevant.

In example 1, the purple (color 8) shape is reflected across the vertical line at the left edge of the purple shape. There is also a yellow (color 4) shape below the purple shape which is unchanged.

In example 2, the purple shape is reflected across the vertical line at the right edge of the purple shape. The purple shape is different from the purple shape in example 1. There is also a yellow shape below the purple shape which is unchanged. The yellow shape is different from the yellow shape in example 1. However, it is just the reflection of the yellow shape in example 1.

In example 3, the purple shape is reflected across the vertical line at the left edge of the purple shape. The purple shape is different from the purple shape in example 1. There is also a yellow shape below the purple shape which is unchanged. The yellow shape is the same as the yellow shape in example 1.

In the additional input, there is a purple shape and there is also a yellow shape. The purple shape is different from the purple shape in example 1. The yellow shape is the same as the yellow shape in example 2 (which isn't the same as example 1).

The rule is to reflect the purple shape across a vertical line at the edge of the purple shape. I need to determine which side of the purple shape to reflect towards. Example 2 differs from examples 1 and 3 in which side the purple shape is reflected towards. Are there any salient commonalities between examples 1 and 3 which differ from example 2? Yes, the yellow shape is the same in examples 1 and 3. The yellow shape is different in example 2. So, the orientation of the yellow shape determines which side the purple shape is reflected towards.

If the yellow shape is:
4|0|0
4|4|4
0|4|0

Then the purple shape is reflected towards the left. If the yellow shape is:
0|0|4
4|4|4
0|4|0

Then the purple shape is reflected towards the right.

The yellow shape always appears in the middle in the bottom 3 rows, so it should be easy to check what orientation it is in. Then, I just need to reflect the purple shape based on the orientation of the yellow shape. If it is the first orientation, I reflect towards the left. If it is the second orientation, I reflect towards the right.

I will now implement this in code.
</reasoning>

```python
import numpy as np

def reflect_shape(grid, shape_coords, direction):
    min_row, min_col = np.min(shape_coords, axis=0)
    max_row, max_col = np.max(shape_coords, axis=0)

    if direction == 'left':
        for row, col in shape_coords:
            new_col = min_col - (col - min_col) - 1
            grid[row, new_col] = grid[row, col]
    elif direction == 'right':
        for row, col in shape_coords:
            new_col = max_col + (max_col - col) + 1
            grid[row, new_col] = grid[row, col]

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    purple_color = 8
    yellow_color = 4

    # Find the yellow shape
    yellow_shape = np.argwhere(grid == yellow_color)

    # Determine the orientation of the yellow shape
    if np.array_equal(yellow_shape, np.array([[3, 3], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'left'
    elif np.array_equal(yellow_shape, np.array([[3, 5], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'right'
    else:
        raise ValueError("Unexpected yellow shape orientation")

    # Find the purple shape
    purple_shape = np.argwhere(grid == purple_color)

    # Reflect the purple shape
    reflect_shape(grid, purple_shape, direction)

    return grid.tolist()
```

# Now, repair the following ARC-AGI task implementation:

{task}

Previous implementation:
{previous_implementation}
"""
prompt_repair_sol_v1 ="""You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by repairing Python code implementations.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs.
You will be given a python function `transform` that was supposed to implement the transformation rule, but it is not working correctly for all inputs.
You role is to fix this `transform` function.

Your solution should be:
- Accurate: Correctly fix the transformation for all given inputs so they give correct outputs as provided (it should also work for all test inputs)
- Comprehensive: Handles all possible input scenarios
- Well-structured: Uses clear, readable, and efficient code

{additional_info}

**Now, repair the following ARC-AGI task implementation:**

{task}

Previous implementation:
{previous_implementation}"""

prompt_repair_cot_fewshot_sol_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by repairing Python code implementations. Your goal is to analyze input-output grid pairs and fix a previously implemented transformation function that wasn't working correctly for all cases.

Begin your response with a analysis of the errors in the previous implementation and how to fix them. Use <reasoning></reasoning> tags for this analysis. Your reasoning should:

1. Critically evaluate the previous implementation
2. Identify patterns and rules in the input-output transformations
3. Recognize any incorrect assumptions in the original code
4. Propose improvements and fixes

In your reasoning, consider the following:
- Analyze all provided input-output pairs, paying special attention to cases that were previously incorrect
- Look for edge cases or special conditions that might not have been handled properly
- Ensure that your proposed solution works for all given examples

After your reasoning, provide the corrected code in triple backticks (```python and ```). Implement a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Your final solution should be:
- Accurate: Correctly implements the transformation for all given examples
- Comprehensive: Handles all possible input scenarios
- Well-structured: Uses clear, readable, and efficient code

{additional_info}

# General Example:
Here are an example on how to solve a task that can help you understand the process of reasoning and can be usefull for fixing an solution:

# Task 1:
## Input 1 (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 0 0 8 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 1 (grid shape: 6 by 9):
[[8 0 8 8 0 8 0 0 0]
 [8 8 0 0 8 8 0 0 0]
 [8 0 0 0 0 8 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Input 2 (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 8 8 8 0 0 0]
 [0 0 0 8 8 0 0 0 0]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 2 (grid shape: 6 by 9):
[[0 0 0 8 0 8 8 0 8]
 [0 0 0 8 8 8 8 8 8]
 [0 0 0 8 8 0 0 8 8]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Input 3 (grid shape: 6 by 9):
[[0 0 0 8 0 0 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 8 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Output 3 (grid shape: 6 by 9):
[[0 0 8 8 0 0 0 0 0]
 [8 8 0 0 8 8 0 0 0]
 [0 0 8 8 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Test Input (grid shape: 6 by 9):
[[0 0 0 8 0 8 0 0 0]
 [0 0 0 0 8 8 0 0 0]
 [0 0 0 8 0 0 0 0 0]
 [0 0 0 0 0 4 0 0 0]
 [0 0 0 4 4 4 0 0 0]
 [0 0 0 0 4 0 0 0 0]]

## Solution:
<reasoning>
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the change from the input to the output in detail for each of the examples. I'll also describe the input for each example as seems relevant.

In example 1, the purple (color 8) shape is reflected across the vertical line at the left edge of the purple shape. There is also a yellow (color 4) shape below the purple shape which is unchanged.

In example 2, the purple shape is reflected across the vertical line at the right edge of the purple shape. The purple shape is different from the purple shape in example 1. There is also a yellow shape below the purple shape which is unchanged. The yellow shape is different from the yellow shape in example 1. However, it is just the reflection of the yellow shape in example 1.

In example 3, the purple shape is reflected across the vertical line at the left edge of the purple shape. The purple shape is different from the purple shape in example 1. There is also a yellow shape below the purple shape which is unchanged. The yellow shape is the same as the yellow shape in example 1.

In the additional input, there is a purple shape and there is also a yellow shape. The purple shape is different from the purple shape in example 1. The yellow shape is the same as the yellow shape in example 2 (which isn't the same as example 1).

The rule is to reflect the purple shape across a vertical line at the edge of the purple shape. I need to determine which side of the purple shape to reflect towards. Example 2 differs from examples 1 and 3 in which side the purple shape is reflected towards. Are there any salient commonalities between examples 1 and 3 which differ from example 2? Yes, the yellow shape is the same in examples 1 and 3. The yellow shape is different in example 2. So, the orientation of the yellow shape determines which side the purple shape is reflected towards.

If the yellow shape is:
4|0|0
4|4|4
0|4|0

Then the purple shape is reflected towards the left. If the yellow shape is:
0|0|4
4|4|4
0|4|0

Then the purple shape is reflected towards the right.

The yellow shape always appears in the middle in the bottom 3 rows, so it should be easy to check what orientation it is in. Then, I just need to reflect the purple shape based on the orientation of the yellow shape. If it is the first orientation, I reflect towards the left. If it is the second orientation, I reflect towards the right.

I will now implement this in code.
</reasoning>

```python
import numpy as np

def reflect_shape(grid, shape_coords, direction):
    min_row, min_col = np.min(shape_coords, axis=0)
    max_row, max_col = np.max(shape_coords, axis=0)

    if direction == 'left':
        for row, col in shape_coords:
            new_col = min_col - (col - min_col) - 1
            grid[row, new_col] = grid[row, col]
    elif direction == 'right':
        for row, col in shape_coords:
            new_col = max_col + (max_col - col) + 1
            grid[row, new_col] = grid[row, col]

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    purple_color = 8
    yellow_color = 4

    # Find the yellow shape
    yellow_shape = np.argwhere(grid == yellow_color)

    # Determine the orientation of the yellow shape
    if np.array_equal(yellow_shape, np.array([[3, 3], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'left'
    elif np.array_equal(yellow_shape, np.array([[3, 5], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'right'
    else:
        raise ValueError("Unexpected yellow shape orientation")

    # Find the purple shape
    purple_shape = np.argwhere(grid == purple_color)

    # Reflect the purple shape
    reflect_shape(grid, purple_shape, direction)

    return grid.tolist()
```


**Now, repair the following ARC-AGI task implementation:**

{task}

Previous implementation:
{previous_implementation}
"""
