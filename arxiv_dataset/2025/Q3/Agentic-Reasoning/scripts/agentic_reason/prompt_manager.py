from typing import List, Dict, Any
from prompts import (
    get_singleqa_search_o1_instruction,
    get_multiqa_search_o1_instruction,
    get_math_search_o1_instruction,
    get_hard_question_instruction,
    get_code_search_o1_instruction,
    get_task_instruction_openqa,
    get_task_instruction_math,
    get_task_instruction_multi_choice,
    get_task_instruction_code
)

def get_instruction_and_prompt(
    dataset_name: str,
    question: str,
    model_path: str = None,
    max_search_limit: int = 5,
    question_title: str = None
) -> tuple[str, str]:
    """
    Get the appropriate instruction and user prompt based on dataset and model.
    
    Args:
        dataset_name: Name of the dataset
        question: The question to process
        model_path: Path to the model (used to determine model type)
        max_search_limit: Maximum number of search operations allowed
        question_title: Optional title for code questions
    
    Returns:
        tuple[str, str]: (instruction, user_prompt)
    """
    is_qwq = model_path and 'qwq' in model_path.lower()
    
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
        if dataset_name in ['nq', 'triviaqa']:
            instruction = get_singleqa_search_o1_instruction(max_search_limit)
        else:
            instruction = get_multiqa_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_openqa(question, model_name='qwq' if is_qwq else None)

    elif dataset_name in ['math500', 'aime', 'amc']:
        instruction = get_math_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_math(question, model_name='qwq' if is_qwq else None)

    elif dataset_name == 'gpqa':
        instruction = get_hard_question_instruction(max_search_limit)
        if model_path:
            if is_qwq:
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        else:
            user_prompt = get_task_instruction_multi_choice(question)

    elif dataset_name == 'livecode':
        instruction = get_code_search_o1_instruction(max_search_limit)
        user_prompt = get_task_instruction_code(
            question, 
            question_title=question_title,
            model_name='qwq' if is_qwq else None
        )
    else: # deep research
        instruction = ""
        user_prompt = ""

    return instruction, user_prompt

def prepare_input_prompts(
    filtered_data: List[Dict[str, Any]],
    dataset_name: str,
    model_path: str,
    tokenizer,
    max_search_limit: int = 5,
    subset_num: int = -1
) -> tuple[List[str], List[Dict]]:
    """
    Prepare input prompts for the model.
    
    Args:
        filtered_data: List of data items to process
        dataset_name: Name of the dataset
        model_path: Path to the model
        tokenizer: Tokenizer instance
        max_search_limit: Maximum number of search operations allowed
        subset_num: Number of items to process (-1 for all)
    
    Returns:
        tuple[List[str], List[Dict]]: (input_prompts, active_sequences)
    """
    input_list = []
    
    for item in filtered_data:
        question = item['Question']
        question_title = item.get('question_title', '') if dataset_name == 'livecode' else None
        
        instruction, user_prompt = get_instruction_and_prompt(
            dataset_name,
            question,
            model_path,
            max_search_limit,
            question_title
        )

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt,
            chat_template="{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
            tokenize=False,
            add_generation_prompt=True
        )
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
    } for item, prompt in zip(filtered_data, input_list)]

    return input_list, active_sequences
