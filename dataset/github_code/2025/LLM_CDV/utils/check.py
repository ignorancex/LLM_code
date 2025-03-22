import json
from .prompt import check_question_semantic
from .tools import get_response, clear_json


def check_semantic(model='gpt-4o', question=None, ori_question=None, ground_truth=None, choices=[], max_tries=5):
    prompt = check_question_semantic(question=question,ori_question=ori_question,ground_truth=ground_truth)
    while max_tries > 0:
        response = get_response(model=model, prompt=prompt,temperature=0.0001)
        try:
            response = json.loads(response)
            break
        except json.JSONDecodeError:
            max_tries -= 1
            print(f"Failed to extract ideas. Retrying... {max_tries} tries left.")
            continue
    
    if max_tries==0 or response['response']=='No':
        return False
        
    return True

