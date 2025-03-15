import json
import random
from .prompt import get_distraction
from .tools import get_response, clear_json

def generate_question(model='gpt-4o', question=None, wrong_option=None, choices=None,ground_truth=None,max_tries=5):
    distraction_prompt = get_distraction(question=question, wrong_answer=wrong_option)
    distraction_origin = get_response(model='gpt-4o-mini', prompt=distraction_prompt, temperature=0.7)
    shuffled_choices = choices[:]
    random.shuffle(shuffled_choices)
    prefix = "Generated Distraction: "
    # extract the content after the prefix
    distraction = distraction_origin[distraction_origin.find(prefix) + len(prefix):]
    if distraction == '':
        return None
    new_question = [
        {
            'question': question + ' ' + distraction,
            'choices': shuffled_choices,
            'ground_truth': ground_truth,
        },
        {
            'question': distraction + ' ' + question,
            'choices': shuffled_choices,
            'ground_truth': ground_truth,
        }
    ]
    random_questions = random.sample(new_question, 1)
    return random_questions
    
        
