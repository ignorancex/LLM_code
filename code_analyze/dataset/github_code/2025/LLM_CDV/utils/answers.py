import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt import answer_question_dk, extract_answer_dk
from .tools import get_response, clear_json, get_structured_response

def entropy(probability):
    return -sum([p * math.log2(p) for p in probability if p > 0])

def simulate(mode='reasoning', question=None, model='gpt-4o', choices=[], ground_truth=None, simulate_times=5, max_tries=5):
    prompt = answer_question_dk(question=question, choices=choices)
    responses = []

    def get_answer_dk():
        tries = max_tries
        response = get_response(model=model, prompt=prompt, temperature=0.0001)
        extract_prompt = extract_answer_dk(question=question, answer=response, choices=choices)
        answer = clear_json(get_response(model='gpt-4o-mini', prompt=extract_prompt, temperature=0.0001))
        try:
            answer = json.loads(answer)['final_answer']
            if answer in choices:
                return answer
            else:
                tries -= 1
                print(f"Out of choices")
                print(answer+'\n')
        except json.JSONDecodeError:
            print(f"Failed to extract answer.")
            return 1  # Return None if all tries are exhausted

    with ThreadPoolExecutor() as executor:
        future_answers = [executor.submit(get_answer_dk) for _ in range(simulate_times)]
        for future in as_completed(future_answers):
            result = future.result()
            if result:
                responses.append(result)
    
    if len(responses) == 0:
        return 0
    correct_number = 0
    for value in responses:
        if value == ground_truth:
            correct_number += 1
    print(f"\tResponses: {responses}")
    correct_rate = correct_number / len(responses)
    return correct_rate
