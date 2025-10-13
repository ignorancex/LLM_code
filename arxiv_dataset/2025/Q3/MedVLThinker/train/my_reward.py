"""
Doc: https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Implementation:
1. EasyR1: third_party/EasyR1/examples/reward_function/r1v.py
2. https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/geo3k.py
"""

import re
from typing import Dict


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, answer: str, answer_label: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict)
        given_answer = (
            content_match.group(1).strip() if content_match else predict.strip()
        )
        if grade_answer(given_answer, answer, answer_label):
            return 1.0

    except Exception:
        pass

    return 0.0


def grade_answer(prediction, answer, answer_label=None):
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower() == answer.strip().lower():
        return True

    return False


def compute_score(data_source, solution_str, ground_truth, extra_info):
    if format_reward(solution_str) == 0.0:
        return 0.0

    answer = extra_info.get("answer")
    answer_label = extra_info.get("answer_label")

    if answer != ground_truth:
        raise ValueError(
            f"Answer mismatch: {answer} != {ground_truth}. "
            "Ensure the ground truth is correctly set in the extra_info."
        )
    return accuracy_reward(solution_str, answer, answer_label)
