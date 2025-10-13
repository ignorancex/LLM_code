import torch
from typing import List, Dict
import math
def doc_to_choice(doc: Dict) -> List[str]:
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]

def doc_to_target(doc: Dict) -> str:
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()

def doc_to_text(doc: Dict) -> int:
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]
