from curses.ascii import isdigit
import re
import json
import os
CACHE_DIR = ".cache"


def jsonlines_load(fname: str):
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]


def extract_relation_triples(prediction):
    parentheses_parts = re.findall(r'\(([^)]*)\)', prediction)
    relation_triples = []
    for element in parentheses_parts:
        parts = element.split(", ")
        if len(parts) >= 3:
            relation_triples.append(f"({element})")
            
    return relation_triples


def extract_num(solution: str):
    raw_ans = solution.strip().split("The final answer:")[-1]
    raw_ans = raw_ans.strip()
    raw_ans = raw_ans.replace(",", "") # remove comma in number
    raw_ans = raw_ans.replace("$", "") # remove dollar sign
    raw_ans = raw_ans.replace("¥", "") # remove yen sign
    raw_ans = raw_ans.replace("£", "") # remove pound sign
    raw_ans = raw_ans.replace("€", "") # remove euro sign
    raw_ans = raw_ans.replace("%", "") # remove rupee sign
    
    if raw_ans[-1] == ".":
        raw_ans = raw_ans[:-1]
    elif not raw_ans[-1].isdigit():
        raw_ans = raw_ans.split()[0]
    
    print("Extracted number: ")
    print(raw_ans)
        
    return float(raw_ans)


def extract_code_from_str(s):
    try:
        num = len(s.split("```python\n"))
        if num == 1:
            # do not have code
            return None
        elif num >= 1:
            latter_part = s.split("```python\n")[-1]
            program = latter_part.split("```")[0]
            return program
    except Exception as e:
        return None
    return None


if __name__ == "__main__":
        
    relation_triples = extract_relation_triples("""To solve this problem, we first need to determine the number of clips Natalia sold in May, (the-number-of-clips-sold-in-May, is, ?). It is given that she sold half as many clips in May as she did in April, (clips-sold-in-May, is, half-of-April).  From the problem, we know clips sold in April is 48, (number-of-clips-sold-in-Apirl, is, 48). Therefore, the number of clips sold in May is half of 48, (number-of-clips-sold-in-May, is, 48/2). 48/2 = 24, (48/2, is, 24). Now, we find the total number of clips sold over both April and May by adding the clips sold in each month, (tatal-clips-sold, is, sum-of-number-of-clips-sold-in-Apirl-and-May). 48+24=72, (48+24, is, 72). Thus, Natalia sold a total of 72 clips in April and May combined.
The final answer: 72.""")
    print("\n".join(relation_triples))
    