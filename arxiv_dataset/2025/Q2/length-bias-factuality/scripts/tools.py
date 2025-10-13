import json
from typing import Union
import re

################################################################################
#                             JSON FILE OPERATION                              #
################################################################################

def jsonlines_load(fname: str):
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]

def jsonlines_dump(fname: str, data: Union[dict, list]):
    try:
        with open(fname, 'a+') as f:
            if isinstance(data, dict):
                f.write(json.dumps(data)+'\n')
            elif isinstance(data, list):
                for d in data:
                    f.write(json.dumps(d)+'\n')

    except (FileNotFoundError, FileExistsError) as e:
        print(f'Error: {e}')
        print(f'Could not write to {fname}')

################################################################################
#                             ABSTENTION DETECTION                             #
################################################################################
"""Detects if the generation is an abstention response."""
def generic_abstain_detect(generation):
    return generation.startswith("I'm sorry") or generation.startswith("I apologize") \
        or generation.startswith("Sorry") or "provide more" in generation or "couldn't find" in generation\
        or generation.startswith("I'm not") or generation.startswith("I\u2019m sorry")\
        or "no publicly available" in generation

################################################################################
#                             STRING MANIPULATION                              #
################################################################################
def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')

def extract_hash_block(
    input_string: str, number_of_blocks: int
) -> str:
    """Extracts the contents of a string under the hash block (###)."""
    pattern = re.compile(r'###\s*(.*?)\s*###\n(.*?)(?=\n###|$)', re.DOTALL)
  
  # search all the matched patterns
    match = pattern.findall(input_string)
   
    if number_of_blocks == 1:
        assert len(match) == 1, f'Error: {match}'
        return strip_string(match[0][1])
    elif number_of_blocks == 2:
        assert len(match) == 2, f'Error: {match}'
        return strip_string(match[0][1]), strip_string(match[1][1])
    else:
        raise ValueError(f'Not implemented. Error: {number_of_blocks}')

