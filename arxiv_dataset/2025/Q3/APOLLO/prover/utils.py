import os
import re
import json
import pytz
from pathlib import Path
from datetime import datetime
from collections import UserDict
from importlib.machinery import SourceFileLoader
from easydict import EasyDict as AttrDict


LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

def non_cot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def non_cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def cot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def cot_goedel_v2_prompt(data):
    # return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}"

    return 'Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}  sorry```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.'.format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def cot_few_shot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def post_process_output(output):
    _find_idx = output.find("```")
    return output[:_find_idx] if _find_idx >= 0 else output

def naive_post_process_output(output):
    return output

def cot_kimina_prompt(data):
    return "Think about and solve the following problem step by step in Lean 4.\n# Problem:{informal_prefix}\n# Formal statement:\n```lean4\n{header}{formal_statement}\n```\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def cot_follow_informal_proof_prompt(data):
    # return "Think about and solve the following problem step by step based on a given informal solution in Lean 4. Focus on writing a good proof sketch based on the given informal proof.\n# Problem:{informal_prefix}\n# Informal solution:\n{informal_solution}\n\n# Formal statement:\n```lean4\n{header}{formal_statement}\n```\n".format(
    # return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}\nInformal Proof:\n{informal_solution}\n\n{formal_statement}\n  sorry```\n\nBefore producing the Lean 4 code to formally prove the given theorem and an informal proof sketch, provide a detailed proof plan outlining the main proof steps and strategies.\n".format(
    # return '\n# Problem statement:\n{informal_prefix}\n\n# Informal solution:\n{informal_solution}\n\n# Objective:\nTranslate the informal steps into a Lean4 proof script. Your proof should:\n\n1. Follow the same high‑level structure as the informal solution.  \n2. Use idiomatic Lean4 tactics (`rw`, `simp`, `induction`, etc.).  \n3. Include explicit lemma/theorem names where needed.  \n\n# Formal statement:\n```lean4\n{header}{formal_statement}\n```\n'.format(
    # return '\n# Problem statement:\n{informal_prefix}\n\n# Informal solution:\n{informal_solution}\n\n# Objective:\nTranslate the informal steps into a Lean4 proof sketch. Your proof should:\n\n- Follow the same high‑level structure as the informal solution.  \n- Use "sorry" statements to fill in the gaps\n- Make sure that the formal proof sketch follows the informal proof\n\n# Formal statement:\n```lean4\n{header}{formal_statement}\n```\n'.format(
    return '\n# Problem statement:\n{informal_prefix}\n\n# Informal solution:\n{informal_solution}\n\n# Objective:\nTranslate the informal steps into a Lean4 proof sketch. Do not attempt to prove the problem. Your proof sketch should:\n\n- Mirror the informal proof’s structure: Break your Lean proof into the same steps (cases, inductive hypothesis, key calculations) in the same order.\n- Try to break down the problem into major subproofs\n- Translate each informal claim: Wherever the informal proof says “clearly,” “hence,” or “by symmetry,” insert a corresponding have or by block with the exact lemma or tactic.\n- Use sorry placeholders for subproofs\n- Declare and name intermediate lemmas: If the informal solution introduces a new fact, turn it into a named lemma or have with that same description.\n\n# Formal statement:\n```lean4\n{header}{formal_statement}\n```\n'.format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        informal_solution=data.get('informal_solution', str()),
        formal_statement=data['formal_statement'],
    )

def cot_kimina_few_shot_prompt(data):
    return "Think about and solve the following problem step by step in Lean 4.\n# Problem:{informal_prefix}\n# Formal statement:\n```lean4\n{header}{formal_statement}{formal_proof}\n```\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
        formal_proof=data['formal_proof'],
    )

def kimina_post_process_output(output):
    def after_by(s: str) -> str:
        """
        Returns everything after the first occurrence of ':= by' (with any amount of space
        between := and by), or an empty string if not found.
        """
        m = re.search(r':=\s*by(.*)', s, flags=re.DOTALL)
        return m.group(1) if m else s
    output = output[:output.find('<\think>')]
    _find_idx = output.find("```")

    # return output[:_find_idx] if _find_idx >= 0 else output
    pattern = r"```lean4\s*(.*?)\s*```"
    match = re.search(pattern, output, flags=re.DOTALL)
    output = match.group(1) if match else output

    output = after_by(output)
    return output

def goedel_v2_post_process_output(output):
    def after_by(s: str) -> str:
        """
        Returns everything after the first occurrence of ':= by' (with any amount of space
        between := and by), or an empty string if not found.
        """
        m = re.search(r':=\s*by(.*)', s, flags=re.DOTALL)
        return m.group(1) if m else s

    # return output[:_find_idx] if _find_idx >= 0 else output
    pattern = r"```lean4\s*(.*?)\s*```"
    all_blocks = re.findall(pattern, output, flags=re.DOTALL)
    if all_blocks:
        output = all_blocks[-1]

    output = after_by(output)
    return output

def cot_ds_v2_prompt(data):
    return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}\n  sorry```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.\n".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )

def ds_v2_post_process_output(output):
    def after_by(s: str) -> str:
        """
        Returns everything after the first occurrence of ':= by' (with any amount of space
        between := and by), or an empty string if not found.
        """
        try:
            m = re.search(r':=\s*by(.*)', s, flags=re.DOTALL)
            return m.group(1) if m else s
        except:
            return s

    def find_last_lean4_block(s: str) -> str | None:
        """
        Find all substrings of the form ```lean4 ... ``` in `s` and return the last one.
        Returns None if no such block is found.
        """
        # Pattern matches ```lean4 followed by any characters (including newlines), non-greedy, up to ```
        pattern = r'```lean4[\s\S]*?```'
        matches = re.findall(pattern, s)
        return matches[-1].replace('```lean4', '').replace('```', '') if matches else None

    output = find_last_lean4_block(output)

    output = after_by(output)
    return output

MODEL_FORMAT = dict(
    non_cot=dict(prompt=non_cot_prompt, output=post_process_output, few_shot=non_cot_few_shot_prompt),
    cot=dict(prompt=cot_prompt, output=post_process_output, few_shot=cot_few_shot_prompt),
    cot_kimina=dict(prompt=cot_kimina_prompt, output=kimina_post_process_output, few_shot=cot_kimina_few_shot_prompt),
    cot_goedel_v2=dict(prompt=cot_goedel_v2_prompt, output=goedel_v2_post_process_output, few_shot=cot_kimina_few_shot_prompt),

    cot_kimina_inference_tokens=dict(prompt=cot_kimina_prompt, output=naive_post_process_output, few_shot=cot_kimina_few_shot_prompt),
    cot_ds_v2_inference_tokens=dict(prompt=cot_ds_v2_prompt, output=naive_post_process_output, few_shot=cot_kimina_few_shot_prompt),

    cot_ds_v2=dict(prompt=cot_ds_v2_prompt, output=ds_v2_post_process_output, few_shot=cot_kimina_few_shot_prompt),
    cot_kimina_follow_informal_proof=dict(prompt=cot_follow_informal_proof_prompt, output=kimina_post_process_output, few_shot=cot_kimina_few_shot_prompt),
    cot_dsv2_follow_informal_proof=dict(prompt=cot_follow_informal_proof_prompt, output=ds_v2_post_process_output, few_shot=cot_kimina_few_shot_prompt),
)


def get_datetime(readable=False):
    if readable:
        return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y/%m/%d %H:%M:%S")
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")

def load_config(fname):
    name = Path(fname).stem
    mod = SourceFileLoader(name, fname).load_module()

    config = {}
    for n in dir(mod):
        if not n.startswith("__"):
            config[n] = getattr(mod, n)
    config = AttrDict(config)

    return config

def load_jsonl_objects(input_path):
    objects = []
    with open(input_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            objects.append(json.loads(line))
    return objects


class ConcurrentJob(object):
    def __init__(self, stage_list):
        assert len(stage_list) > 1
        self.stage_list = stage_list
        self.reset()
    
    def is_idle(self):
        return self._stage_idx is None
    
    def reset(self):
        self._stage_idx = None
        self._stage_cache = None
    
    def start(self, **kwargs):
        self._stage_idx = 1
        self._stage_cache = self.stage_list[0](**kwargs)
    
    def get_status(self):
        assert not self.is_idle()
        while True:
            status = self.stage_list[self._stage_idx](**self._stage_cache)
            if status is None:
                return None
            self._stage_idx += 1
            if self._stage_idx == len(self.stage_list):
                self.reset()
                return status
            self._stage_cache = status