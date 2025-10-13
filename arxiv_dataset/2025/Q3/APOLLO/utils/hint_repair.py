from collections import defaultdict
from tqdm import tqdm
import re
import sys


class ProofRepairer:
    def __init__(self, code: str, verifier, verbose=True):
        self.code = code
        self.verifier = verifier
        self.verbose = verbose
        
        self.possible_tactics = ['nlinarith', 'norm_cast', 'norm_num', 'ring_nf', 'ring']
        self.try_repairer = 'try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith'
    
    def repair_proof(self) -> str:

        code_with_hints = self.code.replace('sorry', 'hint')
        print('Begin HintRepair...')
        err_info = self.verify_lean_code(code_with_hints)

        if 'infos' not in err_info.keys():
            return self.code
        
        hint_correct = []
        
        for i, info in enumerate(err_info['infos'], start=1):
            try:
                suggestions = self.get_hint_tactics(info['data'])
            except:
                suggestions = []
            if len(suggestions) == 1:
                hint_correct.append([i, suggestions[0]])
        
        replacement_accumulation = 0
        for idx, tactic in tqdm(hint_correct, desc='Correcting with hint') if self.verbose else hint_correct:
            idx -= replacement_accumulation
            self.code = self.replace_nth('sorry', tactic, self.code, idx)
            replacement_accumulation += 1
        
        codelines = self.code.splitlines()
        fixed_codelines = []
        for idx, line in enumerate(tqdm(self.code.splitlines(), desc='Correcting with other solvers') if self.verbose else self.code.splitlines()):
            if 'sorry' in line:
                indent = len(line) - len(line.lstrip())
                line = line.replace('sorry', self.try_repairer) + '\n' + ' '*indent + 'sorry'
                fixed_codelines.append(line)
            else:
                fixed_codelines.append(line)
        
        fixed_code = '\n'.join(fixed_codelines)
        err_info = self.verify_lean_code(fixed_code)
        fixed_codelines = fixed_code.splitlines()
        
        lines_to_remove = []
        if not err_info['pass']:
            for e in err_info['errors']:
                if 'no goals to be solved' not in e['data']:
                    continue
                
                r = int(e['pos']['line']-1)
                lines_to_remove.append(r)

        fixed_codelines = self.remove_indices(fixed_codelines, lines_to_remove)

                    
        self.code = '\n'.join(fixed_codelines)
        
            
        return self.code
    
    def remove_indices(self, lst, remove_idx):
        remove_set = set(remove_idx)               # for O(1) membership tests
        return [x for i, x in enumerate(lst) if i not in remove_set]
                


    def apply_hints(self, code: str) -> str:
        return code.replace('sorry', 'hint')
    
    def replace_line_with_hint_line(self, code, line):
        return code.replace(line, line.replace('sorry', 'hint'))
    
    def replace_line_with_suggested_tactic(self, code, line, tactic):
        return code.replace(line, line.replace('sorry', tactic))
    
    def get_hint_tactics(self, info):
        lines = info.splitlines()
        suggestions = []
        current_suggestion = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("•"):
                # Finish any current suggestion
                if current_suggestion is not None:
                    suggestions.append(current_suggestion)
                # Start a new suggestion, removing the bullet and any whitespace
                current_suggestion = stripped[1:].strip()
            else:
                # If this line doesn't start with a bullet and is not empty,
                # consider it a continuation of the current suggestion.
                if current_suggestion is not None and stripped:
                    current_suggestion += " " + stripped
        # Append the last suggestion if it exists.
        if current_suggestion is not None and current_suggestion.strip() != 'aesop' and current_suggestion.strip() != 'intro':
            # Skip aesop and intro, since usually they do not close the goal...
            suggestions.append(current_suggestion)
        return suggestions
    
    def replace_nth_occurrence(self, string, target, replacement, n):
        """
        Replaces the n-th occurrence of 'target' in 'string' with 'replacement'.
        If 'target' does not occur n times, the original string is returned.
        """
        index = -1
        for _ in range(n):
            index = string.find(target, index + 1)
            if index == -1:
                # 'target' doesn't occur n times
                return string
        
        # Rebuild the string with the replacement
        return string[:index] + replacement + string[index + 1:]
    
    def replace_nth(self, sub, repl, txt, nth):
        arr=txt.split(sub)
        part1=sub.join(arr[:nth])
        part2=sub.join(arr[nth:])
        
        return part1+repl+part2

    def verify_lean_code(self, code):
        return self.verifier(code)
            



class LeanServerProofRepairer(ProofRepairer):
    def verify_lean_code(self, code):
        # Verifier is a Scheduler Object, so handle differently
        request_id = self.verifier.verifier_submit_request(code)
        result_list = self.verifier.verifier_get_all_request_outputs([request_id])
        return result_list[0]







    
    
    
    
    
    
    
    
    