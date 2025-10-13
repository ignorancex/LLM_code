import re

from prover.lean.verifier import verify_lean4_file
from tqdm import tqdm

class LemmaExtractor:
    def __init__(self, code: str, verifier, name=None):
        self.code = code
        self.verifier = verifier
        
        self.header = self.get_header(self.code)
        
        self.possible_tactics = ['nlinarith', 'norm_cast', 'norm_num', 'ring_nf', 'ring']

        if name:
            self.name = name
        else:
            self.name = self.get_name(self.code)
        
    
    def get_lemmas(self):
        sorry_states = self.verifier(self.code)['sorries']
                
        lemmas = []
        
        for i, state in enumerate(tqdm(sorry_states), start=1):
            given_block, goal_block = self.get_statement(state['goal'])

            # Purely for better formatting
            if given_block == ':':
                s = self.header + f'lemma {self.name}_{i}'
                s += given_block + '\n' + goal_block
            else:
                s = self.header + f'lemma {self.name}_{i}\n' 
                s += given_block + goal_block
                
            err_info = self.verifier(s)
            if not err_info['pass']:
                s = self.statement_check(s, err_info)
            lemmas.append(s)
        return lemmas
    
    def get_statement(self, state):
        # Split off the 'goal' part
        # Using rsplit with maxsplit=1 to ensure we only split on the last '⊢'
        if '⊢' in state:
            given, goal = state.rsplit('⊢', 1)
        else:
            # No '⊢', treat the entire string as 'given'
            given = state
            goal = ''

        # Split the given part into lines
        lines = given.splitlines()

        # Merge multi-line statements by checking indentation
        merged_lines = []
        for line in lines:
            # If this line starts with whitespace, treat it as a continuation
            # of the previous line.
            if line.strip() and (line[0].isspace() or line.startswith(' ')):
                # Append with a newline to keep indentation in the final output
                merged_lines[-1] += "\n" + line
            else:
                # Start a new line in merged_lines
                merged_lines.append(line)

        # Wrap each merged line in parentheses
        # Only do so for non-empty lines
        wrapped = []
        for ml in merged_lines:
            ml = ml.strip()
            if ml:
                wrapped.append(f"  ({ml})")

        # Join everything with newlines, then add the 'goal' part.
        # For Lean, you often want a " :\n" before the goal chunk:
        # but you can adapt as needed.
        given_block = '\n'.join(wrapped) + ' :\n' if wrapped else ':'
        
        # Format the goal
        if goal.strip():
            # If goal is not empty, prepend two spaces and finish with := by\n  sorry
            goal_block = '  ' + goal.strip() + ' := by\n  sorry'
        else:
            goal_block = ''

        return given_block, goal_block
    
    def get_header(self, code):
        header = []
        for c in code.splitlines():
            for h in ['import', 'open', 'set_option']:
                if c.strip().startswith(h):
                    header.append(c)
        return '\n'.join(header) + '\n\n'
    
    def statement_check(self, code, err_info):
        codelines = code.splitlines()
        
        removed_lines = set()
        
        for e in err_info['errors']:
            idx = e['pos']['line'] - 1
            if idx in removed_lines:
                continue
            
            codelines[idx] = ''
            
            removed_lines.add(idx)
        
        codelines = [c for c in codelines if c.strip() != '']
        
        return '\n'.join(codelines)
    
    def get_name(self, code):
        match = re.search(r'^(theorem|lemma)\s+(\S+)', code, re.MULTILINE)
        if match:
            name = match.group(2)
            return name
        else:
            return None
    














