import os
import re
import pickle
import glob
import textwrap
from pathlib import Path

class LeanProofAssembler:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def assemble_full_proof(self) -> str:
        root_main = self.root_dir / "main_theorem.lean"
        if not root_main.is_file():
            raise FileNotFoundError(f"No main_theorem.lean found in {self.root_dir}")

        # Read file and expand sorries
        with open(root_main, "r", encoding="utf-8") as f:
            file_text = f.read()

        return self._expand_sorries(file_text, self.root_dir, sorry_counter=1)

    def _expand_sorries(self, content: str, parent_dir: Path, sorry_counter: int) -> str:
        lines = content.splitlines(keepends=True)
        result = []

        

        for line in lines:
            if 'sorry' in line:
                if self.starts_with_digits_underscore(parent_dir.name):
                    corrected_name = '_'.join(parent_dir.name.split('_')[1:])
                else:
                    corrected_name = parent_dir.name
                subfolder_name = f"{sorry_counter-1}_{corrected_name}_{sorry_counter}"
                subfolder_path = parent_dir / subfolder_name

                if not subfolder_path.is_dir():
                    print(f"Expected subfolder '{subfolder_name}' for 'sorry' #{sorry_counter} in {parent_dir}. Skipping this sorry...")
                    result.append(line + '\n')
                    print(rf'{line}')
                    sorry_counter += 1
                    continue

                # Retrieve the sub-proof text
                subproof_text = self._get_sub_proof_text(subfolder_path)

                # Remove redundant indentation from partial proofs
                subproof_text = textwrap.dedent(subproof_text)

                indentation = self.count_leading_spaces(line)

                subproof_text = '\n\n' + ''.join([indentation*" " + s for s in subproof_text.splitlines(keepends=True)]) + '\n\n'
                # Recursively expand any nested sorries in the subproof
                subproof_expanded = self._expand_sorries(subproof_text, subfolder_path, sorry_counter=1)

                result.append(subproof_expanded)
                sorry_counter += 1
            else:
                result.append(line)
        print(result)
        return "".join(result)
    
    def count_leading_spaces(self, s: str) -> int:
            return len(s) - len(s.lstrip(" "))
    
    def correct_text_indentation(self, proof_code, line_indentation):
        result = []
        for line in proof_code.splitlines(keepends=True):
            ind = self.count_leading_spaces(line)
            line = ' ' * max(line_indentation - ind, 0) + line.lstrip()
            result.append(line)
        return ''.join(result)
    
    def starts_with_digits_underscore(self, s: str) -> bool:
        pattern = re.compile(r'^\d+_')
        return bool(pattern.match(s))
    
    def split_by_first_by(self, s):
        parts = re.split(r':=\s*by', s, maxsplit=1)
        if len(parts) == 1:
            return s, ''
        return parts[0], parts[1]

    def _get_sub_proof_text(self, subdir: Path) -> str:
        # 1) try a nested main_theorem.lean
        try:
            nested_lean = subdir / "main_theorem.lean"
            if nested_lean.is_file():
                text = nested_lean.read_text(encoding="utf-8")
                proof = text.split(':= by', 1)[1]
                return proof
        except:
            print('Failed to extract from main_theorem.lean file')

        
        # 2) Try success.pkl (could be raw text or an actual pickle)
        pkl_file = glob.glob(os.path.join(subdir, 'success-*.pkl'))
        if pkl_file:
            pkl_file_name = pkl_file[0].split('/')[-1]
            pkl_file = subdir / pkl_file_name
            if pkl_file.is_file():
                with open(pkl_file, "rb") as f:
                    proof_text = pickle.load(f)[0]['proof_code']
                    if "```" in proof_text:
                        proof_text = proof_text.replace("```lean4", "").replace("```", "")
                    if 'import' in proof_text:
                        imports, proof_text = split_by_first_by(proof_text)
                return proof_text
        
        return 'sorry'