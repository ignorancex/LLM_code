import os
import sys
import glob
import json
import time
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

# -- Local imports (adjust paths if needed) --
from utils.syntax_repair import SyntaxCorrector
from utils.sorrify import Sorrifier, ProofTree
from utils.hint_repair import ProofRepairer
from utils.extract_lemmas_from_sorry import LemmaExtractor
from utils.convert_lean_to_json import get_deepseek_format_proofs
from utils.proof_assembler import LeanProofAssembler

from prover.lean.verifier import verify_lean4_file
from prover.launch_solver import launch_llm
from prover.launch_hint_fixer import launch_parallel_hint_search, launch_regular_hint_search



class ApolloRepair:
    def __init__(self,
                 code: str,
                 lemma_name: str,
                 config: str = "configs/baseline_sampling.py",
                 rec_depth: int = 2,
                 lean_version: str = 'v4.17.0',
                 log_dir = None):
        self.code = code


        self.lemma_name = lemma_name
        self.config = config
        self.rec_depth = rec_depth
        self.lean_version = lean_version
        # Where we store logs, partial results, etc.
        if not log_dir:
            self.log_dir = f"logs/{self.lean_version}/rec_depth_{self.rec_depth}/{self.lemma_name}"
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.options = 'set_option pp.instanceTypes true\nset_option pp.numericTypes true\nset_option pp.coercions.types true\nset_option pp.letVarTypes true\nset_option pp.structureInstanceTypes true\nset_option pp.instanceTypes true\nset_option pp.mvars.withType true\nset_option pp.coercions true\nset_option pp.funBinderTypes true\nset_option pp.piBinderTypes true'

        self.default_header = f'import Mathlib\nimport Aesop\n\nopen BigOperators Real Nat Topology Rat\n\n{self.options}'
    
    def preprocess_code(self, code):
        code = code.splitlines()
        for i, line in enumerate(code):
            if line.strip().startswith('theorem') or line.strip().startswith('lemma'):
                line = f'\n{self.options}\n' + line
                code[i] = line
                break
        code = '\n'.join(code)

        return code

    def run(self) -> str:
        # prepend additional options for explicit type ontations for variables
        code = self.preprocess_code(self.code)

        start_time = time.time()

        # 1) Attempt to fix the code (will recurse as necessary)
        self.fix_code(
            code=code,
            attempt=1,
            log_dir=self.log_dir,
            is_root=True
        )

        # 2) Once done, gather everything into a final proof
        proof_root = Path(self.log_dir)
        assembler = LeanProofAssembler(proof_root)
        final_proof_content = assembler.assemble_full_proof()

        # 3) Write out the final assembled proof
        output_file = proof_root / "assembled_main_theorem.lean"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_proof_content)

        print(f"Final assembled proof saved to: {output_file}")
        elapsed = time.time() - start_time
        print(f"ApolloRepair finished in {elapsed:.2f}s.")
        return str(output_file)

    def fix_code(self,
                 code: str,
                 attempt: int,
                 log_dir: str,
                 is_root: bool = False):
        if attempt > self.rec_depth:
            print(f"[{self.lemma_name}] Max attempts ({self.rec_depth}) exceeded. Stopping recursion.")
            return

        main_lean_path = os.path.join(log_dir, "main_theorem.lean")
        data_path = os.path.join(log_dir, f"rec_{attempt}.jsonl")

        # If we haven't already created main_theorem.lean in this attempt
        if not os.path.isfile(main_lean_path):
            # -- Syntax repair --
            code_corrected = SyntaxCorrector(code).correct_text()
            print(f"[{self.lemma_name}] Attempt {attempt}, after syntax fix:\n{code_corrected}\n")

            if not verify_lean4_file(code_corrected)['pass']:
                # -- Sorrify partial sub-proofs --
                pt = ProofTree(code_corrected)
                pt.parse_lean_with_dot_subcases()
                pt.fix_inline_have_text()

                tree = pt.tree
                checker = Sorrifier(pt, verify_lean4_file, clean_empty_lines=True, clean_comments=False)
                code_corrected_sorry = checker.verify_and_fix_tree()
                print(f"[{self.lemma_name}] After sorrification:\n{code_corrected_sorry}\n")
            else:
                code_corrected_sorry = code_corrected

            # -- Automatic proof repair --
            repairer = ProofRepairer(code_corrected_sorry, verify_lean4_file)
            final_code = repairer.repair_proof()
            print(f"[{self.lemma_name}] Final repaired code:\n{final_code}\n")

            # Save the best known main theorem for this attempt
            with open(main_lean_path, "w") as f:
                f.write(final_code)
        else:
            # If main_theorem.lean already exists from a previous call
            with open(main_lean_path, "r") as f:
                final_code = f.read()

        # Extract sub-lemmas only if we haven't yet saved them this attempt
        if not os.path.isfile(data_path):
            lemma_extractor = LemmaExtractor(final_code, verify_lean4_file)
            lemmas = lemma_extractor.get_lemmas()
            formatted_lemmas = get_deepseek_format_proofs(lemmas)

            print(f"[{self.lemma_name}] Attempt {attempt} - extracted {len(lemmas)} lemmas.")
            print(f"[{self.lemma_name}] Attempt {attempt} - formatted into {len(formatted_lemmas)} JSON lines.")

            with open(data_path, "w") as f:
                for item in formatted_lemmas:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Now run the LLM solver on the sub-lemmas
        launch_llm(data_path, self.config, log_dir)

        # Next, update or finalize the main theorem
        self.generate_main_theorem_files(log_dir, attempt)

    def generate_main_theorem_files(self, log_dir: str, attempt: int):
        for entry in os.listdir(log_dir):
            fullpath = os.path.join(log_dir, entry)

            # Skip direct files that are not pkl logs or subdirectories
            if os.path.isfile(fullpath):
                if not entry.endswith(".pkl"):
                    continue

            # If it's a directory that may contain .pkl logs or hint_repair info
            if os.path.isdir(fullpath):
                if 'run' in fullpath or 'hint_repair' in fullpath:
                    continue
                print(fullpath)
                success_file = glob.glob(os.path.join(fullpath, "success-*.pkl"))
                if success_file:
                    with open(success_file[0], "rb") as f:
                        data = pickle.load(f)[0]
                    verified_code = data["result"]["verified_code"]
                    with open(os.path.join(fullpath, "main_theorem.lean"), "w") as f:
                        f.write(verified_code)
                else:
                    failure_file = glob.glob(os.path.join(fullpath, "failure-*.pkl"))[0]
                    fails = self.extract_proof_from_deepseek_pkl(failure_file, log_dir)

                    data_jsonl = get_deepseek_format_proofs(fails)

                    success_path = os.path.join(fullpath, "hint_repair", "hint-solver-success.pkl")
                    failure_path = os.path.join(fullpath, "hint_repair", "hint-solver-failure.pkl")

                    if os.path.isfile(success_path):
                        # If successful hint repaired proof exists, then use it
                        with open(success_path, "rb") as f:
                            data = pickle.load(f)
                        verified_code = data[0]["result"]["verified_code"]
                        with open(os.path.join(fullpath, "main_theorem.lean"), "w") as f:
                            f.write(verified_code)
                    elif os.path.isfile(failure_path):
                        # If failed hint repaired proof exists, then use it for further recursive expansion
                        with open(failure_path, "rb") as f:
                            data = pickle.load(f)
                        
                        try:
                            code = data[0]["result"]["verified_code"]
                        except:
                            # If REPL completely failed, fallback on presaved but unchecked output... (usually happens if REPL crashed/timed out)
                            code = self.default_header + data[0]['formal_statement'] + '\n' + data[0]['proof_code']
                        self.fix_code(code, attempt + 1, fullpath, is_root=False)
                    else:
                        launch_regular_hint_search(data_jsonl, self.config, fullpath)

                        self.generate_main_theorem_files(log_dir, attempt)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def extract_header(self, code):
        header = []
        for line in code.splitlines():
            if any(k in line for k in ("import", "set_option", "open")):
                header.append(line)
        return '\n'.join(header)
    
    def extract_proof_from_deepseek_pkl(self, filepath, log_dir):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        try:
            fails = [d["result"]["verified_code"] for d in data]
        except:
            # If failure pkl file is corrupted/incomplete/REPL outputted nothing, then fallback on extracting the lemmas without relying on REPL
            fails = []
            with open(os.path.join(log_dir, 'main_theorem.lean'), 'r') as f:
                main_theorem = f.read()
            header = self.extract_header(main_theorem)
            for d in data:
                c = header + '\n\n' + d['formal_statement'] + d['proof_code']
                fails.append(c)
        return fails