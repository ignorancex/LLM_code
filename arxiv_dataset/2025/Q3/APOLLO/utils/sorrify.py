from collections import defaultdict
from tqdm import tqdm
import re
import sys


#%%
class ProofTree:
    def __init__(self, code: str):
        self.code = code
        self.tree = None
        
    def print_lean_tree(self, nodes=None, level=0):
        if not nodes:
            nodes = self.tree
            
        for node in nodes:
            indent_prefix = "  " * level
            print(f"{node['line']}:{indent_prefix}{node['text']}")
            if node["children"]:
                self.print_lean_tree(node["children"], level + 1)
    
    def assign_line_numbers(self, nodes=None, start=1):
        if nodes is None:
            nodes = self.tree
            
        for node in nodes:
            node["line"] = start
            start += 1
            if node["children"]:
                start = self.assign_line_numbers(node["children"], start)
        return start
    
    def get_line_by_number(self, line_num: int, nodes=None) -> dict:
        if nodes is None:
            nodes = self.tree
    
        for node in nodes:
            if node.get("line") == line_num:
                return node
            # Recursively search in the children
            result = self.get_line_by_number(line_num, node.get("children", []))
            if result is not None:
                return result
        return None
    
    def replace_node_by_line_number(self, line_num: int, new_node: dict, nodes=None) -> bool:
        if nodes is None:
            nodes = self.tree
            
        for i, node in enumerate(nodes):
            if node.get("line") == line_num:
                nodes[i] = new_node
                return True
            if self.replace_node_by_line_number(line_num, new_node, node.get("children", [])):
                return True
        
        return False
    
    def fix_induction_nodes(self, nodes=None) -> None:
        if nodes is None:
            nodes = self.tree
    
        for node in nodes:
            # Check if the current node's text begins with "induction "
            if node["text"].lstrip().startswith("induction "):
                child_count = len(node.get("children", []))
                if child_count == 1:
                    # Append a new child with text "sorry"
                    new_child = {
                        "indent": node["indent"] + 2,
                        "text": " " * (node["indent"] + 2) + "sorry",
                        "children": []
                    }
                    node["children"].append(new_child)
                elif child_count > 2:
                    # Remove extra children: keep only the first two
                    node["children"] = node["children"][:2]
            
            # Recursively process this node's children
            if node.get("children"):
                self.fix_induction_nodes(node["children"])
    
    def add_sorry_child_if_all_comments(self, nodes=None):
        if nodes is None:
            nodes = self.tree
    
        for node in nodes:
            children = node.get("children", [])
            
            if any(node['text'].lstrip().startswith(prefix) for prefix in ('·', '|', '.')):
                has_tactics = False
                if not children:
                    new_child = {
                        "indent": node["indent"]+2,
                        "text": " " * (node["indent"]+2) + "sorry",
                        "children": [],
                        "line": -1
                    }
                    children.append(new_child)
                else:
                    for child in children:
                        if not child['text'].strip() or child['text'].lstrip().startswith('--'):
                            continue
                        has_tactics=True
                    if not has_tactics:
                        new_child = {
                            "indent": children[-1]["indent"],
                            "text": " " * children[-1]["indent"] + "sorry",
                            "children": [],
                            "line": -1
                        }
                        children.append(new_child)
            elif children and node['text'].lstrip().startswith('have'):
                has_tactics = False
                for child in children:
                    if not child['text'].strip() or child['text'].lstrip().startswith('--'):
                        continue
                    has_tactics=True
                if not has_tactics:
                    new_child = {
                        "indent": children[-1]["indent"],
                        "text": " " * children[-1]["indent"] + "sorry",
                        "children": [],
                        "line": -1
                    }
                    children.append(new_child)
            elif node['text'].lstrip().startswith('have') and node['text'].lstrip().endswith('by') and not children:
                new_child = {
                    "indent": node["indent"]+2,
                    "text": " " * (node["indent"]+2) + "sorry",
                    "children": [],
                    "line": -1
                }
                children.append(new_child)
            if children:
                self.add_sorry_child_if_all_comments(children)
        return nodes
                
    def prepend_char_to_all_nodes(self, char: str, nodes=None):
        if nodes is None:
            nodes = self.tree
        
        if isinstance(nodes, dict):
            nodes = [nodes]
            
        for node in nodes:
            node["text"] = node['indent'] * ' ' + char + node["text"].lstrip()
            if node.get("children"):
                self.prepend_char_to_all_nodes(char, node["children"])
        
        return nodes
    
    def get_max_line_number(self, nodes=None) -> int:
        if nodes is None:
            nodes = self.tree
            
        max_line = 0
        for node in nodes:
            current_line = node.get("lineno", 0)
            max_line = max(max_line, current_line)
            if node.get("children"):
                max_line = max(max_line, self.get_max_line_number(node["children"]))
        return max_line
    
    def prepend_char_to_lines(self, char: str, line_numbers: set, nodes=None) -> None:
        if nodes is None:
            nodes = self.tree
    
        for node in nodes:
            # If the node has a line number and it is in the specified set, modify its text.
            if "line" in node and node["line"] in line_numbers:
                node["text"] = node['indent']*' ' + char + node["text"].lstrip()
            # Recurse on the children if they exist.
            if node.get("children"):
                self.prepend_char_to_lines(char, line_numbers, node["children"])
    
    def find_parent(self, target, nodes=None, parent=None):
        if nodes is None:
            nodes = self.tree
    
        for node in nodes:
            if node is target:
                return parent
            if "children" in node and node["children"]:
                result = self.find_parent(target, node["children"], node)
                if result is not None:
                    return result
        return None
    
    def retrieve_lean_tree_code(self, nodes=None, level=0) -> str:
        if not nodes:
            nodes = self.tree
            
        code_str = []
        for node in nodes:
            indent_prefix = "  " * level
            code_str.append(f"{indent_prefix}{node['text']}")
            if node["children"]:
                child_code = self.retrieve_lean_tree_code(node["children"], level + 1)
                code_str.append(child_code)
        return "\n".join(code_str)

    def parse_lean_with_dot_subcases(self, clean_empty_lines=False,
                                           clean_comments=False) -> list[dict]:
        
        def count_leading_spaces(s: str) -> int:
            return len(s) - len(s.lstrip(" "))
        
        lines = self.code.splitlines()
        root = {"indent": -1, "text": "<ROOT>", "children": []}
        stack = [root]
        
        for idx, line in enumerate(lines, start=1):
            if clean_empty_lines and not line.strip():
                continue
            
            indent = count_leading_spaces(line)
            stripped = line.lstrip(" ")
            node = {"indent": indent, "text": line, "children": []}
            
            if stripped.startswith("| "):
                header = None
                for cand in reversed(stack):
                    cand_str = cand["text"].lstrip()
                    if ((cand_str.startswith("cases ") and cand_str.endswith(" with")) or
                        (cand_str.startswith("match ") and cand_str.endswith(" with")) or 
                        (cand_str.startswith("inductive ") and cand_str.endswith(" where")) or
                        (cand_str.startswith("induction ") and cand_str.endswith(" with"))
                        
                        ):
                        header = cand
                        break
                if header is not None:
                    header["children"].append(node)
                    stack.append(node)
                    continue
            if stripped.startswith(". "):
                header = None
                for cand in reversed(stack):
                    cand_str = cand["text"].lstrip()
                    if (cand_str.startswith("by_cases ") or
                        cand_str.startswith("rcases ") or 
                        cand_str.startswith("cases' ") or
                        cand_str.startswith("induction' ") or
                        cand_str.startswith("interval_cases ")
                        ):
                        header = cand
                        break
                if header is not None:
                    header["children"].append(node)
                    stack.append(node)
                    continue

            if stack and stack[-1]["text"].lstrip().startswith("| "):
                if indent >= stack[-1]["indent"]:
                    stack[-1]["children"].append(node)
                    stack.append(node)
                    continue
                else:
                    while stack and stack[-1]["text"].lstrip().startswith("| "):
                        stack.pop()
            elif stack and stack[-1]["text"].lstrip().startswith(". "):
                if indent >= stack[-1]["indent"]:
                    stack[-1]["children"].append(node)
                    stack.append(node)
                    continue
                else:
                    while stack and stack[-1]["text"].lstrip().startswith(". "):
                        stack.pop()
            
            while stack and indent <= stack[-1]["indent"]:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
        
        self.tree = root["children"]
        self.normalize_indentation(self.tree, 0, 2)
        
        self.fix_inline_have_text()
        
        self.assign_line_numbers()
        return self.tree

    # -------------------------------------------------------------------------
    # POST-PROCESSING STEP
    # -------------------------------------------------------------------------
    def fix_inline_have_text(self, nodes=None) -> None:
        if nodes is None:
            nodes = self.tree  # start from the top

        for node in nodes:
            txt = node["text"]
            # Check conditions:
            if "have" in txt and " := by" in txt and node["children"]:
                # Parse out the portion after " := by"
                prefix, sep, after = txt.partition(" := by")
                extra_text = after.strip()
                if extra_text:
                    # 1) Update this node's text to end at " := by"
                    node["text"] = prefix + sep  # e.g. "have h : X := by"
                    
                    # 2) Create a new child node for that extra text
                    # We'll guess the child's indentation is node["indent"]+2
                    # but you can pick any offset you like
                    new_child = {
                        # "indent": node["indent"] + 2,
                        "indent": node["children"][0]["indent"],
                        "text": node["children"][0]["indent"] * ' ' +  extra_text,
                        "children": []
                    }
                    # 3) Insert it at the *front* of the children list
                    node["children"].insert(0, new_child)

            # Recurse
            if node["children"]:
                self.fix_inline_have_text(node["children"])
                
    def normalize_indentation(self, nodes, current_level=0, step=2):
        for node in nodes:
            # Remove old leading spaces
            stripped_text = node["text"].lstrip(" ")
            node["indent"] = current_level * step
            # Rebuild the text with new indentation
            node["text"] = (" " * node["indent"]) + stripped_text

            # Recurse on children
            if node["children"]:
                self.normalize_indentation(node["children"], current_level + 1, step)
#%%
class Sorrifier:
    """
    A class that sorrifies Lean code with REPL
    """
    def __init__(self, tree: ProofTree, verifier, verbose=False, pbar = True, 
                 clean_empty_lines=False, clean_comments=False):
        self.proof_tree = tree
        self.clean_empty_lines = clean_empty_lines
        self.clean_comments = clean_comments
        if not self.proof_tree.tree:
            self.proof_tree.parse_lean_with_dot_subcases(clean_empty_lines=self.clean_empty_lines,
                                                         clean_comments = self.clean_comments)
        self.verifier = verifier
        self.verbose = verbose
        self.pbar = pbar
    
        # We'll store lines like `import`, `set_option`, `open` in a separate header.
        self._header_code = []
        self.inductive_commands = ['interval_cases', 'cases ', 'by_cases ', 'rcases ', "cases' ", "induction' ", "match ", "inductive ", "induction "]
        
        
    # ----------------------------------------------------------------------
    # Public entry
    # ----------------------------------------------------------------------
    def verify_and_fix_tree(self):
        try:
            return self.sorrify_code()
        except Exception as e:
            main_goal = None
            print(f'Following Error occured in Sorrifier block: {e}')
            for idx, line in enumerate(self.proof_tree.code.splitlines(), start=1):
                if re.search(r":=\s*by", line):
                    main_goal = idx
                    break
            
            if not main_goal:
                raise AttributeError("Theorem is missing a header")
            
            fully_sorrified = self.proof_tree.code.split(':= by')[0]
            fully_sorrified += ':= by\n  sorry'
            return fully_sorrified
            # If any error occurs, then just sorrify everything

    def sorrify_code(self):
        main_goal = None
        main_goal_unsolved_goals = False
        
        for idx, line in enumerate(self.proof_tree.code.splitlines(), start=1):
            if re.search(r":=\s*by", line):
                main_goal = idx
                break
        if not main_goal:
            raise AttributeError("Theorem is missing a header")
        
        attempt = 1
        
        to_check = [l for l in self.proof_tree.code.splitlines()[main_goal:] if not l.strip().startswith('--')]
        attempt_num = min(75, len(to_check))

        if self.pbar:
            pbar = tqdm(desc="Fixing with Sorrifier", unit="cycle",
                        bar_format="{desc}: {n_fmt} cycles [{elapsed}]")
        
        err_info = self.verify_lean_code(self.proof_tree.code)
        
        if err_info['pass']:
            return self.proof_tree.code
        
        while not err_info['pass']:
            if attempt > attempt_num:
                break
            
            errors = self.clean_unsolved_goals(err_info['errors'])
            
            fixed_lines = []
            
            for err in errors:
                if 'unsolved goals' in err['data'] and err['pos']['line'] == main_goal and len(errors) == 1:
                    main_goal_unsolved_goals = True
                    break
                if err['pos']['line'] <= main_goal and 'unsolved goals' not in err['data']:
                    raise AssertionError('Problem statetement corrupted!')
                if "expected '{' or indented tactic sequence" in err['data']:
                    self.proof_tree.add_sorry_child_if_all_comments()
                    break
                    
                start = err['pos']['line']
                
                    
                if start in fixed_lines:
                    continue
                
                if err['endPos']:
                    end = err['endPos']['line']
                else:
                    end = start
                
                if start == end:
                    error_node = self.proof_tree.get_line_by_number(start)
                    fixed_node = self.identify_and_fix_error(error_node, err)
                    if not fixed_node:
                        continue
                    self.proof_tree.replace_node_by_line_number(line_num=start, new_node=fixed_node)
                elif start < end:
                    lines_to_fix = list(range(start, end+1))
                    fixed_nodes, lines_to_fix = self.fix_multi_line_error(lines_to_fix, err=err)
                    
                    assert len(lines_to_fix) == len(fixed_nodes)
                    for line, fixed_node in zip(lines_to_fix, fixed_nodes):
                        self.proof_tree.replace_node_by_line_number(line_num=line, new_node=fixed_node)
                    
                else:
                    raise ValueError("Invalid range: start value must not exceed end value.")
                    
                
                
                fixed_lines.append(start)
            
            # Handle a case when all children are just comments, then add a sorry to fix the error
            self.proof_tree.add_sorry_child_if_all_comments()
            
            code = self.proof_tree.retrieve_lean_tree_code()
            code = self.clean_comments_from_code(code)
            
            if main_goal_unsolved_goals:
                code_splitted = code.splitlines()
                # Put sorry only if it does not exist there
                for i, c in enumerate(code_splitted):
                    pattern = r":=\s*by"
                    match = re.search(pattern, c)
                    if match:
                        try:
                            indentation = len(code_splitted[i+1]) - len(code_splitted[i+1].lstrip())
                        except:
                            raise ValueError('Theorem/Lemma has no body. Unexpected end of proof.')
                        break
                code += '\n' + indentation*' ' + 'sorry'
            
                # Assemble and Disassmble the code
                pt = ProofTree(code)
                pt.parse_lean_with_dot_subcases(clean_empty_lines=self.clean_empty_lines,
                                                clean_comments=self.clean_comments)
                self.proof_tree = pt
            

            self.proof_tree.assign_line_numbers()
            code = self.proof_tree.retrieve_lean_tree_code()
            
            err_info = self.verify_lean_code(code)
            
            if self.pbar:
                pbar.update(1)
            attempt += 1

        if attempt > attempt_num:
            fully_sorrified = code.split(':= by')[0]
            fully_sorrified += ':= by\n  sorry'
            return fully_sorrified
        return code
            
    # ----------------------------------------------------------------------
    # (A) Identify and Fix Errors
    # ----------------------------------------------------------------------
    
    def identify_and_fix_error(self, error_node, err):
        if not error_node:
            return None
        error_line_stripped = error_node['text'].strip()
        if '--'*4 in error_line_stripped:
            raise ValueError('Fell into recursive trap, discard the proof!')
        if ("have" in error_line_stripped) and (":= by" in error_line_stripped) and not error_node["children"]:
            # Case of a have with oneliner proof
            fixed_node = self.fix_oneliner_have_subblock(error_node, err)
            return fixed_node
        elif ("have" in error_line_stripped) and (":= by" in error_line_stripped) and error_node["children"]:
            # Case of a have with children
            fixed_node = self.fix_multiliner_have_subblock(error_node, err)
            return fixed_node
        elif error_line_stripped.startswith('. ') or error_line_stripped.startswith('| '):
            fixed_node = self.fix_induction_child(error_node, err)
        # add inductions here too...
        elif not error_node['children']:
            # Regular line
            fixed_node = self.fix_regular_line(error_node, err)
            return fixed_node
        else:
            # If encounters atypical line, then nuke it
            error_node['text'] = ''
            error_node['children'] = []
            return error_node
            
    
    # ----------------------------------------------------------------------
    # (B) Handle Different Types of Errors and Sub-blocks
    # ----------------------------------------------------------------------
    
    def fix_oneliner_have_subblock(self, error_node, err):
        fixed_node = {}
        
        start = err['pos']['column']
        if err['endPos']:
            end = err['endPos']['column']
        else:
            end = start
        
        pattern = r":=\s*by"
        match = re.search(pattern, error_node['text'])
        
        if not match:
            raise ValueError(f"Pattern {pattern} not found in the string.")
        

        by_idx_start = match.start()
        by_idx_end = match.end()
        
        have_declaration = error_node['text'][:match.end()]
        proof = error_node['text'][match.end():]
        
        if start >= by_idx_end or 'unsolved goals' in err['data']:
            # If error in proof or unsolved goals left, then just set the whole proof to sorry
            fixed_node['line'] = error_node['line']
            fixed_node['text'] = have_declaration
            fixed_node['indent'] = error_node['indent']
            fixed_node['children'] = [{
                'text': (error_node['indent'] + 2) * ' ' + 'sorry',
                'indent': error_node['indent'] + 2,
                'children': [],
                'line': -1,
                }]
        else:
            # Otherwise nuke the line
            fixed_node['line'] = error_node['line']
            fixed_node['text'] = '--' + error_node['text']
            fixed_node['indent'] = error_node['indent']
            fixed_node['children'] = []
        
        return fixed_node
    
    def fix_multiliner_have_subblock(self, error_node, err):
        # If unsolved goal, just append sorry at the end
        if 'unsolved goals' in err['data']:
            fixed_node = self.fix_unsolved_goal(error_node, err)
            return fixed_node
        if 'fix_multiliner_have_subblock' in err['data']:
            sorry_child = {
                'text': (error_node['indent']+2) * ' ' + 'sorry',
                'indent': error_node['indent']+2,
                'children': [],
                'line': -1
                }
            error_node['children'] = [sorry_child]
            return sorry_child
        
        fixed_node = {}
        fixed_node['text'] = ''
        fixed_node['line'] = error_node['line']
        fixed_node['indent'] = error_node['indent']
        fixed_node['children'] = []
        
        
        
        if isinstance(fixed_node, list):
            if len(fixed_node) > 1:
                raise ValueError('Handler received a node with multiple entries, entry length 1 is expected')
            fixed_node = fixed_node[0]
        
        return fixed_node
        
        
            
    
    def fix_regular_line(self, error_node, err):
        # Might need future adjustments...
        if 'unexpected end of input; expected' in err['data']:
            # Handle this particular error in a regular line
            fixed_node, _ = self.fix_unexpected_eof(err)
            return fixed_node
        if 'no goals to be solved' in err['data']:
            error_node['text'] = ''
            error_node['children'] = []
            return error_node
        if 'cases' in error_node['text'] and "cases'" not in error_node['text']:
            # Try cases' before nuking the line
            error_node['text'] = error_node['text'].replace('cases', "cases'")
            return error_node
        
        fixed_node = {}
        
        fixed_node['text'] = ''
        fixed_node['indent'] = error_node['indent']
        fixed_node['children'] = []
        fixed_node['line'] = error_node['line']
        
        return fixed_node
    
    def fix_induction_child(self, error_node, err):
        if 'unsolved goals' in err['data']:
            fixed_node = self.fix_unsolved_goal(error_node, err)
            return fixed_node
        # Write more logic if necessary
        fixed_node = {}
        

        fixed_node['text'] = '' 
        fixed_node['indent'] = error_node['indent']
        fixed_node['children'] = []
        fixed_node['line'] = error_node['line']
        
        return fixed_node
    
    def fix_multi_line_error(self, lines_to_fix: list, err = None):
        '''
        Comment out problematic lines
        '''
        if 'unsolved goals' in err['data']:
            info = self.fix_multiline_unsolved_goal(err)
            lines_to_fix = info['lines_to_fix']
            fixed_nodes = info['fixed_nodes']
            return fixed_nodes, lines_to_fix
        
        if 'unexpected end of input; expected' in err['data']:
            fixed_nodes, lines_to_fix = self.fix_unexpected_eof(err)
            
            return fixed_nodes, lines_to_fix
            
        
        fixed_nodes = []
        
        for line in lines_to_fix:
            error_node = self.proof_tree.get_line_by_number(line)
            if not error_node:
                raise TypeError("Error Node is None!")
            error_node['text'] = error_node['indent']*' ' + '--' + error_node['text'].lstrip()
            
            fixed_nodes.append(error_node)
            
        return fixed_nodes, lines_to_fix
        
    
    
    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    
    def fix_unsolved_goal(self, node, err=None):
        if node['children']:
            sorry_child = {
                'text': node['children'][-1]['indent'] * ' ' + 'sorry',
                'indent': node['children'][-1]['indent'],
                'children': [],
                'line': -1
                }
            node['children'].append(sorry_child)
        else:
            node['text'] = node['text'].split(':= by')[0] + ':= by'
            sorry_child = {
                'text': (node['indent'] + 2) * ' ' + 'sorry',
                'indent': node['indent']+2,
                'children': [],
                'line': -1
                }
            
        return node
    
    def fix_multiline_unsolved_goal(self, err):
        start = err['pos']['line']
        end = err['endPos']['line']
        start_node = self.proof_tree.get_line_by_number(start)
        
        assert start < end
        end_node = self.proof_tree.get_line_by_number(end)
        
        if start_node['children']:
            # Case 1: start node has children, which means that unsolved goal can be fixed by appending sorry as a last child
            
            sorry_child = {
                'text': start_node['children'][-1]['indent'] * ' ' + 'sorry',
                'indent': start_node['children'][-1]['indent'],
                'children': [],
                'line': -1
                }
            start_node['children'].append(sorry_child)
            return {'fixed_nodes': [start_node], 'lines_to_fix': [start]}
        else:
            # Case 2: start node has no children, which means that unsolved goal can be fixed by appending sorry right after the endPos line
            end_node = self.fix_unsolved_goal(end_node)
            return {'fixed_nodes': [end_node], 'lines_to_fix': [end]}
            
    def fix_unexpected_eof(self, err):
        start = err['pos']['line']
        try:
            end = err['endPos']['line']
        except:
            end = None
        
        if end:
            end_node = self.proof_tree.get_line_by_number(end)
            end_node['text'] = end_node['indent'] * ' ' + 'sorry'
            
            lines_to_fix = [end]
            
            return end_node, lines_to_fix
        else:
            
            self.proof_tree.add_sorry_child_if_all_comments()
            
            return None, None
        
        
    
    def clean_unsolved_goals(self, errors):
        if all("unsolved goals" in error['data'] for error in errors):
            return errors
        else:
            return [error for error in errors if "unsolved goals" not in error['data']]
    
    def clean_no_goals_to_be_solved(self, errors):
        if any("no goals to be solved" in error['data'] for error in errors):
            return errors
        else:
            return [error for error in errors if "unsolved goals" not in error['data']]
        
    def clean_comments_from_code(self, code):
        lines = code.splitlines()
        clean_lines = []
        for line in lines:
            if line.lstrip().startswith('--'):
                continue
            clean_lines.append(line)
        return '\n'.join(clean_lines)
        
    
    def verify_lean_code(self, code):
        return self.verifier(code)
    
class LeanServerSorrifier(Sorrifier):
    def verify_lean_code(self, code):
        request_id = self.verifier.verifier_submit_request(code)
        result_list = self.verifier.verifier_get_all_request_outputs([request_id])
        return result_list[0]
























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    