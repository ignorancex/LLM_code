import re


class SyntaxCorrector:
    """
    A baseline class to automatically correct specific text patterns.
    """

    def __init__(self, code: str, verifier = None, error_message: str = None):
        self.code = code
        self.verifier = verifier
        
    def correct_rw(self):
        pattern = r"(rw\s+)(?!\[[^]]*\])(.*?)(?=\s*(?:at|\n))"
        self.code = re.sub(pattern, r"\1[\2]", self.code)
        
    def correct_by_statements(self):
        pattern_repeating_whitespace = re.compile(r':=\s+by')
        pattern_missing_by = re.compile(r':=(?!\s*by\b)')
        correct_header_pattern = re.compile(r"\(([^)]*:= by[^)]*)\)")

        # Replace it with := by (only a single space)
        self.code = pattern_repeating_whitespace.sub(':= by', self.code)
        self.code = pattern_missing_by.sub(':= by ', self.code)
        
        pattern_lean3_by = re.compile(r',\s+by')
        self.code = pattern_lean3_by.sub(':= by', self.code)

        def replacer(match):
            content = match.group(1)  # <-- Now group(1) actually exists
            # Replace ":= by" with ":="
            content = content.replace(":= by", ":=")
            # Put parentheses back
            return f"({content})"
        
        self.code = re.sub(correct_header_pattern, replacer, self.code)

        
    
    def correct_by_statements_in_let(self):
        t = ''
        pattern = re.compile(r":=\s*by")
        
        for c in self.code.split('\n'):
            if 'let' in c:
                t += pattern.sub(":=", c) + '\n'
            else:
                t += c + '\n'
        self.code = t
        
    def remove_comments(self):
        lines = self.code.splitlines()
        
        lines = [line for line in lines if not line.lstrip().startswith('--')]
        
        self.code = '\n'.join(lines)
            
    def correct_multiple_line_have_statements(self):
        pattern = r"(have\s+.*?=)\s*\n\s*(.*?)(\s*:=\s*by)"
        replacement = r"\1 \2\3"
        self.code = re.sub(pattern, replacement, self.code, flags=re.DOTALL)
    
    def remove_redundant_spaces(self):
        def clean_line(line: str) -> str:
            # Split the line into indentation (leading spaces) and the rest of the content.
            match = re.match(r'^(\s*)(.*)$', line)
            if match:
                indent, content = match.groups()
                # Replace sequences of whitespace in the content with a single space.
                cleaned_content = re.sub(r'\s+', ' ', content)
                return indent + cleaned_content
            return line  # If no match (unlikely), return the line unchanged.
        self.code = "\n".join(clean_line(line) for line in self.code.splitlines())
    
    def remove_large_comment_blocks(self):
        pattern = r'/-.*?-/'

        # Use re.sub with the DOTALL flag to remove the blocks
        self.code = re.sub(pattern, '', self.code, flags=re.DOTALL)

    
    def remove_commas(self):
        pattern = re.compile(r",\s*$", flags=re.MULTILINE)

        self.code = pattern.sub("", self.code)
    
    def remove_newlines_inside_brackets(self):
        """
        Find bracketed substrings [ ... ] possibly spanning multiple lines,
        remove newline characters from inside, and re-insert them intact.
        """
        pattern = r'(have.*?:= by)'
    
        # Define a replacement function that removes newlines from the matched substring
        def repl(match):
            substring = match.group(1)
            return substring.replace('\n', '')
        
        # Use re.sub with the replacement function and DOTALL flag to handle newlines
        self.code = re.sub(pattern, repl, self.code, flags=re.DOTALL)
        
    def count_indentation(self, line):
        return len(line) - len(line.lstrip())
    
    def normalise_newline_hypothesis(self):
        lines = self.code.splitlines()
        code = []
        flag = False
        for i, line in enumerate(lines):
            
            if line.lstrip().startswith('have') and not line.lstrip().endswith(':= by'):
                parts = line.split(':= by')
                if i+1 < len(lines):
                    indentation = self.count_indentation(lines[i+1])
                    if indentation < self.count_indentation(lines[i]):
                        indentation = self.count_indentation(lines[i]) + 2
                else:
                    indentation = self.count_indentation(lines[i]) + 2
                try:
                    line = parts[0] + ' := by\n' + indentation *' ' + parts[1] + '\n'
                except:
                    continue
            code.append(line)
        self.code = '\n'.join(code)
        
    ### o3-mini/o4-mini specific corrections
    
    def curly_bracket_fix(self, line):            
        if '{' in line and '}' not in line:
            return line.replace('{', '')
        elif '}' in line and '{' not in line:
            return line.replace('}', '')
        else:
            return line
        
    def normalize_indentation(self):
        lines = self.code.splitlines()
        indents = []
        corrected_lines = []
        
        for line in lines:
            ind = self.count_indentation(line)
            if ind % 2 != 0:
                ind += 1 
            indents.append(ind)
        for ind, line in zip(indents, lines):
            corrected_lines.append(ind * ' ' + line.lstrip())
        
        self.code = '\n'.join(corrected_lines)
            
    def specific_corrections(self):
        lines = self.code.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith('rw') or line.lstrip().startswith('intro'):
                for rogue_string in [':=']:
                    line = line.replace(':=', '')
                lines[i] = line
            if any(identifier for identifier in ['by', 'do', 'begin', 'end', 'exact', 'calc', 'if', 'else'] if identifier == line.strip()):
                lines[i] = ''
                
            
        self.code = '\n'.join(lines)
        
            

    def correct_text(self) -> str:
        """
        Applies the stored corrections to the provided text.
        :param text: The original text to correct.
        :return: The corrected text.
        """
        # o3-mini specific corrections
        self.specific_corrections()
        
        
        self.remove_large_comment_blocks()
        self.remove_comments()
        
        
        self.code = self.code.replace('split', 'constructor').replace('admit', 'sorry').replace('/--', '/-')
        self.correct_rw()
        self.correct_by_statements()
        self.correct_by_statements_in_let()
        self.remove_newlines_inside_brackets()
        self.remove_commas()
        self.correct_multiple_line_have_statements()
        self.remove_redundant_spaces()
        

        self.normalise_newline_hypothesis()
        
        
        self.code = '\n'.join([l for l in self.code.splitlines() if l.strip() != ''])
        self.normalize_indentation()
        
        
        return self.code