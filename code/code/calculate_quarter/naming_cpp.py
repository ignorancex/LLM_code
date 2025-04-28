import os
import re
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import warnings

# Import tree-sitter components
from tree_sitter import Parser, Language
from tree_sitter_languages import get_language, get_parser # Helper to load languages

warnings.filterwarnings("ignore", category=SyntaxWarning) # Keep this if needed elsewhere

# === Define naming ways regular expressions ===
# (Keep the same patterns as before, they apply to identifiers generally)
naming_patterns = {
    "single_letter": r'^[a-zA-Z]$',
    "lowercase": r'^[a-z]+$',
    "UPPERCASE": r'^[A-Z]+$',
    "camelCase": r'^[a-z]+(?:[A-Z][a-z0-9]*)*$', # Adjusted for C/C++ common style
    "snake_case": r'^[a-z]+(?:_[a-z0-9]+)+$', # Adjusted for C/C++ common style
    "PascalCase": r'^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)*$', # Adjusted for C/C++ common style
    "UPPER_SNAKE_CASE": r'^[A-Z]+(?:_[A-Z0-9]+)+$', # Adjusted for C/C++ common style
    "endsWithDigits": r'^[A-Za-z_]+[0-9]+$',
    "startsWithUnderscore": r'^_[A-Za-z0-9_]*$', # Common C/C++ pattern
    "Other": r'.*' # Catch-all
}

def get_naming_pattern(name):
    """Classifies the naming convention of an identifier."""
    name = str(name)
    # Check specific patterns first
    if re.match(naming_patterns["startsWithUnderscore"], name):
        return "startsWithUnderscore"
    if re.match(naming_patterns["single_letter"], name):
        return "single_letter"
    if re.match(naming_patterns["lowercase"], name):
        return "lowercase"
    if re.match(naming_patterns["UPPERCASE"], name):
        return "UPPERCASE"
    if re.match(naming_patterns["camelCase"], name):
        return "camelCase"
    if re.match(naming_patterns["snake_case"], name):
        return "snake_case"
    if re.match(naming_patterns["PascalCase"], name):
        return "PascalCase"
    if re.match(naming_patterns["UPPER_SNAKE_CASE"], name):
        return "UPPER_SNAKE_CASE"
    if re.match(naming_patterns["endsWithDigits"], name):
        return "endsWithDigits"
    # Default to Other if no specific pattern matches
    return "Other"


# --- Tree-sitter Queries for C/C++ ---
# These queries target common function definitions and variable/parameter declarations.
# They might need refinement for very complex C++ code (templates, namespaces, etc.)

# Query for Function Definitions/Declarations (captures the function name)
# Captures identifiers within function declarators
C_CPP_FUNC_QUERY = """
(function_definition declarator: (function_declarator declarator: identifier) @function_name)
(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: identifier)) @function_name)
(declaration type: (_) declarator: (function_declarator declarator: identifier) @function_name) ;; Function prototype
"""

# Query for Variable/Parameter Declarations (captures the variable name)
# Captures identifiers in declaration specifiers (variables) and parameter declarations
C_CPP_VAR_QUERY = """
(declaration declarator: [
    (identifier) @variable_name
    (init_declarator declarator: (identifier) @variable_name)
    (pointer_declarator declarator: (identifier) @variable_name)
    (array_declarator declarator: (identifier) @variable_name)
])

(parameter_declaration declarator: [
    (identifier) @variable_name
    (pointer_declarator declarator: (identifier) @variable_name)
    (array_declarator declarator: (identifier) @variable_name)
])

; Capture field identifiers (struct/class members)
(field_declaration declarator: [
    (field_identifier) @variable_name
    (pointer_declarator declarator: (field_identifier) @variable_name)
    (array_declarator declarator: (field_identifier) @variable_name)
])
"""
# --- End Tree-sitter Queries ---


# Pre-compile queries for efficiency
C_LANG = get_language('c')
CPP_LANG = get_language('cpp')
FUNC_QUERY_C = C_LANG.query(C_CPP_FUNC_QUERY)
VAR_QUERY_C = C_LANG.query(C_CPP_VAR_QUERY)
FUNC_QUERY_CPP = CPP_LANG.query(C_CPP_FUNC_QUERY)
VAR_QUERY_CPP = CPP_LANG.query(C_CPP_VAR_QUERY)

# Initialize parsers once
parser_c = Parser()
parser_c.set_language(C_LANG)

parser_cpp = Parser()
parser_cpp.set_language(CPP_LANG)


def extract_code_info_ts(file_path, skipped_files_log):
    """Parses C/C++ code using tree-sitter, extracts function and variable names."""
    function_names = set()
    variable_names = set()
    parser = None
    func_query = None
    var_query = None

    # Determine language and select parser/queries
    if file_path.endswith(".c") or file_path.endswith(".h"):
        parser = parser_c
        func_query = FUNC_QUERY_C
        var_query = VAR_QUERY_C
        lang_name = "C"
    elif file_path.endswith(".cpp") or file_path.endswith(".hpp") or file_path.endswith(".cc") or file_path.endswith(".hh") or file_path.endswith(".cxx"):
        parser = parser_cpp
        func_query = FUNC_QUERY_CPP
        var_query = VAR_QUERY_CPP
        lang_name = "C++"
    else:
        return function_names, variable_names # Skip unsupported files silently

    try:
        with open(file_path, "rb") as f: # Read as bytes for tree-sitter
            code_bytes = f.read()

        # Basic check for null bytes which can cause issues
        if b"\x00" in code_bytes:
             with open(skipped_files_log, "a", encoding="utf-8") as log:
                 log.write(f"Skipped {file_path}: Contains null bytes\n")
             return set(), set()

        # Attempt to decode for name extraction later (best effort)
        try:
            code_str = code_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                code_str = code_bytes.decode('latin-1') # Try another common encoding
            except UnicodeDecodeError as e:
                 with open(skipped_files_log, "a", encoding="utf-8") as log:
                     log.write(f"Skipped {file_path}: Unicode decode error after trying utf-8 and latin-1 - {str(e)}\n")
                 return set(), set()


        tree = parser.parse(code_bytes)

        # --- Extract names using queries ---
        func_captures = func_query.captures(tree.root_node)
        var_captures = var_query.captures(tree.root_node)

        for node, capture_name in func_captures:
             # Get identifier text using node byte range and decoded string
            identifier = code_str[node.start_byte:node.end_byte]
            if identifier: # Ensure it's not empty
                function_names.add(identifier)

        # Add variable/parameter names
        # Note: This might capture some duplicates if a name is used as both func and var,
        # but sets handle uniqueness. It might also capture type names in some contexts
        # depending on query specifics, but should be reasonably accurate for variable names.
        for node, capture_name in var_captures:
            identifier = code_str[node.start_byte:node.end_byte]
            if identifier: # Ensure it's not empty
                 # Avoid adding names already identified as functions to the variable list
                 # (Helps distinguish, though C/C++ allows same name for var and func in different scopes)
                 # if identifier not in function_names: # Optional: keep them separate if needed
                 variable_names.add(identifier)


    except FileNotFoundError:
         with open(skipped_files_log, "a", encoding="utf-8") as log:
            log.write(f"Skipped {file_path}: File not found\n")
    except Exception as e: # Catch other potential errors (e.g., tree-sitter issues)
        with open(skipped_files_log, "a", encoding="utf-8") as log:
            log.write(f"Skipped {file_path}: Error during {lang_name} parsing - {str(e)}\n")
        return set(), set() # Return empty sets on error

    return function_names, variable_names


def process_project_ts(project_name, quarter_path, skipped_files_log):
    """Processes a single project for C/C++ files, returning local counts."""
    project_path = os.path.join(quarter_path, project_name)

    # No categories needed, just count patterns directly
    local_func_counts = defaultdict(int)
    local_var_counts = defaultdict(int)

    # Define C/C++ file extensions
    c_cpp_extensions = (".c", ".cpp", ".h", ".hpp", ".cc", ".hh", ".cxx")

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.lower().endswith(c_cpp_extensions):
                file_path = os.path.join(root, file)
                functions, variables = extract_code_info_ts(file_path, skipped_files_log)

                for name in functions:
                    pattern = get_naming_pattern(name)
                    local_func_counts[pattern] += 1
                for name in variables:
                    pattern = get_naming_pattern(name)
                    local_var_counts[pattern] += 1

    return local_func_counts, local_var_counts


# === Main Program ===
base_dir = "LLM_code/arxiv_dataset_cpp" # Keep your dataset path
output_dir = "LLM_code/naming_patterns_c_cpp" # New output directory
os.makedirs(output_dir, exist_ok=True)
skipped_files_log = os.path.join(output_dir, "skipped_files_c_cpp.txt")
if os.path.exists(skipped_files_log):
    os.remove(skipped_files_log) # Clear log on new run

# No need to load categories.json

# Final count structure: {quarter: {pattern: count}}
quarter_func_counts = defaultdict(lambda: defaultdict(int))
quarter_var_counts = defaultdict(lambda: defaultdict(int))

# Define relevant years and quarters
# Assuming the same year/quarter structure as before
start_year = 2020
end_year = 2025 # Process up to end of 2024 + Q1 2025
current_year = 2025 # Update if needed
current_quarter = 2 # Update if needed (e.g., 1 for Q1, 2 for Q2...)

for year in range(start_year, end_year + 1):
    # Determine the last quarter to process for the current year
    max_quarter = 4
    if year == current_year:
         max_quarter = current_quarter -1 # Process completed quarters
    if year > current_year:
        continue # Don't process future years


    for q in range(1, max_quarter + 1):
        quarter_name = f"Q{q}"
        year_str = str(year)
        quarter_key = f"{year_str}Q{q}"
        quarter_path = os.path.join(base_dir, year_str, quarter_name)

        if not os.path.isdir(quarter_path):
            print(f"⏭️ Skipping {quarter_key}: Directory not found at {quarter_path}")
            continue

        print(f"\n🔍 Processing {quarter_key}...")

        # List projects in the quarter directory
        try:
            project_list = [d for d in os.listdir(quarter_path) if os.path.isdir(os.path.join(quarter_path, d))]
        except FileNotFoundError:
             print(f"⏭️ Skipping {quarter_key}: Directory disappeared?")
             continue


        if not project_list:
            print(f"  -> No projects found in {quarter_key}.")
            continue

        # Use ThreadPoolExecutor for parallel processing of projects
        # Adjust max_workers based on your system's capabilities
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Submit tasks: process_project_ts doesn't need quarter_key or category info anymore
            futures = [executor.submit(process_project_ts, project_name, quarter_path, skipped_files_log)
                       for project_name in project_list]

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Scanning {quarter_key}"):
                try:
                    result = future.result()
                    if result: # Ensure result is not None (though shouldn't be in this version)
                        local_func_counts, local_var_counts = result
                        # Aggregate counts directly by pattern for the quarter
                        for pattern, count in local_func_counts.items():
                            quarter_func_counts[quarter_key][pattern] += count
                        for pattern, count in local_var_counts.items():
                            quarter_var_counts[quarter_key][pattern] += count
                except Exception as exc:
                    print(f'\n❗️ Project processing generated an exception: {exc}') # Log exceptions from threads

        print(f"✅ Finished {quarter_key}")

# --- Output Generation ---
# Output the raw counts per quarter per naming pattern.

# Ensure all defined patterns are present in the output, even if count is 0
all_patterns = list(naming_patterns.keys())

final_func_output = {}
for quarter, pattern_counts in sorted(quarter_func_counts.items()):
    final_func_output[quarter] = {pattern: pattern_counts.get(pattern, 0) for pattern in all_patterns}

final_var_output = {}
for quarter, pattern_counts in sorted(quarter_var_counts.items()):
    final_var_output[quarter] = {pattern: pattern_counts.get(pattern, 0) for pattern in all_patterns}


# Save the results as JSON files
func_output_path = os.path.join(output_dir, "naming_patterns_c_cpp_function_counts.json")
var_output_path = os.path.join(output_dir, "naming_patterns_c_cpp_variable_counts.json")

print(f"\n💾 Saving function name pattern counts to {func_output_path}")
with open(func_output_path, "w", encoding="utf-8") as f:
    json.dump(final_func_output, f, ensure_ascii=False, indent=2)

print(f"💾 Saving variable name pattern counts to {var_output_path}")
with open(var_output_path, "w", encoding="utf-8") as f:
    json.dump(final_var_output, f, ensure_ascii=False, indent=2)

print(f"\n🎉 All C/C++ processing completed. Results saved in {output_dir}")