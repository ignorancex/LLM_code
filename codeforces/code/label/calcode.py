import json
import re
import pandas as pd
from collections import defaultdict
import argparse

def extract_identifiers_python(data):
    """
    Extract function and variable names from Python code blocks.
    Returns two dicts: funcs_stats and vars_stats, each: {'AC': {...}, 'ANS': {...}, 'REF': {...}}
    """
    py_func = re.compile('def\\s+(\\w+)\\s*\\(')
    assign_var = re.compile('(?<!def\\s)(?<!class\\s)\\b(\\w+)\\b\\s*=')
    funcs = {k: defaultdict(lambda : {'total': 0, 'count': 0}) for k in ('AC', 'ANS', 'REF')}
    vars_ = {k: defaultdict(lambda : {'total': 0, 'count': 0}) for k in ('AC', 'ANS', 'REF')}
    for entry in data:
        blocks = {k: entry.get(field, '') for (k, field) in zip(('AC', 'ANS', 'REF'), ('sourceCode', 'generate_code_block', 'generate_ref_code_block'))}
        for (key, code) in blocks.items():
            seen_funcs = set()
            seen_vars = set()
            for name in py_func.findall(code):
                occ = code.count(name)
                funcs[key][name]['total'] += occ
                if name not in seen_funcs:
                    funcs[key][name]['count'] += 1
                    seen_funcs.add(name)
            for name in assign_var.findall(code):
                occ = code.count(name)
                vars_[key][name]['total'] += occ
                if name not in seen_vars:
                    vars_[key][name]['count'] += 1
                    seen_vars.add(name)
    return (funcs, vars_)

def extract_identifiers_cpp(data):
    """
    Extract function and variable names from C/C++ code blocks.
    Returns two dicts: funcs_stats and vars_stats.
    """
    cpp_func = re.compile('\\b(?:[\\w:<>]+)\\s+(\\w+)\\s*\\([^)]*\\)\\s*\\{')
    cpp_var = re.compile('\\b(?:int|float|double|char|bool|string|auto|long|short|void)\\s+\\*?(\\w+)')
    cpp_types = {'int', 'float', 'double', 'char', 'bool', 'string', 'auto', 'long', 'short', 'void'}
    cpp_keywords = {'if', 'for', 'while', 'switch', 'case', 'return', 'else', 'do', 'goto', 'sizeof', 'typedef', 'static', 'const', 'class', 'struct', 'union', 'namespace', 'public', 'protected', 'private', 'continue', 'break', 'default', 'delete', 'new', 'this', 'operator', 'template', 'typename', 'try', 'catch', 'throw', 'using', 'virtual', 'override', 'constexpr', 'friend', 'explicit', 'export', 'extern', 'inline', 'mutable', 'register', 'static_cast', 'reinterpret_cast', 'const_cast', 'dynamic_cast', 'volatile', 'enum'}
    funcs = {k: defaultdict(lambda : {'total': 0, 'count': 0}) for k in ('AC', 'ANS', 'REF')}
    vars_ = {k: defaultdict(lambda : {'total': 0, 'count': 0}) for k in ('AC', 'ANS', 'REF')}
    for entry in data:
        blocks = {k: entry.get(field, '') for (k, field) in zip(('AC', 'ANS', 'REF'), ('sourceCode', 'generate_code_block', 'generate_ref_code_block'))}
        for (key, code) in blocks.items():
            seen_funcs = set()
            seen_vars = set()
            for name in cpp_func.findall(code):
                if name in cpp_keywords or name in cpp_types:
                    continue
                occ = code.count(name)
                funcs[key][name]['total'] += occ
                if name not in seen_funcs:
                    funcs[key][name]['count'] += 1
                    seen_funcs.add(name)
            for name in cpp_var.findall(code):
                if name in cpp_keywords or name in cpp_types:
                    continue
                occ = code.count(name)
                vars_[key][name]['total'] += occ
                if name not in seen_vars:
                    vars_[key][name]['count'] += 1
                    seen_vars.add(name)
    return (funcs, vars_)

def consolidate_stats(stats):
    """
    Consolidate a stats dict into a pandas DataFrame with ratio columns.
    """
    rows = []
    keys = ('AC', 'ANS', 'REF')
    names = set(stats['AC']) | set(stats['ANS']) | set(stats['REF'])
    for name in names:
        ac_tot = stats['AC'][name]['total']
        ac_cnt = stats['AC'][name]['count']
        ans_tot = stats['ANS'][name]['total']
        ans_cnt = stats['ANS'][name]['count']
        ref_tot = stats['REF'][name]['total']
        ref_cnt = stats['REF'][name]['count']
        total_tot = ac_tot + ans_tot + ref_tot
        total_cnt = ac_cnt + ans_cnt + ref_cnt
        rows.append({'name': name, 'ac_total': ac_tot, 'ac_count': ac_cnt, 'ans_total': ans_tot, 'ans_count': ans_cnt, 'ref_total': ref_tot, 'ref_count': ref_cnt, 'total_total': total_tot, 'total_count': total_cnt, 'ans_total_ratio': ans_tot / ac_tot if ac_tot else None, 'ref_total_ratio': ref_tot / ac_tot if ac_tot else None, 'ans_count_ratio': ans_cnt / ac_cnt if ac_cnt else None, 'ref_count_ratio': ref_cnt / ac_cnt if ac_cnt else None})
    df = pd.DataFrame(rows)
    return df.sort_values('total_total', ascending=False)

def main():
    input_file = 'deepseek_32b_cpp_extract.json'
    output_file_vars = 'deepseek_32b_cpp_vars.csv'
    output_file_funcs = 'deepseek_32b_cpp_funcs.csv'
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    (funcs, vars_) = extract_identifiers_cpp(data)
    df_funcs = consolidate_stats(funcs)
    df_vars = consolidate_stats(vars_)
    df_funcs.to_csv(output_file_funcs, index=False)
    df_vars.to_csv(output_file_vars, index=False)
if __name__ == '__main__':
    main()