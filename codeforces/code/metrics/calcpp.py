import csv
import re, math, sys
import lizard
import json
from statistics import mean

def count_comment_lines(code: str) -> int:
    """
    统计 C++ 源码中的注释行数，包括：
     - 单行注释 // 开头
     - 多行注释 /* ... */ 中的所有行
    """
    lines = code.splitlines()
    comment_lines = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if in_block:
            comment_lines += 1
            if '*/' in stripped:
                in_block = False
            continue
        if stripped.startswith('//'):
            comment_lines += 1
        elif '/*' in stripped:
            comment_lines += 1
            if '*/' not in stripped:
                in_block = True
    return comment_lines

def scan_halstead(code: str):
    """
    扫描 C++ 代码，统计 Halstead 操作符/操作数。
    返回 h1, h2, N1, N2, vocabulary, length, calculated_length, V, D, E, T, B
    """
    op_re = re.compile('\\+\\+|--|->|==|!=|<=|>=|\\+=|-=|\\*=|/=|&&|\\|\\||[+\\-*/%<>&|^~!=]=?|::|\\.|\\?|:|!')
    tokens = re.split('(\\W)', code)
    ops = {}
    operands = {}
    N1 = N2 = 0
    for tok in tokens:
        if not tok or tok.isspace():
            continue
        if op_re.fullmatch(tok):
            ops[tok] = ops.get(tok, 0) + 1
            N1 += 1
        else:
            operands[tok] = operands.get(tok, 0) + 1
            N2 += 1
    h1 = len(ops)
    h2 = len(operands)
    vocabulary = h1 + h2
    length = N1 + N2
    calc_len = 0.0
    if h1 > 0:
        calc_len += h1 * math.log2(h1)
    if h2 > 0:
        calc_len += h2 * math.log2(h2)
    V = length * math.log2(vocabulary) if vocabulary > 0 else 0.0
    D = h1 / 2.0 * (N2 / h2) if h2 > 0 else 0.0
    E = D * V
    T = E / 18.0
    B = V / 3000.0
    return {'h1': h1, 'h2': h2, 'N1': N1, 'N2': N2, 'vocabulary': vocabulary, 'length': length, 'calculated_length': calc_len, 'volume': V, 'difficulty': D, 'effort': E, 'time_sec': T, 'bugs': B}

def compute_mi_std(V, G, L, comment_rate):
    """
    标准 Visual Studio MI（含注释项），归一化到 0–100
    MI = max(0, (171 -5.2 ln(V) -0.23 G -16.2 ln(L) +50 sin(sqrt(2.4 C)))/171 *100)
    C = comment_rate*100（度）
    """
    Cdeg = comment_rate * 100.0
    C = math.radians(Cdeg)
    raw = 171.0 - 5.2 * math.log(V or 1) - 0.23 * G - 16.2 * math.log(L or 1) + 50.0 * math.sin(math.sqrt(2.4 * C))
    return max(0.0, raw * 100.0 / 171.0)

def compute_mi_custom(V, G, L):
    """
    简化版 MI（不含注释项），归一化到 0–100
    MI_custom = max(0, (171 -5.2 ln(V) -0.23 G -16.2 ln(L))/171 *100)
    """
    raw = 171.0 - 5.2 * math.log(V or 1) - 0.23 * G - 16.2 * math.log(L or 1)
    return max(0.0, raw * 100.0 / 171.0)

def analyze_code(code: str):
    lines = code.splitlines()
    sloc = sum((1 for ln in lines if ln.strip()))
    ana = lizard.analyze_file.analyze_source_code('filename.cpp', code)
    cyclomatic = sum((fn.cyclomatic_complexity for fn in ana.function_list))
    lloc = sum((fn.nloc for fn in ana.function_list))
    comment_lines = count_comment_lines(code)
    comment_rate = comment_lines / (comment_lines + lloc) if comment_lines + lloc > 0 else 0.0
    hal = scan_halstead(code)
    mi_std = compute_mi_std(hal['volume'], cyclomatic, sloc, comment_rate)
    mi_custom = compute_mi_custom(hal['volume'], cyclomatic, sloc)
    return {'h1': hal['h1'], 'h2': hal['h2'], 'N1': hal['N1'], 'N2': hal['N2'], 'vocabulary': hal['vocabulary'], 'length': hal['length'], 'calculated_length': round(hal['calculated_length'], 2), 'volume': round(hal['volume'], 2), 'difficulty': round(hal['difficulty'], 2), 'effort': round(hal['effort'], 2), 'time_sec': round(hal['time_sec'], 2), 'bugs': round(hal['bugs'], 4), 'cyclomatic': cyclomatic, 'sloc': sloc, 'lloc': lloc, 'comment_rate': round(comment_rate, 2), 'mi_std': round(mi_std, 2), 'mi_custom': round(mi_custom, 2)}

def process_field(records, field_name, output_csv):
    if not records:
        return
    sample_metrics = analyze_code(records[0][field_name])
    headers = ['submission_id'] + list(sample_metrics.keys())
    rows = []
    cols = {k: [] for k in sample_metrics.keys()}
    for rec in records:
        sid = rec.get('submission_id', '<no_id>')
        code = rec.get(field_name, '')
        try:
            metrics = analyze_code(code)
        except Exception as e:
            continue
        for (k, v) in metrics.items():
            cols[k].append(v)
        rows.append([sid] + [metrics[k] for k in sample_metrics.keys()])
    if rows:
        avg_row = ['average'] + [round(mean(cols[k]), 4) for k in sample_metrics.keys()]
    else:
        avg_row = ['average'] + [''] * len(sample_metrics)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(avg_row)
if __name__ == '__main__':
    with open('../gemma_27b_cpp_extract.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data
    process_field(records, 'sourceCode', 'gemma_AC_cpp.csv')
    process_field(records, 'generate_code_block', 'gemma_ANS_cpp.csv')
    process_field(records, 'generate_ref_code_block', 'gemma_REF_cpp.csv')