import json
import csv
import traceback
from radon.metrics import h_visit, mi_visit, mi_parameters, mi_compute
from radon.raw import analyze as raw_analyze
from radon.visitors import ComplexityVisitor
from statistics import mean

def analyze_code(code: str):
    """
    返回一个 dict，包括：
      - Halstead Overall: h1, h2, N1, N2, vocabulary, length, calculated_length, volume, difficulty, effort, time, bugs
      - Cyclomatic Complexity (总和)
      - SLOC, LLOC, comment_rate
      - MI Score (标准版)
      - MI Custom (重算版)
    """
    hal = h_visit(code)
    total = hal.total
    visitor = ComplexityVisitor.from_code(code)
    cyclomatic = sum([block.complexity for block in visitor.blocks])
    raw = raw_analyze(code)
    sloc = raw.sloc
    lloc = raw.lloc
    comment_rate = raw.comments / raw.sloc * 100 if raw.sloc else 0.0
    mi_std = mi_visit(code, True)
    (hal_vol, _, _, _) = mi_parameters(code)
    mi_custom = mi_compute(hal_vol, cyclomatic, lloc, comment_rate)
    return {'h1': total.h1, 'h2': total.h2, 'N1': total.N1, 'N2': total.N2, 'vocabulary': total.vocabulary, 'length': total.length, 'calculated_length': round(total.calculated_length, 2), 'volume': round(total.volume, 2), 'difficulty': round(total.difficulty, 2), 'effort': round(total.effort, 2), 'time_sec': round(total.time, 2), 'bugs': round(total.bugs, 4), 'cyclomatic': cyclomatic, 'sloc': sloc, 'lloc': lloc, 'comment_rate': round(comment_rate, 2), 'mi_std': round(mi_std, 2), 'mi_custom': round(mi_custom, 2)}

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
    with open('../deepseek_32b_python_extract.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data
    process_field(records, 'sourceCode', 'deepseek_AC.csv')
    process_field(records, 'generate_code_block', 'deepseek_ANS.csv')
    process_field(records, 'generate_ref_code_block', 'deepseek_REF.csv')