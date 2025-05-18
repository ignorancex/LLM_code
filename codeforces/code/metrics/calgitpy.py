import os
import csv
from radon.metrics import h_visit, mi_visit, mi_parameters, mi_compute
from radon.visitors import ComplexityVisitor

def analyze_py(path: str):
    """对单个 .py 文件计算各项指标，出错则返回 None。"""
    try:
        code = open(path, encoding='utf-8').read()
        hal = h_visit(code)
        t = hal.total
        mi_std = mi_visit(code, True)
        (hal_vol, cyclo, sloc, comment_rate) = mi_parameters(code)
        mi_custom = mi_compute(hal_vol, cyclo, sloc, comment_rate)
        return {'h1': t.h1, 'h2': t.h2, 'N1': t.N1, 'N2': t.N2, 'vocabulary': t.vocabulary, 'length': t.length, 'calculated_length': t.calculated_length, 'volume': t.volume, 'difficulty': t.difficulty, 'effort': t.effort, 'time_sec': t.time, 'bugs': t.bugs, 'cyclomatic': cyclo, 'sloc': sloc, 'lloc': sloc, 'comment_rate': comment_rate, 'mi_std': mi_std, 'mi_custom': mi_custom}
    except Exception:
        return None

def average_dicts(dicts):
    """将一组同结构字典求平均。"""
    if not dicts:
        return None
    keys = dicts[0].keys()
    avg = {k: sum((d[k] for d in dicts)) / len(dicts) for k in keys}
    return avg

def main(root='../../LLM_code/arxiv_dataset', out_csv='github_py_metrics.csv'):
    fields = ['quarter', 'h1', 'h2', 'N1', 'N2', 'vocabulary', 'length', 'calculated_length', 'volume', 'difficulty', 'effort', 'time_sec', 'bugs', 'cyclomatic', 'sloc', 'lloc', 'comment_rate', 'mi_std', 'mi_custom']
    rows = []
    for year in range(2020, 2021):
        quarters = ['Q1', 'Q2', 'Q3', 'Q4'] if year < 2020 else ['Q1']
        for q in quarters:
            path = os.path.join(root, str(year), q)
            metrics = []
            for (dirpath, _, files) in os.walk(path):
                for fn in files:
                    if fn.endswith('.py'):
                        full = os.path.join(dirpath, fn)
                        res = analyze_py(full)
                        if res:
                            metrics.append(res)
            avg = average_dicts(metrics)
            if avg:
                avg['quarter'] = f'{year}{q}'
                rows.append(avg)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()