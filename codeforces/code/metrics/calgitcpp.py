import os
import re
import math
import sys
import csv
import traceback
import lizard


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
            if "*/" in stripped:
                in_block = False
            continue

        if stripped.startswith("//"):
            comment_lines += 1
        elif "/*" in stripped:
            comment_lines += 1
            if "*/" not in stripped:
                in_block = True

    return comment_lines


def scan_halstead(code: str):
    """
    扫描 C++ 代码，统计 Halstead 操作符/操作数。
    """
    op_re = re.compile(
        r"\+\+|--|->|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\||"
        r"[+\-*/%<>&|^~!=]=?|::|\.|\?|:|!"
    )
    tokens = re.split(r"(\W)", code)
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
    D = (h1 / 2.0) * (N2 / h2) if h2 > 0 else 0.0
    E = D * V
    T = E / 18.0
    B = V / 3000.0

    return {
        "h1": h1, "h2": h2, "N1": N1, "N2": N2,
        "vocabulary": vocabulary, "length": length,
        "calculated_length": calc_len,
        "volume": V, "difficulty": D,
        "effort": E, "time_sec": T, "bugs": B
    }


def compute_mi_std(V, G, L, comment_rate):
    """
    标准 Visual Studio MI（含注释项），归一化到 0–100
    """
    Cdeg = comment_rate * 100.0
    C = math.radians(Cdeg)
    raw = (171.0
           - 5.2 * math.log(V or 1)
           - 0.23 * G
           - 16.2 * math.log(L or 1)
           + 50.0 * math.sin(math.sqrt(2.4 * C)))
    return max(0.0, raw * 100.0 / 171.0)


def compute_mi_custom(V, G, L):
    """
    简化版 MI（不含注释项），归一化到 0–100
    """
    raw = 171.0 \
          - 5.2 * math.log(V or 1) \
          - 0.23 * G \
          - 16.2 * math.log(L or 1)
    return max(0.0, raw * 100.0 / 171.0)


def analyze_cpp_file(path: str):
    """
    分析单个 C++ 文件，返回指标或抛出异常。
    """
    code = open(path, encoding='utf-8', errors='ignore').read()
    lines = code.splitlines()
    sloc = sum(1 for ln in lines if ln.strip())

    ana = lizard.analyze_file(path)
    cyclomatic = sum(fn.cyclomatic_complexity for fn in ana.function_list)
    lloc = sum(fn.nloc for fn in ana.function_list)

    comment_lines = count_comment_lines(code)
    comment_rate = comment_lines / (comment_lines + lloc) if (comment_lines + lloc) > 0 else 0.0

    hal = scan_halstead(code)

    mi_std = compute_mi_std(hal["volume"], cyclomatic, sloc, comment_rate)
    mi_custom = compute_mi_custom(hal["volume"], cyclomatic, sloc)

    # 合并结果
    result = {
        **hal,
        "cyclomatic": cyclomatic,
        "sloc": sloc,
        "lloc": lloc,
        "comment_rate": comment_rate,
        "mi_std": mi_std,
        "mi_custom": mi_custom
    }
    return result


def main(root_dir='/mnt/nvme1/hsm/LLM_code/arxiv_dataset_cpp', output_csv='github_cpp_metrics.csv'):
    years = range(2020, 2026)
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    header = [
        'quarter', 'h1', 'h2', 'N1', 'N2', 'vocabulary', 'length',
        'calculated_length', 'volume', 'difficulty', 'effort', 'time_sec', 'bugs',
        'cyclomatic', 'sloc', 'lloc', 'comment_rate', 'mi_std', 'mi_custom'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for year in years:
            for q in quarters:
                # 2025 only has Q1
                if year == 2025 and q != 'Q1':
                    continue

                quarter_label = f"{year}{q}"
                quarter_path = os.path.join(root_dir, str(year), q)
                if not os.path.isdir(quarter_path):
                    continue

                sums = {key: 0.0 for key in header if key != 'quarter'}
                count = 0

                # 遍历所有子目录，查找 .cpp 文件
                for dirpath, _, filenames in os.walk(quarter_path):
                    print(quarter_path,dirpath)
                    for fname in filenames:
                        if fname.lower().endswith('.cpp'):
                            file_path = os.path.join(dirpath, fname)
                            try:
                                metrics = analyze_cpp_file(file_path)
                                for key in sums:
                                    sums[key] += metrics.get(key, 0.0)
                                count += 1
                            except Exception:
                                # 跳过出错文件
                                traceback.print_exc(file=sys.stderr)
                                continue

                if count == 0:
                    continue

                averages = [quarter_label] + [sums[key] / count for key in header if key != 'quarter']
                writer.writerow(averages)

    print(f"已生成季度指标 CSV: {output_csv}")


if __name__ == '__main__':
    main()
