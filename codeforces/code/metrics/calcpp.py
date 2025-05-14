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
            # 在多行注释块内部
            comment_lines += 1
            if "*/" in stripped:
                in_block = False
            continue

        # 检查是否是单行注释
        if stripped.startswith("//"):
            comment_lines += 1
        # 检查是否是多行注释开始
        elif "/*" in stripped:
            comment_lines += 1
            # 如果同一行有结束标志，就不进入块内状态
            if "*/" not in stripped:
                in_block = True

    return comment_lines

def scan_halstead(code: str):
    """
    扫描 C++ 代码，统计 Halstead 操作符/操作数。
    返回 h1, h2, N1, N2, vocabulary, length, calculated_length, V, D, E, T, B
    """
    # 操作符正则（可按需扩展）
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
    # 计算 ĤN（calculated_length）
    calc_len = 0.0
    if h1 > 0:
        calc_len += h1 * math.log2(h1)
    if h2 > 0:
        calc_len += h2 * math.log2(h2)
    # Halstead 核心度量
    V = length * math.log2(vocabulary) if vocabulary > 0 else 0.0
    D = (h1 / 2.0) * (N2 / h2)       if h2 > 0 else 0.0
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
    MI = max(0, (171 -5.2 ln(V) -0.23 G -16.2 ln(L) +50 sin(sqrt(2.4 C)))/171 *100)
    C = comment_rate*100（度）
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
    MI_custom = max(0, (171 -5.2 ln(V) -0.23 G -16.2 ln(L))/171 *100)
    """
    raw = 171.0 \
          - 5.2 * math.log(V or 1) \
          - 0.23 * G \
          - 16.2 * math.log(L or 1)
    return max(0.0, raw * 100.0 / 171.0)

def analyze_code(code: str):
    # 读取源码
    lines = code.splitlines()
    sloc = sum(1 for ln in lines if ln.strip())  # 非空即为 SLOC

    # 1. Lizard 分析
    ana = lizard.analyze_file.analyze_source_code("filename.cpp", code)
    cyclomatic = sum(fn.cyclomatic_complexity for fn in ana.function_list)
    lloc       = sum(fn.nloc                 for fn in ana.function_list)
    comment_lines = count_comment_lines(code)
    comment_rate  = (comment_lines / (comment_lines + lloc)
                     if (comment_lines + lloc) > 0 else 0.0)

    # 2. Halstead 指标
    hal = scan_halstead(code)

    # 3. Maintainability Index
    mi_std    = compute_mi_std(hal["volume"], cyclomatic, sloc, comment_rate)
    mi_custom = compute_mi_custom(hal["volume"], cyclomatic, sloc)

    # # 4. 输出
    # print("=== Halstead Metrics ===")
    # for k in ["h1","h2","N1","N2","vocabulary","length","calculated_length",
    #           "volume","difficulty","effort","time_sec","bugs"]:
    #     print(f"{k:20s}: {hal[k]:>10.2f}" if isinstance(hal[k], float)
    #           else f"{k:20s}: {hal[k]:>10d}")
    #
    # print("\n=== Code Complexity & Size ===")
    # print(f"{'cyclomatic':20s}: {cyclomatic:>10d}")
    # print(f"{'sloc':20s}: {sloc:>10d}")
    # print(f"{'lloc':20s}: {lloc:>10d}")
    # print(f"{'comment_rate':20s}: {comment_rate*100:>9.2f}%")
    #
    # print("\n=== Maintainability Index ===")
    # print(f"{'mi_std':20s}: {mi_std:>10.2f}")
    # print(f"{'mi_custom':20s}: {mi_custom:>10.2f}")

    return {
        "h1": hal["h1"],
        "h2": hal["h2"],
        "N1": hal["N1"],
        "N2": hal["N2"],
        "vocabulary": hal["vocabulary"],
        "length": hal["length"],
        "calculated_length": round(hal["calculated_length"], 2),
        "volume": round(hal["volume"], 2),
        "difficulty": round(hal["difficulty"], 2),
        "effort": round(hal["effort"], 2),
        "time_sec": round(hal["time_sec"], 2),
        "bugs": round(hal["bugs"], 4),
        "cyclomatic": cyclomatic,
        "sloc": sloc,
        "lloc": lloc,
        "comment_rate": round(comment_rate, 2),
        "mi_std": round(mi_std, 2),
        "mi_custom": round(mi_custom, 2),
    }

# 2. 主流程：读取 JSON，分别处理三个字段，输出三个 CSV
def process_field(records, field_name, output_csv):
    # 如果没有记录则直接返回
    if not records:
        print(f"No records to process for field '{field_name}'.")
        return

    # 用第一条记录来构造表头
    sample_metrics = analyze_code(records[0][field_name])
    headers = ["submission_id"] + list(sample_metrics.keys())

    rows = []
    cols = {k: [] for k in sample_metrics.keys()}

    for rec in records:
        sid = rec.get("submission_id", "<no_id>")
        code = rec.get(field_name, "")
        try:
            metrics = analyze_code(code)
        except Exception as e:
            print(f"[Warning] Skipping submission_id={sid} in field '{field_name}' due to error:")
            #traceback.print_exc()
            continue

        # 累积用于平均值计算
        for k, v in metrics.items():
            cols[k].append(v)

        rows.append([sid] + [metrics[k] for k in sample_metrics.keys()])

    # 计算平均值行
    if rows:
        avg_row = ["average"] + [round(mean(cols[k]), 4) for k in sample_metrics.keys()]
    else:
        avg_row = ["average"] + [""] * len(sample_metrics)

    # 写入 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(avg_row)

if __name__ == "__main__":
    # 读取 JSON 文件
    with open("../gemma_27b_cpp_extract.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 假设 JSON 的顶层是一个列表
    records = data

    # 分别生成三个 CSV
    process_field(records, "sourceCode", "gemma_AC_cpp.csv")
    process_field(records, "generate_code_block", "gemma_ANS_cpp.csv")
    process_field(records, "generate_ref_code_block", "gemma_REF_cpp.csv")

    print("三个 CSV 文件已生成：")
    print(" - sourceCode_metrics.csv")
    print(" - generate_code_block_metrics.csv")
    print(" - generate_ref_code_block_metrics.csv")
