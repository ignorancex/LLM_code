import re, math, sys
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

def analyze_cpp_file(path: str):
    # 读取源码
    code = open(path, encoding='utf-8').read()
    lines = code.splitlines()
    sloc = sum(1 for ln in lines if ln.strip())  # 非空即为 SLOC

    # 1. Lizard 分析
    ana = lizard.analyze_file(path)
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

    # 4. 输出
    print("=== Halstead Metrics ===")
    for k in ["h1","h2","N1","N2","vocabulary","length","calculated_length",
              "volume","difficulty","effort","time_sec","bugs"]:
        print(f"{k:20s}: {hal[k]:>10.2f}" if isinstance(hal[k], float)
              else f"{k:20s}: {hal[k]:>10d}")

    print("\n=== Code Complexity & Size ===")
    print(f"{'cyclomatic':20s}: {cyclomatic:>10d}")
    print(f"{'sloc':20s}: {sloc:>10d}")
    print(f"{'lloc':20s}: {lloc:>10d}")
    print(f"{'comment_rate':20s}: {comment_rate*100:>9.2f}%")

    print("\n=== Maintainability Index ===")
    print(f"{'mi_std':20s}: {mi_std:>10.2f}")
    print(f"{'mi_custom':20s}: {mi_custom:>10.2f}")

# 脚本入口
if __name__ == "__main__":
    file="ceshi.cpp"
    analyze_cpp_file(file)
