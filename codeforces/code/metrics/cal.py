from radon.metrics import h_visit, mi_visit, mi_parameters, mi_compute
from radon.raw import analyze
from radon.visitors import ComplexityVisitor

def analyze_file(path: str):
    """
    计算并打印指定 Python 文件的 Halstead 指标与可维护性指数。
    """
    # 读取源代码
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Halstead 指标（整体）
    # Halstead 整体与函数级计算
    hal = h_visit(code)
    total = hal.total

    # 1. 打印总体指标
    print("=== Halstead Overall Metrics ===")
    print(f"Distinct Operators (h1)     : {total.h1}")
    print(f"Distinct Operands (h2)      : {total.h2}")
    print(f"Total Operators (N1)        : {total.N1}")
    print(f"Total Operands (N2)         : {total.N2}")
    print(f"Vocabulary (η)              : {total.vocabulary}")
    print(f"Length (N)                  : {total.length}")
    print(f"Calculated Length (ĤN)      : {total.calculated_length:.2f}")
    print(f"Volume (V)                  : {total.volume:.2f}")
    print(f"Difficulty (D)              : {total.difficulty:.2f}")
    print(f"Effort (E)                  : {total.effort:.2f}")
    print(f"Time (T, sec)               : {total.time:.2f}")
    print(f"Estimated Bugs (B)          : {total.bugs:.4f}\n")

    # 2. 遍历并打印函数级报告
    if hal.functions:
        print("=== Halstead Metrics per Function ===")
        for func_name, rep in hal.functions:
            print(f"-- Function: {func_name}")
            print(f"   h1={rep.h1}, h2={rep.h2}, N1={rep.N1}, N2={rep.N2}, V={rep.volume:.2f}, D={rep.difficulty:.2f}, E={rep.effort:.2f}, B={rep.bugs:.4f}")
        print()

    # 3. 快速可维护性指数（Visual Studio 版本）
    mi_score = mi_visit(code,True)       # 直接返回 MI 分数（0–100 区间）
    print("=== Maintainability Index (MI) ===")
    print(f"MI Score                 : {mi_score:.2f}\n")

    # 4. 拆分参数并自定义计算
    hal_vol, cyclo, sloc, comment_rate = mi_parameters(code)
    mi_custom = mi_compute(hal_vol, cyclo, sloc, comment_rate)
    print("=== Custom MI Computation ===")
    print(f"Halstead Volume (V)      : {hal_vol:.2f}")
    print(f"Cyclomatic Complexity    : {cyclo}")
    print(f"Logical SLOC (LLOC)      : {sloc}")
    print(f"Comment Percentage       : {comment_rate:.2f}%")
    print(f"Recomputed MI            : {mi_custom:.2f}")

if __name__ == "__main__":
    analyze_file("../arrange.py")
