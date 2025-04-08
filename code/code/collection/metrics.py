import re

# 命名风格分类
def classify_naming(name):
    """简化命名规范分类"""
    if re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', name):
        return 'camelCase'
    elif re.match(r'^[a-z]+(?:_[a-z]+)+$', name):
        return 'snake_case'
    elif re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', name):
        return 'PascalCase'
    elif re.match(r'^[A-Z]+(?:_[A-Z]+)+$', name):
        return 'UPPER_SNAKE_CASE'
    else:
        return 'Other'


# 提取函数名
def extract_function_names(code):
    pattern = r'^\s*def\s+(\w+)\s*\('
    return re.findall(pattern, code, re.MULTILINE)


# 提取变量名
def extract_variable_names(code):
    pattern = r'^\s*(\w+)\s*=\s*[^=]'
    return re.findall(pattern, code, re.MULTILINE)


# 提取类名
def extract_class_names(code):
    pattern = r'^\s*class\s+(\w+)\s*(?:\(|:)?'
    return re.findall(pattern, code, re.MULTILINE)


# 提取常量名
def extract_constant_names(code):
    pattern = r'^\s*([A-Z][A-Z_0-9]*)\s*=\s*'
    return re.findall(pattern, code, re.MULTILINE)


# 分析代码并计算指标
def analyze_code(code):
    lines = code.splitlines()

    metrics = {
        "function_naming_consistency": 0.0,
        "variable_naming_consistency": 0.0,
        "class_naming_consistency": 0.0,
        "constant_naming_consistency": 0.0,
        "indentation_consistency": 0.0,
        "avg_function_length": 0.0,
        "avg_nesting_depth": 0.0,
        "comment_ratio": 0.0,
        "avg_function_name_length": 0.0,
        "avg_variable_name_length": 0.0,
        "function_naming_counts": {},
        "variable_naming_counts": {},
        "class_naming_counts": {},
        "constant_naming_counts": {}
    }

    # 提取名字
    function_names = extract_function_names(code)
    variable_names = extract_variable_names(code)
    class_names = extract_class_names(code)
    constant_names = extract_constant_names(code)

    # 命名一致性计算
    def calculate_naming_consistency(names, naming_counts_key):
        naming_counts = {
            "camelCase": 0,
            "snake_case": 0,
            "PascalCase": 0,
            "UPPER_SNAKE_CASE": 0,
            "Other": 0
        }
        for name in names:
            naming_style = classify_naming(name)
            if naming_style in naming_counts:
                naming_counts[naming_style] += 1
            else:
                naming_counts["Other"] += 1

        total_names = sum(naming_counts.values())
        metrics[naming_counts_key] = naming_counts  # Store counts
        if total_names > 0:
            most_common_style_count = max(naming_counts.values())
            return most_common_style_count / total_names
        else:
            return 0.0

    # 计算每种命名一致性
    metrics["function_naming_consistency"] = calculate_naming_consistency(function_names, "function_naming_counts")
    metrics["variable_naming_consistency"] = calculate_naming_consistency(variable_names, "variable_naming_counts")
    metrics["class_naming_consistency"] = calculate_naming_consistency(class_names, "class_naming_counts")
    metrics["constant_naming_consistency"] = calculate_naming_consistency(constant_names, "constant_naming_counts")

    # 计算平均名字长度
    metrics["avg_function_name_length"] = sum(len(name) for name in function_names) / len(function_names) if function_names else 0.0
    metrics["avg_variable_name_length"] = sum(len(name) for name in variable_names) / len(variable_names) if variable_names else 0.0

    # 分析缩进一致性
    def calculate_indentation_consistency(lines):
        indent_unit_counts = {}
        total_indented_lines = 0
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line or stripped_line.startswith(('#', '//', '/*', '*')):
                continue
            indent = line[:len(line)-len(stripped_line)]
            if indent:
                total_indented_lines += 1
                indent = indent.replace('\t', '    ')
                indent_length = len(indent)
                if indent_length in indent_unit_counts:
                    indent_unit_counts[indent_length] += 1
                else:
                    indent_unit_counts[indent_length] = 1

        if total_indented_lines == 0:
            return 1.0

        most_common_indent_count = max(indent_unit_counts.values())
        consistency = most_common_indent_count / total_indented_lines
        return consistency

    metrics["indentation_consistency"] = calculate_indentation_consistency(lines)

    # 分析函数长度
    function_lengths = []
    function_pattern = r'^\s*def\s+\w+\s*\(.*\):'

    function_starts = [i for i, line in enumerate(lines) if re.match(function_pattern, line)]
    for start_line in function_starts:
        length = 0
        i = start_line
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())
            start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            if i > start_line and stripped_line and (len(line) - len(line.lstrip())) <= start_indent:
                break
            length += 1
            i += 1
        function_lengths.append(length)
    metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths) if function_lengths else 0.0

    # 分析嵌套深度
    nesting_depths = []
    indent_levels = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        current_indent = len(line) - len(line.lstrip())
        while indent_levels and current_indent < indent_levels[-1]:
            indent_levels.pop()
        if indent_levels and current_indent == indent_levels[-1]:
            pass
        elif current_indent > (indent_levels[-1] if indent_levels else 0):
            indent_levels.append(current_indent)
        nesting_depths.append(len(indent_levels))
    metrics["avg_nesting_depth"] = sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0

    # 计算注释比例
    comment_lines = 0
    code_lines = 0
    in_block_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith('#'):
                comment_lines += 1
        elif re.match(r'(\'\'\'|\"\"\")', stripped_line):
            comment_lines += 1
            if stripped_line.count('\'\'\'') % 2 == 1 or stripped_line.count('\"\"\"') % 2 == 1:
                in_block_comment = not in_block_comment
        elif in_block_comment:
            comment_lines += 1
        else:
            code_lines += 1
        

    total_code_lines = code_lines + comment_lines
    metrics["comment_ratio"] = comment_lines / total_code_lines if total_code_lines > 0 else 0.0
    return metrics


# 主函数
def main():
    # 示例代码片段
    code = """
    def my_function(a, b):
        # This is a comment
        result = a + b
        return result
    """
    
    # 分析代码
    metrics = analyze_code(code)
    
    # 输出指标
    print(metrics)


if __name__ == "__main__":
    main()
