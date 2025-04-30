import pandas as pd

# === 1. 读取 CSV 文件 ===
python_df = pd.read_csv("python_repo_file_counts.csv")
cpp_df = pd.read_csv("cpp_repo_file_counts.csv")

# === 2. Pivot 为：quarter -> category -> value ===
def pivot_and_format(df, lang):
    df_pivot = df.pivot(index="quarter", columns="category", values=["num_repos", "num_files"]).fillna(0).astype(int)
    df_pivot.columns = [f"{lang}_{cat}_{typ}" for typ, cat in df_pivot.columns]
    return df_pivot.reset_index()

py_df = pivot_and_format(python_df, "py")
cpp_df = pivot_and_format(cpp_df, "cpp")

# === 3. 合并两个表格 ===
merged_df = pd.merge(py_df, cpp_df, on="quarter", how="outer").fillna(0)

# 只对数值列做 int 转换
int_columns = [col for col in merged_df.columns if col != "quarter"]
merged_df[int_columns] = merged_df[int_columns].astype(int)


# === 4. 补齐可能缺失的列 ===
for lang in ["py", "cpp"]:
    for cat in ["cs", "non_cs"]:
        for typ in ["num_repos", "num_files"]:
            col = f"{lang}_{cat}_{typ}"
            if col not in merged_df.columns:
                merged_df[col] = 0

# === 5. 生成 LaTeX 表格代码 ===
latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\begin{tabular}{l|cccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\textbf{Quarter} & \textbf{Py-cs} & \textbf{Py-non-cs} & \textbf{C++-cs} & \textbf{C++-non-cs} \\")
latex_lines.append(r"& (\#R/\#F) & (\#R/\#F) & (\#R/\#F) & (\#R/\#F) \\")
latex_lines.append(r"\midrule")

for _, row in merged_df.iterrows():
    line = f"{row['quarter']} & " \
           f"{row['py_cs_num_repos']}/{row['py_cs_num_files']} & " \
           f"{row['py_non_cs_num_repos']}/{row['py_non_cs_num_files']} & " \
           f"{row['cpp_cs_num_repos']}/{row['cpp_cs_num_files']} & " \
           f"{row['cpp_non_cs_num_repos']}/{row['cpp_non_cs_num_files']} \\\\"
    latex_lines.append(line)

# === 6. 添加总计行 ===
total_row = {
    col: merged_df[col].sum()
    for col in merged_df.columns if col.startswith(("py_", "cpp_"))
}

total_line = "Total & " + \
             f"{total_row['py_cs_num_repos']}/{total_row['py_cs_num_files']} & " + \
             f"{total_row['py_non_cs_num_repos']}/{total_row['py_non_cs_num_files']} & " + \
             f"{total_row['cpp_cs_num_repos']}/{total_row['cpp_cs_num_files']} & " + \
             f"{total_row['cpp_non_cs_num_repos']}/{total_row['cpp_non_cs_num_files']} \\\\"

latex_lines.append(r"\midrule")
latex_lines.append(total_line)
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\caption{Number of repositories and Python/C++ files per quarter and category}")
latex_lines.append(r"\label{tab:quarter_repo_stats}")
latex_lines.append(r"\end{table}")

# === 7. 保存 LaTeX 文件 ===
with open("repo_quarter_table.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))

print("✅ LaTeX 表格代码已保存至 repo_quarter_table.tex")
