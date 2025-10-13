# 将latex中表格的行统一加粗或下划线

# 计算平均值：
latex_input = input("输入计算平均值的字符串：")# latex_input = "1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10"
latex_input = latex_input.split("&")
print(latex_input)
print("overall:", sum([float(i) for i in latex_input]) / len(latex_input))
# 间隔三个算一次
R1 = sum([float(i) for i in latex_input[0::3]]) / 4
R2 = sum([float(i) for i in latex_input[1::3]]) / 4
R3 = sum([float(i) for i in latex_input[2::3]]) / 4
print(R1, R2, R3)

inp = input("输入字符串：")# inp = "1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10"
rmv = input("输入要去掉的字符：") # rmv = " & "
inp = inp.replace(rmv, "")
prefix = input("输入前缀：")# prefix = " \underline "

# 返回 \underline {1} & \underline {2} & \underline {3} & \underline {4} & \underline {5} & \underline {6} & \underline {7} & \underline {8} & \underline {9} & \underline {10}
def add_prefix(inp, prefix):
    inp = inp.split("&")
    out = []
    for i in inp:
        out.append(prefix + "{" + i.strip() + "}")
    return " & ".join(out)

print(add_prefix(inp, prefix))