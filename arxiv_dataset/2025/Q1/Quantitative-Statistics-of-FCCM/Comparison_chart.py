import pandas as pd
import matplotlib.pyplot as plt

def map_to_range(value, from_min, from_max, to_min, to_max):
    # 确保输入值在原始范围内
    value = max(min(value, from_max), from_min)

    # 计算映射比例
    from_range = from_max - from_min
    to_range = to_max - to_min
    scale = to_range / from_range

    # 计算映射后的值
    mapped_value = (value - from_min) * scale + to_min

    # 返回映射后的值
    return int(mapped_value)

def normalization(i):
    if i >= 0 and i < 1000:
        mapped_value = map_to_range(i, 0, 1000, 0, 1000)
    elif i >= 1000 and i < 3500:
        mapped_value = map_to_range(i, 1000, 3500, 1000, 2000)
    elif i >= 3500 and i < 5000:
        mapped_value = map_to_range(i, 3500, 5000, 2000, 2500)
    elif i >= 5000 and i < 10000:
        mapped_value = map_to_range(i, 5000, 10000, 2500, 3000)
    elif i >= 10000 and i < 50000:
        mapped_value = map_to_range(i, 10000, 50000, 3000, 3500)
    elif i >= 50000 and i < 150000:
        mapped_value = map_to_range(i, 50000, 150000, 3500, 5000)
    else:
        mapped_value = 5000

    return mapped_value

def Comparison_chart(list1, list2, save_path):
    # 对两个列表排序并删除小于100的数据
    sorted_list1 = sorted(filter(lambda x: x >= 100, list1))
    sorted_list2 = sorted(filter(lambda x: x >= 100, list2))

    # 获取两个列表的最大长度
    max_length = max(len(sorted_list1), len(sorted_list2))

    # 补充零使得两个列表长度相同
    sorted_list1 = [0] * (max_length - len(sorted_list1)) + sorted_list1
    sorted_list2 = [0] * (max_length - len(sorted_list2)) + sorted_list2

    # 创建一个DataFrame
    df = pd.DataFrame({'Previous Inspection': sorted_list1, 'Follow-up Inspection': sorted_list2})

    # 将DataFrame写入xlsx文件
    df.to_excel(save_path + '/Pre_Follow_general_trend.xlsx', index=False)

    # 读取Excel表格数据
    excel_file_path = save_path + '/Pre_Follow_general_trend.xlsx'  # 替换成你的Excel文件路径
    df = pd.read_excel(excel_file_path)

    # 将数据存储为列表
    data1 = df['Previous Inspection'].tolist()
    data2 = df['Follow-up Inspection'].tolist()

    data1_1 = []
    data2_1 = []

    for n, i in enumerate(data1):
        c = normalization(i)
        data1_1.append(c)

    for n, i in enumerate(data2):
        c = normalization(i)
        data2_1.append(c)

    # 生成自定义的y轴刻度
    custom_y_ticks = [0, 1000, 2000, 3000, 4000, 5000]

    # 将5000刻度显示为15000
    custom_y_tick_labels = [0, 1000, 2000, 3000, 4000, 5000]
    custom_y_tick_labels[custom_y_ticks.index(0)] = '0'
    custom_y_tick_labels[custom_y_ticks.index(1000)] = '1000'
    custom_y_tick_labels[custom_y_ticks.index(2000)] = '3500'
    custom_y_tick_labels[custom_y_ticks.index(3000)] = '10000'
    custom_y_tick_labels[custom_y_ticks.index(4000)] = ' . . .   '
    custom_y_tick_labels[custom_y_ticks.index(5000)] = '150000'

    # 绘制折线图
    plt.plot(data1_1, marker='+', label='Previous Inspection')
    plt.plot(data2_1, marker='+', label='Followup Inspection')

    # 设置y轴刻度
    plt.yticks(custom_y_ticks, custom_y_tick_labels)

    # 添加标题和标签
    plt.title('Previous and followup inspection condition comparison')
    plt.xlabel('Lesion')
    plt.ylabel('Volume')

    # 添加图例
    plt.legend()

    # 保存为png图片，设置dpi为300
    plt.savefig(save_path + '/Pre_Follow_general_trend.png', dpi=500)

    plt.tight_layout()


def Comparison_chart_1(data, save_path):
    # plt.figure(1)  # 创建新的图形对象
    list1, list2 = [], []
    for i in data:
        if type(i) is list:
            if i[0] in list1:
                list2[list1.index(i[0])] = list2[list1.index(i[0])] + i[1]
            else:
                list1.append(i[0])
                list2.append(i[1])
    # 对两个列表排序并删除小于100的数据
    sorted_list1 = sorted(filter(lambda x: x >= 100, list1))
    sorted_list2 = sorted(filter(lambda x: x >= 100, list2))

    # 获取两个列表的最大长度
    max_length = max(len(sorted_list1), len(sorted_list2))

    # 补充零使得两个列表长度相同
    sorted_list1 = [0] * (max_length - len(sorted_list1)) + sorted_list1
    sorted_list2 = [0] * (max_length - len(sorted_list2)) + sorted_list2

    # 创建一个DataFrame
    df = pd.DataFrame({'Previous Inspection': sorted_list1, 'Follow-up Inspection': sorted_list2})

    # 将DataFrame写入xlsx文件
    df.to_excel(save_path + '/Pre_Follow_match.xlsx', index=False)

    # 读取Excel表格数据
    excel_file_path = save_path + '/Pre_Follow_match.xlsx'  # 替换成你的Excel文件路径
    df = pd.read_excel(excel_file_path)

    # 将数据存储为列表
    data1 = df['Previous Inspection'].tolist()
    data2 = df['Follow-up Inspection'].tolist()

    data1_1 = []
    data2_1 = []

    for n, i in enumerate(data1):
        c = normalization(i)
        data1_1.append(c)

    for n, i in enumerate(data2):
        c = normalization(i)
        data2_1.append(c)

    # 生成自定义的y轴刻度
    custom_y_ticks = [0, 1000, 2000, 3000, 4000, 5000]

    # 将5000刻度显示为15000
    custom_y_tick_labels = [0, 1000, 2000, 3000, 4000, 5000]
    custom_y_tick_labels[custom_y_ticks.index(0)] = '0'
    custom_y_tick_labels[custom_y_ticks.index(1000)] = '1000'
    custom_y_tick_labels[custom_y_ticks.index(2000)] = '3500'
    custom_y_tick_labels[custom_y_ticks.index(3000)] = '10000'
    custom_y_tick_labels[custom_y_ticks.index(4000)] = ' . . .   '
    custom_y_tick_labels[custom_y_ticks.index(5000)] = '150000'

    # 绘制折线图
    plt.plot(data1_1, marker='+', label='Previous Inspection')
    plt.plot(data2_1, marker='+', label='Followup Inspection')

    # 设置y轴刻度
    plt.yticks(custom_y_ticks, custom_y_tick_labels)

    # 添加标题和标签
    plt.title('Matching lesion volume comparison')
    plt.xlabel('Lesion')
    plt.ylabel('Volume')

    # 添加图例
    plt.legend()

    # 保存为png图片，设置dpi为300
    plt.savefig(save_path + '/Pre_Follow_match.png', dpi=500)

    plt.tight_layout()

    plt.close()


def Comparison_chart_2(data, list1, save_path):
    # plt.figure(2)  # 创建新的图形对象
    list2 = []
    for i in data:
        if type(i) is int:
            list2.append(i)
        elif i[0] in list1:
            list1.remove(i[0])

    # 对两个列表排序并删除小于100的数据
    sorted_list1 = sorted(filter(lambda x: x >= 100, list1))
    sorted_list2 = sorted(filter(lambda x: x >= 100, list2))

    # 获取两个列表的最大长度
    max_length = max(len(sorted_list1), len(sorted_list2))

    # 补充零使得两个列表长度相同
    sorted_list1 = [0] * (max_length - len(sorted_list1)) + sorted_list1
    sorted_list2 = [0] * (max_length - len(sorted_list2)) + sorted_list2

    # 创建一个DataFrame
    df = pd.DataFrame({'Previous Inspection': sorted_list1, 'Follow-up Inspection': sorted_list2})

    # 将DataFrame写入xlsx文件
    df.to_excel(save_path + '/Pre_Follow_nomatch.xlsx', index=False)

    # 读取Excel表格数据
    excel_file_path = save_path + '/Pre_Follow_nomatch.xlsx'  # 替换成你的Excel文件路径
    df = pd.read_excel(excel_file_path)

    # 将数据存储为列表
    data1 = df['Previous Inspection'].tolist()
    data2 = df['Follow-up Inspection'].tolist()

    data1_1 = []
    data2_1 = []

    for n, i in enumerate(data1):
        c = normalization(i)
        data1_1.append(c)

    for n, i in enumerate(data2):
        c = normalization(i)
        data2_1.append(c)

    # 生成自定义的y轴刻度
    custom_y_ticks = [0, 1000, 2000, 3000, 4000, 5000]

    # 将5000刻度显示为15000
    custom_y_tick_labels = [0, 1000, 2000, 3000, 4000, 5000]
    custom_y_tick_labels[custom_y_ticks.index(0)] = '0'
    custom_y_tick_labels[custom_y_ticks.index(1000)] = '1000'
    custom_y_tick_labels[custom_y_ticks.index(2000)] = '3500'
    custom_y_tick_labels[custom_y_ticks.index(3000)] = '10000'
    custom_y_tick_labels[custom_y_ticks.index(4000)] = ' . . .   '
    custom_y_tick_labels[custom_y_ticks.index(5000)] = '150000'

    # 绘制折线图
    plt.plot(data1_1, marker='+', label='Previous Inspection')
    plt.plot(data2_1, marker='+', label='Followup Inspection')

    # 设置y轴刻度
    plt.yticks(custom_y_ticks, custom_y_tick_labels)

    # 添加标题和标签
    plt.title('Volume comparison of unmatched lesions')
    plt.xlabel('Lesion')
    plt.ylabel('Volume')

    # 添加图例
    plt.legend()

    # 保存为png图片，设置dpi为300
    plt.savefig(save_path + '/Pre_Follow_nomatch.png', dpi=500)

    plt.tight_layout()

    plt.close()


def Volume_proportion_statistics(values, save_path):
    # 定义数据范围
    ranges = ['<200', '200-3500', '>=3500']
    num = [0, 0, 0]

    # 统计数据在不同范围内的个数
    for value in values:
        if value < 200:
            num[0] += 1
        elif 200 <= value < 3500:
            num[1] += 1
        else:
            num[2] += 1

    Proportion = [0, 0, 0]
    num_sum = num[0] + num[1] + num[2]
    Proportion[0] = str(round(num[0] / (num_sum) * 100)) + '%'
    Proportion[1] = str(round(num[1] / (num_sum) * 100)) + '%'
    Proportion[2] = str(round(num[2] / (num_sum) * 100)) + '%'

    # 创建一个DataFrame
    df = pd.DataFrame({'Volume range': ranges, 'Number': num, 'Proportion': Proportion}, )

    # 将DataFrame写入xlsx文件
    df.to_excel(save_path + '/Volume_proportion_statistics.xlsx', index=False)

