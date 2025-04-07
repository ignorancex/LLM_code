import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import time
import math
from tools import add_fname, correct


def write_to_file(result_file, array):
    with open(result_file, 'w', newline='', encoding='UTF-8-sig') as ff:
        for i in tqdm(range(len(array))):
            writer = csv.writer(ff)
            writer.writerow(array[i])


def not_nan(s):
    try:
        f = float(s)
        if f != f or f == float('inf') or f == float('-inf'):
            return False
        return True

    except ValueError:
        return False


def count_columns(path):
    all_lines = pd.read_csv(path, header=None)
    all_lines = np.array(all_lines)
    column_num = []
    for i in range(all_lines.shape[1]):
        column_num.append(str(i))
    return column_num


# 去重
def unique(source, result=None):
    if result is None:
        result = source
    all_lines = pd.read_csv(source)
    all_lines = np.array(all_lines)
    values = []
    new_lines1 = []
    for data in all_lines:
        if (data[0] not in values) and not_nan(data[0]):
            new_lines1.append(data)
            values.append(data[0])

    write_to_file(result, new_lines1)


def wash(source, result=None):
    if result is None:
        result = source
    all_lines = pd.read_csv(source, header=None)
    all_lines = np.array(all_lines)
    start_column = 3
    end_column = 7
    new_lines = []

    for data in all_lines:
        for i in range(start_column, end_column + 1):
            if (not not_nan(data[i])) or (data[i] is np.nan):
                break
            if i == end_column:
                new_lines.append(data)

    write_to_file(result, new_lines)


# 原始文件转化
# bili表头内容
# id 分区 标题 粉丝数 1d评论数 点赞 收藏 投币 分享
def rewrite(source, result):
    all_lines = pd.read_csv(source)
    all_lines = np.array(all_lines)
    all_lines = all_lines[:, 1:]
    new_lines = []
    for data in all_lines:
        # tname 1
        # title 3
        text_cols = [1, 3]
        for num in text_cols:
            if (data[num] == '\\N' or '') or (data[num] is np.nan):
                data[num] = "0"

        meta_cols = range(4, 14)
        for num in meta_cols:
            if (data[num] == '\\N' or '') or (data[num] is np.nan) or (data[num] is math.nan):
                data[num] = 0

        new_lines.append(data)

    new_lines = np.array(new_lines)
    new_lines = new_lines[:, [0, 1, 3, 11, 5, 9, 6, 7, 8]]

    write_to_file(result, new_lines)


def add_views(source, add_file, result, col):
    path1 = source
    path2 = add_file
    result_path = result
    ###参数合并
    a = count_columns(path1)
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=count_columns(path1)))
    df11 = pd.DataFrame(pd.read_csv(path2, names=count_columns(path2)))
    df11 = df11.drop(0)
    df11[['0']] = df11[['0']].astype(float)
    df11 = df11[['0', str(col)]]
    #
    merge2 = pd.merge(left=df10, right=df11, on=['0'],
                      how="inner")  # 实现满足点数与区域都匹配两个条件时候合并表格，只匹配一个条件采用left_on="plot_no",right_on="plot_no"
    merge2.to_csv(result_path, index=False, header=None, encoding='utf-8-sig')


def merge_scores(source, add_file, result, mode):
    # mode = 0 合并NIMA
    # mode = 1 合并IIPA
    path1 = source
    path2 = add_file
    result_path = result
    ###参数合并
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=count_columns(path1)))
    df11 = pd.DataFrame(pd.read_csv(path2, header=None, names=count_columns(path2)))

    merge2 = pd.merge(left=df10, right=df11, on=['0'],
                      how="inner")  # 实现满足点数与区域都匹配两个条件时候合并表格，只匹配一个条件采用left_on="plot_no",right_on="plot_no"
    if mode == 0:
        cols = merge2.columns[[0, 1, 2, 3, 4, 5, 12, 7, 8, 9, 10, 11]]
    if mode == 1:
        cols = merge2.columns[[0, 1, 2, 3, 4, 5, 6, 12, 8, 9, 10, 11]]

    merge2 = merge2[cols]
    merge2.to_csv(result_path, index=False, header=None, encoding='utf-8-sig')


def get_ln(source, result=None):
    if result is None:
        result = source

    path1 = source
    # result = path1
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    # bili = [] || youtube = [6,9,10]
    ln_cols = [6,9,10]
    for data in all_lines:
        for i in ln_cols:
            data[i] = np.log(correct(float(data[i])) + 1)
    write_to_file(result, all_lines)


def get_pop(source, result=None):
    if result is None:
        result = source

    path1 = source
    # result = path1
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)

    num1, day1 = -2, 1
    num2, day2 = -1, 7

    for data in all_lines:
        data[num1] = math.log(float(data[num1] / day1) + 1, 2)
        data[num2] = math.log(float(data[num2] / day2) + 1, 2)
    write_to_file(result, all_lines)


def get_pop_seq(source, result=None):
    if result is None:
        result = source

    path1 = source
    # result = path1
    all_lines = pd.read_csv(path1, header=None)
    all_lines = np.array(all_lines)
    _, cols = all_lines.shape
    # yt=11 || bili=8
    start_col = 11
    days = cols - start_col

    for i in range(days):
        col = i + start_col
        for data in all_lines:
            data[col] = np.log2(float(data[col]) / (i + 1) + 1)

    write_to_file(result, all_lines)


def all_normalize(source, result=None):
    if result is None:
        result = source

    all_lines = pd.read_csv(source, header=None)
    all_lines = np.array(all_lines)
    # bili=3 || yt=6
    start_column = 6

    # bili=9 || yt=11
    end_column = 10
    new_lines = []
    for i in range(start_column, end_column + 1):
        col = all_lines[:, i]
        col_mean = np.mean(col)
        col_std = np.std(col)
        for data in all_lines:
            data[i] = (data[i] - col_mean) / col_std

    write_to_file(result, all_lines)


def other_normal(source, other, result=None):
    if result is None:
        result = other

    all_lines = pd.read_csv(source, header=None)
    all_lines = np.array(all_lines)
    new_lines = pd.read_csv(other, header=None)
    new_lines = np.array(new_lines)
    start_column = 4
    end_column = 10
    for i in range(start_column, end_column + 1):
        col = all_lines[:, i]
        col_mean = np.mean(col)
        col_std = np.std(col)
        for data in new_lines:
            data[i] = (data[i] - col_mean) / col_std

    write_to_file(result, new_lines)


def merge(path1, path2, result):
    ###参数合并
    df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=count_columns(path1)))
    # df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1', '2', '3', '4', '5','6']))
    # df10 = pd.DataFrame(pd.read_csv(path1, header=None, names=['key', '1', '2']))
    # df10 = df10[['key', '1', '2', '4', '6', '7']]
    # df10 = df10[['key']]
    df11 = pd.DataFrame(pd.read_csv(path2, header=None, names=count_columns(path2)))
    # df11 = df11[['0', '1']]
    # df11[['key']] = df11[['key']].astype(int)
    # df10.to_csv(result_path, index=False, header=None)
    merge2 = pd.merge(left=df10, right=df11, on=['0'],
                      how="inner")  # 实现满足点数与区域都匹配两个条件时候合并表格，只匹配一个条件采用left_on="plot_no",right_on="plot_no"
    # merge2 = pd.concat([df10, df11], axis=1)
    # cols = merge2.columns[[0,1,2,3,4,5,12,7,8,9,10,11]]
    # merge2 = merge2[cols]
    merge2.to_csv(result, index=False, header=None, encoding='utf-8-sig')


if __name__ == '__main__':
    source = '1230_1d.csv'
    ori_file = '1230_ori.csv'
    add_file1 = 'views0204_cl.csv'
    result = '1230_30seq.csv'
    source_u = add_fname(source, "_u")
    #
    # unique(source, source_u)
    # rewrite(source_u, ori_file)  # 记得取粉丝数自然对数
    # add_views(ori_file, add_file1, result, 5)
    # add_views(result, add_file2, result, 4)
    # unique(result)
    # wash(ori_file)
    # get_pop(result, add_fname(result, '_pn'))
    # merge(ori_file, 'views0204_cl.csv', result)

    # get_ln(result)
    # get_pop_seq(result, add_fname(result, '_p'))
    # all_normalize(add_fname(result, '_p'), add_fname(result, '_pn'))

    # fname = '0918_17_.csv'
    # for i in range(4):
    #     result = add_fname(fname, str(i))
    #     get_pop(result, add_fname(fname, 'p'+str(i)))
    #     all_normalize(add_fname(fname, 'p'+str(i)),
    #                   add_fname(fname, 'pn'+str(i)))

    get_ln("0711_yt_seq.csv", "0711_yt_seq_pn1.csv")
    get_pop_seq("0711_yt_seq_pn1.csv")
    all_normalize("0711_yt_seq_pn1.csv")
