import csv
import os
import math
from experiments_data import folderName2Sections, VOADataRaw


def diffBetween2Points(x1, y1, x2, y2):
    return abs(math.sqrt(x1**2 + y1**2) - math.sqrt(x2**2 + y2**2))


def csv2dots(csv_file, req_len, full_path_index=-1):
    list_of_rows_dic = []
    list_of_dots = []

    if full_path_index == -1:
        x_index = 0
        y_index = 1
    else:
        x_index = full_path_index*3
        y_index = full_path_index*3 + 1

    for index, row in enumerate(csv.reader(csv_file)):
        if row[x_index] == '' or row[y_index] == '':
            break
        list_of_rows_dic.append(
            {'line_num': index, 'x': float(row[x_index]), 'y': float(row[y_index])})

    took_a_point = False        # checking if we already took a point
    wanted_diff = 0.01          # the wanted difference between points

    for index in range(len(list_of_rows_dic)-1):
        diff = diffBetween2Points(list_of_rows_dic[index]['x'],
                                  list_of_rows_dic[index]['y'],
                                  list_of_rows_dic[index + 1]['x'],
                                  list_of_rows_dic[index + 1]['y'])

        if diff < wanted_diff:
            if not took_a_point:    # if it is the first point with a close neighbor
                dot = [list_of_rows_dic[index]['x'],
                       list_of_rows_dic[index]['y']]
                list_of_dots.append(dot)
                took_a_point = True
        else:
            took_a_point = False

    if len(list_of_dots) == req_len - 1:
        dot = [list_of_rows_dic[-1]['x'],
               list_of_rows_dic[-1]['y']]
        list_of_dots.append(dot)

    if len(list_of_dots) != req_len:
        print(csv_file)
    return list_of_dots


def csvc2csv(directory):
    for file in os.listdir(directory):
        if file.endswith(".csv^C"):
            pre, ext = os.path.splitext(file)
            os.rename(pre + ".csv^C", pre + ".csv")


def startPathCorrect(path, folder_name):
    x_mean, y_mean, _, _ = VOADataRaw()
    initial_location = folderName2Sections(folder_name)[0]
    if initial_location == "A":
        index = 0
    elif initial_location == "B":
        index = 1
    elif initial_location == "C":
        index = 2
    elif initial_location == "D":
        index = 3
    elif initial_location == "E":
        index = 4

    diff_x = x_mean[index] - path[0][0]
    diff_y = y_mean[index] - path[0][1]

    for i in range(len(path)):
        path[i][0] += diff_x
        path[i][1] += diff_y

    return path


def folder2Paths(folder_name, folder_location):
    req_len = len(folderName2Sections(folder_name)) + 1
    paths = []

    for file in os.listdir(folder_location + '/' + folder_name):
        if file.endswith(".csv"):
            if folder_name == "full_path":
                for i in range(47):
                    csv_file = open(folder_location + '/' +
                                    folder_name + '/' + file)
                    list_of_dots = csv2dots(csv_file, req_len, i)
                    list_of_dots = startPathCorrect(list_of_dots, folder_name)
                    paths.append(list_of_dots)
            elif folder_name == "csv_correction":
                csv_file = open(folder_location + '/' +
                                folder_name + '/' + file)
                list_of_dots = csv2dots(csv_file, req_len)
                paths.append(list_of_dots)
            else:
                csv_file = open(folder_location + '/' +
                                folder_name + '/' + file)
                list_of_dots = csv2dots(csv_file, req_len)
                list_of_dots = startPathCorrect(list_of_dots, folder_name)
                paths.append(list_of_dots)

    return paths
