#!/usr/bin/env python3

# TODO: rename into full dis axis

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# glob
import glob
# regular expression
import re

def parse():
    parser = argparse.ArgumentParser(description='Post process library that contains most post processing functions.')
    return parser.parse_args()

def get_file_header(file_name):
    head_str = ""
    with open(file_name) as f:
        head_str = f.readline()
    head_str = head_str.replace("#", "")
    head_str = head_str.replace(" ", "")
    head_str = head_str.replace("\n", "")
    head = head_str.split(",")
    return np.array(head_str.split(","), dtype=str)

def get_time():
    time_dir_list = glob.glob("time/*")
    time_dir_list.sort()
    time_list = [float(s.split("/")[1]) for s in time_dir_list]
    return np.array(time_dir_list, dtype=str), np.array(time_list)

def get_equation_property_over_time(equation_name, prop, time_dir_array):
    property_array_over_time = []
    for time_dir in time_dir_array:
        equation = get_equation(time_dir, equation_name)
        property_array = get_equation_properties(equation, [prop])
        info = property_array["info"]
        if info.size > 0:
            property_array = property_array["value"]
            property_array_over_time.append(property_array)
    return property_array_over_time

def get_equation(time_dir, equation_name):
    post_filename = time_dir + "/" + equation_name + ".csv"
    return {"info":get_file_header(post_filename), "value":np.loadtxt(post_filename, delimiter=",")}

def get_equation_properties(obj, property_list):
    regex = ""
    for prop in property_list:
        regex += "{prop}$|".format(prop=prop)
    regex = regex[:-1]
    # extract indexs
    # indexs = [index for index, name in enumerate(obj["info"]) if re.search(regex, name)]
    indexes = np.vectorize(lambda x: bool(re.compile(regex).match(x)))(obj["info"]).nonzero()
    # return
    return {"info":obj["info"][indexes], "value":obj["value"][indexes]}

def get_equation_names():
    post_names = []
    time_dirs = glob.glob("time/*")
    if time_dirs:
        post_paths= glob.glob(time_dirs[0] + "/*")
        for path in post_paths:
            post_names.append(path.split("/")[-1].split(".")[0])
    else:
        raise NameError('No data found')
    return post_names

def get_properties_from_string(name):
    # init
    properties = {}
    # extract properties
    name_properties = name.replace(r"/", "__").split(".")[0].split("__")
    for prop in name_properties:
        prop_split = prop.split("_")
        if len(prop_split) > 1:
            try:
                properties["_".join(prop_split[:-1])] = float(prop_split[-1].replace("o", ".").replace("m", "-"))
            except:
                pass
    # return
    return properties

if __name__ == '__main__':
    parse()
