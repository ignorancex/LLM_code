#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:16:04 2022

@author: anirban
"""


import random 
import os
import math


outfile = "outfile.txt"
eviction_file = "eviction_status.txt"
base = 16777216

if os.path.exists(outfile):
    os.remove(outfile)

with open("config.ini", "r") as f:
    lines = f.readlines()

cache_size = [16, 32, 64, 128]
throws_list = [100000, 200000, 800000, 1500000]
my_dict = {}

for j in range(len(cache_size)):
    if os.path.exists(eviction_file):
        os.remove(eviction_file)
    with open("config.ini", "w") as f:
        for line in lines:
            if line.strip("\n").startswith("cache-size"):
                f.write("cache-size="+str(int(base * (int(math.pow(2,j)))))+"\n")
            else:
                f.write(line)
    num = []
    for i in range(throws_list[j]):
        num.append(random.randint(0, 100000000))
    
    with open('address_list.txt', 'w') as filehandle:
        filehandle.writelines(str(num))
    os.system("python3 main.py")
    with open(eviction_file) as f:
        first_line = f.readline().strip('\n')
    if first_line.strip("\n").startswith("valid"):
        my_dict[str(cache_size[j])] = throws_list[j]
    else:
        my_dict[str(cache_size[j])] = -1

with open(outfile, "a") as fout:
    for key, value in my_dict.items(): 
        fout.write('%s:%s\n' % (key, value))
