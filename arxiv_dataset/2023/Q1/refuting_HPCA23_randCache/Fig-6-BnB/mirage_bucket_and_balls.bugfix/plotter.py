#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:19:51 2022

@author: anirban
"""

import matplotlib.pyplot as plt
import numpy as np


with open("mirage_bucket_ball_data.txt", 'r') as f1:
    file = f1.readlines()
    


timing = list(map(int, list(map(lambda each:each.strip("\n"), file))))
index = [8, 9, 10, 11, 12, 13, 14]

fig=plt.figure(figsize=(7, 3))
ax = plt.gca()
plt.bar(index, timing, width=0.5)
plt.xlabel("Number of ways (associativity)", fontweight='bold', fontsize=12)
plt.ylabel("Number of trials per\ncollision (bucket spill)", fontweight='bold', fontsize=12)
ax.xaxis.set_tick_params(labelsize=11)
ax.yaxis.set_tick_params(labelsize=11)
plt.xticks(weight = 'bold')
plt.yticks(weight = 'bold')
plt.yscale("log")
plt.ylim(10**4,10**9)
plt.text(13.5, 10**5, "No bucket spills\nobserved", ha='center', va='center', weight='bold', size=12.8)

plt.savefig("mirage_bucket_and_balls.pdf", dpi=1200, bbox_inches = 'tight')
plt.show()
