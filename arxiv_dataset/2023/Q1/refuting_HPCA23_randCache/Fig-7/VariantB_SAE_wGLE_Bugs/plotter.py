#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:19:51 2022

@author: anirban
"""
import matplotlib.pyplot as plt
import numpy as np


with open("outfile.txt", 'r') as f1:
    file = f1.readlines()
   
X = []; Y = []

for line in file:
    cache_size = line.strip("\n").split(":")[0]
    throw_count = line.strip("\n").split(":")[1]
    X.append(cache_size)
    Y.append(int(throw_count))
    
FIG_SIZE = (6,2) #Horizantal and Vertical dimension of figure
GRID_ALPHA = 0.5 #Intensity of gridlines

obj = ['100K','200K','800K','1.5M'] #labelling of points
fig, ax = plt.subplots(figsize = FIG_SIZE)
#Comment this for removing gridlines
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
#Set the labelsize of the tick values
ax.tick_params(colors='black',labelsize=10)
#X labelsize and values
ax.set_xlabel('Size of Cache in MB', fontsize = 10, fontweight='bold')
#Y labelsize and values
ax.set_ylabel("Number of"+"\n"+"balls thrown", fontsize = 10, fontweight='bold')
#Plotting with markertype, color, and linewidth
ax.plot(X, Y, marker="o", markerfacecolor='blue', ms=5, color="blue", linewidth=1)
for i,label in enumerate(obj):

    plt.annotate(label, # this is the text
                 (X[i],Y[i]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,8), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 fontsize=10) #fontsize

fig.savefig('Figure7_HPCAPaper.png', bbox_inches = "tight", dpi=200)
fig.show()