#!/usr/bin/python

import sys
import os

hwufile=sys.argv[1]

plots = []

for l in open(hwufile):
    if not l.strip():
        continue
    if  l.startswith('#'):
        name = l[1:].strip().replace(' ','_').replace('(','').replace(')','')
        while name in plots:
            name=name+'_1'
        print 'DOING', name
        thisplot = open(os.path.join(os.path.dirname(hwufile), name+'.dat'), 'w')
        plots.append(name)
    elif not l.strip():
        thisplot.close()
    else:
        thisplot.write(l.replace('D','E'))
