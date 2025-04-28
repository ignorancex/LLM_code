# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:10:32 2022

@author: wgrgw
"""

from Server import analyzePhage

fastafile = r"./95% non redundant.fasta"
outputcsv = r"./pdbScores"
   
analyzePhage(fastafile, outputcsv)
    
    



