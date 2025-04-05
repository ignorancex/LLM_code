# histo_positron_M_THEMASS_N_NUMBVAL.dat
from math import log, pi, sqrt
from itertools import izip
with open('FILE1') as inf, open('FILE2') as inf2:
    for line1, line2 in izip(inf, inf2):
        parts = line1.split() # split line into parts
        parts2 = line2.split() # split line into parts
        if len(parts) > 1:   # if at least 2 parts/columns
            Val1 = parts[3]
            Err1 = parts[4]
            bin = parts[0]
            binbis = parts[1]
            Val2 = parts2[1]
            Err2 = parts2[2]
            bin2 = parts2[0]
            if float(Err1)**2 + float(Err2)**2 != 0:
                print "value", float(bin2)-0.5, Val1, Err1, float(Err1)/(float(Val1)+0.0000000001)*100, "|", Val2, Err2, "|", "Sigma", abs(float(Val1)-float(Val2))/sqrt(float(Err1)**2 + float(Err2)**2)
                if abs(float(Val1)-float(Val2))/sqrt(float(Err1)**2 + float(Err2)**2) > 3.0:
                    print "########################### Above 3 sigma ######################"
#                    print "value", (float(binbis)+float(bin))/2.0, bin2, Val1, Err1, "|", Val2, Err2, "|", "Sigma", abs(float(Val1)-float(Val2))/sqrt(float(Err1)**2 + float(Err2)**2)

# print "End"
