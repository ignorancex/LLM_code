#!/usr/bin/python

import sys
import os

infile = sys.argv[1]
outfile = infile

content = open(infile).read()


#content = content.replace('set terminal pdf font "Helvetica,14" enhanced dashed size 12 cm, 18 cm', \
#                          'set terminal pdf font "Helvetica,12" enhanced dashed size 8 cm, 12 cm')
#content = content.replace('set label "e^+mu^+{/Symbol nn}jj production at the LHC, 13 TeV" font ",12" at graph 0.03, graph 0.94', \
#                          '')
#content = content.replace('set label "MonteCarlo comparison, NLO fixed order" font ",12" at graph 0.1, graph 0.88', \
#                          'set label "MonteCarlo comparison, NLO fixed order" font ",10" at graph 0.03, graph 0.88')
content = content.replace('set label "NLO" font ",10" at graph 0.03, graph 0.88', \
                          'set label "NLO" font ",10" at graph 0.03, graph 0.94')
#content = content.replace('set key at graph 1, graph 0.9 noautotitles spacing 2.4', \
#                          'set key at graph 1, graph 0.8 noautotitles spacing 2.4')
#content = content.replace('set ylabel \'Ratio /VBFNLO\'', \
#                          'set ylabel \'Ratio /VBFNLO\' offset 1')

open(outfile, 'w').write(content)
