import pstats

## to produce the monitoring_output.txt file first in terminal run this code:
## python -m cProfile -o tools/monitoring_output.txt tools/train.py      ## change the max_epochs variable  to 1
## more information for python 3.6 at https://docs.python.org/3.6/library/profile.html

p = pstats.Stats('monitoring_output.txt')
p.sort_stats('cumulative').print_stats(100)  #This sorts the profile by cumulative time in a function, and then only prints the ten most significant lines. If you want to understand what algorithms are taking time, this line is what you would use.
p.sort_stats('time').print_stats(100)         # looking to see what functions were looping a lot, and taking a lot of time
#p.print_callers(.5, 'call')
#p.sort_stats('cumulative').print_stats('__init__')
#p.print_callees()

