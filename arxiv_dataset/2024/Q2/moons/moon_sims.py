#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rebound
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as c

# local imports
import heartbeat
import globs
import metasim


# In[2]:


def run_sim(filestem='test/test',tmoons=1e3):
    
    # global variables

    globs.initialise()

    globs.glob_dclo = 1. #CE distance to check in rH
    #dir = 'test/'
    globs.glob_archive = ''
    #globals.glob_names = []
    globs.glob_is_close = False

    system = metasim.MetaSim(filestem=filestem,tmoons=tmoons)
    system.run_planets()
    if globs.glob_is_close:
        system.rewind()
        system.add_moons()
        system.run_moons()
        system.analyse()
        system.make_timeline()


# In[3]:


#Nsys = 1

#for i in range(Nsys):
#    run_sim(filestem=f'test/test{i:04d}')

i = 0
tmoons = 1e4
run_sim(filestem=f'test/test{i:04d}',tmoons=tmoons)


# In[ ]:




