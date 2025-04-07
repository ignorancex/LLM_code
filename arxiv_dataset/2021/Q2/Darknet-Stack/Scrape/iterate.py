# Filename:     iterate.py
# Description:  Utility for iterating through links to scrape
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Forked from: Abhishek Singh, absingh31
# Modifed by: Edward Crowder & Jay Lansiquot

import os

name = os.path.abspath("") + '/onions' + '.txt'

for i in range (0,10000):
    file = open(name,'a+')
    file.write(str("http://icarus5l5r3vnn5u.onion/listing/view/{} \n".format(i)))
