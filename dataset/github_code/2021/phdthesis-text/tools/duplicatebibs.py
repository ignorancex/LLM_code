#!/bin/python

"""Find duplicate BibTex entries.

This small script finds duplicate "@artcicle" entries in a given BibTex file.

Example:
		To find duplicate entries in 'papers.bib' simply run:

				$ python duplicatebibs.py papers.bib
		
		Which will output something like this if you have duplicates:
				$                 Galenkamp-2018: 2
    		$               VanMeervelt-2017: 2
    		$                     Wloka-2017: 2
    		$       Willems-VanMeervelt-2017: 2
		Or if you have no duplicates:
				$ No duplicate entries found.

	@author Khalil Muhammad (https://github.com/micaleel)
	@author Kherim Willems (https://github.com/willemsk)

"""

import sys
import os
from collections import Counter, OrderedDict
from pprint import pprint


def extract_id(line):
	x = line.index('{')
	y = line.index(',')
	return line[x+1: y].strip()


def main():
	filepath = sys.argv[1]
	assert os.path.exists(filepath)

	with open(filepath, 'r') as fp:
		lines = [extract_id(line) for line in fp.readlines() 
		if (line.startswith('@article') or line.startswith('@Article'))]

	counts = Counter(lines)
	dups = {}
	for k,v in counts.items():
		if v > 1:
			dups[k] = v
	
	dups = OrderedDict(sorted(dups.items()))
	
	if len(dups) > 0:
		for k,v in dups.items():
					print(f'{k:>35}: {v:d}')
		print(f'Found {len(dups):d} entries with duplicates.')
	else:
		print('No duplicate entries found.')

if __name__ == '__main__':
	main()