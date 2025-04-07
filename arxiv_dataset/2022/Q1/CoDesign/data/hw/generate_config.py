#!/usr/bin/env python3
import random
import textwrap

num_pes = [512, 256, 128, 64, 32, 16]
noc_bw = [300, 400, 500, 600, 700, 800, 900, 1000]
off_chip_bw = [50, 100, 150, 200, 250, 275, 300, 325, 350]

candidate = set()

for i in range(2, 52):
	pe = random.choice(num_pes)
	noc = random.choice(noc_bw)
	off = random.choice(off_chip_bw)
	
	if (pe, noc, off) not in candidate:
		candidate.add((pe, noc, off))
		
		with open('./accelerator_'+str(i)+'.m', 'w') as f:
			f.write(textwrap.dedent('''\
				num_pes: {}
				l1_size_cstr: 100
				l2_size_cstr: 3000
				noc_bw_cstr: {}
				offchip_bw_cstr: {}
					''').format(pe, noc, off))
			
print(candidate)