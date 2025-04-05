#!/usr/bin/env python3

"""
This code is for



	or

	BLOCK: offset: 4 len: 14 phased: 14 SPAN: 25449 fragments 14
	4	0	1	chr1	801628	C	T	0/1:69:28	0	.	31.00
	5	0	1	chr1	805678	A	T	0/1:21:70	0	.	100.00

"""

from sys import argv
import sys
sys.path.append('/anaconda3/lib/python3.7/site-packages')
import numpy


def parse_vcf(file):
	""" extracting position vcf file

	input: vcf file name
	output: dictionary
	"""

	vcf_file = open(file, "r")
	dic_vcf = {}
	indx = 0
	for line in vcf_file:
		if not line.startswith('#'):  # 22	17056280	.	C	A	.	.	AC=2;AF=1.00;AN=2
			indx += 1
			slices = line.strip().split("\t")
			chr_name = slices[0]
			var_pos = int(slices[1])
			ref_all = slices[3]
			alt_all = slices[4]
			dic_vcf[indx] = (chr_name,var_pos,ref_all,alt_all)

	return dic_vcf




def parse_hap_estimated_blocks(file):

	""" extracting estimated haplotype blocks from  algorithm's output file

	input: file name
	output: dictionary, label: block number,  value: a dictionary of idx and hap

	the inptut can be like this
	BLOCK	2	7
	2	0
	3	1

	"""
	hap_blocks = open(file, "r")
	dic_blocks = dict()  # dic_blocks = {1: dic_block1, 2: dic_block2}
	dic_ablock = dict()  # dic_block = {2:0 , 3:1 , 28:1}
	block_idx = 0
	start = True
	for line in hap_blocks:
			line = line.strip()
			if line.startswith('B'):  # new block started
				if not start:
					block_idx += 1
					dic_blocks[block_idx] = dic_ablock
				start = False
				dic_ablock = {}
			elif not line.startswith('*'):
				slices_line = line.split("\t")
				if '-' not in slices_line[1]:
					allele = int(slices_line[1])
					idx = int(slices_line[0])
					dic_ablock[idx] = allele
	block_idx += 1
	dic_blocks[block_idx] = dic_ablock
	return dic_blocks




if __name__ == "__main__":

	address = argv[1]
	filename_vcf= argv[2]
	filename_hap = argv[3]
	dic_vcf = parse_vcf(address+filename_vcf)

	dic_hap = parse_hap_estimated_blocks(address+filename_hap)

	fil = open(address+filename_hap[:-4]+'_hpctformat.txt', 'w+')
	for dic_block in dic_hap.values():
		list_ind = list(dic_block)
		list_ind_sort = numpy.sort(list_ind)
		fil.write('BLOCK: offset: {} \n'.format(list_ind_sort[0]))
		for indx, allel in dic_block.items():
			tup_vcf = dic_vcf[indx]
			fil.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(indx, allel, int(not bool(allel)), tup_vcf[0], tup_vcf[1], tup_vcf[2], tup_vcf[3]))
		A = 2
	fil.close()


# BLOCK: offset: 58 len: 50 phased: 38 SPAN: 74289 fragments 72
	# 58	0	1	22	17092688	C	T


	a = 2

#

# 	filename_hap_estimated_alg1 = address+argv[3]
# 	filename_hap_estimated_alg2 = address+argv[4]
# hap_blocks = open(file, "r")
# set_indx = set()
# dic_blocks = dict()  # dic_blocks = {1: dic_block1, 2: dic_block2}
# dic_ablock = dict()  # dic_block = {2:0 , 3:1 , 28:1}
# block_idx = 0
# start = True
# for line in hap_blocks:
#     line = line.strip()
#     if line.startswith('B'):  # new block started
#         if not start:
#             block_idx += 1
#             dic_blocks[block_idx] = dic_ablock
#         start = False
#         dic_ablock = {}
#     elif not line.startswith('*'):
#         slices_line = line.split("\t")
#         if '-' not in slices_line[1]:
#             allele = int(slices_line[1])
#             idx = int(slices_line[0])
#             if not len(list_positions):  # for comparing based on index, not variation position
#                 dic_ablock[idx] = allele
#                 set_indx.add(idx)
#             else:  # for comparing based on variation position
#                 var_position = list_positions[idx - 1]
#                 set_indx.add(var_position)
#                 dic_ablock[var_position] = allele
#             # For future  if len(slices_line) > 3:  # hapcut2: 4	0	1	chr1	801628	C ...
#             # var_position = int(line_slice[4])
# block_idx += 1
# dic_blocks[block_idx] = dic_ablock
