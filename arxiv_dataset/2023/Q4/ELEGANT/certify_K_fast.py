import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from decimal import Decimal
from numba import jit, njit
from numba.typed import List
from time import time
from multiprocessing import Pool


# # @njit
# def first_loop(complete_cnt, alpha, beta):
# 	outcome = []
# 	raw_cnt = 0
# 	for ((s, t), c) in complete_cnt:
# 		outcome.append(((alpha / beta, t - s), c, s, t))
# 		outcome.append(((alpha / beta, s - t), c, t, s))

# 	return outcome


# # @njit
# def second_loop(outcome, index, alpha, beta, global_d, p_Z, total_Z):
# 	p_given = 0
# 	q_given = 0
# 	for i in index:
# 		ratio, cnt, s, t = outcome[i]
# 		cnt = int(cnt)
# 		p = powe(alpha, global_d - s) * powe(beta, s)
# 		q = powe(alpha, global_d - t) * powe(beta, t)
# 		q_delta = q * cnt
# 		p_delta = p * cnt

# 		if p_given + p_delta < p_Z:
# 			p_given += p_delta
# 			q_given += q_delta
# 		else:
# 			q_given += (p_Z - p_given) / (Decimal(ratio[0]) ** ratio[1])
# 			break
# 	q_given /= Decimal(total_Z)
# 	return q_given


# def inner_loop(args):
# 	p_l, frac_alpha, global_d, v, fn = args
# 	frac_beta = (1 - frac_alpha)
	
# 	alpha = int(frac_alpha * 100)
# 	beta = 100 - alpha

# 	plower_Z = powe(int(p_l * 100), 10) * powe(100, global_d-10)
# 	pupper_Z = powe(int((1-p_l) * 100), 10) * powe(100, global_d-10)
# 	total_Z = 100 ** global_d

# 	complete_cnt = []
# 	cnt = np.load('list_counts/{}/complete_count_{}.npy'.format(fn, v), allow_pickle=True)
# 	complete_cnt += list(cnt)
	
# 	outcome = first_loop(complete_cnt, alpha, beta)
	
# 	outcome.sort(key=lambda x: x[0][1])
# 	index_reverse = range(len(outcome)-1, -1, -1)
# 	q_given_lower = second_loop(outcome, index_reverse, alpha, beta, global_d, plower_Z, total_Z)
	
# 	index = range(len(outcome))
# 	q_given_upper = second_loop(outcome, index, alpha, beta, global_d, pupper_Z, total_Z)
	
# 	if q_given_lower - q_given_upper < 0:
# 		return True
# 	else:
# 		return False


# def certify_K(p_l, frac_alpha, global_d, v_range, fn, idx):

# 	start = time()
# 	with Pool() as pool:
# 		args_list = [(p_l, frac_alpha, global_d, v, fn) for v in range(v_range[0], v_range[1])]
# 		result_list = pool.map(inner_loop, args_list)
# 		print(f'Finish: {idx}, Time: {time() - start}', flush=True)
# 		if True in result_list:
# 			return result_list.index(True)


def certify_K(args):
	
	# if os.path.exists('my_powe.json'):
	# 	myfile = open('my_powe.json', 'r')
	# 	global_powe = json.load(myfile)
	# else:
	# 	global_powe = dict()
	# global_powe = dict()
	global_powe = np.load('my_powe.npy', allow_pickle=True).item()

	def my_powe(k, p):
		if (k, p) not in global_powe:
			global_powe[(k, p)] = k ** p

		return global_powe[(k, p)]
	
	def test_v(p_l, alpha, beta, global_d, v, fn):
		plower_Z = int(my_powe(p_l*100, 10)) * my_powe(100, global_d-10)
		pupper_Z = int(my_powe((1-p_l)*100, 10)) * my_powe(100, global_d-10)
		total_Z = my_powe(100, global_d)
		# plower_Z = int(p_l * 100 ** 10) * (100 ** (global_d-10))
		# pupper_Z = int((1-p_l) * 100 ** 10) * (100 ** (global_d-10))
		# total_Z = 100 ** global_d

		complete_cnt = []
		cnt = np.load('list_counts/{}/complete_count_{}.npy'.format(fn, v), allow_pickle=True)
		complete_cnt += list(cnt)

		# raw_cnt = 0

		outcome = []
		for ((s, t), c) in complete_cnt:
			outcome.append(((alpha / beta, t - s), c, s, t))
			if s != t:
				outcome.append(((alpha / beta, s - t), c, t, s))
		# for ((s, t), c) in complete_cnt:
		# 	outcome.append((
		# 		# likelihood ratio x flips s, x bar flips t
		# 		# and then count, s, t
		# 		(alpha ** (t - s)) * (beta ** (s - t)), c, s, t
		# 	))
		# 	if s != t:
		# 		outcome.append((
		# 			(alpha ** (s - t)) * (beta ** (t - s)), c, t, s
		# 		))

		# 	raw_cnt += c
		# 	if s != t:
		# 		raw_cnt += c

		# sort likelihood ratio in a descending order, i.e., r1 >= r2 >= ...
		outcome.sort(key=lambda x: x[0][1])
		# outcome_descend = sorted(outcome, key = lambda x: -x[0])
		p_given_lower = 0
		q_given_lower = 0
		# print('Begin Forward...')
		for i in tqdm(range(len(outcome)-1, -1, -1)):
			ratio, cnt, s, t = outcome[i]
			# cnt = int(cnt)
			p = int(my_powe(alpha, global_d-s)) * int(my_powe(beta, s))
			q = int(my_powe(alpha, global_d-t)) * int(my_powe(beta, t))
			# p = (alpha ** (global_d - s)) * (beta ** s)
			# q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_lower = q * cnt
			p_delta_lower = p * cnt

			if p_given_lower + p_delta_lower < plower_Z:
				p_given_lower += p_delta_lower
				q_given_lower += q_delta_lower
			else:
				q_given_lower += (plower_Z - p_given_lower) / (Decimal(ratio[0]) ** ratio[1])
				#q_given_lower += q * (plower_Z - p_given_lower) / Decimal(p)
				break
		q_given_lower /= total_Z

		# sort likelihood ratio in a ascending order
		# outcome_ascend = sorted(outcome, key = lambda x: x[0])
		p_given_upper = 0
		q_given_upper = 0
		# print('Begin Backward...')
		for i in tqdm(range(len(outcome))):
			ratio, cnt, s, t = outcome[i]
			# cnt = int(cnt)
			p = int(my_powe(alpha, global_d-s)) * int(my_powe(beta, s))
			q = int(my_powe(alpha, global_d-t)) * int(my_powe(beta, t))
			# p = (alpha ** (global_d - s)) * (beta ** s)
			# q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_upper = q * cnt
			p_delta_upper = p * cnt

			if p_given_upper + p_delta_upper < pupper_Z:
				p_given_upper += p_delta_upper
				q_given_upper += q_delta_upper
			else:
				q_given_upper += (pupper_Z - p_given_upper) / (Decimal(ratio[0]) ** ratio[1])
				#q_given_upper += q * (pupper_Z - p_given_upper) / Decimal(p)
				break
		q_given_upper /= total_Z

		# print(q_given_lower, q_given_upper)

		if Decimal(q_given_lower) - Decimal(q_given_upper) < 0:
			return True
		else:
			return False

	p_l, frac_alpha, global_d, v_range, fn, ind = args

	alpha = int(frac_alpha * 100)
	beta = 100 - alpha

	def binary_search(v_range):

		if v_range[1] >= v_range[0]:
			if v_range[1] - v_range[0] <= 1:
				# myfile = open('my_powe.json', 'w')
				# json.dump(global_powe, myfile)
				# myfile.close()
				return v_range[1]
			mid = int(v_range[0] + (v_range[1] - v_range[0]) / 2)
			start = time()
			res_mid = test_v(p_l, alpha, beta, global_d, mid, fn)
			end = time()
			if res_mid:
				v_range[1] = mid
			else:
				v_range[0] = mid
			# print(f"No.{ind}: v_range: {v_range}, iter time: {end-start}", flush=True)
			return binary_search(v_range)
		else:
			return None
	
	def exp_search(v):
		if v == 0:
			return 1
		res_v = test_v(p_l, alpha, beta, global_d, v, fn)
		if res_v:
			return exp_search(int(v // 2))
		else:
			return int(2 * v)
	
	start = time()
	res_start = test_v(p_l, alpha, beta, global_d, v_range[0], fn)
	if res_start:
		# print(f"No.{ind}: Stop at head {v_range[0]}, time: {time()-start}", flush=True)
		# myfile = open('my_powe.json', 'w')
		# json.dump(global_powe, myfile)
		# myfile.close()
		return v_range[0]
	# print(f"No.{ind}: test start, time: {time()-start}")
	start = time()
	res_end = test_v(p_l, alpha, beta, global_d, v_range[1], fn)
	if not res_end:
		# print(f"No.{ind}: Stop at tail {v_range[1]}, time: {time()-start}", flush=True)
		# myfile = open('my_powe.json', 'w')
		# json.dump(global_powe, myfile)
		# myfile.close()
		return None
	# print(f"v_range: {v_range}, time: {time()-start}")
	# return binary_search(v_range)
	return exp_search(v_range[1])

	# res = test_v(p_l, alpha, beta, global_d, 8, fn)
	# if res:
	# 	return 8
	# return None
