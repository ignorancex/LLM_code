import numpy as np



# find pareto-optimal models

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
	"""
	Find the pareto-efficient points
	:param costs: An (n_points, n_costs) array
	:return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
	"""
	is_efficient = np.ones(costs.shape[0], dtype = bool)
	for i, c in enumerate(costs):
		#print(i, c)
		#print(np.any(costs[is_efficient]<c, axis=1))
		if is_efficient[i]:
			is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
			is_efficient[i] = True  # And keep self
		#print(is_efficient[is_efficient])
	return is_efficient

	
def find_pareto(lat, acc, k):
	arr = np.zeros(shape=(len(lat), 2))
	for i in range(len(lat)):
		arr[i][0] = lat[i]
		arr[i][1] = -1*acc[i]
	
	topk_index = [[] for _ in range(k)]
	topk_lat = [[] for _ in range(k)]
	topk_acc = [[] for _ in range(k)]
		
	for idx in range(k):
		res = is_pareto_efficient_simple(arr)
		
		for i in range(len(res)):
			if res[i]:
				arr[i][1] = float('inf')
				topk_lat[idx].append(lat[i])
				topk_acc[idx].append(acc[i])
				topk_index[idx].append(i)
				
	if k == 1:
		return topk_lat[0], topk_acc[0], topk_index[0]	
				
	return topk_lat, topk_acc, topk_index
	

def select_model(index, arr, acc, k=0):
	index = sorted(index, key=lambda i: arr[i])
	
	select = []
	i = 1
	while i < len(index):
		select.append(index[i])
		i += 3
	
#	step = (arr[index[-1]] - arr[index[0]]) / 10
#	
#	bound = step + arr[index[0]]
#	cur_max = float('-inf')
#	cur_index = None
#	select = []
#	for i in range(len(index)):
#		if arr[index[i]] < bound:
#			if arr[index[i]] > cur_max:
#				cur_max = arr[index[i]]
#				cur_index = index[i]
#		else:
#			select.append(cur_index)		
#			bound += step 
#			cur_max = arr[index[i]]
#			cur_index = index[i] 
			
	select.append(index[-1])
	#select.append(index[1])
#	if acc[select[0]] < 10:
#		select.remove(select[0])
		
	return [arr[i] for i in select], [acc[i] for i in select], select
	
	
def select_pareto_optimal(index, arr, acc, lat_diff=0.3):
	#lat_diff = 0.3
	acc_diff = 0.3
	
	#print([arr[i] for i in index])
	index = sorted(index, key=lambda i: arr[i])
		
	pareto_lat = []
	pareto_acc = []
	pareto_index = []
		
	last_lat = float('inf')
	last_acc = float('-inf')	
	
	for i in range(len(index)):
		temp1 = arr[index[i]]
		temp2 = acc[index[i]]	
		select = i
		for j in range(i+1, len(index)):
			if arr[index[j]] > temp1 and abs(arr[index[j]] - temp1) < lat_diff and acc[index[j]] > temp2:
				temp1 = arr[index[j]]
				temp2 = acc[index[j]]
				select = j
		
		if abs(arr[index[select]] - last_lat) >= lat_diff and acc[index[select]] >= last_acc and abs(acc[index[select]] - last_acc) >= acc_diff:
			pareto_lat.append(arr[index[select]])
			pareto_acc.append(acc[index[select]])
			last_lat = arr[index[select]]
			last_acc = acc[index[select]]
			pareto_index.append(index[select])
			
	return [arr[i] for i in pareto_index], [acc[i] for i in pareto_index], pareto_index
	
# top-k models around each index	
def topk_pareto(index, lat, acc, k):
	
	lat_diff = 0.3
	acc_diff = 0.3
	
	res = []
	
	for i in range(len(index)):
		cur_idx = index[i]
		cur_lat = lat[cur_idx]
		cur_acc = acc[cur_idx]
		
		n_idx = [cur_idx]
		n_lat = [cur_lat]
		n_acc = [cur_acc]
		
		for j in range(len(lat)):
			if abs(lat[j] - cur_lat) < lat_diff and i != j:
				if len(n_idx) < k:
					n_idx.append(j)
					n_lat.append(lat[j])
					n_acc.append(acc[j])
				elif acc[j] > min(n_acc):
					n_idx[n_acc.index(min(n_acc))] = j
					n_lat[n_acc.index(min(n_acc))] = lat[j]
					n_acc[n_acc.index(min(n_acc))] = acc[j]
					
		res.extend(n_idx)
		
	return [lat[i] for i in res], [acc[i] for i in res], res	