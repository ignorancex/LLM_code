# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import comb
from fractions import Fraction
from os import listdir
from os.path import isdir, isfile, join
from sklearn import linear_model
from scipy.stats import chisquare
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter

CACHED_FNN_DIR="fnn_fractions"
USHMM_BIRTH_YEARS_DIR="ushmm_birth_years"

def load_lines(path, encoding=None):
	
	lines=[]
	if (encoding is None):
		with open(path, "r") as f:
			lines=[l.strip("\r\n") for l in f.readlines()]
	else:
		with open(path, "r", encoding=encoding) as f:
			lines=[l.strip("\r\n") for l in f.readlines()]
	
	return lines

def load_lines_generator(path, encoding=None):
	
	if (encoding is None):
		with open(path, "r") as f:
			for line in f:
				yield line.strip("\r\n")
	else:
		with open(path, "r", encoding=encoding) as f:
			for line in f:
				yield line.strip("\r\n")

def multinomial(n, ks):
	result=np.math.factorial(n)
	for k in ks:
		result//=np.math.factorial(k)
	return result
	
def f2N(N):
	
	return (2**(1-N))*((N+1)//2)*comb(N, N//2, exact=True)

def fnn_exact(n, N, use_cached=False):
	
	if (use_cached is True):
		path=join(CACHED_FNN_DIR, "fnn_%d_%d.txt"%(n, N))
		if (isfile(path) is True):
			values=list(map(int, load_lines(path)))
			return values[0]/values[1]
	if (n==2):
		return f2N(N)
	
	result=0
	for k2 in range(1, N+1):
		for k1 in range(0, min(k2, N-k2)):
			if (N-k1-k2<0):
				print(k2, k1, N)
			result+=Fraction(multinomial(N, [k1, k2, N-k1-k2])*(k2-k1), ((n-2)**(k1+k2)))
	
	result*=2*(n-1)*(Fraction(n-2, n))**N
	
	return result.numerator/result.denominator

def fnn_approximate(n, N):
	return 2*(n-1)*np.sqrt(N/(n*np.pi))

def get_tvor_coef(xs, ys, use_ransac=False):
	model_x3=np.array([xs, np.sqrt(np.array(xs))]).T
	
	if (use_ransac is False):
		model=linear_model.LinearRegression()
	else:
		model=linear_model.RANSACRegressor()
	model.fit_intercept=False
	model.fit(model_x3, ys)
	
	if (use_ransac is False):
		cs=model.coef_
	else:
		cs=model.estimator_.coef_
	
	return cs

def calculate_tvor_scores(xs, ys, names, use_ransac=False, print_count=10):
	
	model_x3=np.array([xs, np.sqrt(np.array(xs))]).T
	if (use_ransac is False):
		model=linear_model.LinearRegression()
	else:
		model=linear_model.RANSACRegressor()
	model.fit_intercept=False
	model.fit(model_x3, ys)
	
	distances=[]
	for i in range(len(names)):
		x=xs[i]
		y=ys[i]
		y_m=model.predict([[x, np.sqrt(x)]])[0]
		s=np.sqrt(x)
		distances.append((abs(y-y_m)/s, names[i]))
	distances=sorted(distances, key=lambda x:-x[0])
	if (print_count is not None):
		for i in range(print_count):
			print(i+1, *distances[i])
	
	return distances
	
def calculate_baseline_scores(data, names, eps=1e-6, print_count=10):
	
	data=np.array(data)
	distances=[]
	indices=np.array(range(len(names)))
	for i in range(len(names)):
		target=data[i, :]
		other=data[indices!=i, :]
		other=np.sum(data, axis=0)
		st, p=chisquare(f_obs=target, f_exp=other/np.sum(other)*np.sum(target)+eps)
		distances.append((st, names[i]))
	distances=sorted(distances, key=lambda x:-x[0])
	if (print_count is not None):
		for i in range(print_count):
			print(i+1, *distances[i])
	
	return distances

def simulate_normal_test_point(b, bins_count, in_m, in_std, out_m, out_std, samples_count, other_samples_count, lower_size, upper_size, point_repetitions):
	
	lower_bound=-b
	upper_bound=b
	
	get_histogram=lambda x:np.array(np.histogram(x, bins=np.linspace(lower_bound, upper_bound, bins_count+1))[0])
	
	tvor_results=[]
	tvor_ransac_results=[]
	baseline_results=[]
	
	for pri in range(point_repetitions):
		xs=[]
		ys=[]
		names=[]
		histograms=[]
		
		for i in range(samples_count):
			size=np.random.randint(lower_size, upper_size+1)
			sample=np.random.normal(in_m, in_std, size)
			sample[sample<lower_bound]=lower_bound
			sample[upper_bound<sample]=upper_bound
			bins=get_histogram(sample)
			histograms.append(bins)
			xs.append(size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("1")
		
		for i in range(other_samples_count):
			size=np.random.randint(lower_size, upper_size+1)
			sample=np.random.normal(out_m, out_std, size)
			sample[sample<lower_bound]=lower_bound
			sample[upper_bound<sample]=upper_bound
			bins=get_histogram(sample)
			histograms.append(bins)
			xs.append(size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("0")
		
		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=False, print_count=None)
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_results.append(np.mean(indices))
	
		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=True, print_count=None)
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_ransac_results.append(np.mean(indices))
		
		distances=calculate_baseline_scores(data=histograms, names=names, print_count=None)
		scores=[]
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				scores.append(distances[i][0])
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		baseline_results.append(np.mean(indices))
	
	return np.mean(tvor_results), np.mean(tvor_ransac_results), np.mean(baseline_results)

def create_normal_plot_bins(b, bins_counts, in_m, in_std, out_m, out_std, samples_count, other_samples_count, lower_size, upper_size, point_repetitions, show_ransac=True, output_path=None, show=True, verbose=True):
	tvor_xs=[]
	tvor_ys=[]
	tvor_ransac_xs=[]
	tvor_ransac_ys=[]
	baseline_xs=[]
	baseline_ys=[]
	
	for bins_count in bins_counts:
		tr, trr, br=simulate_normal_test_point(b=b, bins_count=bins_count, in_m=in_m, in_std=in_std, out_m=out_m, out_std=out_std, samples_count=samples_count, other_samples_count=other_samples_count, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions)
		if (verbose is True):
			print(bins_count)
			print("  TVOR:          ", tr)
			print("  TVOR + RANSAC: ", trr)
			print("  Baseline:      ", br)
			print()
		
		tvor_xs.append(bins_count)
		tvor_ys.append(tr)

		tvor_ransac_xs.append(bins_count)
		tvor_ransac_ys.append(trr)

		baseline_xs.append(bins_count)
		baseline_ys.append(br)

	label_fontsize=18
	tick_fontsize=12
	matplotlib.rc("font", size=tick_fontsize)
	plt.figure(figsize=(7, 5))
	plt.plot(tvor_xs, tvor_ys, "-o", label="TVOR")
	if (show_ransac is True):
		plt.plot(tvor_ransac_xs, tvor_ransac_ys, "-o", label="TVOR + RANSAC")
	plt.plot(baseline_xs, baseline_ys, "-o", label="baseline")
	plt.xticks(list(range(5, 50+1, 5)))

	plt.xlabel("Bins", fontsize=label_fontsize)
	plt.ylabel("Average outlier rank", fontsize=label_fontsize)
	plt.legend()
	if (output_path is not None):
		plt.savefig(output_path, format="png", dpi=300)
	if (show is True):
		plt.show()
	plt.clf()

def create_normal_plot_sizes(b, bins_count, size_base, size_powers, lower_to_upper_size_factor, in_m, in_std, out_m, out_std, samples_count, other_samples_count, point_repetitions, show_ransac=True, output_path=None, show=True, verbose=True):
	tvor_xs=[]
	tvor_ys=[]
	tvor_ransac_xs=[]
	tvor_ransac_ys=[]
	baseline_xs=[]
	baseline_ys=[]
	
	prepare_plot_x=lambda x:int(round(np.log(x)/np.log(size_base)))
	
	ls=list(size_base**p for p in size_powers)
	for lower_size in ls:
		upper_size=lower_size*lower_to_upper_size_factor

		tr, trr, br=simulate_normal_test_point(b=b, bins_count=bins_count, in_m=in_m, in_std=in_std, out_m=out_m, out_std=out_std, samples_count=samples_count, other_samples_count=other_samples_count, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions)
		
		if (verbose is True):
			print(lower_size)
			print("  TVOR:          ", tr)
			print("  TVOR + RANSAC: ", trr)
			print("  Baseline:      ", br)
			print()
		
		plot_x=prepare_plot_x(lower_size)

		tvor_xs.append(plot_x)
		tvor_ys.append(tr)

		tvor_ransac_xs.append(plot_x)
		tvor_ransac_ys.append(trr)

		baseline_xs.append(plot_x)
		baseline_ys.append(br)

	label_fontsize=18
	tick_fontsize=12
	matplotlib.rc("font", size=tick_fontsize)
	plt.figure(figsize=(7, 5))
	plt.plot(tvor_xs, tvor_ys, "-o", label="TVOR")
	if (show_ransac is True):
		plt.plot(tvor_ransac_xs, tvor_ransac_ys, "-o", label="TVOR + RANSAC")
	plt.plot(baseline_xs, baseline_ys, "-o", label="baseline")
	logs=size_powers
	plt.xticks(logs, list(map(lambda x:"$"+str(size_base)+"^{"+str(x)+"}$", logs)))

	plt.xlabel("Lower size bound", fontsize=label_fontsize)
	plt.ylabel("Average outlier rank", fontsize=label_fontsize)
	plt.legend()
	if (output_path is not None):
		plt.savefig(output_path, format="png", dpi=300)
	if (show is True):
		plt.show()
	plt.clf()

def visualize_bins(bins, lower_bound, upper_bound, output_path=None, show=True):
	
	label_fontsize=18
	tick_fontsize=12
	
	matplotlib.rc("font", size=tick_fontsize)
	
	show_sample=[]
	for bi in range(len(bins)):
		show_sample.extend([(bi+0.5)/len(bins)]*bins[bi])
	plt.figure(figsize=(7, 5))
	plt.hist(show_sample, bins=np.linspace(lower_bound, upper_bound, len(bins)+1), edgecolor="black")
	plt.xlabel("Value", fontsize=label_fontsize)
	plt.ylabel("Count", fontsize=label_fontsize)
	if (output_path is not None):
		plt.savefig(output_path, format="png", dpi=300)
	if (show is True):
		plt.show()
	plt.clf()
	
def simulate_age_heaping(bins, age_heaping_amount, target=5):
	
	out_size=sum(bins)
	bins_count=len(bins)
	
	take=int(round(out_size*age_heaping_amount))
	indices=[]
	for bi in range(bins_count):
		if ((bi+1)%target!=0 and bins[bi]>0):
			indices.extend([bi]*bins[bi])
	
	take=min(take, len(indices))
	subtract_idx=np.random.choice(indices, size=take, replace=False)
	
	add_idx=((subtract_idx+target//2)//target)*target
	add_idx[add_idx>=len(bins)]=len(bins)-len(bins)%target-1
	
	np.add.at(bins, subtract_idx, -1)
	np.add.at(bins, add_idx, 1)

def simulate_beta_ah_test_point(bins_count, samples_count, other_samples_count, lower_size, upper_size, age_heaping_amount, target, p_a, p_b, point_repetitions):
	in_a=p_a
	in_b=p_b
	
	out_a=p_a
	out_b=p_b

	out_upper_size=upper_size
	out_lower_size=lower_size
	
	lower_bound=0
	upper_bound=1
	
	get_histogram=lambda x:np.array(np.histogram(x, bins=np.linspace(lower_bound, upper_bound, bins_count+1))[0])
	
	tvor_results=[]
	tvor_ransac_results=[]
	baseline_results=[]
	for pri in range(point_repetitions):
		xs=[]
		ys=[]
		names=[]
		histograms=[]

		for i in range(samples_count):
			size=np.random.randint(lower_size, upper_size+1)
			sample=np.random.beta(in_a, in_b, size)
			bins=get_histogram(sample)
			histograms.append(bins)
			xs.append(size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("1")

		for i in range(other_samples_count):
			out_size=np.random.randint(out_lower_size, out_upper_size+1)
			sample=np.random.beta(out_a, out_b, out_size)
			bins=get_histogram(sample)
			simulate_age_heaping(bins=bins, age_heaping_amount=age_heaping_amount, target=target)
			
			histograms.append(bins)
			
			xs.append(out_size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("0")
		
		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=False, print_count=None)
		scores=[]
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				scores.append(distances[i][0])
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_results.append(np.mean(indices))

		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=True, print_count=None)
		scores=[]
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				scores.append(distances[i][0])
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_ransac_results.append(np.mean(indices))

		distances=calculate_baseline_scores(data=histograms, names=names, print_count=None)
		scores=[]
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				scores.append(distances[i][0])
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		baseline_results.append(np.mean(indices))
	
	tr=np.mean(tvor_results)
	trr=np.mean(tvor_ransac_results)
	br=np.mean(baseline_results)
	
	return tr, trr, br

def create_beta_ah_plot(bins_count, samples_count, other_samples_count, lower_size, upper_size, age_heaping_amounts, target, p_a, p_b, point_repetitions, show_ransac=True, output_path=None, show=True, verbose=True):
	tvor_xs=[]
	tvor_ys=[]
	tvor_ransac_xs=[]
	tvor_ransac_ys=[]
	baseline_xs=[]
	baseline_ys=[]
	for age_heaping_amount in age_heaping_amounts:
		tr, trr, br=simulate_beta_ah_test_point(bins_count=bins_count, samples_count=samples_count, other_samples_count=other_samples_count, lower_size=lower_size, upper_size=upper_size, age_heaping_amount=age_heaping_amount, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions)
		
		if (verbose is True):
			print(age_heaping_amount)
			print("  TVOR:          ", tr)
			print("  TVOR + RANSAC: ", trr)
			print("  Baseline:      ", br)
			print()
		
		tvor_xs.append(age_heaping_amount)
		tvor_ys.append(tr)
		
		tvor_ransac_xs.append(age_heaping_amount)
		tvor_ransac_ys.append(trr)
		
		baseline_xs.append(age_heaping_amount)
		baseline_ys.append(br)
	
	label_fontsize=18
	tick_fontsize=12
	matplotlib.rc("font", size=tick_fontsize)
	
	plt.xlabel("Amount of heaped values", fontsize=label_fontsize)
	plt.ylabel("Average outlier rank", fontsize=label_fontsize)

	plt.plot(tvor_xs, tvor_ys, "-o", label="TVOR")
	if (show_ransac is True):
		plt.plot(tvor_ransac_xs, tvor_ransac_ys, "-o", label="TVOR + RANSAC")
	plt.plot(baseline_xs, baseline_ys, "-o", label="Baseline")
	plt.legend()
	xticks=[0.1, 0.2, 0.3, 0.4, 0.5]
	plt.xticks(xticks, list(map(lambda x:str(round(x*100))+"%", xticks)))
	if (output_path is not None):
		plt.savefig(output_path, format="png", dpi=300)
	if (show is True):
		plt.show()
	plt.clf()

def simulate_beta_triangular_test_point(bins_count, in_a, in_b, out_left, out_right, out_mode, samples_count, other_samples_count, lower_size, upper_size, point_repetitions):
	
	lower_bound=0
	upper_bound=1
	
	get_histogram=lambda x:np.array(np.histogram(x, bins=np.linspace(lower_bound, upper_bound, bins_count+1))[0])
	
	tvor_results=[]
	tvor_ransac_results=[]
	baseline_results=[]
	
	for pri in range(point_repetitions):
		xs=[]
		ys=[]
		names=[]
		histograms=[]
		
		for i in range(samples_count):
			size=np.random.randint(lower_size, upper_size+1)
			sample=np.random.beta(in_a, in_b, size)
			bins=get_histogram(sample)
			histograms.append(bins)
			xs.append(size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("1")
		
		for i in range(other_samples_count):
			size=np.random.randint(lower_size, upper_size+1)
			sample=np.random.triangular(left=out_left, mode=out_mode, right=out_right, size=size)
			bins=get_histogram(sample)
			histograms.append(bins)
			xs.append(size)
			ys.append(np.sum(np.abs(bins[:-1]-bins[1:])))
			names.append("0")
		
		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=False, print_count=None)
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_results.append(np.mean(indices))
	
		distances=calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=True, print_count=None)
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		tvor_ransac_results.append(np.mean(indices))
		
		distances=calculate_baseline_scores(data=histograms, names=names, print_count=None)
		scores=[]
		indices=[]
		for i in range(len(distances)):
			if (distances[i][1]=="0"):
				scores.append(distances[i][0])
				indices.append(i)
				if (len(indices)==other_samples_count):
					break
		baseline_results.append(np.mean(indices))
	
	return np.mean(tvor_results), np.mean(tvor_ransac_results), np.mean(baseline_results)

def create_beta_triangular_plot_bins(bins_counts, in_a, in_b, out_left, out_right, out_mode, samples_count, other_samples_count, lower_size, upper_size, point_repetitions, show_ransac=True, output_path=None, show=True, verbose=True):
	tvor_xs=[]
	tvor_ys=[]
	tvor_ransac_xs=[]
	tvor_ransac_ys=[]
	baseline_xs=[]
	baseline_ys=[]
	
	for bins_count in bins_counts:
		tr, trr, br=simulate_beta_triangular_test_point(bins_count=bins_count, in_a=in_a, in_b=in_b, out_left=out_left, out_right=out_right, out_mode=out_mode, samples_count=samples_count, other_samples_count=other_samples_count, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions)
		if (verbose is True):
			print(bins_count)
			print("  TVOR:          ", tr)
			print("  TVOR + RANSAC: ", trr)
			print("  Baseline:      ", br)
			print()
		
		tvor_xs.append(bins_count)
		tvor_ys.append(tr)

		tvor_ransac_xs.append(bins_count)
		tvor_ransac_ys.append(trr)

		baseline_xs.append(bins_count)
		baseline_ys.append(br)

	label_fontsize=18
	tick_fontsize=12
	matplotlib.rc("font", size=tick_fontsize)
	plt.figure(figsize=(7, 5))
	plt.plot(tvor_xs, tvor_ys, "-o", label="TVOR")
	if (show_ransac is True):
		plt.plot(tvor_ransac_xs, tvor_ransac_ys, "-o", label="TVOR + RANSAC")
	plt.plot(baseline_xs, baseline_ys, "-o", label="baseline")
	plt.xticks(list(range(5, 50+1, 5)))

	plt.xlabel("Bins", fontsize=label_fontsize)
	plt.ylabel("Average outlier rank", fontsize=label_fontsize)
	plt.legend()
	if (output_path is not None):
		plt.savefig(output_path, format="png", dpi=300)
	if (show is True):
		plt.show()
	plt.clf()

def load_numbers(path):
	return list(map(int, load_lines(path)))

def get_counts(data):
	return dict(Counter(data))

def get_ushmm_data(input_dir=USHMM_BIRTH_YEARS_DIR, input_path=None, lower_limit=100, lower_bound=1850, upper_bound=1945):
	
	if (isdir(input_dir) is False):
		return None, None, None, None
	
	if (input_path is None):
		paths=listdir(input_dir)
	else:
		ids=load_lines(input_path)
		paths=[]
		for id in ids:
			name="Source_"+str(id)+"_birth_years.txt"
			path=join(input_dir, name)
			if (isfile(path) is True):
				paths.append(name)
	
	xs=[]
	ys=[]
	names=[]
	all_years=[]
	histograms=[]
	for name in paths:
		years=load_numbers(join(input_dir, name))
		years=list(filter(lambda x:lower_bound<=x and x<=upper_bound, years))
		if (len(years)<lower_limit):
			continue
		counts=get_counts(years)
		
		y1=min(years)
		y2=max(years)
		y1=lower_bound
		y2=upper_bound
		gm=max(counts.values())
		ds=[]
		histograms.append([counts.get(y, 0) for y in range(y1, y2+1)])
		for y in range(y1, y2):
			c=counts.get(y, 0)
			c2=counts.get(y+1, 0)
			ds.append(abs(c-c2))
		
		xs.append(len(years))
		ys.append(sum(ds))
		names.append(name)
		all_years.append(years)
	
	return xs, ys, names, all_years, histograms

def plot_birth_year_histogram(years_counts, xlabel="Birth year", ylabel="Number of born people", legend_patches=None, figsize=(13, 8), tick_fontsize=20, label_fontsize=30, selected_years=None, default_year_color="#1F77B4", year_colors=None, save_path=None, dpi=300, show=True, repaints=None):
	
	years_counts=sorted(years_counts.items(), key=lambda x:x[0])
	years, counts=zip(*[(x[0], x[1]) for x in years_counts])
	
	colors=[default_year_color]*len(years)
	
	if (year_colors is not None):
		year_index={y:i for y, i in zip(years, range(len(years)))}
		for year, color in year_colors.items():
			if (year in year_index.keys()):
				colors[year_index[year]]=color
	
	matplotlib.rc("font", size=tick_fontsize)
	fig=plt.figure(figsize=figsize)
	ax=fig.add_subplot(1, 1, 1)
	ax.bar(years, height=counts, width=1.0, align="center", edgecolor="black", color=colors)
	if (repaints is not None):
		for x, h, c in repaints:
			if (c is None):
				c=[default_year_color]*len(x)
			ax.bar(x, height=h, width=1.0, align="center", edgecolor="black", color=c)
		
	font={"family":"normal", "weight":"bold", "size":tick_fontsize}
	plt.xlabel(xlabel, fontsize=label_fontsize)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	
	if (selected_years is None):
		selected_years=sorted(set([years[0], *filter(lambda x:x%10==0, range(years[0], years[-1])), years[-1]]))
	
	plt.xticks(selected_years)
	
	if (legend_patches is not None):
		handles=[]
		for label, color in legend_patches:
			handles.append(mpatches.Patch(label=label, color=color))
		ax.legend(handles=handles, loc=2)
	
	plt.tight_layout()
	if (save_path is not None):
		p=save_path.rfind(".")
		if (p!=-1):
			format=save_path[p+1:]
		plt.savefig(save_path, format=format, dpi=dpi)
	
	if (show is True):
		plt.show()
	
	plt.clf()

def plot_default_birth_year_histogram(years=None, counts=None, lower_bound=1850, upper_bound=1945, figsize=(13, 8), show=True, save_path=None, ww1_label="World War I", xlabel="Birth year", ylabel="Number of born people"):
	
	if (years is None and counts is None):
		return
	
	if (years is not None):
		years=list(filter(lambda x:lower_bound<=x and x<=upper_bound, years))
	if (counts is None):
		counts=get_counts(years)
	
	ww1_color="#555555"
	
	legend_patches=[
					(ww1_label, ww1_color)
					]
	
	plot_birth_year_histogram(counts, selected_years=list(range(lower_bound, 1945, 10)), year_colors={1915:ww1_color, 1916:ww1_color, 1917:ww1_color, 1918:ww1_color}, xlabel=xlabel, ylabel=ylabel, legend_patches=legend_patches, figsize=figsize, show=show, save_path=save_path)

def get_ushmm_birth_years_path(id):
	return join(USHMM_BIRTH_YEARS_DIR, "Source_"+str(id)+"_birth_years.txt")

def create_ushmm_plot(id=None, path=None, lower_bound=1850, upper_bound=1945, figsize=(21, 13), show=True, save_path=None):
	
	if (id is None and path is None):
		return
	
	from os.path import join
	
	if (path is None):
		path=get_ushmm_birth_years_path(id)
	
	years=load_numbers(path)
	plot_default_birth_year_histogram(years=years, lower_bound=lower_bound, upper_bound=upper_bound, figsize=figsize, show=show, save_path=save_path)

def get_jms_birth_years():
	return load_numbers(get_ushmm_birth_years_path(45409))

def create_jms_plot(figsize=(13, 8), tick_fontsize=20, label_fontsize=30, nationality="", camp="", place="", take_males=True, take_females=True, must_have_in_remarks=[], must_not_have_in_remarks=[], additional_colors=True, build_base=True, show=True, save_path=None):
	
	years=get_jms_birth_years()
	
	years=list(filter(lambda x:1850<=x and x<=1945, years))
	counts=get_counts(years)
	
	ww1_color="#555555"
	years0_color="#FF0000"
	years2_color="#FFC90E"
	pattern_color="#5FB7F4"
	
	legend_patches=[
					("World War I", ww1_color)]
	
	if (additional_colors is True):
		legend_patches.extend(
					[
					("Years ending with a 0", years0_color),
					("Years ending with a 2", years2_color),
					("Mid-decade pattern", pattern_color)
					])
	
	repaints=[]
	
	if (build_base is True):
		#0s
		x=[]
		h=[]
		keys=sorted(counts.keys())
		for year in keys:
			if (year%10==0 and year!=keys[0] and year!=keys[-1]):
				x.append(year)
				ch=(counts.get(year-1, 0)+counts.get(year+1, 0))//2
				if (counts[year]<ch):
					ch=counts[year]
				h.append(ch)
		repaints.append((x, h, None))
		
		#2s
		x=[]
		h=[]
		keys=sorted(counts.keys())
		for year in keys:
			if (year%10==2 and year!=keys[0] and year!=keys[-1]):
				x.append(year)
				ch=(counts.get(year-1, 0)+counts.get(year+1, 0))//2
				if (counts[year]<ch):
					ch=counts[year]
				h.append(ch)
		repaints.append((x, h, None))
		
	pattern_bounds=[
					(1874, 1878),
					(1884, 1888),
					(1894, 1898),
					(1904, 1908),
					(1924, 1928)
					]
	
	for lower_bound, upper_bound in pattern_bounds:
		left=counts.get(lower_bound-1, 0)
		right=counts.get(upper_bound+1, 0)
		
		x=[]
		h=[]
		n=upper_bound-lower_bound+2
		for year in range(lower_bound, upper_bound+1):
			x.append(year)
			f=(year-lower_bound+1)/float(n)
			ch=int((1-f)*left+f*right)
			if (counts.get(year, 0)<ch):
				ch=counts.get(year)
			h.append(ch)
		repaints.append((x, h, None))
	
	x=[]
	h=[]
	keys=[1914]
	for year in keys:
		x.append(year)
		ch=(counts.get(year-1, 0)+counts.get(year+1, 0))//2
		if (counts.get(year, 0)<ch):
			ch=counts.get(year, 0)
		h.append(ch)
	repaints.append((x, h, None))
	
	year_colors=dict()
	year_colors[1915]=ww1_color
	year_colors[1916]=ww1_color
	year_colors[1917]=ww1_color
	year_colors[1918]=ww1_color
	if (additional_colors is True):
		year_colors[1914]=pattern_color
		for lower_bound, upper_bound in pattern_bounds:
			for i in range(lower_bound, upper_bound+1):
				year_colors[i]=pattern_color
		for year in range(1850, 1945, 10):
			year_colors[year]=years0_color
		for year in range(1852, 1945, 10):
			year_colors[year]=years2_color
	else:
		repaints=None
	
	plot_birth_year_histogram(counts, selected_years=list(range(1850, 1945, 10)), year_colors=year_colors, legend_patches=legend_patches, repaints=repaints, show=show, save_path=save_path, figsize=figsize, tick_fontsize=tick_fontsize, label_fontsize=label_fontsize)

def create_ushmm_lists_plot(input_dir="ushmm_birth_years", input_path=None, model_path=None, labeled_points=None, distances_output_path=None, lower_limit=100, lower_bound=1850, upper_bound=1945, targets=[45409], target_label="Jasenovac", figsize=(21, 13), tick_fontsize=20, label_fontsize=30, add_annotation_x=8000, add_annotation_y=-1000, xlabel="List size", ylabel="Total variation of histogram", logx=False, logy=False, global_maximum=True, normalize=False, save_path=None, dpi=300, show=True, draw_tvor=False, legend=None):
	
	if (isdir(input_dir) is False):
		return
	
	if (input_path is None):
		paths=listdir(input_dir)
	else:
		ids=load_lines(input_path)
		paths=[]
		for id in ids:
			name="Source_"+str(id)+"_birth_years.txt"
			path=join(input_dir, name)
			if (isfile(path) is True):
				paths.append(name)
	
	xs=[]
	ys=[]
	txs=[]
	tys=[]
	
	all_xs=[]
	all_ys=[]
	
	labels=[]
	for name in paths:
		target=False
		for t in targets:
			if (name.find(str(t))!=-1):
				target=True
				break
		years=load_numbers(join(input_dir, name))
		years=list(filter(lambda x:lower_bound<=x and x<=upper_bound, years))
		if (len(years)<lower_limit):
			continue
		counts=get_counts(years)
		
		if (normalize is True):
			counts_sum=sum(counts.values())
			for k in counts.keys():
				counts[k]/=counts_sum
		
		y1=min(years)
		y2=max(years)
		y1=lower_bound
		y2=upper_bound
		gm=max(counts.values())
		ds=[]
		for y in range(y1, y2):
			c=counts.get(y, 0)
			c2=counts.get(y+1, 0)
			cm=max(c, c2)
			m=cm
			if (global_maximum is True):
				m=gm
			if (m==0 or global_maximum is None):
				m=1
			ds.append(float(abs(c-c2))/float(m))
		ds=sorted(ds)
		xv=len(years)
		
		yv=sum(ds)
		
		label=int(name.split("_")[1])
		labels.append(label)
		if (target is False):
			xs.append(xv)
			ys.append(yv)
		else:
			txs.append(xv)
			tys.append(yv)
		
		all_xs.append(xv)
		all_ys.append(yv)
		
	matplotlib.rc("font", size=tick_fontsize)
	fig=plt.figure(figsize=figsize)
	ax=fig.add_subplot(1, 1, 1)
	ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position:"{:,}".format(int(value))))
	
	if (logx is True):
		ax.set_xscale("log")
	if (logy is True):
		ax.set_yscale("log")
	
	plt.scatter(x=xs, y=ys, marker="o")
	if (target_label is None):
		plt.scatter(x=txs, y=tys, color="r", marker="o")
	else:
		plt.scatter(x=txs, y=tys, color="r", marker="o")
		plt.annotate(target_label, xy=(txs[0], tys[0]), xytext=(txs[0]+add_annotation_x, tys[0]+add_annotation_y), fontsize=16)
	
	plt.xlabel(xlabel, fontsize=label_fontsize)
	plt.ylabel(ylabel, fontsize=label_fontsize)
	
	model=None
	
	if (model_path is not None):
		model_data=np.loadtxt(model_path)
		model_x=model_data[:, 0]
		model_y=model_data[:, 1]
		
		model_points=[(model_x[0], model_y[0])]
		
		model_sigma=model_data[:, 2]
		
		model=dict()
		for i in range(1, len(model_x)):
			x1=model_x[i-1]
			x2=model_x[i]
			y1=model_y[i-1]
			y2=model_y[i]
			
			s1=model_sigma[i-1]
			s2=model_sigma[i]
			
			for x in range(int(x1), int(x2)+1):
				l=(x-x1)/(x2-x1)
				y=(1-l)*y1+l*y2
				s=(1-l)*s1+l*s2
				model[x]=(y, s)
		
		plt.plot(model_x, model_y, color="red")
	
	if (draw_tvor is True):
		cs=get_tvor_coef(all_xs, all_ys, use_ransac=False)
		
		model_x=np.array(range(1, max(all_xs)+1))
		model_y=cs[0]*model_x+cs[1]*np.sqrt(model_x)
		plt.plot(model_x, model_y, color="orange")
		
	if (labeled_points is not None):
		for coordinates, label, color in labeled_points:
			px, py=coordinates
			plt.scatter(px, py, color=color, marker="o")
			plt.annotate(label, xy=(px, py), xytext=(px+add_annotation_x, py), fontsize=16)
	
	if (legend is not None):
		plt.legend(legend)
	
	if (save_path is not None):
		p=save_path.rfind(".")
		if (p!=-1):
			format=save_path[p+1:]
		plt.savefig(save_path, format=format, dpi=dpi)
	
	if (show is True):
		plt.show()

def get_german1939_birth_year_counts(path="germany1939.txt"):
	lines=load_lines(path)
	counts=dict()
	for l in lines:
		parts=list(map(int, l.split(" ")))
		year=parts[0]
		count=parts[1]
		if (year>=1850 and year<=1939):
			counts[year]=count
	return counts

def simulate_dtv_calculation_for_germany1939(trials=100, lower_bound=100, upper_bound=5000, points=None, output_path="germany1939_dtvs.txt"):
	
	import random
	
	if (points is not None):
		start=np.log(lower_bound)
		end=np.log(upper_bound)
		points=np.exp(np.arange(start, end+1e-12, (end-start)/points))
		points=map(int, map(round, points))
	else:
		points=range(lower_bound, upper_bound+1)
	
	counts=get_german1939_birth_year_counts()
	years=[]
	for y, c in counts.items():
		years.extend(c*[y])
	
	with open(output_path, "w") as f:
		for sample_size in points:
			dtvs=[]
			for i in range(trials):
				sample=random.sample(years, sample_size)
				bins, edges=np.histogram(sample, bins=np.arange(1850, 1945+1e-12, 1))
				dtv=sum(np.abs(bins[:-1]-bins[1:]))
				dtvs.append(dtv)
			m=np.mean(dtvs)
			s=np.std(dtvs)
			f.write(str(sample_size)+" "+str(m)+" "+str(s)+"\n")
			f.flush()
			print(sample_size, m, s)

def calculate_whipple_index(counts):
	d=sum([counts.get(i, 0) for i in range(25, 60+1, 5)])
	n=sum([counts.get(i, 0) for i in range(23, 62+1, 1)])
	
	if (n==0):
		n=1
	
	return 500*d/n

def calculate_myer_blended_index(counts, lower_year=10, upper_year=89):
	s1=[0]*10
	s2=[0]*10
	for y in range(lower_year, upper_year+1):
		s1[y%10]+=counts.get(y, 0)
	for y in range(lower_year+10, upper_year+1):
		s2[y%10]+=counts.get(y, 0)
	f1=list(range(1, 10+1))
	f2=list(range(9, 0-1, -1))
	s=[f1[i]*s1[i]+f2[i]*s2[i] for i in range(10)]
	sa=sum(s)
	
	return np.sum(np.abs(100*np.array(s).astype(np.uint64)/sa-10))/2

def test1():
	
	n=2
	N=400
	
	use_cached=False
	
	exact=fnn_exact(n, N, use_cached=use_cached)
	print("Exact value:      ", exact)
	
	approximate=fnn_approximate(n, N)
	print("Approximate value:", fnn_approximate(n, N))

def test2():
	
	bins_counts=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	
	in_m=0
	in_std=1
	
	out_m=0
	
	lower_size=500
	upper_size=1000
	
	point_repetitions=100
	
	#create_normal_plot_bins(b=5, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=0.9, samples_count=100, other_samples_count=1, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=False, output_path="sc_100_osc_1_b_5_std_0_9_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_normal_plot_bins(b=5, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=1.5, samples_count=100, other_samples_count=1, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=False, output_path="sc_100_osc_1_b_5_std_1_5_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_normal_plot_bins(b=5, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=0.5, samples_count=100, other_samples_count=1, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=False, output_path="sc_100_osc_1_b_5_std_0_5_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)

	#create_normal_plot_bins(b=10, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=0.5, samples_count=100, other_samples_count=1, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=False, output_path="sc_100_osc_1_b_10_std_0_5_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	create_normal_plot_bins(b=5, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=0.9, samples_count=100, other_samples_count=90, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_90_b_5_std_0_9_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_normal_plot_bins(b=5, bins_counts=bins_counts, in_m=in_m, in_std=in_std, out_m=out_m, out_std=1.5, samples_count=100, other_samples_count=90, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_90_b_5_std_1_5_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)

def test3():

	b=5
	
	size_base=2
	size_powers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	lower_to_upper_size_factor=10
	
	in_m=0
	in_std=1
	
	out_m=0
	
	samples_count=100
	
	point_repetitions=100
	
	create_normal_plot_sizes(b=b, bins_count=15, size_base=size_base, size_powers=size_powers, lower_to_upper_size_factor=lower_to_upper_size_factor, in_m=in_m, in_std=in_std, out_m=out_m, out_std=0.9, samples_count=samples_count, other_samples_count=1, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_1_b_5_std_0_9_bc_15_pr_"+str(point_repetitions)+".png", show=True, verbose=True)

def test4():
	
	out_size=10000
	out_a=2
	out_b=3
	
	age_heaping_amount=0.1
	target=5
	
	lower_bound=0
	upper_bound=1
	
	bins_count=100
	
	get_histogram=lambda x:np.array(np.histogram(x, bins=np.linspace(lower_bound, upper_bound, bins_count+1))[0])
	
	sample=np.random.beta(out_a, out_b, out_size)
	bins=get_histogram(sample)
	
	visualize_bins(bins=bins, lower_bound=lower_bound, upper_bound=upper_bound)
	
	simulate_age_heaping(bins=bins, age_heaping_amount=age_heaping_amount, target=target)
	
	visualize_bins(bins=bins, lower_bound=lower_bound, upper_bound=upper_bound)

def test5():
	
	bins_count=100
	
	samples_count=100
	
	lower_size=500
	upper_size=1000
	
	target=5
	
	age_heaping_amounts=np.linspace(0.025, 0.5, 20)
	
	p_a=2
	p_b=3
	
	point_repetitions=100
	
	#create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=1, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_1_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=10, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_10_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=30, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_30_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=50, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_50_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=70, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_70_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)
	
	#create_beta_ah_plot(bins_count=bins_count, samples_count=samples_count, other_samples_count=90, upper_size=upper_size, lower_size=lower_size, age_heaping_amounts=age_heaping_amounts, target=target, p_a=p_a, p_b=p_b, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_100_osc_90_a_"+str(p_a)+"_b_"+str(p_b)+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)

def test6():
	
	b=5
	bins_counts=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	
	in_a=7
	in_b=1
	
	out_left=0.0
	out_right=1.0
	out_mode=0.5
	
	samples_count=100
	other_samples_count=90
	
	lower_size=500
	upper_size=1000
	
	point_repetitions=100
	
	create_beta_triangular_plot_bins(bins_counts=bins_counts, in_a=in_a, in_b=in_b, out_left=out_left, out_right=out_right, out_mode=out_mode, samples_count=samples_count, other_samples_count=other_samples_count, lower_size=lower_size, upper_size=upper_size, point_repetitions=point_repetitions, show_ransac=True, output_path="sc_"+str(samples_count)+"_osc_"+str(other_samples_count)+"_a_"+str(in_a)+"_b_"+str(in_b)+"_mode_"+(str(out_mode).replace(".", "_"))+"_low_"+str(lower_size)+"_up_"+str(upper_size)+"_pr_"+str(point_repetitions)+".png", show=True, verbose=True)

def test7():
	
	use_ransac=False
	
	xs, ys, names, all_years, histograms=get_ushmm_data(input_dir=USHMM_BIRTH_YEARS_DIR, input_path=None, lower_limit=100, lower_bound=1850, upper_bound=1945)
	calculate_tvor_scores(xs=xs, ys=ys, names=names, use_ransac=use_ransac, print_count=100)

def test8():
	create_ushmm_plot(20781, figsize=(13, 8), save_path="20781.png")

def test9():
	create_jms_plot(save_path="jms_marked.png", show=True, figsize=(13, 8), tick_fontsize=10, label_fontsize=15)

def test10():
	create_ushmm_lists_plot(global_maximum=None, input_path=None, model_path="germany1939_10000_dtvs_parts__.txt", labeled_points=None, lower_limit=1, lower_bound=1850, upper_bound=1945, figsize=(13, 8), normalize=False, xlabel="Histogram size", ylabel="Discrete total variation of the histogram", logx=True, logy=True, tick_fontsize=10, label_fontsize=15, show=True, save_path="ushmm.png", draw_tvor=True, legend=["Monte Carlo model", "TVOR model"])

def test11():
	simulate_dtv_calculation_for_germany1939()

def test12():
	
	counts={i:1000 for i in range(0, 100+1)}
	
	make_uneven=True
	
	if (make_uneven is True):
		for digit in [0, 1, 2]:
			for i in range(30+digit, 90+digit+1, 20):
				counts[i]+=100+digit*100
			for i in range(40+digit, 90+digit+1, 20):
				counts[i]-=100+digit*100
		
		for digit in [5, 6, 7, 8, 9]:
			for i in range(20+digit, 60+digit+1, 20):
				counts[i]+=100+digit*100
			for i in range(30+digit, 70+digit+1, 20):
				counts[i]-=100+digit*100

		for digit in [3]:
			for i in range(20+digit, 60+digit+1, 20):
				counts[i]+=100+digit*100
			for i in range(30+digit, 70+digit+1, 20):
				counts[i]-=100+digit*100

		
	print("Whipple's index:", calculate_whipple_index(counts))
	print("Myers' index:   ", calculate_myer_blended_index(counts))
	
	years=[]
	for k, v in counts.items():
		years.extend([1945-k]*v)
	plot_default_birth_year_histogram(years=years, lower_bound=1850, upper_bound=1945, figsize=(13, 8), show=True, save_path=None)
	
def main():
	
	#compare the exact and the approximate value of function F(n, N)
	test1()
	
	#check TVOR's performance in distribution outlier detection for various parameters of the normal distribution 
	#test2()
	
	#check the influence of the random samples' size on TVOR's performance in distribution outlier detection
	#test3()
	
	#age heaping simulation
	#test4()
	
	#check TVOR's performance in DTV outlier detection
	#test5()
	
	#check TVOR's performance in distribution outlier detection for inliers drawn from the beta distribution and the outliers drawn from the triangular distribution
	#test6()
	
	#check TVOR's performance on the dataset of USHMM birth years
	#test7()
	
	#create the plot for the histogram of a USHMM's list birth years
	#test8()
	
	#create the marked histogram of Jasenovac inmates' birth years
	#test9()
	
	#create the USHMM lists plot
	#test10()
	
	#use an auxiliary function to generate the Monte Carlo model for the German 1939 census
	#test11()
	
	#show how Whipple's and Myers' indices fail
	#test12()

if (__name__=="__main__"):
	main()
