import pickle,os

combined = {}
count = 0
for f in os.listdir('./'):
	if f.endswith('pkl') and f.startswith('evaluated_decisions'):
		with open(f,'rb') as f:
			res = pickle.load(f)
		for key in res:
			if key not in combined:
				perf,size,bitops = res[key]
				for k in perf:
					try:
						perf[k] = perf[k].cpu()
					except:
						pass
				combined[key] = (perf,size,bitops)
		print(f,'combined')
		count += 1
print(f'total {count} files combined')

with open('evaluated_decisions_combined.pkl','wb') as f:
	pickle.dump(combined,f)

