f = open("cora_sampling_stats.dat")
content = f.readlines()
f.close()
foo = list(map(lambda x: x.split(),content))
#del(foo[-1])
bar = np.array(foo).T
baz = str(bar).replace("[","").replace("]","").replace("'","")

f2 = open("transposed.dat", "w")
f2.write(baz)
f2.close()
str(total).replace("[","").replace("]","").replace("'","").replace(","," ")
