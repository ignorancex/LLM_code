import json
import matplotlib.pyplot as plt 


fin1 = open('experiments/nosparse_item_e100_b10_nh100_100k-u1xu1.json','rb')
fin2 = open('experiments/nosparse_gitem_e100_b10_nh100_100k-u1xu1.json','rb')
fin3 = open('experiments/sparse_item_e100_b10_nh100_005p_100k-u1xu1.json','rb')

data1 = fin1.read()
data2 = fin2.read()
data3 = fin3.read()

dic1 = json.loads(data1)
dic2 = json.loads(data2)
dic3 = json.loads(data3)

x = []
y1 = []
y2 = []
y3 = []

for iter in dic1["results"]:
    x.append(iter["iteration"])
    y1.append(iter["mae"])

for iter in dic2["results"]:
    y2.append(iter["mae"])

for iter in dic3["results"]:
    y3.append(iter["mae"])

print min(y1),min(y2),min(y3)

plt.plot(x,y1,color='r',label='User RBM')
plt.plot(x,y2,color='b',label='User metadata RBM')
plt.plot(x,y3,color='g',label='User Sparse RBM')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('MAE')
plt.title('RBM')
plt.savefig('item_100h.svg', format='svg', dpi=1200)
plt.close()