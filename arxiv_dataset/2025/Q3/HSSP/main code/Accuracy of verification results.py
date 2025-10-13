m = np.zeros((Max,Max),dtype=float)#创建验证用空列表
row=[]
fin=[]
fin1=[]
x=0
for i in parent[0]:   
    #i = parent[x]
    dindex1 = nested_d[0].index(i)#获取父节点的index，为了便于验证，加上子节点的数量)
    A=subset_list1[x::lend]#根据父节点的个数隔位取对应的子集
    x=x+1
    list.clear(fin)
    print(dindex1)
    for i in A:
        for j in i:
            for s in j:
                fin.append(j)
                maxlabel = max(fin,key=fin.count)
                maxlabel1=[i for j in maxlabel for i in j]
    for item in maxlabel1:
        if item != dindex1:
            if m[:,item].any()== 0:
                row.clear()
                row.append(item)
                print(row)
                m[dindex1][row]=1
                
mt=m.T
m=m+mt
m

flatten_arr1 = np.ravel(M)
flatten_arr2 = np.ravel(m)

print (flatten_arr1)
print (flatten_arr2)#将两个矩阵平铺
test = np.equal(flatten_arr1,flatten_arr2)
count1 = sum(test[test == True])#两个矩阵相同数据的个数
H=m.shape[0]
L=m.shape[1]
R=H*L#矩阵的长×宽=矩阵数据数量
accuracy=count1/R
accuracy#准确率：相同数据的个数/原有数据数量
