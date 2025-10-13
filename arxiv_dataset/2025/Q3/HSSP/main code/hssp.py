from collections import Counter
from statistics import mode
from operator import itemgetter
import itertools
result_dic={}
lst1=[]
lst2=[]
Index=[]
subset_list1=[]
x1 = -1
for t in nested_d:
    x = 0
    x1 = x1+1
    for num in parent[x]:
        num = parent[x1][x]
        print(nested_d[x1].index(num,x))
        x = x+1
        y = 0
        subset_list = []
        for i in nested_i[x1]:
            newnums=[i for i in nested_i[x1] if i<num]
            a=subset1(newnums[y:],num)#克服动态规划只能输出第一个匹配的结果，每次遍历完子列表后将第一个数据删除
            lst1.append(a)
            y=y+1
            list.clear(Index)
            list.clear(lst2)
            if a != None:   
                for n in a:
                    if a in Input_one[x1]:
                        A=[i for i,x in enumerate(Input_one[x1]) if x==n]#获取子节点的index，但是由于同一个数字可能有多个index
                    else:
                        A=[i for i,x in enumerate(Input_two[x1]) if x==n]#获取子节点的index，但是由于同一个数字可能有多个index
                    Index.append(A)
                B=Index
                C=list(itertools.product(*B))#所以把所有index排列组合得到结果            
            lst2.append(C)
            result_dic.clear()
            for item_str in lst2:
                for m in item_str:
                    if m not in result_dic:
                        result_dic[m]=1
                    else:
                        result_dic[m]+=1#新建一个字典，对子节点index合集投票
                        values_list = list(result_dic.values())
                counts = Counter(result_dic)
                maxcount = [value for value, count in counts.most_common()]
                subset_list.append(maxcount)
        subset_list1.append(subset_list)
            
