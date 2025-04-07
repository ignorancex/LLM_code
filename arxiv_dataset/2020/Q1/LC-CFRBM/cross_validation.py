import json
from os import listdir

experiments = listdir('experiments')

min_lst = []
for e in experiments:
    if 'nosparse_item_e100_b10_nh100_' in e:
        fin = open('experiments/' + e,'rb')
        data = fin.read()
        dic = json.loads(data)

        minn = 100
        for iter in dic["results"]:
            if iter["mae"] < minn:
                minn = iter["mae"]
        min_lst.append(minn)


print len(min_lst)
print sum(min_lst) / (len(min_lst) * 1.0), min(min_lst)

