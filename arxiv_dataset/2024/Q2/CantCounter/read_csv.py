import csv
need=[]
former=[]
latter=[]
with open('True.csv',encoding='UTF-8') as f:
    f_csv = csv.reader(f)
    headers=next(f_csv)

    for row in f_csv:
        site=row[1].find('Trump')
        if site== -1:
            continue
        length=len(row[1])
        need.append(row[1][0:site+5])
        need.append(row[1][site+5:length])

    a=1
