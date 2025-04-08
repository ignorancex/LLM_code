import csv
import datetime as dt

result = dict()
with open('COVID19_vaccination_timeline_v202210.csv','r') as f:
    r = csv.reader(f,delimiter=';')
    next(r)
    for line in r:
        if line[1]=='0' or line[1]=='10' or line[1]=='11':
            continue
        date = dt.datetime.strptime(line[0][:10],'%Y-%m-%d')
        fed = 'AT-'+line[1]
        dose = int(line[4][0])-1
        count = int(line[5])
        if not date in result.keys():
            result[date]={'AT-'+str(i):[0,0,0,0] for i in range(1,10)}
        result[date][fed][dose]+=count
        
X = list()
X.append(['date','region','shotNo','doses'])
dates = list(result.keys())
dates.sort()
for date in dates:
    for i in range(1,10):
        for j in range(4):
            fed = 'AT-'+str(i)
            if date==dates[0]:
                X.append([date.strftime('%Y-%m-%d'),fed,j+1,result[date][fed][j]])
            else:
                X.append([date.strftime('%Y-%m-%d'),fed,j+1,result[date][fed][j]-result[date-dt.timedelta(1)][fed][j]])
with open('vaccination_data.csv','w',newline='') as f:
    w=csv.writer(f,delimiter=';')
    w.writerows(X)