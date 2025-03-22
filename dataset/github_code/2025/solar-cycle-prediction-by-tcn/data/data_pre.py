from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

datas = pd.read_csv("data.csv")
column = datas.columns[0]
datas = datas[column]
years = []
months = []
year_months = []
SSN_data = []
for data in datas:
    item = data.split(";")
    if(float(item[3].strip())>-1):
        years.append((item[0]))
        months.append((item[1]))
        year_months.append(float(item[2]))
        SSN_data.append(float(item[3].strip()))

plt.plot(years, SSN_data, color='red')
plt.show()
#写入到新文件

with open("13-month-smoothed-sunspots.csv","w",newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Month","Sunspots"])
    for year,month, ssn in zip(years,months, SSN_data):
        writer.writerow([year+month,ssn])

file.close()
