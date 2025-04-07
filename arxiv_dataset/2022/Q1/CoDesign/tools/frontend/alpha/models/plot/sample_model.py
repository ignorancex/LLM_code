#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
import matplotlib
import math
import random
from numpy import polyfit
from plot_utils import find_pareto

accuracy = pickle.load(open("../accuracy.pickle", "rb"))
latency = pickle.load(open("../flops.pickle", "rb"))

coeff = polyfit(latency, accuracy, 1)
print(coeff)

pareto_lat, pareto_acc, pareto_index = find_pareto(latency, accuracy, 1)

plt.rcParams["font.family"] = "Linux Biolinum O"
plt.rcParams.update({'font.size': 33,'font.weight':'bold','pdf.fonttype':42})
plt.rcParams["legend.handlelength"]=1.5

fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_axes([0.174, 0.235, 0.779, 0.68])
ax1.yaxis.set_label_coords(-0.142, 0.48)
ax1.ticklabel_format(style='sci', axis='x')


ax1.scatter(latency, accuracy, c='#adadad', s=10, zorder=6)
ax1.scatter(latency[9000:], accuracy[9000:], c='green', s=10, zorder=6)
ax1.plot(latency, [(coeff[0]*i+coeff[1]) for i in latency], 'r-', zorder=10)
ax1.scatter(pareto_lat, pareto_acc, c='blue', zorder=7, s=10, marker='^', label=r'Note8')

#ax1.set_xticks(xtick)
#ax1.set_xlim(xlim)
#ax1.set_yticks([i for i in range(72, 79, 2)])
#ax1.set_ylim([72, 78])

#ax1.set_xlabel(xname +  ' Lat. (ms)',weight='bold')
ax1.set_xlabel('FLOPs',weight='bold')
ax1.set_ylabel('Predicted Acc. (%)', weight='bold')

ax1.yaxis.grid(zorder=-1,color='lightgray',dashes=(5,10),linewidth=1,linestyle='--')
ax1.xaxis.grid(zorder=-1,color='lightgray',dashes=(5,10),linewidth=1,linestyle='--')

#ax1.yaxis.get_major_formatter().set_powerlimits((0,1))

ax1.tick_params(axis='both',direction='out', length=10, width=3, colors='k', grid_alpha=1)
for axis in ['top','bottom','left','right']:
	ax1.spines[axis].set_linewidth(3.0)
	
#legend=fig.legend(scatterpoints=1, frameon=True,ncol=1,bbox_to_anchor=(0.95,0.52),fontsize='small',facecolor='w',edgecolor='k',handletextpad=0.1,labelspacing=0,columnspacing = 0.1)
#legend.get_frame().set_alpha(0.5)


plt.show()

fig.savefig('./sample_model.pdf')


def euclidean_distance(k, h, pointIndex):
	'''
	计算一个点到某条直线的euclidean distance
	:param k: 直线的斜率，float类型
	:param h: 直线的截距，float类型
	:param pointIndex: 一个点的坐标，（横坐标，纵坐标），tuple类型
	:return: 点到直线的euclidean distance，float类型
	'''
	x=pointIndex[0]
	y=pointIndex[1]
	theDistance=math.fabs(h+k*(x-0)-y)/(math.sqrt(k*k+1))
	return theDistance


distance = []
for i in range(len(latency)):
	distance.append(euclidean_distance(coeff[0], coeff[1], (latency[i], accuracy[i])))
	
print(max(distance))


#####################################################
# select models above the line
#####################################################
select_above_line = []
for i in range(len(latency)):
	line = coeff[0]*latency[i]+coeff[1]
	if accuracy[i] > line:
		select_above_line.append(i)
		
print(len(select_above_line))
print(len(pareto_index))

random.seed(10)
select = random.sample(select_above_line, 1000)
select.extend(pareto_index)
select = list(set(select))
print(select)
print(len(select))

select_accuracy = [accuracy[i] for i in select]
pickle.dump(select, open("./select_model_index.pickle", "wb"))
#pickle.dump(select_accuracy, open("./select_accuracy.pickle", "wb"))
print(select_accuracy)