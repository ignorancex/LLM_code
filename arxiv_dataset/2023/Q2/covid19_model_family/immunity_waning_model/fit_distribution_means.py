"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


import os
from collections import Callable
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

ITERS = 2000

def estimate_kaplan_meier_kurve(samplefun:Callable,maxT:int,base:float,mean:float) -> np.array:
    """
    The function estimates the ratio of persons to be immune t days after the immunization event. This is done using a kaplan meier estimator given by the 'lifelines' package.
    :param samplefun: function to sample from
    :param maxT: evaluate the kaplan meier curve for all t in [0,1,...,maxT]
    :param base: base probability that intervention leads to immunity
    :param mean: mean value parameter for the waning distribution
    :return: array of the survival function to times [0,1,...,maxT]
    """
    np.random.seed(12345)
    X = np.array([samplefun(mean) for i in range(ITERS)])
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(X)
    return kmf.survival_function_at_times(list(range(maxT+1)))*base

def calculate_average_effectiveness(t1s:np.array,t2s:np.array,kaplan_meier:np.array) -> np.array:
    """
    Function evaluates the survival function to the given time intervals and calculates the average effectiveness of the intervention on the given time interval [t1,t2). This is done by numeric integration of the survival function (daily discretization) divided by the integral length.
    This stretegy is limited by the idea, that the population is homogeneously spread among the time interval. I.e. the number of people having had the intervention x days ago is more less equivalent for all x in the interval.
    :param t1s: array of left interval bounds
    :param t2s: array of right interval bounds
    :param kaplan_meier: estimated survival function given as daily values
    :return: array with the same length as t1 and t2 containing the average effectiveness values
    """
    values = list()
    for t1,t2 in zip(t1s,t2s):
        avg = sum(kaplan_meier[t1:t2])/(t2-t1)
        values.append(avg)
    return np.array(values)

def error_fun(referenceValues:np.array,modelValues:np.array) -> float:
    """
    Error function calculates the relative error between the given reference effectivenss values and the modeled ones
    :param referenceValues: array of reference values
    :param modelValues: array of modeled values
    :return: sum of abolute relative errors
    """
    return sum(abs(referenceValues-modelValues)/abs(referenceValues))

def fit_distribution_mean(references:list,samplefun:Callable) -> Tuple[float, float]:
    """
    In literature, effectiveness of an pharmaceutical intervention is typically defined for an observed time span after the event. E.g. vaccine effectiveness 2 to 4 weeks after the vaccination is 0.8, meaning, that compared to a cohort without vaccination, numbers of infections is reduced by 4/5th. We may interpret this result in that way, that the intervention causes that 80% of all vaccinated persons are rendered immune withn the regarded time-period. I.e. the average fraction of immunes is f_data(t)=0.8 for 14<=t<28.
    We may model this behaviour individually by regarding two distinct processes:
    First, there is a chance, that the intervention does not work at all. We regard this as 'base' probability. Second, the immunity is waning with a certain distribution. I.e. if the intervention caused immunity at all, there is a time X, after which the immunity is lost again. We characterize X by a given distribution and a distribution parameter, tapically the 'mean' - i.e. E(X)=mean.
    Goal of this fitting process is to find values 'mean' and 'base' s.t. the resulting distribution causes, on average, 80% immunity 15 to 28 days after the event.
    According to the model, the fraction of persons being immune t days after the immunization event, i.e. the value of the Kaplan-Meier survival curve, computes to f_model(t) = base*(1-F_X(t)), whereas F_X is the cumulative distribution function of X with E(X)=mean. In the fitting process we calculate the relative error between the average effectivenss: \int f_model(t) dt  and \int f_data(t) dt for t in the interval [14,28).
    In the typical case, effectiveness is not only given for one time interval, but for a series of intervals. This causes f_data to be a piecewise constant function.

    The optimization is done using the Nelder-Mead downhill simplex algorithm. The survival curve f_model is estimated using sampling of waning dates and fit of the Kaplan Meier curve via a suitable python package.

    :param references: list to specify the reference data in the format [[dayStart,dayEnd,measuredEffectivenss],...]
    :param samplefun: function handle to the sample the waning dates from
    :return: fitted base and mean value
    """
    maxT = int(max([x[1] for x in references]))
    refValues = np.array([x[2] for x in references])
    t1s = [int(x[0]) for x in references]
    t2s = [int(x[1]) for x in references]
    minimizefun = lambda p: error_fun(refValues,calculate_average_effectiveness(t1s,t2s,estimate_kaplan_meier_kurve(samplefun,maxT,p[0],p[1])))
    #plt.plot(tValues,estimate_kaplan_meier_kurve(samplefun,tValues,0.3058114908088022,50))
    #plt.plot(tValues,refValues,'r')
    #plt.show()
    opt = minimize(minimizefun,np.array([refValues[0],100.0]),bounds=[[0.0,1.0],[10.0,1000.0]],method='Nelder-Mead')
    return opt.x[0],opt.x[1]

def adjust_vacc_values(samplefuns:list,means:list,aPriorBases:list,vaccIntervals:list)->list:
    """
    In the model, cascading immunization events will provide stacking immunity levels. I.e. chance to become immune after two vaccine doses (+14 days) equals to
    P2 = p1*(1-p12)+(1-p1*(1-p12))*p2
    whereas p1 and p2 refer to the base probabilities that the corresponding first and second shot leads to immunity, whereas p12 refers to the probability to lose immunity betweem shot one and two
    Clearly, only values for P2 are given in literature. Given p1 = P1 and p12 via the estimated waning distribution, we may to calculate p2 in an inverse process. (Analogously for p3,p4,...)
    :param samplefuns: functions to sample waning of immunity after first, second, ... vaccination
    :param means: mean values of the corresponding sample functions
    :param aPriorBases: Values for P1,P2,...
    :param vaccIntervals: Typical intervals between first, second,... dose
    :return: values for p1,p2,...
    """
    aPosteriorBases=[aPriorBases[0]]
    for i in range(0,len(aPriorBases)-1):
        loseCount = [samplefuns[i](means[i]) for k in range(ITERS)]
        fractionImm = len([x for x in loseCount if x > vaccIntervals[i]]) / ITERS
        # overallProb2 = overallProb1*waningfactor + (1-overallProb1*waningfactor)*prob2 -> transform to prob2
        prob2 = (aPriorBases[i+1] - aPriorBases[i] * fractionImm) / (
                1 - aPosteriorBases[-1] * fractionImm)
        aPosteriorBases.append(prob2)
    return aPosteriorBases

class FitPlotter:
    def __init__(self,causes:int,targets:int):
        self.X = causes
        self.Y = targets
        self.currRow = 0
        self.currCol = 0
        self.fig = plt.figure(figsize=[20, 20])
        self.gs = GridSpec(self.X+2,self.Y,self.fig)
        self.labels = dict()
        self.currLabelid = 1

    def plot_fit(self,distributionName:str, references: list, samplefun:Callable, base:float, mean:float, nameStamp:str, label:str) -> None:
        """
        Plots the modeled fraction of persons immune after t days. If a fitting process was performed, the fitting data is displayed as well.
        :param resultFolder: folder to save plot into
        :param distributionName: name of the distribution used
        :param references: list to specify the reference data in the format [[dayStart,dayEnd,measuredEffectivenss],...]
        :param samplefun: function handle to the sample the waning dates from
        :param base: base probability that the immunization process works
        :param mean: mean value for the lose-immunity-date
        :param nameStamp: title of the corresponding subplot
        :param label: source of the data as label to the plot
        :return:
        """
        if label!='':
            if label not in self.labels.keys():
                self.labels[label]="["+str(self.currLabelid)+"]"
                self.currLabelid+=1
        labelid = self.labels[label]

        self.fig.add_subplot(self.gs[self.currRow,self.currCol])
        tmx = 1000
        effs = estimate_kaplan_meier_kurve(samplefun,tmx-1, base, mean)
        pl = plt.bar(range(tmx), effs,width=1.0,color=[0,0,1],alpha=0.35)
        t1s = [int(x[0]) for x in references]
        t2s = [int(x[1]) for x in references]
        avgs = calculate_average_effectiveness(t1s,t2s,effs)
        pl2 = None
        for (t1, t2, prob),avg in zip(references,avgs):
            pl2,= plt.plot([t1, t2], [prob, prob], 'r',linewidth=1,zorder=1)
            #plt.plot([t1, t2], [avg, avg], 'b',linewidth=2,zorder=0)
        plt.xlim([0, 365])
        plt.ylim([0,1])
        plt.text(360,0.8,nameStamp+'\n'+'{:.02f}*'.format(base)+distributionName+'(1.5,{:,.0f})'.format(mean)+'\n'+'(see '+labelid+')',ha='right',va='top')#,bbox={'alpha':0.5,'facecolor':[1,1,1]})

        self.currCol+=1
        if self.currCol==self.Y:
            self.currCol = 0
            self.currRow +=1

    def finish_plot_fit(self,resultFolder:str)-> None:

        self.fig.add_subplot(self.gs[self.X-3:, :])
        l = list(self.labels.items())
        l.sort(key=lambda x:int(x[1][1]))

        txt = '\n'.join([x[1]+': '+x[0] for x in l])
        plt.text(0,0,txt)
        plt.axis('off')
        plt.savefig(os.path.join(resultFolder, 'distribution_fit.png'), dpi=400)
        plt.savefig(os.path.join(resultFolder, 'distribution_fit.pdf'))