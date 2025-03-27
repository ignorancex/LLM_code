#!/usr/bin/env python3
# coding: utf-8
"""
This work is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/

Updated on July 15, 2024


Generates data for a BS for which the preceding BS can provide HO statistics. 
These HO data (number of HO calls and their duration) are used together with 
the PeMS metrics (flow and speed) to generate network data for the target BS.

Similar to the script that generates network data for the first BS at a highway, 
this code generates network data for a target BS on a highway different from 
the first BS. In contrast to 'data_gen_1BS.py', the data generation for 
the target BS in this case requires network statistics from a preceding BS. 
The traffic is assumed unidirectional. 
The statistics provided by the preceding BS are HO -- number of HO calls
and their duration -- which are used together with the PeMS metrics 
to generate network data for the target BS. 
A sample consists of (1) the flow and speed measured in the vicinity yet
outside the BS duting time interval 't' and (2) the calls observed in   
the BS during the same time slot 't'. In summary, both are measurements
concerning the same time interval. However, the road variables are measured
outside the BS and the calls inside the BS.
The flow and speed measured outside the target BS during time 't' are
assumed to be observed in the BS during time slot 't+1'. This flow
generates calls during time slots 't+1' and the slots that followed it.

@author: vesseln1
"""


__title__			 = 'Mobile traffic forecasting'
__description__		 = 'Generates mobile traffic load on a highway.'
__version__			 = '2.0.0'
__date__			 = 'July 2024'
__author__			 = 'Natalia Vesselinova'
__author_email__	 = 'natalia.vesselinova@aalto.fi'
__institution__ 	 = 'Alto University'
__department__		 = 'Mathematics and Systems Analysis'
__url__				 = 'https://github.com/nvassileva/DataMatters/'
__license__          = 'CC BY 4.0'


# Import the necessary packages
import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd
import ruamel.yaml
yaml = ruamel.yaml.YAML()


# Parse command-line options, arguments and sub-commands
parser = argparse.ArgumentParser()

parser.add_argument('--outputRoot', '-o', required=True, type=str, help='Path to the directory where the output will be stored.')
parser.add_argument('--precedingNodeRoot', '-d',  required=True, type=str, help='Path to a directory with PeMS and network data of the preceding base station.')
parser.add_argument('--targetNodeRoot', '-t', required=True, type=str, help='Path to the target node (BS/RSU/sensor/other) for which data is generated.')
parser.add_argument('--distance', '-m', required=True, type=float, help='The range of the base station in miles.')
parser.add_argument('--distribution', '-distr',  required=True, type=int, help='Call distribution (an int): (1) exponential, (2) lognormal, (3) a mixture of two lognormals.')
parser.add_argument('--arrivalRate', '-a', type=int, help='Call interarrival rate of a Poisson process.', default=1)
parser.add_argument('--meanCallDuration', '-c1', type=float, help='Mean call duration.', default=1)
parser.add_argument('--timeInterval', type=int, help='PeMS time granularity in minutes.', default=5)
parser.add_argument('--standardDeviationCallDuration', '-std1', type=float, help='Standard deviation of the call duration distribution.', default=np.sqrt(3))
parser.add_argument('--probability', '-p', type=float, help='Probability parameter (fairness/bias) in the mixture of two lognormals PDF.', default=0.5)
parser.add_argument('--meanCallDuration2', '-c2', type=float, help='Mean call duration in minutes for the second lognormal in the mixture of 2 lognorm distr.', default=1)
parser.add_argument('--standardDeviationCallDuration2', '-std2', type=float, help='Standard deviation of the call duration distribution of the second lognormal in the mixture of 2 lognormal distr.', default=np.sqrt(3))
parser.add_argument('--ratio', type=float, help='A user defined threshold for deciding when adjustment between the flow of the preceeding and the flow of target base station is needed.', default=1)
parser.add_argument('--maxCalls', type= int, help='Needed for coding purposes.', default=100)
parser.add_argument('--beginWeek', '-b', type=int, help='Week number of the first week with data.', default=13)
parser.add_argument('--endWeek', '-e', type=int, help='Week number of the last week with data.', default=36)

args = parser.parse_args()

outputR           = args.outputRoot
precedingNodeRoot = args.precedingNodeRoot
targetNodeRoot    = args.targetNodeRoot
timeInt           = args.timeInterval
distance          = args.distance
distr             = args.distribution
rateArr           = args.arrivalRate             
meanCallD         = args.meanCallDuration               
stdDevCallD       = args.standardDeviationCallDuration
prob              = args.probability                        
meanCallD2        = args.meanCallDuration2
stdDevCallD2      = args.standardDeviationCallDuration2  
ratio             = args.ratio
maxCalls          = args.maxCalls
beginWeek         = args.beginWeek
endWeek           = args.endWeek



# Vehicular arrival traffic: emulating a Poisson process 
def vehFlow(flow, timeInt
           ):
    """ Emulating the arrival times of the vehicles
        using the real data measurements from PeMS and 
        assuming Poisson arrival process with mean=flow/timeInt.
        The Poisson arrival process is generated by 
        exponentially distributed interarrival times with mean=timeInt/flow.
        
        Returns an numpy array with ordered intearrival times.
    """
    expArrT = np.random.exponential(timeInt/flow, flow)

    for i in range(1, len(expArrT)):
        expArrT[i] += expArrT[i - 1]
    
    while not expArrT[-1] < float(timeInt):
        timeInt -= 0.05
        expArrT = np.random.exponential(timeInt/flow, flow)

        for i in range(1, len(expArrT)):
            expArrT[i] += expArrT[i - 1]
    
    return expArrT



# Vehicular dwell time in the cell: calculation
def dwellTime(speed, distance, flow=-1, noiseMean=0, noiseStd=0.05
             ):
    ''' Calculates the dwell time of the vehicle in the cell
        using the PeMS average speed of the vehicle and assuming 
        certain range of the BS (distance traversed by the 
        vehicle when under the coverage of the BS).
        
        Input: flow:     its value defines the version of this function
                         (that is, it works as a flag)
                           = when set to its default value (flow = -1), 
                             the average speed is an array and the function
                             returns an array of a constant dwell time per 5-min interval
                           = flow >=0: speed is a single value and the function
                             returns the average dwell time per vehicle from this flow
               speed:    average speed of veh (PeMS) per 5-minute-interval; 
                         it is univariate or a list
               distance: user defined (the BS range in miles)
               noiseMean and noiseStd of normal distr to make the dwellTime non-deterministic
               under the same speed and distance values 
               when the flow during a 5-minute-interval is provided as an input
               
        Output: the dwell times of the vehicles in the flow (flow=-1) or 
        the average dwell time (flow != -1) based on the PeMS speed and distance metrics.
        To avoid making this parameter deterministic when calculated with flow >= 0, 
        a Gaussian noise is added. The noise with default mean=0 and std=0.05
        means that the r.v. belongs to [-0.15, 0.15] with prob = 0.997.
        
        Returns the dwell time (of type numpy array) of each veh in the flow or
                the average dwell time in the BS for 
                a specific 5-minute-interval, speed and distance.
    '''

    speed = np.array(speed) # speed = np.where(speed==0, 1, speed)
    speed[speed==0] = 1     # to avoid division by 0; this does not change the end results as speed is discretised
    
    if flow == -1: 
        distance = np.full(len(speed), distance)
        dwellT = (distance / speed) * 60
    else:
        avDwellT = (distance / speed) * 60
        dwellT = abs(np.random.normal(noiseMean, noiseStd, flow) + avDwellT)

    return dwellT




# Generating new call arrivals
def callNewGen(rateArr, timeInt, maxCalls
           ):
    ''' Generate (new) call requests per vehicle (indVeh)
        emulating Poisson arrivals at the vehicle
        with rateArr.
        
        maxCalls is to know how many to generate yet
        the exact number of calls depends on the Poisson 
        rateArr and on the veh dwell time
        
        Return the number of call arrival time(s). 
    '''  
    # Create an exception when the max calls is not a positive integer
    # to notify that this is not an accepted value
    if maxCalls < 0 or rateArr <= 0:  
        print("Enter valid values for the arrival rate and maxCalls. ")
        raise ValueError("The arrival rate must be positive and maxCalls non-negative!")

    # Create an array with call arrival times      
    callArrTimeNew = np.random.exponential((timeInt/rateArr), (maxCalls))
    for t in range(1, len(callArrTimeNew)):
        callArrTimeNew[t] += callArrTimeNew[t - 1]   # Poisson process / exp interarrival times

    return callArrTimeNew





# Calculate the lognormal distribution parameters from its mean and variance
def lognormal_parameters(mean, std
            ):
    ''' Determing the parameters of the lognormal distribution
        mu_lognorm and sigma_lognorm
        from its mean and standard deviation.
        
        Input: the mean and the standard deviation of the lognormal.
        
        Return mu_lognorm and sigma_lognorm.  
    '''
    sigma_lognorm = np.sqrt(np.log(1 + np.square(std) / np.square(mean)))
    
    mu_lognorm    = np.log(np.square(mean) / np.sqrt(np.square(std) + np.square(mean)))
    
    return mu_lognorm, sigma_lognorm





# Call duration -- exponential, lognormal and a mixture of two lognormals at present
def callDuration(numCalls, distr, meanCallD, stdDevCallD, meanCallD2, stdDevCallD2, probability
            ):
    ''' Generate call duration for numCalls.
        
        Input: 
        -- numCalls := total number of calls, a positive integer 
        -- distr := the probability distribution, which is encoded as follows:
            1 := exponential(mean)
            2 := lognormal(mean, std)
            3 := a mixture of two lognormals with parameters mean and std, mean2 and std2, probability
                 p * lognormal(mean, std) + (1 - p) * lognormal(mean2, std2)
        
        Return the call duration array.  
    '''  
    if distr == 1:
        callD = np.random.exponential(meanCallD, numCalls)
    elif distr == 2:
        mu, sigma = lognormal_parameters(meanCallD, stdDevCallD)
        callD = np.random.lognormal(mu, sigma, numCalls)
    elif distr == 3:
        callD = np.zeros(0)
        mu, sigma = lognormal_parameters(meanCallD, stdDevCallD)
        mu2, sigma2 = lognormal_parameters(meanCallD2, stdDevCallD2)
        for _ in range(numCalls):
            if np.random.random() < probability:
                callD = np.append(callD, np.random.lognormal(mu, sigma))
            else:
                callD = np.append(callD, np.random.lognormal(mu2, sigma2))
    else:
        raise ValueError('Choose between three probability distributions: 1:=exp, 2:=lognormal, 3:=a mixture of two lognormals.')
            
    return callD



# Determine the call type and the timeslot it belongs to
def callClass(timeInt, callArrT, vehArrT, vehDepT, distr, meanCallD, stdCallD, meanCallD2, stdCallD2, probability
           ):
    ''' Determine the call type -- 'new' or 'handover' -- and the timeslot 
        it belongs to.
    
        callArrT:  call arrival times (an array)
        vehArrT:   vehicule's arrival time (a const)
                    needed for shifting the call arr times
        vehDepT:   vehicle's departure time (a const)
                    to determine the class of the call,
                    either new of handover
        meanCallD: mean call duration (a const)
        
        Return 3 arrays: the number of new calls per time slot,
        the number of handover calls (if any) per time slot
        and their (HO's) remaining (excess) times (a list). 
    '''  

    # Create an array of size four
    # arr[0] accounts for num calls during TS0 
    # arr[1] for calls initiated by the veh 
    # while in the coverage of the BS but during TS1, 
    # i.e., the next 5-min slot; continue up to TS3
    newCalls = np.zeros(4)
    callsHO = np.zeros(4)

    # An array for keeping track of the excess time
    # per each time slot TS0 to TS3
    excessT0 = np.empty([0])
    excessT1 = np.empty([0])
    excessT2 = np.empty([0])
    excessT3 = np.empty([0])
   
    # Shift call arrivals with the time of the vehicle's arrival
    callArrT += vehArrT
    
    # Generate the duration of the calls  
    # callEndT contains the time instants when the calls end
    callEndT = callDuration(len(callArrT), distr, meanCallD, stdCallD, meanCallD2, stdCallD2, probability) + callArrT 
    
    
    # Which time slot do the calls belolong to?
    for call, arrT in enumerate(callArrT):
        if arrT > vehDepT:
            break
        # determine the time slot the new call belongs to
        if arrT <= timeInt: 
            newCalls[0] += 1
        elif arrT > timeInt and arrT <= 2*timeInt:
            newCalls[1] += 1
        elif arrT > 2*timeInt and arrT <= 3*timeInt: 
            newCalls[2] += 1
        elif arrT > 3*timeInt:
            newCalls[3] += 1
        
        # determine the time slot the handover call belongs to
        if callEndT[call] > vehDepT:
            if vehDepT <= timeInt:
                callsHO[0] += 1
                excessT0 = np.append(excessT0, callEndT[call] - vehDepT)   
            elif vehDepT > timeInt and vehDepT <= 2*timeInt:
                callsHO[1] += 1
                excessT1 = np.append(excessT1, callEndT[call] - vehDepT)  
            elif vehDepT > 2*timeInt and vehDepT <= 3*timeInt:
                callsHO[2] += 1
                excessT2 = np.append(excessT2, callEndT[call] - vehDepT)
            elif vehDepT > 3*timeInt:
                callsHO[3] += 1
                excessT3 = np.append(excessT3, callEndT[call] - vehDepT) 
        
    excessT = [excessT0, excessT1, excessT2, excessT3] 
    # if excessT must be an array, then excessT = np.asarray([excessT0, excessT1, excessT2, excessT3], dtype=object)
                    
    return newCalls, callsHO, excessT





# Convert the averaged speed into a speed level
def speedConv(speed
             ):
    """ The PeMS averaged speed is converted into speed levels as such 
        discretisation decreases the prediction error. The speed is discretised
        at the very end, after all calculations that involve the speed are done.        
        The MAE was decreased by 2 % by discretising the speed alone
        for the very first case study (the patent).       
    """
    speedLevel = []
    for speedV in speed:
        if speedV < 20:
            speedLevel.append(1)
        elif speedV >= 20 and speedV < 25:
            speedLevel.append(2)
        elif speedV >= 25 and speedV < 30:
            speedLevel.append(3)
        elif speedV >= 30 and speedV < 35:
            speedLevel.append(4)
        elif speedV >= 35 and speedV < 40:
            speedLevel.append(5)
        elif speedV >= 40 and speedV < 50:
            speedLevel.append(6)
        elif speedV >= 50 and speedV < 60:
            speedLevel.append(7)
        else: 
            speedLevel.append(8)
          
    return speedLevel



# Determine if the handover calls continue to the subsequent BS --
# assume we are in BS2 and there were some HO calls from BS1 to BS3.
# This function answers the question whether these HO calls, 
# namely calls initiated in BS1 but not finalised there, continue 
# into BS3 or they end before crossing the BS2 BS3 border.
def contHO(avDwellT, callsHO1, timeHO1
          ):
    ''' Calls that were handed over from BS1 to BS2 may continue to BS3.
        This function checks this by using the average dwell time in BS2
        per five-minute-time interval and the call remaining time. 
        
        avDwellT: an array with the time spent by the veh in the cell on average 
                  during a specific time slot (an array with such av dwell times)
        callsHO1: an array with the number of HOs from BS1 to BS2
                  per five-minute-time slot 
        timeHO1:  a dataframe with the excess (call remianing) time
                  of the HOs from BS1 to BS2
        
        Returns callsHO1 and timeHO1 updated. 
        They are named callsHO2 and timeHO2 
        as they continue from BS2 to BS3.
    
    '''

    callsHO2 = callsHO1.copy()
    timeHO1 = timeHO1.assign(Index=range(len(timeHO1))).set_index('Index') # needed so that indexing can be done properly
    # Create lists that at the end will be merged into a timeHO2 dataframe
    # as lists are more computationally efficient to deal with 
    # in contrast to dataframes
    tInd = []
    tHO2 = []    

    flag = 0
    for call, valueHO1 in enumerate(callsHO1):
        for time in range(valueHO1):
            if timeHO1.loc[time + flag, '(Next BS) Remaining Time'] < avDwellT[call]:
                callsHO2[call] -= 1
            else:
                tInd.append(timeHO1.loc[time + flag, 'Five-Minute Time Interval Index'])
                tHO2.append(timeHO1.loc[time + flag, '(Next BS) Remaining Time'])
        flag += valueHO1
        
    timeHO2 = pd.DataFrame(columns=['Five-Minute Time Interval Index', '(Next BS) Remaining Time'])
    timeHO2.loc[:, 'Five-Minute Time Interval Index'] = tInd
    timeHO2.loc[:, '(Next BS) Remaining Time'] = tHO2
                    
    return callsHO2, timeHO2



# Network statistics adjusted to the vehicular statistics of BS2
def netStats(flow1, flow2, ratio, ho1, rt1
            ):
    """ Network metrics nameley, 
        number of HO calls from BS1 to BS2 
        and correspondingly, their remaining times 
        are adjusted to the flow in BS2.
        
        Input:   flow1:    an array with the number of vehicles in BS1 during deltaT = 5 min
                 flow2:    an array with the number of veh in BS2 during the 5-minute intervals
                 ratio:    a user defined threshold for deciding when adjustment is needed
                 ho1:      a list with the estimated number of HO from BS1 to BS2
                 rt1:      a dataframe with the reamining times for the HO calls per 5-min time slot
                          (the 'Time Index' column indicates the measured 5-min period (row))
               
        Output:  the newly estimated number of HOs to BS2. It will be 0 if
                 the flow in BS2 equals 0. It remains unchanged if the flow 
                 in BS2 is larger than the flow in BS1. It will be decreased
                 by a fraction if smaller than that in BS1.
        Returns: the updated ho1 and rt1 metrics.
        
    """
    ### Add an exception for the case when len(flow1) != len(flow2)
    # Both flows measured on the same time interval
    if len(flow1) != len(flow2):
        raise Exception("Unequal number of measurements. Computation has been halted.")
    
    for m, flow1V in enumerate(flow1):
        if flow2[m] < ratio * flow1V and flow1V != 0:
            shift = 0
            weight = 1 - flow2[m] / flow1V                      # proportion between the two flows (weight)
            droppedNumHO = np.random.binomial(1, weight, ho1[m])  # flipping a weighted coin ho1 times  
            ho1[m] -= sum(droppedNumHO)                           # delete a corresponding num of HOs
            begin = sum(ho1[:m])                                  
            for h, droppedNumHOV in enumerate(droppedNumHO):
                if droppedNumHOV == 1:                          # according to the coin's outcome 
                    rt1 = rt1.drop(rt1.index[begin+h+shift])      # and their excess times from the list
                    shift -= 1

    return ho1, rt1





def dataProcess(file1, file2): 
    ''' Reads network and road statistics from the preceding BS (BS1)
        and road statistics relevant to the current BS (BS2) to obtain
        network statistics for the present BS2
    
        Input:  file1:   Reads road and network data from the preceding BS, namely
                         Flow, HO Calls per time slot and Remaining Time per time slot (from TS0 to TS3).
                
                file2:   Reads data from a PeMS file and generates the corresponding 
                         service requests and needed statistics.
        
        Returns 5 pandas dataframes:
                dataML:  the input to the ML model -- Flow, Speed Level, Total Num Calls = Total New Calls + Total HOs
                data:    Time, Flow, Speed, Total New Calls, Total HOs from current BS (BS2) to next BS (BS3)
                         (BS2 to BS3 Total HOs together with their corresponding Remaining Time array 
                         are used in BS3 net metrics generation) (Time, Flow, Speed are PeMS data)
                dataRT:  an array with the HOs' Remaining Time per five-minute-time interval
                dataBS1: HOs from BS1 to BS2 after being updated according to the flow1/flow2 ratio (see netStats)
                dataRT1: the array with the BS1 to BS2 HOs' Remaining time (the HOs listed in dataBS1)
        '''
    # Read from a PeMS Excel file2
    data = pd.read_excel(file2, header=0, usecols=['5 Minutes', 'Flow (Veh/5 Minutes)', 'Speed (mph)'])
    numM = data.shape[0]
    
    if numM > 1441:
        data.drop(data.head(11).index,inplace=True)    # remove the first eleven rows corresponding to Sunday from 23:00 h to 23:50 h
        data = data.assign(Index=range(len(data))).set_index('Index')
    numM = data.shape[0]

    # Extend by 4 rows: the first row is needed because we read flow and speed
    # measured in time slot 't' and network metrics in 't+1'.
    # We also make computations (generate data) for 3 time slots (TSs) ahead  
    # to account for the case when the vehicle is still within the BS coverage 
    # (after the time slot the vehicle entered the BS range) 
    # and requests BS service (that is, makes a call).
    for i in range(4):
        data.loc[len(data.index)] = [0, 0, 0]  
    
    # Create a data frame for the HO calls remaining time
    dataRT = pd.DataFrame(columns=['Five-Minute Time Interval Index', '(Next BS) Remaining Time'])
        
    # Generate the network statistic for BS2 and HO ones for BS3
    
    # Accummulates the new and ho calls 
    newCallsTotal = np.zeros(data.shape[0])
    hoCallsTotal = np.zeros(data.shape[0]) 

    for row in range(numM):
        if data.at[row, 'Flow (Veh/5 Minutes)'] != 0:
            indA = np.arange(1, data.at[row, 'Flow (Veh/5 Minutes)'] + 1).astype(int)
            df = pd.DataFrame(indA, columns=['Total Num Veh'])

            flow = int(data.at[row, 'Flow (Veh/5 Minutes)'])  
            speed = data.at[row, 'Speed (mph)']

            df['Veh Arr Time'] = vehFlow(flow, timeInt) 
            df['Veh Dwell Time'] = dwellTime(speed, distance, flow)
            df['Veh Dep Time'] = df['Veh Arr Time'] + df['Veh Dwell Time']  
            
            # Create an auxiliary data frame for saving the remaining times
            # from all vehicles during a 5-min interval
            dfRT = pd.DataFrame(columns=['Five-Minute Time Interval Index', '(Next BS) Remaining Time'])  

            for veh in range(flow):
                callArrT = callNewGen(rateArr, timeInt, maxCalls)
                newCalls, callsHO, timeHO = callClass(timeInt, callArrT, df['Veh Arr Time'][veh], df['Veh Dep Time'][veh], distr, meanCallD, stdDevCallD, meanCallD2, stdDevCallD2, prob)
                for i in range(4):
                    indexR = row + 1
                    newCallsTotal[indexR+i] = newCallsTotal[indexR+i] + newCalls[i]
                    hoCallsTotal[indexR+i] = hoCallsTotal[indexR+i] + callsHO[i]
                    # Recording the remaining HO times
                    width = len(timeHO[i])
                    if width != 0:
                        indTS = np.full((width), indexR+i)
                        dfRTemp = pd.DataFrame(indTS, columns=['Five-Minute Time Interval Index']) # for saving intermediate (per veh) results
                        dfRTemp['(Next BS) Remaining Time'] = timeHO[i]
                        frames = [dfRT, dfRTemp]
                        dfRT = pd.concat(frames)
            frames = [dataRT, dfRT]
            dataRT = pd.concat(frames)
    dataRT = dataRT.assign(Index=range(len(dataRT))).set_index('Index')
    
    data['Total New Calls'] = newCallsTotal
    data['Total HO Calls'] = hoCallsTotal          # these are the HOs from this (BS2) to the next BS3
    data.drop(data.tail(4).index, inplace=True)    # remove the rows added at the beginning
        
    # Flow and speed metrics for the present BS (BS2)
    # needed when adjusting the HO calls from BS1 to BS2
    flow2  = data.loc[:, 'Flow (Veh/5 Minutes)'].tolist()    
    speed2 = data.loc[:, 'Speed (mph)'].tolist()
    
    # Read the required data from the preceeding BS (BS1)
    # into two dataframes: dfBS with flow and total # HOs
    # and another with the HO calls remaining time
    dfBS = pd.read_excel(file1, sheet_name='Data', header=0, 
                         usecols=['Flow (Veh/5 Minutes)', 'Total HO Calls'])
    timeHO1 = pd.read_excel(file1, sheet_name='RemainingTime', header=0)

    # Convert them into lists for ease of use
    flow1 = dfBS.loc[:, 'Flow (Veh/5 Minutes)'].tolist() 
    callsHO1 = dfBS.loc[:, 'Total HO Calls'].tolist()
    
    # Adjust the network variables considering the flows in each of the two BSs
    # print('len flow1 = ', len(flow1))
    # print('len flow2 = ', len(flow2))
    callsHO1, timeHO1 = netStats(flow1, flow2, ratio, callsHO1, timeHO1)
    dfcallsHO1 = pd.DataFrame()   # this is the df with the HO calls from the preceeding BS1 to the current BS2
    dfcallsHO1['Total HO Calls'] = callsHO1
    timeHO1 = timeHO1.assign(Index=range(len(timeHO1))).set_index('Index')                            
            
    # Remaining time of the handovers from BS1 to BS2 that potentially continue to BS3
    avDT = dwellTime(speed2, distance)
    
    callsHO2, timeHO2 = contHO(avDT, callsHO1, timeHO1)
    data.loc[:, 'Total HO Calls'] = data.loc[:, 'Total HO Calls'] + callsHO2
    speedL = data.loc[:, 'Speed (mph)']                       
    
    dataML = pd.DataFrame()
    dataML['Flow (Veh/5 Minutes)'] = data.loc[:, 'Flow (Veh/5 Minutes)']
    dataML['Speed Level'] = speedConv(speedL)
    dataML['Total Number Calls'] = data.loc[:, 'Total New Calls'] + dfcallsHO1.loc[:,'Total HO Calls']
    dataML = dataML.assign(Index=range(len(dataML))).set_index('Index')   
    
    frames = [dataRT, timeHO2]
    dataRT = pd.concat(frames)
    dataRT = dataRT.sort_values('Five-Minute Time Interval Index')
    dataRT = dataRT.assign(Index=range(len(dataRT))).set_index('Index')
    
    data = data.assign(Index=range(len(data))).set_index('Index')   

    dfcallsHO1 = dfcallsHO1.assign(Index=range(len(dfcallsHO1))).set_index('Index')  
   
    return dataML, data, dataRT, dfcallsHO1, timeHO1



# Change to the dataroot directory that contains 
# the preceding BS PeMS+networking data
os.chdir(precedingNodeRoot)
# List all files in this dir and save the list
filesPreceding = os.listdir(os.getcwd())

# Change to the current/target node
os.chdir(targetNodeRoot)
filesTarget = os.listdir(os.getcwd())

# Set a random seed and save it for reproducability 
now = datetime.datetime.now()
seed = int(now.timestamp())
np.random.seed(seed)


f1 = str()
f2 = str()

for num in range(beginWeek, (endWeek+1), 1):
    print(num)
    for file in sorted(filesPreceding, key=lambda x: x.split(".")[0]):
        if file.endswith(str(num) + '.xlsx'):
            f1 = file
            print('f1: ', f1)
            break
    if not f1 or int(f1.split(".")[0]) != num:          
        print('Preceding file not found. Program execution stopped.')
        sys.exit()
        
    for fl in sorted(filesTarget, key=lambda x: x.split(".")[0]):   
        if fl.endswith(str(num) + '.xlsx'):
            f2 = fl
            print('f2: ', f2)
            break
    if not f2 or int(f2.split(".")[0]) != num:              
        print('Target file not found. Program aborded.')
        sys.exit()
    print(f1, f2)
    path1 = precedingNodeRoot + "/" + f1
    path2 = targetNodeRoot + "/" + f2

    dataML, data, dataRT, dfcallsHO1, timeHO1  = dataProcess(path1, path2)
    # Remember to switch to a writing mode 'w' so that only the generated data
    # is saved once in a 'production' state 
    # as with mode append we are adding new sheets to the existing file
    # mode ='a', if_sheet_exists='replace'
    # if apend mode, if_sheet_exists='replace'
    pathOut = outputR + "/" + f2 
    with pd.ExcelWriter(pathOut, mode='w', engine='openpyxl') as writer: 
        dataML.to_excel(writer, sheet_name='MLData')
        data.to_excel(writer, sheet_name='Data')
        dataRT.to_excel(writer, sheet_name='RemainingTime', index=False)
        dfcallsHO1.to_excel(writer, sheet_name='AdjustedHOsfromBS1')
        timeHO1.to_excel(writer, sheet_name='RemainingTimeHOsfromBS1', index=False)

# Change to the output directory to store the generated data 
os.chdir(outputR)

# Save the input parameters into a yaml file 
argsDict = vars(args)
argsDict['standardDeviationCallDuration'] = str(stdDevCallD)
argsDict['standardDeviationCallDuration2'] = str(stdDevCallD2)
argsDict['seed'] = seed
argsDict['date and time'] = now

print("Saving the input values into an yaml file.")
print('ArgsDict', argsDict)
with open('input_data_values.yml', 'w') as f:
    yaml.dump(argsDict, f)
    