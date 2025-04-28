#PART1 - Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#____________________Code block for getting microjams from initial condition for rho<0.5
def calc_total_runs_ones(arr):
    
    total_length = 0
    current_length = 0

    for num in arr:
        if num == 1:
            current_length += 1
        else:
            if current_length > 1:  # Only consider runs longer than 1
                total_length += current_length
            current_length = 0

    # Add the last run if it exists
    if current_length > 1:
        total_length += current_length
    if arr[0]*arr[-1] > 0 and arr[1] == 0 :
        total_length += 1
    if arr[0]*arr[-1] > 0 and arr[-2] == 0 :
        total_length += 1

    return total_length

def min_max_cumulative_sums(arr):
    cumulative_sum = 0
    min_cumulative_sum = float('inf')
    max_cumulative_sum = float('-inf')

    for num in arr:
        cumulative_sum += num
        min_cumulative_sum = min(min_cumulative_sum, cumulative_sum)
        max_cumulative_sum = max(max_cumulative_sum, cumulative_sum)

    return min_cumulative_sum, max_cumulative_sum

def microjams(ini, rho, N):
    total_runs_ones =  calc_total_runs_ones(ini)#since we reversed it for algo, but number of jams unchanged..
    total_runs_zeros = calc_total_runs_ones(np.ones(len(ini))-ini)

    #print('0: ', total_runs_zeros, ' 1: ', total_runs_ones)

    #deciding value of switch

    switch = 0

    if(total_runs_zeros > total_runs_ones):
        switch = 1
    ###print('switch: ', switch)

    #Now depending on switch we change ini

    #Now we do modification of ini and double and reverse it

    mod_ini = 2*ini - np.ones(N)
    mod_ini_double = np.concatenate((mod_ini,mod_ini))
    rev = mod_ini_double[::-1]

    ##Now we permute ini to avoid issues with the boundary
    level_func = np.cumsum(rev)

    min_index = np.argmin(level_func)

    actual_min_index = N - (int(min_index)%N) - 1

    rev = np.concatenate(( rev[min_index:], rev[:min_index]))

    ini = np.concatenate((ini[actual_min_index:], ini[:actual_min_index]))
    
    ###if (switch == 1):
        ###ini = np.ones(len(ini)) - ini
        ###ini = ini[::-1]

    #loop stopper should be total_runs_ones in the original ini

    total_jammed_cars =  total_runs_ones

    #min and max_posns from rev

    pos_min, pos_max = min_max_cumulative_sums(rev)
    
    

    ##2.3 Loop data structures - 

    #start_list
    start_list = [[] for _ in range(int(pos_max - pos_min + 1))]
    #close_list
    ###close_list = -1*np.ones(int(pos_max- pos_min +1))

    #microjams_store
    micro_jams_store = np.zeros(N)

    #posn variable
    posn = 0 - pos_min

    #cleared cars variable
    cleared_cars = 0
    #index variable
    i = 0

    #PART-3 Loop

    #While loop condition
    while (cleared_cars < total_jammed_cars):
        #update position
        
        posn = int(posn+ rev[i])
        

        #if condition for start_list append
        if(rev[i%N] == 1 and rev[(i+1)%N] == 1) or (rev[i%N] == 1 and rev[(i-1)%N] == 1):

            #make close_list empty/-1
            ###close_list[posn] = -1
            #append to start list
            #print('start candidate added: ', i)####
            start_list[posn].append(i)
            #print(start_list)####

        #if condition for close_list append
        if (((rev[i%N] == -1 and rev[(i+1)%N] == -1) or (rev[i%N] == -1 and rev[(i-1)%N] == -1))
    and (posn+1 < len(start_list)) and len(start_list[posn+1]) != 0):
            #print('close candidate: ', (N-i-1)%N)####
            #while start list at posn not empty condition(takes care of when start list is empty)
            if(switch == 0):
                while (len(start_list[posn+1]) != 0):
                    pop_index = start_list[posn+1].pop()
                    micro_jams_store[int(N-pop_index)%N] = (i - pop_index-1)/2 +1
                    cleared_cars += 1
                    #print('cleared_cars = ', cleared_cars)####
                    #pop start list index and get microjam
                    #add it to microjam array
                    #increment cleared_cars variable by 1
            if(switch == 1):
                max_val = float('-inf')
                while (len(start_list[posn+1]) != 0):
                    pop_index = start_list[posn+1].pop()
                    max_val = max(max_val,(i - pop_index-1)/2 +1)
                micro_jams_store[int(N - i)%N] = max_val
                cleared_cars += 1

        #increase index by 1
        #print('--------------------')####
        i = i+1
    return micro_jams_store

#Above code in this block are necessary functions to run for getting the microjams from the initial condition; it works for rho<0.5 - 
