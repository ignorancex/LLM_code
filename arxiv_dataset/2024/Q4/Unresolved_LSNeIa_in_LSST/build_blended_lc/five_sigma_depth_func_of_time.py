#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:53:05 2020

@author: shuber
"""

import numpy as np
import matplotlib.pyplot as plt

class five_sigma_depth():
    def __init__(self,filter_):
        self.filter_ = filter_
        self.duration_between_two_full_moons = 29.5
    
    def f_five_sigma_depth_as_function_of_time_sinus(self,time_to_get_five_sigma_depth,mean_depth,variation,start_in_moon_phase = "random"):

        
        min_ = np.min(time_to_get_five_sigma_depth)- self.duration_between_two_full_moons
        max_ = np.max(time_to_get_five_sigma_depth)+ self.duration_between_two_full_moons
        
        periodizitaet = self.duration_between_two_full_moons/2/np.pi


        # 0 assumes first quarter, 29.5/4 assumes  full moon, 29.5/2 assumes third quarter, 29.5*3/4 assumes new moon, 29.5 is then again firt quarter       
        if start_in_moon_phase == "random":
            start = np.random.uniform(0,self.duration_between_two_full_moons)
        else:
            start = start_in_moon_phase 

        x = np.linspace(min_,max_,num=10000)
        # 0 assumes first quarter, 29.5/4 assumes  full moon, 29.5/2 assumes third quarter, 29.5*3/4 assumes new moon, 29.5 is then again firt quarter       
        y = - variation * np.sin(x/periodizitaet) + mean_depth
        
        #plt.plot(x,y)        
        y_interp = np.interp(time_to_get_five_sigma_depth+start,x,y)

        
        return y_interp 
        
    def f_five_sigma_depth_as_function_of_time_ESO(self,time_to_get_five_sigma_depth,reference_depth,start_in_moon_phase = "random"):
        
        min_ = np.min(time_to_get_five_sigma_depth)- self.duration_between_two_full_moons
        max_ = np.max(time_to_get_five_sigma_depth)+ self.duration_between_two_full_moons
        
        #time_reference = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,28,self.duration_between_two_full_moons]) # time 0 is first quarter
        #five_sigma_reference = np.array([25.5,25.0,24.8,24.2,23.7,23.8,24.3,24.8,25.5,25.5,25.5,25.5,25.5,25.5,25.5])


        time_reference = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,self.duration_between_two_full_moons]) # time 0 is first quarter

        #print self.filter_
        if self.filter_ == "u":
            five_sigma_reference = np.array([24.2,23.9,23.6,23.3,22.8,23.3,23.6,23.8,24.1,24.4,24.7,24.8,24.8,24.7,24.4,24.2])
        elif self.filter_ == "g":
            five_sigma_reference = np.array([25.4,25.2,24.7,24.4,23.9,24.4,24.7,25,25.3,25.6,25.9,26.1,26.1,26,25.7,25.4])
        elif self.filter_ == "r":
            five_sigma_reference = np.array([25.1,24.8,24.5,24.2,23.8,24.2,24.5,24.7,25.0,25.2,25.3,25.4,25.4,25.4,25.2,25.1])
        elif self.filter_ == "i":
            five_sigma_reference = np.array([24.5,24.4,24.2,24,23.6,24,24.2,24.3,24.5,24.6,24.7,24.7,24.7,24.7,24.6,24.5])
        elif self.filter_ == "z":
            five_sigma_reference = np.array([23.8,23.7,23.6,23.6,23.3,23.6,23.6,23.7,23.8,23.8,23.8,23.8,23.8,23.8,23.8,23.8])
        elif self.filter_ == "y":
            five_sigma_reference = np.array([22.5,22.5,22.4,22.4,22.3,22.4,22.4,22.5,22.5,22.5,22.5,22.5,22.5,22.5,22.5,22.5])
        elif self.filter_ == "J":
            five_sigma_reference = np.array([22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2,22.2])
        elif self.filter_ == "H":
            five_sigma_reference = np.array([21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3])

        if self.filter_ == "u" or self.filter_ == "g" or self.filter_ == "r" or self.filter_ == "i":
            time_new_moon = 23
            indizes_around_full_moon = np.where((time_reference >= time_new_moon-7.5) & (time_reference <= time_new_moon+7.5))
            ref_depth_around_intresting_time = np.mean(five_sigma_reference[indizes_around_full_moon])
        
        if self.filter_ == "z" or self.filter_ == "y" or self.filter_ == "J" or self.filter_ == "H":
            time_full_moon = 8
            indizes_around_full_moon = np.where((time_reference >= time_full_moon-7.5) & (time_reference <= time_full_moon+7.5))
            ref_depth_around_intresting_time = np.mean(five_sigma_reference[indizes_around_full_moon])
        
        shift_of_depth = reference_depth - ref_depth_around_intresting_time
        
        #print reference_depth
        
        five_sigma_reference = five_sigma_reference + shift_of_depth
        
        #print five_sigma_reference[indizes_around_full_moon]
        #print np.mean(five_sigma_reference[indizes_around_full_moon])
        
        #print five_sigma_reference
        
        time = time_reference
        five_sigma_depth = five_sigma_reference



        # 0 assumes first quarter, 29.5/4 assumes  full moon, 29.5/2 assumes third quarter, 29.5*3/4 assumes new moon, 29.5 is then again firt quarter       
        if start_in_moon_phase == "random":
            start = np.random.uniform(0,self.duration_between_two_full_moons)
        else:
            start = start_in_moon_phase 


        
        
        counter = 0
        while np.min(time) >= min_:
            counter += 1
            time_extend_lower_end = time_reference - counter * self.duration_between_two_full_moons 
            time = np.concatenate((time_extend_lower_end[:-1],time))
            
            
            five_sigma_depth = np.concatenate((five_sigma_reference[:-1],five_sigma_depth))
            
        counter = 0 
        while np.max(time) <= max_:
            counter += 1
            time_extend_higher_end = time_reference + counter * self.duration_between_two_full_moons 
    
            time = np.concatenate((time,time_extend_higher_end[1:]))
    
            five_sigma_depth = np.concatenate((five_sigma_depth,five_sigma_reference[1:]))
            


        #plt.plot(time,five_sigma_depth)
        
        five_sigma_depth_interp = np.interp(time_to_get_five_sigma_depth+start,time,five_sigma_depth)
        
        return five_sigma_depth_interp

        





if __name__ == "__main__":
    time_to_get_five_sigma_depth = np.linspace(0,29.5,num=1000)    
    
    """
    fsd = five_sigma_depth_sinus(filter_="g",mean_depth=24.5,variation=1)
    
    five_sigma_depth_interpol = fsd.f_five_sigma_depth_as_function_of_time(time_to_get_five_sigma_depth,start_in_moon_phase=0)
    """
    plt.rcParams.update({'font.size': 14})
    depth = {"u": 23.3, 
             "g": 24.7, 
             "r": 24.3, 
             "i": 23.7, 
             "z": 22.8, 
             "y": 22.0,
             "J": 22.0-0.3,
             "H": 22.0-1.2}   
    
    d_color = {"u":"blueviolet", "g":"blue", "r": "turquoise", "i":"darkgreen","z":"lime", "y": "gold","J": "darkorange","H": "red"}




    for filter_ in ["u","g","r","i","z","y","J","H"]:
        fsd = five_sigma_depth(filter_=filter_)
        
        five_sigma_depth_interpol = fsd.f_five_sigma_depth_as_function_of_time_ESO(time_to_get_five_sigma_depth,reference_depth=depth[filter_]+1,start_in_moon_phase=0)
        
        plt.plot(time_to_get_five_sigma_depth,five_sigma_depth_interpol,label=filter_, color = d_color[filter_])
        plt.ylabel("mag")
        plt.xlabel("time [days]")
        plt.ylim(26.3,21)
        plt.legend(title='filter', bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("/afs/mpa/home/shuber/research/MicrolensSN/plot/moon_phase.png",format="png",dpi=200)


    
    """    
    time_1 = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,28]) # time 0 is first quarter
    five_sigma_1 = np.array([25.5,25.0,24.8,24.2,23.7,23.8,24.3,24.8,25.5,25.5,25.5,25.5,25.5,25.5])
    
    plt.plot(time_1,five_sigma_1,label = "Ra = 1, dec = -20")
    
    time_2 = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,28]) # time 0 is first quarter
    five_sigma_2 = np.array([25.4,25.4,25.3,24.8,24.4,24.5,24.6,24.8,24.9,25.2,25.4,25.4,25.4,25.4]) 
    
    plt.plot(time_2,five_sigma_2,label = "Ra = 6, dec = -20")
    
    plt.legend()
    plt.savefig("/afs/mpa/home/shuber/research/MicrolensSN/plot/moon_phase.png",format="png",dpi=200)
    """