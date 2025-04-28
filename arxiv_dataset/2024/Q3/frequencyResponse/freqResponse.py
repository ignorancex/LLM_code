# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:51:33 2023

@author: hsauro

MIT License

Copyright (c) 2024 UW Sauro Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import roadrunner
import numpy as np
import math
import matplotlib.pyplot as plt

class FreqencyResponse:
    
    def __init__(self, r):
        self.r = r
        self.r.conservedMoietyAnalysis = True
    
    # Gets a range of values spread along a log scale
    def getLogSpace (self, startW, numDecades, numPoints):
        d1 = 0
        y = np.zeros (numPoints)
        for i in range (numPoints-1):
            y[i] = i*(numDecades-d1)
            y[i] = y[i]/(numPoints-1)
            y[i] = d1 + y[i]
            y[i] = (10**y[i]) * startW
        y[numPoints-1] = (10**numDecades)*startW
        return y
    
    def getPhase (self, val):
        if (val.real == 0.0) and (val.imag == 0.0):
           return 0.0
        else:
           return math.atan2(val.imag, val.real)

    # Compute the frequency response with respect to a flux
    # H_J (jw) = dv/ds H(jw) + dvdp
    def getFluxFrequencyResponse(self, startFrequency, numberOfDecades, numberOfPoints,
                      parameterId, variableId, useDB=True, useHz = True):
        """
           Compute the flux frequency response 
 
           Parameters
           ----------
           startFrequency : float
               Start frequency for the frequency response
           numberOfDecades : integer
               How many decades to use beyond the starting frequency
           numberOfPoints : integer
               How many data points to generate, too few and the Bode plots will not look smooth
           parameterId : string
               Name of the input/parameter in the model
           variableId : string
               Name of the reaction which carries the flux
           useDB : boolean
               Use Decibels onthe y-axis in the amplitude plot
            useHz  boolean
               Use Hz or rads/sec for the frequency x-axis  
         
          Returns
          -------
           numpy array
             Returns a numpy array containins frequency response
             First column = frequency 
             Second column = amplitude
             Third column = phase
       """
    
        numReactions = self.r.getNumReactions()
        numFloatingSpecies = self.r.getNumIndFloatingSpecies()
        reactionIds = self.r.getReactionIds()
        boundarySpeciesIds = self.r.getBoundarySpeciesIds()
        globalParameterIds = self.r.getGlobalParameterIds()
        
        if variableId not in reactionIds:
           raise Exception ("Can't find the reaction:" + variableId + " in the model")     
               
        # Check that the parameter name exists
        if parameterId not in boundarySpeciesIds:
           if parameterId not in globalParameterIds:
               raise Exception ("Can't find the parameter/input:" + parameterId + " in the model")     
       
        try:
           self.r.steadyState()
        except Exception as e:
            raise Exception ("Can't find the steady state: " + str (e))
       
        linkMatrix = self.r.getLinkMatrix()
        linkMatrix = np.array (linkMatrix)
        linkMatrix = linkMatrix.astype (complex)        
        
        ee = self.r.getUnscaledElasticityMatrix()
        ee = np.array (ee)
        ee = ee.astype (complex)
        
        # Convert stoich matrix to complex matrix
        Nr = self.r.getReducedStoichiometryMatrix()
        Nr = np.array (Nr)
        Nr = Nr.astype (complex)
        
        # Compute the Jacobian Nr dv/ds L
        Jac = np.matmul (Nr, ee)
        Jac = np.matmul (Jac, linkMatrix)
             
        resultArray = np.zeros ((numberOfPoints, 3))
    
        identMatrix = np.identity(numFloatingSpecies, dtype=complex)
        
        w = self.getLogSpace (startFrequency, numberOfDecades, numberOfPoints)   
        for i, wf in enumerate(w):
            
            # Compute H(jw) = L (jw I - Nr dv/dx L )^{-1} Nr dv/dp
            jw_val = complex (0, wf)
            T1 = identMatrix*jw_val         # = jwI
            T2 = T1 - Jac                   # = jwI - Jac
            inverse = np.linalg.inv(T2)     # = (jwI - Jac)^(-1)
            T3 = np.matmul(inverse, Nr)     # = (jwI - Jac)^(-1) Nr
            T3 = np.matmul(linkMatrix, T3)  # = L (Nr dv/ds L)^(-1) Nr
            
            dvdp = np.zeros(numReactions, dtype=complex)
            for k in range (numReactions): 
                val = self.r.getUnscaledParameterElasticity(reactionIds[k], parameterId);
                dvdp[k] = val+0j
            T4 = np.matmul (T3, dvdp)       
            
            # Next compute H^v (jw) = dv/ds H(jw) + dv/dp            
            T4 = np.matmul (ee, T4) + dvdp

            # we need to pick out the requested output flux
            for  j in range (numReactions):
                if (reactionIds[j] == variableId):
                    dw = abs(T4[j])
                    if useDB:
                       dw = 20.0 * math.log10(dw)
                    resultArray[i, 1] = dw
        
                    val = complex (T4[j], 0)
                    phase = self.getPhase(val)
                    resultArray[i, 2] = phase
                    break
    
            if (useHz):
               # Store frequency, convert to Hz by dividing by 2Pi
               resultArray[i,0] = w[i] / (2. * math.pi);
            else:
               # Store frequency, leave as rad/sec
               resultArray[i,0] = w[i]
            
        # This is to prevent discontinous jumps at the 180 degree points
        resultArray[:,2] = np.unwrap(resultArray[:,2])
        # Convert to degrees
        resultArray[:,2] = (180/math.pi)*resultArray[:,2]
        # Store result internally so that it can be used by plot
        self.results = resultArray
        return resultArray
               
    # Implement this:
    # This should be able to handle models with conserved cycles
    #   H(jw) = L (jw I - Nr dv/dx L )^{-1} Nr dv/dp
    def getSpeciesFrequencyResponse(self, startFrequency, numberOfDecades, numberOfPoints,
                      parameterId, variableId, useDB=True, useHz = True):
        """
           Compute the species concentration frequency response 
 
           Parameters
           ----------
           startFrequency : float
               Start frequency for the frequency response
           numberOfDecades : integer
               How many decades to use beyond the starting frequency
           numberOfPoints : integer
               How many data points to generate, too few and the Bode plots will not look smooth
           parameterId : string
               Name of the input/parameter in the model
           variableId : string
               Name of the species that will be the output for the frequency response
           useDB : boolean
               Use Decibels onthe y-axis in the amplitude plot
            useHz  boolean
               Use Hz or rads/sec for the frequency x-axis  
         
          Returns
          -------
           numpy array
             Returns a numpy array containins frequency response
             First column = frequency 
             Second column = amplitude
             Third column = phase
       """
    
        numReactions = self.r.getNumReactions()
        numIndependentFloatingSpecies = self.r.getNumIndFloatingSpecies()
        numFloatingSpecies = self.r.getNumFloatingSpecies()
        reactionIds = self.r.getReactionIds()
        speciesIds = self.r.getFloatingSpeciesIds()
        print ('sp = ', speciesIds)
        boundarySpeciesIds = self.r.getBoundarySpeciesIds()
        globalParameterIds = self.r.getGlobalParameterIds()
    
        if variableId not in speciesIds:
           raise Exception ("Can't find species:" + variableId + " in the model")     
               
        # Check that parameter name exists
        if parameterId not in boundarySpeciesIds:
           if parameterId not in globalParameterIds:
               raise Exception ("Can't find parameter:" + parameterId + " in the model")     
               
        try:
           self.r.steadyState()
        except Exception as e:
            raise Exception ("Can't find the steady state: " + str (e))

        # Convert link matrix to complex matrix                     
        linkMatrix = self.r.getLinkMatrix()
        linkMatrix = np.array (linkMatrix)
        linkMatrix = linkMatrix.astype (complex)        
        
        # Convert elasticity matrix to complex matrix
        ee = self.r.getUnscaledElasticityMatrix()
        ee = np.array (ee)
        ee = ee.astype (complex)
        
        # Convert stoich matrix to complex matrix
        Nr = self.r.getReducedStoichiometryMatrix()
        Nr = np.array (Nr)
        Nr = Nr.astype (complex)
        
        # Compute the Jacobian Nr dv/ds L
        Jac = np.matmul (Nr, ee)
        Jac = np.matmul (Jac, linkMatrix)
       
        resultArray = np.zeros ((numberOfPoints, 3))
    
        identMatrix = np.identity(numIndependentFloatingSpecies, dtype=complex)
        
        w = self.getLogSpace (startFrequency, numberOfDecades, numberOfPoints)   
        for i, wf in enumerate(w):
               
            # Compute H(jw) = L (jw I - Nr dv/dx L )^{-1} Nr dv/dp
            jw_val = complex (0, wf)
            T1 = identMatrix*jw_val         # = jwI
            T2 = T1 - Jac                   # = jwI - Jac
            inverse = np.linalg.inv(T2)     # = (jwI - Jac)^(-1)
            T3 = np.matmul(inverse, Nr)     # = (jwI - Jac)^(-1) Nr
            T3 = np.matmul(linkMatrix, T3)  # = L (Nr dv/ds L)^(-1) Nr
                       
            dvdp = np.zeros(numReactions, dtype=complex)
            for k in range (numReactions): 
                val = self.r.getUnscaledParameterElasticity(reactionIds[k], parameterId);
                dvdp[k] = val+0j
            T4 = np.matmul (T3, dvdp)    # = L (Nr dv/ds L)^(-1) Nr dvdp
            
            # we need to pick out the requested output species
            for  j in range (numFloatingSpecies):
                if (speciesIds[j] == variableId):
                    dw = abs(T4[j])
                    if useDB:
                       dw = 20.0 * math.log10(dw)
                    resultArray[i, 1] = dw
        
                    val = complex (T4[j], 0)
                    phase = self.getPhase(val)
                    resultArray[i, 2] = phase
                    break
    
            if (useHz):
               # Store frequency, convert to Hz by dividing by 2Pi
               resultArray[i,0] = w[i] / (2. * math.pi);
            else:
               # Store frequency, leave as rad/sec
               resultArray[i,0] = w[i]
               
        # This is to prevent discontinous jumps at the 180 degree points
        resultArray[:,2] = np.unwrap(resultArray[:,2])
        # Convert to degrees
        resultArray[:,2] = (180/math.pi)*resultArray[:,2]
        # Store result internally so that it can be used by plot
        self.results = resultArray
        return resultArray


    def plot (self, aresult=None, pdf=None, amplitude=True, phase=True):
        """
           Plot the Bode plots for the last frequency analysis carried out
 
           Parameters
           ----------
           aresult : numpy array
              Use this to supply your own numy array to plot. Not required very often
           pdf : string
               Set to the name of file name (with path if required) if you want a pdf of the plots
           ampltiude, phase : boolean
               Can be use to indicate that you only want the amplitude plot or the phase plot.
               By default both amplitude and phase responses are plotted. 
       """
       
        if aresult == None:
           aresult = self.results
           
        if amplitude:
            plt.subplot(211)
            plt.xscale("log")
            plt.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.3)
            plt.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.3)
            plt.minorticks_on()
            #plt.ylim((-100,0))
            plt.plot (aresult[:,0], aresult[:,1], 'r', linewidth=1.5)
            plt.ylabel('Amp')
            # In case they only ask for amplitude plot, we will need a x label
            if not phase:
               plt.xlabel ('Frequency') 
            
        if phase:
            plt.subplot(212)
            plt.xscale("log")
            plt.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.3)
            plt.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.3)
            plt.minorticks_on()
            plt.plot (aresult[:,0], aresult[:,2], 'r', linewidth=1.5)
            plt.ylabel('Phase (Degrees)')
            plt.xlabel ('Frequency') 
        
        if pdf != None:
            plt.savefig(pdf)
        plt.show()
    