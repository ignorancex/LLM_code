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
import pickle
import numpy as np
import datetime as dt

from base_immunization_parameters import BaseImmunizationParameters
from person import Person
from case_parameters import CaseParameters
from config import Config
from loss_parameters import LossParameters
from population_parameters import PopulationParameters
from utils import *
from vaccination_parameters import VaccinationParameters
from variant_parameters import VariantParameters

class Simulation:
    def __init__(self,config:Config) -> None:
        """
        Class to instantiate a new immunization level simulation
        :param config: config instance of the simulation
        """
        #initialize parameter classes
        self.vaccinations = VaccinationParameters(config)
        self.lossParameters = LossParameters(config)
        self.baseImmunizationParameters = BaseImmunizationParameters(config)
        self.caseParameters = CaseParameters(config)
        self.variantParameters = VariantParameters(config)
        self.populationParameters = PopulationParameters(config)
        self.config = config

    def add_together(self,array1:np.array, index1:int, array2:np.array) -> None:
        """
        Function to add two arrays together, whereas the second one is shifted back by index1. The result will have the length of array1. E.g. add_together([1,2,3,4,5,6],3,[1,1,1,1]) will result in [1,2,3,5,6,7].

        The method operates on array1 and does not return anything.
        :param array1:
        :param index1:
        :param array2:
        :return:
        """
        n = len(array1)
        m = len(array2)
        n2 = n-index1
        if n2<0:
            None
        elif n2>m:
            array1[index1:index1+m]+=array2
        else:
            array1[index1:] += array2[:n2]

    def get_cache_filename(self) -> str:
        """
        Return a filename for saving and loading cached simulation results
        :return: filepath as string
        """
        folder = 'cache'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filename = os.path.join(folder,self.config.hash() + '.pickle')
        return filename

    def try_to_load_from_cached(self) -> dict:
        """
        Attempts to load a simulation result from a cached file.
        :return: empty dict, if no file was found, a simulation result as dict otherwise
        """
        filename = self.get_cache_filename()
        try:
            with open(filename,'rb') as f:
                result = pickle.load(f)
            print('loaded cached result')
            return result
        except:
            print('no cached result found')
            return {}

    def save_as_pickle(self,result:dict) -> None:
        """
        Save a simulation result to a pickle binary file for caching.
        :param result: result as dictionary
        :return:
        """
        filename = self.get_cache_filename()
        with open(filename, 'wb') as f:
            pickle.dump(result,f)

    def run(self) -> dict:
        """
        Routine to run the simulation. Automatically iterates over the simulation time window specified in the config and evaluates infections and vaccinations for immunization.
        :return: simulation result as dictionary
        """
        np.random.seed(self.config.seed) #set the seed of the random number generator for reproducibility reasons
        OBSERVABLES = list(self.config.observables.keys())
        result = self.try_to_load_from_cached() #try to load a cached result
        if result != {}:
            return result
        else:
            print('start simulation')
            steps = (self.config.tend-self.config.t0).days+1
            times = [self.config.t0 + dt.timedelta(x) for x in range(steps)]

            #specify arrays for output
            ImmunesVaccinated = dict()
            ImmunesRecovered = dict()
            ImmunesRecoveredUndet = dict()
            ImmunesVaccinatedAndRecovered = dict()
            ImmunesVaccinatedAndRecoveredUndet = dict()
            for o in OBSERVABLES:
                ImmunesVaccinated[o] = np.zeros(steps)
                ImmunesRecovered[o] = np.zeros(steps)
                ImmunesRecoveredUndet[o] = np.zeros(steps)
                ImmunesVaccinatedAndRecovered[o] = np.zeros(steps)
                ImmunesVaccinatedAndRecoveredUndet[o] = np.zeros(steps)
            DetReinfections = dict()
            for v in self.variantParameters.get_variants():
                for v2 in self.variantParameters.get_variants():
                    DetReinfections[(v,v2)] = np.zeros(steps)
            for v2 in self.variantParameters.get_variants():
                DetReinfections[(None, v2)] = np.zeros(steps)
            Vaccinated = np.zeros(steps)
            Recovered = np.zeros(steps)
            RecoveredUndet = np.zeros(steps)
            VaccinatedAndRecovered = np.zeros(steps)
            VaccinatedAndRecoveredUndet = np.zeros(steps)
            Active = np.zeros(steps)
            ActiveUndet = np.zeros(steps)

            #initialize population
            fed = self.config.federalstate
            N = self.populationParameters.get_population(fed)
            scale = self.config.scale
            Persons = [Person(x) for x in range(int(N*scale))]
            for p in Persons:
                p.immune = {o:False for o in OBSERVABLES}
                p.immDate = {o:None for o in OBSERVABLES}
                p.lossDate = {o:None for o in OBSERVABLES}

            #main loop
            for i in range(steps):
                print('\r{: 4d}/{: 4d}'.format(i+1,steps),end='')

                t = times[i]
                #get cases and vaccinations for the current date
                v1 = int(round(self.vaccinations.get(t, 1,fed)*scale,0))
                v2 = int(round(self.vaccinations.get(t, 2,fed)*scale,0))
                v3 = int(round(self.vaccinations.get(t, 3,fed)*scale,0))
                v4 = int(round(self.vaccinations.get(t, 4,fed)*scale,0))
                c1 = int(round(self.caseParameters.get(t,True,fed)*scale,0))
                c2 = int(round(self.caseParameters.get(t,False,fed)*scale,0))

                #shuffle the list - expensive but necessary
                np.random.shuffle(Persons)

                #loop ofer agents
                for p in Persons:
                    #as long as cases (c1,c2) or vaccines (v1,v2,v3) are available, we try to distribute them among the persons
                    if c1>0 and not p.active:
                        cause = self.variantParameters.sample_variant(t.date())
                        if p.immune[cause]==False:
                            p.confDate = i + np.random.choice(self.config.detDelay) # day of detection
                            rday = i + np.random.choice(self.config.recoveryDelay[0]) # day of recovery
                            p.recDate = rday
                            baseImm = self.baseImmunizationParameters.sample_base_immunity_all(cause,
                                                                                               OBSERVABLES)  # sample where recovery leads to immunity at all
                            loseDays = self.lossParameters.sample_loss(cause,
                                                                       OBSERVABLES)  # sample day of immunity loss
                            for target, bi, ld in zip(OBSERVABLES, baseImm, loseDays):
                                if bi:
                                    p.immDate[target] = rday # render immune after recovery
                                    lossDate = rday+ld
                                    # if the agent currently has a future immunity loss date defined, take the maximum
                                    if (p.lossDate[target]!=None):
                                        p.lossDate[target] = max(p.lossDate[target],lossDate)
                                    else:
                                        p.lossDate[target] = lossDate
                            #state changes
                            p.conf = False
                            p.active = True
                            DetReinfections[(p.variant,cause)][i]+=1
                            p.variant = cause #p.variant is only CONFIRMED variant
                            c1-=1 #reduce number of confirmed infections
                    elif c2>0 and not p.active: #analogous to detected infections
                        cause = self.variantParameters.sample_variant(t.date())
                        if p.immune[cause] == False:
                            rday = i + np.random.choice(self.config.recoveryDelay[1])
                            p.recDate = rday
                            baseImm = self.baseImmunizationParameters.sample_base_immunity_all(cause,
                                                                                 OBSERVABLES)
                            loseDays = self.lossParameters.sample_loss(cause, OBSERVABLES)
                            for target,bi,ld in zip(OBSERVABLES,baseImm,loseDays):
                                if bi:
                                    p.immDate[target] = rday
                                    lossDate = rday + ld
                                    if (p.lossDate[target] != None):
                                        p.lossDate[target] = max(p.lossDate[target], lossDate)
                                    else:
                                        p.lossDate[target] = lossDate
                            p.conf = False
                            p.active = True
                            c2-=1
                    elif p.vacc==0 and p.active==False and v1>0: #first vaccinations only for persons who are not active and are not vaccinated yet
                        p.vaccDate = i
                        baseImm = self.baseImmunizationParameters.sample_base_immunity_all('VACC1',
                                                                                           OBSERVABLES)
                        loseDays = self.lossParameters.sample_loss('VACC1', OBSERVABLES)
                        for target, bi, ld in zip(OBSERVABLES, baseImm, loseDays):
                            if bi:
                                p.immDate[target] = i + self.config.vaccDelay
                                lossDate = i + self.config.vaccDelay + ld
                                if (p.lossDate[target] != None):
                                    p.lossDate[target] = max(p.lossDate[target], lossDate)
                                else:
                                    p.lossDate[target] = lossDate
                                p.lossDate[target] = lossDate
                        p.vacc = 1
                        v1 -= 1
                    elif p.vacc==1 and p.active== False and v2>0 and (i-p.vaccDate)>self.config.vaccIntervals[0]: #second vaccinations only for persons who are not active, have already got a first shot, and time between vaccinations is at least x -days
                        p.vaccDate = i
                        baseImm = self.baseImmunizationParameters.sample_base_immunity_all('VACC2',
                                                                                           OBSERVABLES)
                        loseDays = self.lossParameters.sample_loss('VACC2', OBSERVABLES)
                        for target, bi, ld in zip(OBSERVABLES, baseImm, loseDays):
                            if bi:
                                p.immDate[target] = i + self.config.vaccDelay
                                lossDate = i + self.config.vaccDelay + ld
                                if (p.lossDate[target] != None):
                                    p.lossDate[target] = max(p.lossDate[target], lossDate)
                                else:
                                    p.lossDate[target] = lossDate
                                p.lossDate[target] = lossDate
                        p.vacc = 2
                        v2 -= 1
                    elif p.vacc==2 and p.active== False and v3>0 and (i-p.vaccDate)>self.config.vaccIntervals[1]: #third vaccinations only for persons who are not active, have already got a second shot, and time between vaccinations is at least x -days
                        p.vaccDate = i
                        baseImm = self.baseImmunizationParameters.sample_base_immunity_all('VACC3',
                                                                                           OBSERVABLES)
                        loseDays = self.lossParameters.sample_loss('VACC3', OBSERVABLES)
                        for target, bi, ld in zip(OBSERVABLES, baseImm, loseDays):
                            if bi:
                                p.immDate[target] = i + self.config.vaccDelay
                                lossDate = i + self.config.vaccDelay + ld
                                if (p.lossDate[target] != None):
                                    p.lossDate[target] = max(p.lossDate[target], lossDate)
                                else:
                                    p.lossDate[target] = lossDate
                                p.lossDate[target] = lossDate
                        p.vacc = 3
                        v3 -= 1
                    elif p.vacc==3 and p.active== False and v4>0 and (i-p.vaccDate)>self.config.vaccIntervals[2]: #fourth vaccinations only for persons who are not active, have already got a second shot, and time between vaccinations is at least x -days
                        p.vaccDate = i
                        baseImm = self.baseImmunizationParameters.sample_base_immunity_all('VACC4',
                                                                                           OBSERVABLES)
                        loseDays = self.lossParameters.sample_loss('VACC4', OBSERVABLES)
                        for target, bi, ld in zip(OBSERVABLES, baseImm, loseDays):
                            if bi:
                                p.immDate[target] = i + self.config.vaccDelay
                                lossDate = i + self.config.vaccDelay + ld
                                if (p.lossDate[target] != None):
                                    p.lossDate[target] = max(p.lossDate[target], lossDate)
                                else:
                                    p.lossDate[target] = lossDate
                                p.lossDate[target] = lossDate
                        p.vacc = 4
                        v4 -= 1
                    #################### EVALUATE STATE CHANGES ###################
                    for target in OBSERVABLES:
                        if p.immDate[target] == i:
                            p.immDate[target] = None
                            p.immune[target] = True
                        if p.lossDate[target] == i:
                            p.lossDate[target] = None
                            p.immune[target] = False
                    if p.recDate == i:
                        p.recDate = None
                        p.active = False
                        p.rec = True
                    if p.confDate == i:
                        p.confDate = None
                        p.conf = True
                    #################### SUMMARIZE ###################
                    if p.active == True:
                        if p.conf:
                            Active[i] += 1
                            if p.vacc > 0:
                                VaccinatedAndRecovered[i] += 1
                                for target in OBSERVABLES:
                                    ImmunesVaccinatedAndRecovered[target][i] += 1
                            else:
                                Recovered[i] += 1
                                for target in OBSERVABLES:
                                    ImmunesRecovered[target][i] += 1
                        else:
                            ActiveUndet[i] += 1
                            if p.vacc>0:
                                VaccinatedAndRecoveredUndet[i] += 1
                                for target in OBSERVABLES:
                                    ImmunesVaccinatedAndRecoveredUndet[target][i] += 1
                            else:
                                RecoveredUndet[i] += 1
                                for target in OBSERVABLES:
                                    ImmunesRecoveredUndet[target][i] += 1
                    elif p.vacc>0 and not p.rec:
                        Vaccinated[i]+=1
                        for target in OBSERVABLES:
                            if p.immune[target]:
                                ImmunesVaccinated[target][i]+=1
                    elif p.vacc>0 and p.rec:
                        if p.conf:
                            VaccinatedAndRecovered[i]+=1
                            for target in OBSERVABLES:
                                if p.immune[target]:
                                    ImmunesVaccinatedAndRecovered[target][i]+=1
                        else:
                            VaccinatedAndRecoveredUndet[i] += 1
                            for target in OBSERVABLES:
                                if p.immune[target]:
                                    ImmunesVaccinatedAndRecoveredUndet[target][i] += 1
                    elif p.rec:
                        if p.conf:
                            Recovered[i]+=1
                            for target in OBSERVABLES:
                                if p.immune[target]:
                                    ImmunesRecovered[target][i]+=1
                        else:
                            RecoveredUndet[i] += 1
                            for target in OBSERVABLES:
                                if p.immune[target]:
                                    ImmunesRecoveredUndet[target][i] += 1
            # simulation results are given in relative numbers. I.e. divide numbers by N*scale
            Vaccinated /= (scale)
            Recovered /= (scale)
            RecoveredUndet /= (scale)
            VaccinatedAndRecovered /= (scale)
            VaccinatedAndRecoveredUndet /= (scale)
            for target in OBSERVABLES:
                ImmunesVaccinated[target] /= (scale)
                ImmunesRecovered[target] /= (scale)
                ImmunesRecoveredUndet[target] /= (scale)
                ImmunesVaccinatedAndRecovered[target] /= (scale)
                ImmunesVaccinatedAndRecoveredUndet[target] /= (scale)
            Active /= (scale)
            ActiveUndet /= (scale)
            for v1 in self.variantParameters.variants:
                for v2 in self.variantParameters.variants:
                    DetReinfections[(v1,v2)]/= (scale)
            for v2 in self.variantParameters.variants:
                DetReinfections[(None, v2)] /= (scale)

            #setup result dictionary
            result = {'vaccinated':Vaccinated,
                'past detected':Recovered,
                'past undetected':RecoveredUndet,
                'past detected + vaccinated':VaccinatedAndRecovered,
                'past undetected + vaccinated':VaccinatedAndRecoveredUndet,
                'active detected':Active,
                'active undetected':ActiveUndet,
                'time':times}

            variants = self.variantParameters.get_variants()
            ratiosList = self.variantParameters.get_variant_ratios(times)
            ratios = {v: list() for v in variants}
            for rl in ratiosList:
                for v, r in zip(variants, rl):
                    ratios[v].append(r)
            for k, v in ratios.items():
                result['active detected ' + k] = Active * np.array(v)
                result['active undetected ' + k] = ActiveUndet * np.array(v)

            for target in OBSERVABLES:
                result['vaccinated immune '+target] = ImmunesVaccinated[target]
                result['past detected immune '+target] = ImmunesRecovered[target]
                result['past undetected immune '+target] = ImmunesRecoveredUndet[target]
                result['past detected + vaccinated immune '+target] = ImmunesVaccinatedAndRecovered[target]
                result['past undetected + vaccinated immune '+target] = ImmunesVaccinatedAndRecoveredUndet[target]

                result['vaccinated susceptible ' + target] = Vaccinated - ImmunesVaccinated[target]
                result['past detected susceptible ' + target] = Recovered - ImmunesRecovered[target]
                result['past undetected susceptible ' + target] = RecoveredUndet - ImmunesRecoveredUndet[target]
                result['past detected + vaccinated susceptible ' + target] = VaccinatedAndRecovered - ImmunesVaccinatedAndRecovered[target]
                result['past undetected + vaccinated susceptible ' + target] = VaccinatedAndRecoveredUndet - ImmunesVaccinatedAndRecoveredUndet[target]
                result['immune' + target] = ImmunesRecovered[target] + ImmunesRecoveredUndet[target] + ImmunesVaccinated[target] + ImmunesVaccinatedAndRecovered[target] + ImmunesVaccinatedAndRecoveredUndet[target]

            for v1 in self.variantParameters.variants:
                for v2 in self.variantParameters.variants:
                    result['detected reinfection ({},{})'.format(v1,v2)] = DetReinfections[(v1,v2)]
            for v2 in self.variantParameters.variants:
                result['detected reinfection (None,{})'.format(v2)] = DetReinfections[(None,v2)]

            fed = self.config.federalstate
            for t in times:
                if fed == None:
                    try:
                        cases = self.caseParameters.cases[t][True]
                    except:
                        cases = 0
                else:
                    try:
                        cases = self.caseParameters.casesFed[fed][t][True]
                    except:
                        cases = 0
                cases2 = self.caseParameters.get(t, True, fed) + self.caseParameters.get(t, False, fed)
                for v, s in zip(self.variantParameters.get_variants(), self.variantParameters.get_variant_ratio(t.date())):
                    try:
                        result['new confirmed ' + v].append(int(cases * s))
                        result['new infected ' + v].append(int(cases2 * s))
                    except:
                        result['new confirmed ' + v] = [int(cases * s)]
                        result['new infected ' + v] = [int(cases2 * s)]
                try:
                    result['new confirmed'].append(int(cases))
                    result['new infected'].append(int(cases2))
                except:
                    result['new confirmed'] = [int(cases)]
                    result['new infected'] = [int(cases2)]
            result['population'] = [N for x in result['time']]

            self.save_as_pickle(result)
            print() #interrupt \r printing from time-counter
            return result