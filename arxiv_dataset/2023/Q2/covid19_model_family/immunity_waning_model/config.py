"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

import hashlib
import json
import os
import shutil
import datetime as dt
from loss_sampler import LossSampler
from fit_distribution_means import fit_distribution_mean, adjust_vacc_values, FitPlotter
from utils import vname_function


class Config:
    def __init__(self,filename:str,experimentTimestamp:str):
        """
        Configuration for the immunity loss model. Parses the json config file. File will be semantically hashed and copied to the experiment result folder.

        :param filename: path to json config file as string
        :param experimentTimestamp: timestamp when the experiment is started
        """
        with open(filename,'r') as f:
            self.file_content = json.load(f)
        # make hash from json
        dumped = json.dumps(self.file_content, sort_keys=True).encode("utf-8")
        self._hash = hashlib.md5(dumped).hexdigest()

        self._experimentTimestamp = experimentTimestamp
        self.seed = int(self.file_content['seed']) #integer seed for RNG
        self.t0 = dt.datetime.strptime(self.file_content['t0'],'%Y-%m-%d') #startdate of simulation
        self.tend = dt.datetime.strptime(self.file_content['tend'],'%Y-%m-%d') #enddate of simulation. Can be later than the last date in the case and vaccination data
        if 'scenario' in self.file_content.keys() and self.file_content['scenario']!='': #name of the scenario.
            self.scenario = self.file_content['scenario']
        else: #if not defined manually, set to default name
            self.scenario = 'immunizationLevel_'+self.tend.strftime('%Y%m%d')

        if 'federalstate' in self.file_content.keys(): #restrict to certain federalstate. Some plot routines might not work for specific federalstate
            self.federalstate = self.file_content['federalstate']
            self._folderstamp = self.scenario+'_'+self.federalstate+'_'+self._experimentTimestamp
        else:
            self.federalstate = None
            self._folderstamp = self.scenario +'_'+self._experimentTimestamp

        # setup resultfolder
        if not os.path.isdir(self.file_content['resultFolder']):
            os.mkdir(self.file_content['resultFolder'])
        self.resultfolder = os.path.join(self.file_content['resultFolder'],self._folderstamp)
        if not os.path.isdir(self.resultfolder):
            os.mkdir(self.resultfolder)

        self.detectionProbability = {dt.datetime.strptime(x,'%Y-%m-%d'):y for x,y in self.file_content['detectionProbability'].items()}
        #probability that a case is detected
        """
        Either waning rate and distribution are extracted from given data or given via mean value and base value
        Then: P(Immune t days after event) = (U<base)*P(X>t), whereas E(X)=mean and U~U(0,1)
        """
        self.vaccDelay = int(self.file_content[
                                 'vaccDelay'])  # delay (in days) after which a vaccination can lead to immunity (three element vector for first, second and third shot)
        self.vaccIntervals = [int(x) for x in self.file_content['vaccIntervals']]  # minimum time between vaccinations
        self.detDelay = [int(x) for x in
                         self.file_content['detDelay']]  # time (in days) between infection and confirmation of a case
        self.recoveryDelay = [[int(y) for y in self.file_content[
            'recoveryDelay']]]  # recovery time (in days) of detected cases. A random vector entry is drawn. Eac vector entry must be >=detDelay
        self.recoveryDelay.append([int(y) for y in self.file_content[
            'recoveryDelayUndet']])  # recovery time (in days) of undetected cases. A random vector entry is drawn.

        self.observables = self.file_content['observables']
        for key1,value1 in self.observables.items():
            for key2, value2 in value1.items():
                ls = LossSampler(value2['distribution'], 1)
                samplefun = lambda x:ls.sample_with_mean(x)
                if 'mean' not in value2.keys():
                    print('fitting '+key2+' against '+key1)
                    if 'VACC' in key2:
                        refs = [[int(x[0])-self.vaccDelay, int(x[1])-self.vaccDelay, float(x[2])] for x in value2['values']]
                    else:
                        refs = [[int(x[0]),int(x[1]),float(x[2])] for x in value2['values']]
                    ls = LossSampler(value2['distribution'],1)
                    base,mean = fit_distribution_mean(refs,samplefun)
                    print([base,mean])
                    value2['mean'] = mean
                    value2['base'] = base
                else:
                    value2['mean'] = float(value2['mean'])
                    value2['base'] = float(value2['base'])
                    if 'values' not in value2.keys():
                        value2['values']=[]

        self.plotPdfs = self.file_content['plotPdfs']
        self.plotFits = self.file_content['plotFits']

        if self.plotFits:
            print('plot distributions and fitted data')
            print('(sorry, need to to this before the simulation starts for technical reasons)')
            causes = list(self.observables.keys())
            causes.remove('HOSPITALISATION')
            causes.extend(['VACC'+str(i) for i in range(1,5)])
            targets = list(self.observables.keys())
            targets.remove('ALPHA') #not really interesing anymore...
            targets.remove('WILDTYPE') #not really interesing anymore...
            plotter = FitPlotter(len(causes),len(targets))
            for cause in causes:
                for target in targets:
                    if cause in self.observables[target].keys():
                        value2 = self.observables[target][cause]
                    else:
                        value2 = self.observables[target]['DEFAULT']
                    print(vname_function(cause) + ' against ' + vname_function(target))
                    ls = LossSampler(value2['distribution'], 1)
                    samplefun = lambda x: ls.sample_with_mean(x)
                    if 'VACC' in cause:
                        refs = [[int(x[0]) - self.vaccDelay, int(x[1]) - self.vaccDelay, float(x[2])] for x in
                                value2['values']]
                    else:
                        refs = [[int(x[0]), int(x[1]), float(x[2])] for x in value2['values']]
                    if "source" in value2.keys():
                        label = value2["source"]
                    else:
                        label = ""
                    plotter.plot_fit( value2['distribution'], refs, samplefun, value2['base'],value2['mean'], vname_function(cause) + ' against ' + vname_function(target), label)
            plotter.finish_plot_fit(self.resultfolder)

        for key1,value1 in self.observables.items():
            aPriorBases = list()
            samplefuns = list()
            means = list()
            for i in range(1,100):
                if 'VACC'+str(i) in value1.keys():
                    vacc = value1['VACC'+str(i)]
                    means.append(vacc['mean'])
                    aPriorBases.append(vacc['base'])
                    ls = LossSampler(vacc['distribution'], 1)
                    samplefun = lambda x: ls.sample_with_mean(x)
                    samplefuns.append(samplefun)
                else:
                    break
            aPosteriorBases = adjust_vacc_values(samplefuns,means,aPriorBases,self.vaccIntervals)
            for i in range(len(aPosteriorBases)):
                vacc = value1['VACC' + str(i+1)]
                vacc['base']=aPosteriorBases[i]

        self.scale = float(self.file_content['scale']) #the model is run with scale*population agents. Heavy impact on computation time. Typically ~100000 agents is sufficient. So scale 0.01 is ok for AUstria

        self.filenameEpidata = self.file_content['filenameEpidata'] #path to file with COVID-19 case data
        self.filenameVaccdata = self.file_content['filenameVaccdata'] #path to file with vaccination data
        self.filenameVariantdata = self.file_content['filenameVariantdata'] #path to file with variant information
        self.filenamePopulationdata = self.file_content['filenamePopulationdata'] #path to file with population information

        # copy config file into result folder
        name = os.path.split(filename)[-1]
        shutil.copy(filename,os.path.join(self.get_result_folder(),name))

    def get_result_folder(self):
        """
        :return: folder for experiment results
        """
        return self.resultfolder

    def get_csv_filename(self) -> str:
        """
        :target: target observable
        :return: filename for writing the simulation results into
        """
        return self.scenario+'.csv'

    def hash(self):
        """
        :return: semantic hash for config json file to check for cached results
        """
        return self._hash