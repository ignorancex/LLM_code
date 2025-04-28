import numpy as np
import datetime as dt

class Settings:
    def __init__(self):
        self.NU = np.linspace(50, 200, 151)
        self.PATH21TS = "./../../data/TS21/lfcal_training_set_8-2020.hdf5"
        # ! Remove the name of antenna from PATHFGTS. e.g. a_logspiral.h5 --> a
        self.PATHFGTS = "./../../data/TSFG/consFgTs_2019-10-01_00:00:00_06:00:00_30_gsm_reg-1_"
        self.PATHFGSIM = "./../../data/simFg/cons_simFg_2019-10-01_00:00:00_06:00:00_30_gsm_"
        self.LST = 1
        self.ANT = ["logspiral"]
        self.dNU = 1.
        self.INT_TIME = 24.
        self.MODES_FG = 80
        self.MODES_21 = 80
        self.QUANTITY = "DIC"
        self.VISUALS = True
        self.SAVE = False
        self.INDFG = 0
        self.IND21 = 1000
        self.FNAME = f"./{self.QUANTITY}_search.txt"
        self.startTime = dt.datetime(2019, 10, 1, 0, 0, 0) 
        self.endTime = dt.datetime(2019, 10, 1, 5, 30, 0)
    
    @property
    def timeList(self):
        obsTimes = []
        for dtime in self.datetime_range(self.startTime, self.endTime, dt.timedelta(minutes=360/self.LST)):
            obsTimes.append(dtime)
        return obsTimes
    
    @property
    def dT(self):
        return self.INT_TIME/self.LST
    
    @property
    def intBins(self):
        return int(12/self.LST)
    
    @staticmethod
    def datetime_range(start, end, delta):
        current = start
        while current <= end:
            yield current
            current += delta

    def printSettings(self):
        print('\n------------------ Settings for the pipeline ------------------\n')
        print('Frequencies: Between (%d - %d) MHz with step size of %d MHz.'
                %(self.NU[0], self.NU[-1], self.NU[1] - self.NU[0]))
        print('21 TS:', self.PATH21TS)
        print('FG TS:', self.PATHFGTS)
        print('Number of time bins:', self.LST)
        print('Start time of each time bin:', *[value.time() for value in self.timeList])
        print(f'Each value is this list is integrated for {self.intBins} bins')
        print('Antenna design:', self.ANT)
        print(f'Integration time for each time bin: {self.dT} h')
        print('Total number of foreground modes:', self.MODES_FG)
        print('Total number of 21cm modes:', self.MODES_21)
        print('Information Criterion:', self.QUANTITY)
        print('Filename to store the info criteria:', self.FNAME)
        print('Visualization:', self.VISUALS)
        print(f'Save figures: {self.SAVE} \n')


class Settings_P2:
    def __init__(self):
        self.NU = np.linspace(50, 200, 151)
        self.PATH21TS = "./../../data/TS21/lfcal_training_set_8-2020.hdf5"
        # ! Remove the name of antenna from PATHFGTS. e.g. a_logspiral.h5 --> a
        self.PATHFGTS = "./../../data/TSFG/consFgTs_2019-10-01_00:00:00_06:00:00_30_gsm_reg-1_"
        self.PATHFGSIM = "./../../data/simFg/cons_simFg_2019-10-01_00:00:00_06:00:00_30_gsm_"
        self.LST_2_FIT = 1
        self.ANT1 = ["dipole"]
        self.ANT2 = ["logspiral"]
        self.dNU = 1.
        self.INT_TIME = 24.
        self.MODES_FG = 80
        self.MODES_21 = 80
        self.QUANTITY = "DIC"
        self.VISUALS = True
        self.SAVE = False
        self.INDFG = 0
        self.IND21 = 1000
        self.FNAME = f"./{self.QUANTITY}_search.txt"
        self.startTime1 = dt.datetime(2019, 10, 1, 0, 0, 0) 
        self.endTime1 = dt.datetime(2019, 10, 1, 2, 30, 0)
        self.startTime2 = dt.datetime(2019, 10, 1, 3, 0, 0)
        self.endTime2 = dt.datetime(2019, 10, 1, 5, 30, 0)
        
    @property
    def ANT(self):
        return ["-".join(self.ANT1 + self.ANT2)]
        
    @property
    def LST(self):
        return self.LST_2_FIT*2
    
    @property
    def timeList1(self):
        obsTimes = []
        for dtime in self.datetime_range(self.startTime1, self.endTime1, dt.timedelta(minutes=360/self.LST)):
            obsTimes.append(dtime)
        return obsTimes

    @property
    def timeList2(self):
        obsTimes = []
        for dtime in self.datetime_range(self.startTime2, self.endTime2, dt.timedelta(minutes=360/self.LST)):
            obsTimes.append(dtime)
        return obsTimes
    
    @property
    def dT(self):
        return self.INT_TIME/self.LST
    
    @property
    def intBins(self):
        return int(12/self.LST)
    
    @staticmethod
    def datetime_range(start, end, delta):
        current = start
        while current <= end:
            yield current
            current += delta

    def printSettings(self):
        print('\n------------------ Settings for the pipeline ------------------\n')
        print('Frequencies: Between (%d - %d) MHz with step size of %d MHz.'
                %(self.NU[0], self.NU[-1], self.NU[1] - self.NU[0]))
        print('21 TS:', self.PATH21TS)
        print('FG TS:', self.PATHFGTS)
        print('Number of time bins for 1 antenna:', self.LST_2_FIT)
        print('Start time of each time bin ant-1:', *[val.time() for val in self.timeList1])
        print('Start time of each time bin ant-2:', *[val.time() for val in self.timeList2])
        print(f'Each value is this list is integrated for {self.intBins} bins of 30 min.')
        print('Antenna design:', self.ANT)
        print(f'Integration time for each time bin: {self.dT} h')
        print('Total number of foreground modes:', self.MODES_FG)
        print('Total number of 21cm modes:', self.MODES_21)
        print('Information Criterion:', self.QUANTITY)
        print('Filename to store the info criteria:', self.FNAME)
        print('Visualization:', self.VISUALS)
        print(f'Save figures: {self.SAVE} \n')