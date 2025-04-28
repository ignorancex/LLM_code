'''
This defines a model for a steel ladle...
'''
import numpy as np
import pandas as pd
import math
import pickle
from scipy import linalg
import vtk
import matplotlib.pyplot as plt
from massmod import mass_step, k_eff_C, k_eff_mgo, MgOsol
from properties import cp_steel,Rho_steel
from parameters import rho_steel, rho_slag, rho_wall, rho_bottom, Cp_steel, Cp_slag
from parameters import Cp_wall, Cp_bottom, lam_slag, lam_wall, lam_bottom, alpha_carbon
from parameters import alpha_mgo, rho_carbon, rho_mgo, rho_mgoc
from heattransfer import q_electric3, lidT, enthalpy, q_rad, q_radlid, h_steel_inner
from heattransfer import h_slag_inner, h_slag_metal, h_rad
from utau import u_tau, u_tauP
from utils import find_nearest_idx
from influxdbqueries import db_temperature, db_tap_temperature, db_time, db_timep
from influxdbqueries import db_power, db_gas, db_additiontime, db_additionweight
from influxdbqueries import db_vacuum_pressure, db_amount_steel, db_temperature_points
from influxdbqueries import db_temperature_timepoints, db_save
NI = 37  # Number of vertical elements
NJ = 7  # Number of radial elements
NK = 1
NM = 7  # Fow now we assume The same number of layer in the botttom
R  = 2.9/2.0  # m ladle inner radius
L  = 3.5  # m total height of refractory
bricksize = 171.0
piR2 = np.pi*R**2

'''
'''
class LadleModel():
    # Set initial state
    def __init__(self):
        self.dy = np.zeros((NI+1, NJ))
        self.db = np.zeros((NM))
#       bottom      # steel ins alum dol mgo   mgo   mgo
        self.db[:] = [30.0, 5.0, 65.0, 20.0, 60.0, 60.0, 60.0]
        self.db = self.db/1000.0
        self.dx = np.zeros(NI)
        self.dz = np.zeros(NK)
        self.dz[:] = 0.8  # m - width of shown slab
        self.yerode  = np.zeros(NI + 1)
        self.lossC   = np.zeros(NI)
        self.lossMgO = np.zeros(NI)
        self.new_erode = 0.0
        self.old_erode = 0.0
        self.diff_erode = 0.0
        self.maxwear = 0.75
        self.widx = 10  # Hight where wall temperature is plotted
        # define grid
        self.XU  = np.zeros(NI+1)
        self.XU2 = np.zeros(NI+1)
        self.YV  = np.zeros((NI+1, NJ+1))
        self.ZW  = np.zeros(NK+1)
        self.X   = np.zeros(NI)
        self.Y   = np.zeros((NI, NJ))
        self.Z   = np.zeros(NK)
        self.r   = np.zeros((NI+1, NJ+1))
        self.rb  = np.zeros((NM+1))
        self.rm  = np.zeros((NI+1, NJ+1))
        self.rmb = np.zeros((NM+1))
        self.hsi = np.zeros(NI+1)
        self.setwall()
        #
        self.Tside       = np.zeros((NI, NJ, NK))
        self.TsideT0     = np.zeros((NI, NJ, NK))
        self.Tbottom     = np.zeros((NM))
        self.TbottomT0   = np.zeros((NM))
        self.alpha_steel = np.zeros(NI)
        self.alpha_slag  = np.zeros(NI)
        self.alpha_steel1 = np.zeros(NI)
        self.alpha_slag1  = np.zeros(NI)
        self.Qrad_wall   = np.zeros(NI)
        self.Qrad_out    = np.zeros(NI)
        self.Qext_wall   = np.zeros(NI)
        #
        self.lam = np.zeros((NI, NJ+1))
        self.lamb = np.zeros((NM+1))
        self.xCbulk = 0.0
        self.xmgobulk = 0.0
        self.u_tau = u_tau(0.0)
        self.rad = True   # Switch for radiation from ladle to environment
        self.irad = True  # Switch for radiation internally in ladle, from metal to wall/top
        self.islidon = True
        self.isempty = False
        self.hext_c    = 15
        self.hbottom_c = 15
        self.EAFheatloss = 50
        self.EAFvalid = 1
        self.time = 0.0
        self.eps1 = 0.9  # Must be changed!
        self.eps2 = 0.9   # Must be changed!
        self.s_pore = 0.00048  # 0.00048  # C length of pore channel between MgO particles in MgOC
        self.diff_Mgo_in_slag = 3e-9  # From Amini et.al
        self.xeqC = 0.1
        self.xeqmgo = 0.1
        self.lamwalltune = np.ones(NJ)
        self.cp_steel_tune = 1.0
        self.rho_steel_tune = 1.0
        self.lam_steel_tune = 1.0
        self.savevtk = False
        # Arrays to store results
        self.TsteelV    = []
        self.TslagV     = []
        self.TInnerW    = []
        self.TW         = []
        self.TOuterW    = []
        self.masssteelV = []
        self.massslagV  = []
        self.TimeV      = []
        self.TmeasV     = []
        self.PowerV     = []
        self.ElectV     = []
        self.qslagV     = []
        self.qsteelV    = []
        self.qwallV     = []
        self.qbottomV   = []
        self.qtot       = []
        self.TsideV     = []
        self.TbottomV   = []

    def set_parameter(self, key, value):
        if key == 'lamstsh':
            self.lamwalltune[6] = value
        elif key == 'lamins':
            self.lamwalltune[5] = value
        elif key == 'lamal':
            self.lamwalltune[4] = value
        elif key == 'lamdol':
            self.lamwalltune[3] = value
        elif key == 'lammgoc':
            self.lamwalltune[0:2] = value
        elif key == 'eps1':
            self.eps1 = value
        elif key == 'eps2':
            self.eps2 = value
        elif key == 'hext':
            self.setHext(value)
        elif key == 'hextb':
            self.setHbottom(value)
        elif key == 'rhosteel':
            self.rho_steel_tune = value
        elif key == 'cpsteel':
            self.cp_steel_tune = value
        elif key == 'lamsteel':
            self.lam_steel_tune = value
        elif key == 's_pore':
            self.s_pore = value
        elif key == 'diffMgo':
            self.diff_Mgo_in_slag = value
        elif key == 'xeqC':
            self.xeqC = value
        elif key == 'xeqMgO':
            self.xeqmgo = value
        else:
            print("did not find key: ", key)


    def erode_wall(self):
        for i in range(0, NI):
            mass = np.pi*self.dx[i]*rho_mgoc*((R+0.18)**2-R**2)
            self.yerode[i] += self.lossC[i]/alpha_carbon/mass
            self.yerode[i] += self.lossMgO[i]/alpha_mgo/mass
            self.lossC[i] = 0.0
            self.lossMgO[i] = 0.0
            if self.yerode[i] > self.maxwear:
                print("Refractory in index", i, "is too thin")
                print("yerode[i],maxwear:",self.yerode[i] , self.maxwear)
                raise Exception("Sorry, no refractory left")
        self.setwall()

    def setwall(self):
        shelltunefac = 2.0
        for i in range(0, NI+1):
            self.dy[i, :] = [bricksize/3.0*(1.-self.yerode[i]),
                             bricksize/3.0*(1.-self.yerode[i]),
                             bricksize/3.0*(1.-self.yerode[i]),
                             20.0, 65.0, 5.0, shelltunefac*30.0]
        self.dy = self.dy/1000.0
        self.dx[:] = L*1.0/NI
        # outer radius of steel shell
        self.R_outer = R + (bricksize+20.0+65.0+5.0+30.0)/1000.0

        self.XU[0] = 0.0
        for i in range(1, NI+1):
            self.XU[i] = self.XU[i-1]+self.dx[i-1]
        self.XU2[0] = self.dx[0]/2.0
        for i in range(1, NI+1):
            self.XU2[i] = self.XU2[i-1]+self.dx[i-1]
        for i in range(0, NI+1):
            self.YV[i, 0] = 0.0 + self.yerode[i]*bricksize/1000.
        for i in range(0, NI+1):
            for j in range(1, NJ+1):
                self.YV[i, j] = self.YV[i, j-1]+self.dy[i, j-1]
        self.ZW[0] = 0.0
        for i in range(1, NK+1):
            self.ZW[i] = self.ZW[i-1]+self.dz[i-1]
        # define X, Y, Z positions for scalar cells:
        for i in range(0, NI):
            self.X[i] = (self.XU[i]+self.XU[i+1])/2.0
        for i in range(0, NI):
            for j in range(0, NJ):
                self.Y[i, j] = (self.YV[i, j]+self.YV[i, j+1])/2.0
        for k in range(0, NK):
            self.Z[k] = (self.ZW[k]+self.ZW[k+1])/2.0
        for i in range(0, NI):
            totwall = 0.0
            for j in range(0, NJ):
                totwall = totwall + self.dy[i, j]
            self.rm[i, 0] = self.R_outer-totwall
        for i in range(0, NI):
            for j in range(0, NJ):
                self.rm[i, j+1] = self.rm[i, j] + self.dy[i, j]
        for i in range(0, NI):
            for j in range(0, NJ):
                self.r[i, j] = self.rm[i, j]+self.dy[i, j]/2.0
        self.rmb[0] = 0.0
        for i in range(0, NM):
            self.rmb[i+1] = self.rmb[i] + self.db[i]
        for i in range(0, NM):
            self.rb[i] = self.rmb[i]+self.db[i]/2.0

    def setInitialCondition(self, T0wallin, T0wallout):
        for i in range(0, NI):
            for j in range(0, NJ):
                Tj = T0wallin - (T0wallin-T0wallout)*(self.Y[i, j] - self.Y[i, 0])/self.Y[i, NJ-1]
                for k in range(0, NK):
                    self.TsideT0[i, j, k] = Tj
        for j in range(0, NM):
            self.TbottomT0[NM-1-j] = self.TsideT0[0, j, 0]
        self.Tside[:, :, :] = self.TsideT0
        self.Tbottom[:] = self.TbottomT0

    def setHext(self, hext):
        self.hext_c = hext

    def setHbottom(self, hbot):
        self.hbottom_c = hbot

    def setDataFrame(self, heat_number, ladle_number=0, use_number=0, source=""):

        self.Tslagin = 25.0 + 273.15
        self.EAFvalid = 1
        self.heat_number = heat_number
        if source == 'db':
            self.temperatura = db_temperature(heat_number)
            self.Tsteelin = db_tap_temperature(heat_number, ladle_number, use_number)
            self.time_data = db_time(heat_number)
            self.timep_data = db_timep(heat_number)
            self.use_number = use_number
            self.ladle_number = ladle_number
            self.power_data = db_power(heat_number)
            self.caudal_gas = db_gas(heat_number)
            self.addtime = db_additiontime(heat_number, self.ladle_number, self.use_number)
            self.addweight = db_additionweight(heat_number, self.ladle_number, self.use_number)
            self.vacuum_pressure = db_vacuum_pressure(heat_number)
        else:
            df1 = pd.read_csv('fielddata/2019_Process Parameters.csv')
       #    df1 = pd.read_csv('fielddata/2018_Process Parameters.csv')
            df2 = pd.read_csv('fielddata/Alloys addition time and weights v2.csv')
            df = pd.read_csv('fielddata/' + str(heat_number) + '.gz')
            df3 = pd.read_csv('fielddata/tt.csv')

            try:
                self.Tsteelin = df3.loc[df3['heat number'] == \
                        heat_number]['tapping T (ºC)'].values[0] + \
                        273.15 - self.EAFheatloss
                if math.isnan(self.Tsteelin ):
                    raise Exception("Temperature is nan")
            except:
#                 print("Warning: No EAF steel temperature available")
                self.EAFvalid = 0
                self.Tsteelin = df.TEMPERATURA[0] + 273.15
            self.tmax = np.max(df.time)  # in seconds

            self.df = df
            self.df2 = df2.loc[df2['Heat Number'] == heat_number]
            self.df1 = df1
            idx = self.df1.index[self.df1.ncol == heat_number][0]
            self.use_number = df1['nusos'][idx]
            self.power_data = df.CONSUMO_ELECTRICO
            self.time_data  = df.time.to_numpy()
            self.addtime = pd.to_datetime(self.df2['Addition Time'],dayfirst=True).to_numpy()
#            self.P = df.PRESION_VACIO
            self.temperatura = self.df.TEMPERATURA
            self.timep_data = self.df.FECHAHORA
            self.caudal_gas = self.df.CAUDAL_GAS
            self.vacuum_pressure = self.df.PRESION_VACIO
            self.addweight = self.df2['Mass added per heat (ton)'].to_numpy()*1000.0
            self.ladle_number = self.df1['ncuch'][idx]
        self.reset_state(source)

    def reset_state(self, source):
        self.time = 0.0
        self.tmax = np.max(self.time_data)  # in seconds
        self.Tsteel = self.Tsteelin
        self.Tslag  = self.Tsteelin
        self.walltemptro = []
        self.walltemptri = []
        # Arrays to store results
        self.TsteelV    = []
        self.TslagV     = []
        self.TInnerW    = []
        self.TW         = []
        self.TOuterW    = []
        self.masssteelV = []
        self.massslagV  = []
        self.TimeV      = []
        self.TmeasV     = []
        self.PowerV     = []
        self.ElectV     = []
        self.qslagV     = []
        self.qsteelV    = []
        self.qwallV     = []
        self.qbottomV   = []
        self.qtot       = []
        self.TsideV     = []
        self.TbottomV   = []
        # Initial condition
        if source == "db":
            self.mass_steel = db_amount_steel(self.heat_number, self.ladle_number, self.use_number)
            self.TemperaturePoints = db_temperature_points(self.heat_number, self.ladle_number, \
                    self.use_number)
            self.TimePoints = db_temperature_timepoints(self.heat_number, self.ladle_number, \
                    self.use_number, self.timep_data[0])
            tmp1 = []
            tmp2 = []
            for i in range(0,len(self.TemperaturePoints)-1):
                if self.TemperaturePoints[i] > 1.0 and self.TimePoints[i] > 0.0:
                    tmp1.append(self.TemperaturePoints[i])
                    tmp2.append(self.TimePoints[i])
            self.TemperaturePoints = tmp1
            self.TimePoints = tmp2
        else:
            idx = self.df1.index[self.df1.ncol == self.heat_number][0]
            self.mass_steel = self.df1.acero_liquido[idx]*1000.0  # Ton to kg
            self.TemperaturePoints = self.df.TEMPERATURA.drop_duplicates().values.tolist()
            self.TimePoints = self.df.TEMPERATURA.drop_duplicates().index.tolist()
        #  We need the final value as well
        if self.mass_steel < 1.0:
            print("Warning, no steel in case, a default of 125 ton is set")
            self.mass_steel = 125000.0
        if len(self.TimePoints) > 0:
            self.TimePoints.append(self.tmax)
            self.TemperaturePoints.append(self.TemperaturePoints[-1])
        self.mass_slag = 0.0
        self.lossC[:] = 0.0
        self.lossMgO[:] = 0.0
        self.xCbulk = 0.0
        self.xmgobulk = 0.0

    def __calc_lamb(self, Tsteel, Text):
        # Compute self.lam coeffients
        hbottom = self.hbottom_c
        utau = self.u_tau(0.0)
        if self.rad:
            hbottom += h_rad(self.Tbottom[0], Text)
        for j in range(-1, NM):
            if j == -1:  # To external
                self.lamb[j+1] = 2.0*(lam_bottom[j+1]/self.db[j+1])*hbottom/(2.0*\
                        lam_bottom[j+1]/self.db[j+1] + hbottom)
            elif j == NM-1:  # To metal
                hinner = self.alpha_steel[0]*h_steel_inner(Tsteel, self.TbottomT0[0],\
                        R, self.db[0], utau) + self.alpha_slag[0]*\
                        h_slag_inner(lam_slag, self.db[0])
                self.lamb[j+1] = 2.0*lam_bottom[j]/self.db[j]*hinner/(2.0*\
                        lam_bottom[j]/self.db[j] + hinner)
            else:  # Inside wall
                self.lamb[j+1] = 2.0*lam_bottom[j+1]/self.db[j+1]*2.0*\
                        lam_bottom[j]/self.db[j]/(2.0*lam_bottom[j+1]/self.db[j+1] +\
                        2.0*lam_bottom[j]/self.db[j])

    # Compute lam coeffients
    def __calc_lam(self, Tsteel, Text):
        for i in range(0, NI):
            hext = self.hext_c
            if self.rad:
                hext += h_rad(self.TsideT0[i, NJ-1, 0], Text)
            for j in range(-1, NJ):
                if j == -1:  # To metal
                    hwp = 2.0*lam_wall[j+1]/self.dy[i, j+1]*self.lamwalltune[j+1]
                    hinner = self.alpha_steel[i]*self.hsi[i] + self.alpha_slag[i]*\
                            h_slag_inner(lam_slag, self.dy.item(i, 0))
                    self.lam[i, j+1] = hwp*hinner/(hwp + hinner)
                elif j == NJ-1:  # To external
                    hw = 2.0*lam_wall[j]/self.dy[i, j]*self.lamwalltune[j]
                    self.lam[i, j+1] = hw*hext/(hw + hext)
                else:  # Inside wall
                    hw  = 2.0*lam_wall[j]/self.dy[i, j]*self.lamwalltune[j]
                    hwp = 2.0*lam_wall[j+1]/self.dy[i, j+1]*self.lamwalltune[j]
                    self.lam[i, j+1] = hw*hwp/(hw + hwp)

    def __prepare(self, dt, efficiency, rho_steel, Text):
        mfr_insl  = self.__feed_slag(dt)
        self.mass_slag = mass_step(self.mass_slag, mfr_insl, 0.0, dt)
        Hsm = self.mass_steel/rho_steel/(piR2)
        self.delta_slag = self.mass_slag/rho_slag/(piR2)
        Hgsm = Hsm + self.delta_slag
        if Hgsm > L:
            print(f'Warning: There is too much metal in the ladle {self.ladle_number}')
            print(f'Height of metal = {Hgsm}, but maximum ladle height is {L}')
            print(f'Amount of steel: {self.mass_steel/1000.0} tons, Amount of slag {self.mass_slag/1000.0} tons')
        self.Qslag  = efficiency*1.0*q_electric3(self.time, self.time_data, self.power_data, dt)
        self.Qsteel = efficiency*0.0*q_electric3(self.time, self.time_data, self.power_data, dt)
        if np.floor(self.time) < len(self.caudal_gas):
            self.u_tau = u_tau(self.caudal_gas[int(np.floor(self.time))])
        # Calculate alpha_steel and self.alpha_slag, XU is the position at top of cell
        self.Qrad_wall[:] = 0.0
        self.Qrad_out[:]  = 0.0
        Tlid = lidT(self.Tslag, Hgsm, self.X[self.X > Hgsm], self.Tside[(self.X >\
                Hgsm), 0, 0], L, R)
        for i in range(0, NI):
            if np.floor(self.time) < len(self.caudal_gas):
                gas_rate = self.caudal_gas[int(np.floor(self.time))]
                P        = self.vacuum_pressure[int(np.floor(self.time))]/100.0 # kPa => bar
            else:
                gas_rate = 0.0
                P        = 1.0
            utau = u_tauP(self.XU2[i]/L, gas_rate, P, rho_steel)
            self.hsi[i] = h_steel_inner(self.Tsteel, self.TsideT0[i, 0, 0], R,\
                    self.dy.item(i, 0), utau)
            if self.XU[i] >= Hgsm:
                if self.widx == -1:
                    self.widx = i
                # No metal in cell
                # Calculate radiation flux from slag/metal to wall
                if self.irad:
                    self.Qrad_wall[i] = q_rad(self.Tslag, self.Tside[i, 0, 0], R, \
                            self.dx[i], Hgsm, self.XU[i+1], lam_wall[0]*self.lamwalltune[0],\
                            self.dy[i, 0], self.eps1, self.eps2)
                    if self.islidon:
                        self.Qrad_wall[i] += q_radlid(Tlid, self.Tside[i, 0, 0], L, self.dx[i],\
                                self.XU[i+1], L, self.eps1, self.eps2)  # From lid
                    else:
                        self.Qrad_out[i]  = h_rad(self.Tside[i, 0, 0], Text)*2.0*np.pi*\
                                self.R_outer*L
                self.alpha_steel[i] = 0.0
                self.alpha_slag[i]  = 0.0
            elif self.XU[i+1] <= Hsm:
                # Only steel in cell
                self.alpha_steel[i] = 1.0
                self.alpha_slag[i]  = 0.0
            elif(self.XU[i] > Hsm and self.XU[i + 1] < Hgsm):  # Only slag in cell
                self.alpha_steel[i] = 0.0
                self.alpha_slag[i]  = 1.0
            else:
                # Here is either slag, slag+steel or steel+air, or all three
                V_cell = piR2*self.dx[i]
                V_steel = max(self.mass_steel/rho_steel - piR2*self.XU[i], 0.0)
                V_slag  = (min(Hgsm, self.XU[i+1]) - max(self.XU[i], Hsm))*piR2
                self.alpha_steel[i] = V_steel/V_cell
                self.alpha_slag[i]  = V_slag/V_cell
                if (self.alpha_steel[i]+self.alpha_slag[i] < 0.99 and self.irad):
                    self.Qrad_wall[i] = q_rad(self.Tslag, self.Tside[i, 0, 0], R, \
                            self.dx[i], Hgsm, self.XU[i+1], lam_wall[0]*self.lamwalltune[0],\
                            self.dy[i, 0], self.eps1, self.eps2)
                    if self.islidon:
                        self.Qrad_wall[i] += q_radlid(Tlid, self.Tside[i, 0, 0], R,\
                                self.dx[i], self.XU[i+1], L, self.eps1, self.eps2)  # From lid
                    else:
                        self.Qrad_out[i]  = h_rad(self.Tside[i, 0, 0], Text)*2.0*\
                                np.pi*self.R_outer*L
            if not self.isempty:
                self.xeqmgo = MgOsol(self.Tslag) # use temperature dependent MgO solubility
                self.masscarbon += self.__carbonloss(dt, gas_rate, P, i)
                self.massmgo += self.__mgoloss(dt, gas_rate, P, i)

            for i in range(1,NI-1):
                self.alpha_slag1[i] = self.alpha_slag[i-1]*0.25 + self.alpha_slag[i]*0.5 + self.alpha_slag[i+1]*0.25
                self.alpha_steel1[i] = self.alpha_steel[i-1]*0.25 + self.alpha_steel[i]*0.5 + self.alpha_steel[i+1]*0.25
            self.alpha_steel1[0] = self.alpha_steel[0]*0.75 + self.alpha_steel[1]*0.25
            self.alpha_slag1[0] = self.alpha_slag[0]*0.75 + self.alpha_slag[1]*0.25
            self.alpha_slag1[NI-1] = self.alpha_slag[NI-1]*0.75 + self.alpha_slag[NI-2]*0.25
            self.alpha_steel1[NI-1] = self.alpha_steel[NI-1]*0.75 + self.alpha_steel[NI-2]*0.25

    def __solve_metal(self, dt, Tsteel0, Tslag0, rho_steel, Cp_steel, Text):
        mfr_insl  = self.__feed_slag(dt)
        if self.mass_slag > 0.0:
            self.xmgobulk = (self.masscarbon/alpha_carbon + self.massmgo/alpha_mgo)*\
                    alpha_mgo/self.mass_slag
            self.xCbulk = (self.massmgo/alpha_mgo + self.masscarbon/alpha_carbon)*\
                    alpha_carbon/self.mass_steel
        if self.isempty:
            # The model will fail with no metal, therefore set to bottom temperature
            # if the ladle is empty
            self.Tsteel = self.Tbottom[6]
            self.Tslag  = self.Tbottom[6]
        else:
            self.__calc_lam(Tsteel0, Text)
            self.__calc_lamb(Tsteel0, Text)
            ##
            #  Calculate steel and slag temperature by enthalpy
            ##
            delt = 0.35
            hsteeln  = enthalpy(Cp_steel, Cp_steel, Tsteel0)
            hsteeln1 = enthalpy(Cp_steel, Cp_steel, Tsteel0 + delt)
            hslagn   = enthalpy(Cp_slag,  Cp_slag,  Tslag0)
            hslagn1  = enthalpy(Cp_slag,  Cp_slag,  Tslag0  + delt)
            Ast = Tsteel0 - hsteeln*delt/(hsteeln1 - hsteeln)
            Asl = Tslag0 - hslagn*delt/(hslagn1 - hslagn)

            Bst = delt/(hsteeln1 - hsteeln)
            Bsl = delt/(hslagn1 - hslagn)

            Hst = piR2*h_slag_metal()*Bst
            Hsl = piR2*h_slag_metal()*Bsl

            Est = self.mass_steel/dt + Bst*piR2*(h_slag_metal())
            Esl = self.mass_slag/dt + mfr_insl + Bsl*piR2*(h_slag_metal())
            Fst = self.mass_steel/dt*hsteeln + self.Qsteel + piR2*h_slag_metal()*(Asl - Ast)
            Fsl = self.mass_slag/dt*hslagn   + self.Qslag  + mfr_insl*enthalpy(Cp_slag,\
                    Cp_slag, self.Tslagin)    + piR2*h_slag_metal()*(Ast - Asl) +\
                    self.Qrad_wall.sum()#+ lid
            Gsl = 0.0  # wall and bottom
            Gst = 0.0  # wall and bottom
            ##
            #  End calculate steel and slag temperature by enthalpy
            ##
            # Calculate metal temperatures
            if not self.islidon:
                Fsl += h_rad(self.Tslag, Text)*np.pi*R**2
            hinnerb = self.lamb[NM]
            Qwall = 0.0
            Est += Bst*piR2*hinnerb
            Gst += piR2*hinnerb*(self.Tbottom[NM-1] - Ast)
            for i in range(0, NI):
                tmp = 2.0*np.pi*self.rm[i, 0]*self.dx[i]
                hwp = 2.0*lam_wall[0]*self.lamwalltune[0]/self.dy[i, 0]
                hinner = h_slag_inner(lam_slag, self.dy[i, 0])
                tmplam = self.alpha_slag[i]*hwp*hinner/(hwp + hinner)
                # New terms slag
                Esl += tmp*tmplam*Bsl
                Gsl += tmp*tmplam*(self.TsideT0[i, 0, 0] - Asl)
                # End new terms slag
                if np.floor(self.time) < len(self.caudal_gas):
                    gas_rate = self.caudal_gas[int(np.floor(self.time))]
                    P        = self.vacuum_pressure[int(np.floor(self.time))]/100.0 # kPa => bar
                else:
                    gas_rate = 0.0
                    P        = 1.0
                utau = u_tauP(self.XU2[i]/L, gas_rate, P, rho_steel)
                hinner = self.hsi[i]
                tmplam = self.alpha_steel[i]*hwp*hinner/(hwp + hinner)
                # New terms steel
                Est += tmp*tmplam*Bst
                Gst += tmp*tmplam*(self.TsideT0[i, 0, 0] - Ast)
                # End new terms steel
            hhsl = (Est*(Fsl+Gsl) + Hst*(Fst + Gst))/(Est*Esl - Hst*Hsl)
            hhst = (Hsl*hhsl + Fst + Gst)/Est
            self.Tsteel = Tsteel0 + (hhst - hsteeln)/(hsteeln1 - hsteeln)*delt
            self.Tslag  = Tslag0 + (hhsl - hslagn)/(hslagn1 - hslagn)*delt

    def __solve_wall(self, dt, Text):
        #  Define arrays for equation system
        bw  = np.zeros((NI, NJ, NK))
        AE  = np.zeros((NI, NJ, NK))
        AW  = np.zeros((NI, NJ, NK))
        AP  = np.zeros((NI, NJ, NK))
        MA  = np.zeros((NJ, NJ))
        k = 0
        for i in range(0, NI):
            for j in range(0, NJ):
                # We have to update conductivity based on
                tmp = 2.0*np.pi*self.dx[i]
                Vol = tmp*self.r[i, j]*self.dy[i, j]
                AP[i, j, k] = Vol*rho_wall[j]*Cp_wall[j]/dt
                bw[i, j, k] = AP[i, j, k]*self.TsideT0[i, j, k]

                if j == NJ-1:  # To external
                    bw[i, j, k] += tmp*self.rm[i, j+1]*self.lam[i,j+1]*Text
                    AE[i, j, k] = 0.0
                    AW[i, j, k] = tmp*self.lam[i, j]*self.rm[i, j]
                    AP[i, j, k] += AE[i, j, k] + AW[i, j, k] + tmp*self.rm[i, j+1]*self.lam[i,j+1]
                elif j == 0:  # To metal (Disregard radiation losses)
                    hwp = 2.0*lam_wall[j]*self.lamwalltune[j]/self.dy[i, j]
                    hinner = self.hsi[i]
                    tmplam = self.alpha_steel[i]*hwp*hinner/(hwp + hinner)
                    bw[i, j, k] += tmp*self.rm[i, j]*tmplam*(self.Tsteel-self.TsideT0[i, j, k])
                    hinners = h_slag_inner(lam_slag, self.dy.item(i, 0))
                    tmplams= self.alpha_slag[i]*hwp*hinners/(hwp + hinners)
                    bw[i, j, k] += tmp*self.rm[i, j]*tmplams*(self.Tslag-self.TsideT0[i, j, k])
                    bw[i, j, k] -= self.Qrad_wall[i]
                    bw[i, j, k] -= self.Qrad_out[i]
                    AE[i, j, k] = tmp*self.lam[i, j+1]*self.rm[i, j+1]
                    AW[i, j, k] = 0.0
                    AP[i, j, k] += AE[i, j, k] + AW[i, j, k]
                else:  # Inside wall
                    AE[i, j, k] = tmp*self.lam[i, j+1]*self.rm[i, j+1]
                    AW[i, j, k] = tmp*self.lam[i, j]*self.rm[i, j]
                    AP[i, j, k] += AE[i, j, k] + AW[i, j, k]

            # Building matrix, for cell i
            for j in range(0, NJ):
                MA[j, j] = AP[i, j, k]
                if j == 0:
                    MA[j, j+1] = -AE[i, j, k]
                elif j == NJ - 1:
                    MA[j, j-1] = -AW[i, j, k]
                else:
                    MA[j, j+1] = -AE[i, j, k]
                    MA[j, j-1] = -AW[i, j, k]
            self.Tside[i, :, k] = linalg.solve(MA, bw[i, :, k])

    def __solve_bottom(self, dt, Text):
        #  Define arrays for equation system
        AN  = np.zeros((NM))
        AS  = np.zeros((NM))
        APb = np.zeros((NM))
        MAb = np.zeros((NM, NM))
        bb  = np.zeros((NM))
        for i in range(0, NM):
            Vslice = piR2*self.db[i]  # Volume of cylinder
            bb[i]  = Vslice*rho_bottom[i]*Cp_bottom[i]/dt*self.TbottomT0[i]
            APb[i] = 0.0
            if i == 0:  # To external
                bb[i] += piR2*self.lamb[0]*Text
                AN[i] = piR2*self.lamb[i+1]
                AS[i] = 0.0
                APb[i] += piR2*self.lamb[0]
            elif i == NM-1:  # To metal
                bb[i] += np.pi*R**2*self.lamb[i+1]*(self.Tsteel-self.TbottomT0[i])
                AN[i] = 0.0
                AS[i] = np.pi*R**2*self.lamb[i]
            else:
                bb[i] += 0.0
                AN[i] = np.pi*R**2*self.lamb[i+1]
                AS[i] = np.pi*R**2*self.lamb[i]
                APb[i] += 0.0
            APb[i] += Vslice*rho_bottom[i]*Cp_bottom[i]/dt + AN[i] + AS[i]
        for j in range(0, NM):
            MAb[j, j] = APb[j]
            if j == 0:
                MAb[j, j+1] = -AN[j]
            elif j == NJ-1:
                MAb[j, j-1] = -AS[j]
            else:
                MAb[j, j+1] = -AN[j]
                MAb[j, j-1] = -AS[j]
        self.Tbottom[:] = linalg.solve(MAb, bb)
        #Set T0
        self.TsideT0[:, :, :] = self.Tside
        self.TbottomT0[:] = self.Tbottom

    def run(self, dt, Text, efficiency):
        Nsteps = int(self.tmax//dt)
        Tslag  = self.Tslagin
        self.Tslag  = self.Tsteelin
        self.Tsteel = self.Tsteelin

        Cp_steel = cp_steel(self.Tsteel)*self.cp_steel_tune
        rho_steel = Rho_steel(self.Tsteel)*self.rho_steel_tune
        self.maxwalltemp = 0.0
        #  Prepare result vectors (not so nice, consider moving)
        self.TPindex = 0
        self.Qsteel = 0.0
        self.Qslag  = 0.0

        #  Save values for output
        self.__save_data()

        self.masscarbon = 0.0
        self.massmgo    = 0.0
        for NT in range(0, Nsteps):

            self.time = self.time + dt #advance time
            self.__prepare(dt, efficiency, rho_steel, Text)
            #  Mass balance
            self.__solve_metal(dt, self.Tsteel, self.Tslag, rho_steel, Cp_steel, Text )
            # Wall
            self.__solve_wall(dt, Text)
            # Bottom
            self.__solve_bottom(dt, Text)
            # Save values for output
            if (not self.isempty and self.savevtk):
                self.save_vtk(NT)
            self.__save_data()
            # End time loop
        self.old_erode = self.new_erode
        self.new_erode = np.max(self.yerode)
        self.diff_erode = self.new_erode - self.old_erode
#         print("CASE FINISHED !")

    def save_vtk(self, NT):
        # https://visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
        # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
        self.sg = vtk.vtkStructuredGrid()
        # Number of corner points in each direction, hard coded from dispmod.py
        self.sg.SetDimensions([NI+1, NJ+1, NK+1])
        stime = vtk.vtkDoubleArray()
        stime.SetName('TIME')
        stime.SetNumberOfTuples(1)
        WallTemperature = vtk.vtkFloatArray()
        WallTemperature.SetName('WallTemperature')
        Steelfrac = vtk.vtkFloatArray()
        Steelfrac.SetName('Steelfrac')
        Slagfrac = vtk.vtkFloatArray()
        Slagfrac.SetName('Slagfrac')
        Metfrac = vtk.vtkFloatArray()
        Metfrac.SetName('Metfrac')
        MetalTemperature = vtk.vtkFloatArray()
        MetalTemperature.SetName('MetalTemperature')
        stime.SetTuple1(0, self.time)
        self.sg.GetFieldData().AddArray(stime)

        points = vtk.vtkPoints()
        for k in range(0, NK+1):
            for j in range(0, NJ+1):
                for i in range(0, NI+1):
                    points.InsertNextPoint(float(self.XU[i]),
                                           float(self.YV[i, j]),
                                           float(self.ZW[k]))

        for k in range(0, NK):
            for j in range(0, NJ):
                for i in range(0, NI):
                    Steelfrac.InsertNextTuple1(float(self.alpha_steel[i]))
                    Slagfrac.InsertNextTuple1(float(self.alpha_slag[i]))
                    Metfrac.InsertNextTuple1(float(self.alpha_steel[i]+self.alpha_slag[i]))
                    WallTemperature.InsertNextTuple1(float(self.TsideT0[i, j, k]))
                    MetalTemperature.InsertNextTuple1(float(self.alpha_steel[i]*self.Tsteel +\
                            self.alpha_slag[i]*self.Tslag))

        self.sg.SetPoints(points)
        self.sg.GetCellData().AddArray(WallTemperature)
        self.sg.GetCellData().AddArray(Steelfrac)
        self.sg.GetCellData().AddArray(Slagfrac)
        self.sg.GetCellData().AddArray(Metfrac)
        self.sg.GetCellData().AddArray(MetalTemperature)

        w = vtk.vtkXMLStructuredGridWriter()  # to get the xml version, use XMLWriter
        w.SetInputData(self.sg)
        w.SetFileName('data/' + str(self.heat_number)+'_' + '%05d' % NT + '.vts')
        print("*********** saving vtk data*************")
        w.Write()

    def __save_data(self):
        self.maxwalltemp = max(self.maxwalltemp, max(self.Tside[:, 6, 0]))
        self.walltemptro.append(self.Tside[5, 6, 0]-273.15)
        self.walltemptri.append(self.Tside[5, 0, 0]-273.15)
        self.TInnerW.append(self.Tside[5, 0, 0]-273.15)
        self.TW.append(self.Tside[self.widx, 0, 0]-273.15)
        self.TOuterW.append(self.Tside[5, 6, 0]-273.15)
        self.TsteelV.append(self.Tsteel-273.15)
        self.TslagV.append(self.Tslag-273.15)
        self.masssteelV.append(self.mass_steel)
        self.massslagV.append(self.mass_slag)
        self.qsteelV.append(self.mass_steel*self.Tsteel*Cp_steel)
        self.qslagV.append(self.mass_slag*self.Tslag*Cp_slag)
        qwallV = 0.0
        for k in range(0, NK):
            for j in range(0, NJ):
                for i in range(0, NI):
                    qwallV += (2.0*np.pi*self.dx[i]*self.rm[i, j]*self.dy[i, j] +\
                            self.dy[i, j]**2*np.pi*self.dx[i])*rho_wall[j]*\
                            Cp_wall[j]*self.Tside[i, j, k]
        self.qwallV.append(qwallV)
        if len(self.TsideV)==0:
            self.TsideV = [self.Tside]
            self.TbottomV =[self.Tbottom[:]]
        else:
            self.TsideV = np.vstack([self.TsideV, [self.Tside]])
            self.TbottomV = np.vstack([self.TbottomV,self.Tbottom[:]])

        qbottomV = 0.0
        for i in range(0, NM):
            Vslice = np.pi*R**2*self.db[i]  # Volume of cylinder
            qbottomV += Vslice*rho_bottom[i]*Cp_bottom[i]*(self.Tbottom[i])
        self.qbottomV.append(qbottomV)
        self.TimeV.append(self.time)
        idx = find_nearest_idx(self.time_data, self.time)
        TPindex = self.TPindex
        if idx < len(self.temperatura):
            idx = find_nearest_idx(self.time_data, self.time)
            if len(self.TimePoints) > 0:
                if idx <= self.TimePoints[TPindex+1]:
                    grad = (self.TemperaturePoints[TPindex+1]-self.TemperaturePoints[TPindex])/\
                           (self.TimePoints[TPindex+1]-self.TimePoints[TPindex])
                    self.TmeasV.append(self.TemperaturePoints[TPindex] + grad*\
                                      (self.time-self.TimePoints[TPindex]) )
                else:
                    self.TmeasV.append(self.temperatura[idx])
                if self.time > self.TimePoints[TPindex+1] and TPindex+2 < len(self.TimePoints):
                    self.TPindex += 1
            self.PowerV.append(self.Qslag + self.Qsteel)
            self.ElectV.append(self.power_data[int(idx)] - self.power_data[0])
            qtot = qbottomV+qwallV+self.qslagV[len(self.qslagV)-1]+self.mass_steel*\
                    self.Tsteel*Cp_steel +self.mass_slag*self.Tslag*Cp_slag -\
                    (self.power_data[int(idx)] - self.power_data[0])*3.6e6
            self.qtot.append(qtot)
        else:
            self.TmeasV.append(0.0)
            self.PowerV.append(0.0)
            self.ElectV.append(0.0)
            self.qtot.append(0.0)

    def save_db(self):
        if db_check_model_data(self.heat_number, self.ladle_number, self.use_number):
            for t in self.TimeV:
                db_save("Time", self.heat_number, self.ladle_number, self.use_number, t)
            for t in self.TsteelV:
                db_save("T_steel", self.heat_number, self.ladle_number, self.use_number, t)
            for t in self.TslagV:
                db_save("T_slag", self.heat_number, self.ladle_number, self.use_number, t)
            for t in self.yerode:
                db_save("brick_left", self.heat_number, self.ladle_number, self.use_number, (1.0-t)*180)
            for i in range(0,NI):
                db_save("brickID", self.heat_number, self.ladle_number, self.use_number, i)
    
    def calc_residual0(self):
        res0 = 0.0
        if not hasattr(self, 'TimePoints'):
            return res0
        elif len(self.TimeV) < 1:
            return res0
        for i in range(1,len(self.TimePoints)-1):
            idx = find_nearest_idx(self.TimeV, self.TimePoints[i])
            res0 += self.TsteelV[idx]-self.TemperaturePoints[i]
        samples = max(1,(len(self.TimePoints)-2))
        res0 = res0/samples
        return res0

    def calc_residual1(self):
        res1 = 0.0
        for i in range(1,len(self.TimePoints)-1):
            idx = find_nearest_idx(self.TimeV, self.TimePoints[i])
            res1 += (self.TsteelV[idx]-self.TemperaturePoints[i])**2
        samples = max(1,(len(self.TimePoints)-2))
        res1 = np.sqrt(res1/samples)
        return res1

    def plot3(self):

        font = {'weight': 'bold',
                'size': 22}

        plt.rc('font', **font)
        fig, ax = plt.subplots()
        ax.plot(self.TimeV, self.TsteelV, label='Steel Temp', linewidth=4)
        ax.plot(self.TimeV, self.TslagV, label='Slag Temp', linewidth=4)
        ax.plot(self.TimeV, self.TW, label='Inner Wall Temp(I=10) ', linewidth=4)
        ax.grid()
        ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Temperature [C]', fontsize=20)
        ax2 = ax.twinx()
        ax2.plot(self.TimeV, np.array(self.PowerV)/1.0e6, color='tab:red', \
                label='Power', linewidth=4)
        ax2.set_ylabel('Power [MW]', color='tab:red', fontsize=20)
        fig.legend(fontsize=20)
        plt.title(self.heat_number)
#         plt.show()

    def __feed_slag(self, dt):
        retval = 0.0
        if self.isempty:
            return retval
        tmp = self.addtime
        idx = int(np.floor(self.time))
        if idx < len(self.timep_data):
            for i in range(0, len(tmp)-1):
                if (np.datetime64(self.timep_data[int(self.time-dt)]) <= tmp[i] and\
                        np.datetime64(self.timep_data[int(self.time)]) > tmp[i]):
                    retval += self.addweight[i]/dt  # Convert to kg/s
        return retval

    def plot2(self):

        font = {'weight': 'bold',
                'size': 22}

        plt.rc('font', **font)
        fig, ax = plt.subplots()
        ax.plot(self.TimeV, self.TOuterW, label='Outer Wall Temp', linewidth=4)
        ax.plot(self.TimeV, self.TInnerW, label='Inner Wall Temp', linewidth=4)
        ax.grid()
        ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Temperature [C]', fontsize=20)
        ax2 = ax.twinx()
        ax2.plot(self.TimeV, np.array(self.PowerV)/1.0e6, color='tab:red',\
                label='Power', linewidth=4)
        ax2.set_ylabel('Power [MW]', color='tab:red', fontsize=20)
        fig.legend(fontsize=20)
        plt.title(self.heat_number)
#         plt.show()

    def plot(self, savefolder=None, namestring=''):
        font = {'weight': 'bold',
                'size': 22}

        plt.rc('font', **font)
        fig, ax = plt.subplots()
        ax.plot(self.TimeV, self.TsteelV, label='Steel Temp', linewidth=4)
        # ax.plot(TimeV, TslagV, label='Slag Temp')
        if len(self.TimePoints) > 0:
            ax.plot(self.TimeV, self.TmeasV, label='Measured Temp', linewidth=4)
            ax.plot(self.TimePoints, [x for x in self.TemperaturePoints], '*')
        ax.set_ylim([1450.0, 1750.0])
        ax.grid()
        # ax.legend()
        ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Temperature [C]', fontsize=20)
        ax2 = ax.twinx()
        ax2.plot(self.TimeV, self.ElectV, color='tab:red', label='Power', linewidth=4)
        # ax2.legend()
        ax2.set_ylabel('Power [kWh]', color='tab:red', fontsize=20)
        fig.legend(fontsize=20)
        fig.set_size_inches(18.5,10.5)
        plt.title(str(self.heat_number) + ', ' + str(self.use_number) + ',Er ' +\
                str(round(np.max(self.yerode),3)) + ',R0: ' + \
                str(round(self.calc_residual0(),0)) + ',R1: ' + \
                str(round(self.calc_residual1(),0)))

        if savefolder:
            plt.savefig(savefolder+str(self.heat_number)+namestring+'.png',\
                    bbox_inches='tight',dpi=150)
            plt.close(fig)
#         plt.show()

    def save(self):
        data = {'Time': self.TimeV,
                'T steel': self.TsteelV,
                'T slag': self.TslagV,
                'Mass steel': self.masssteelV,
                'Mass slag': self.massslagV,
                'T measurement': self.TmeasV,
                'Power': self.PowerV,
                'Electric cons': self.ElectV,
                'Energy steel': self.qsteelV,
                'Energy slag': self.qslagV,
                'Energy wall': self.qwallV,
                'Energy bottom': self.qbottomV,
                'Total energy': self.qtot}
        for i in range(0, NI):
            for j in range(0, NJ):
                for k in range(0, NK):
                    data['Twall[' + str(i) + ', ' + str(j) + ', ' + str(k) +\
                            ']'] = self.TsideV[:, i, j, k]
        for i in range(0, NM):
            data['Tbottom[' + str(i) + ']'] = self.TbottomV[:, i]

        dfr = pd.DataFrame(data)
        dfr.to_csv('sidenor.csv', index=False)
        dfr.to_excel('sidenor.xlsx')

    # Loss of C for each cell. The unit here is m refractory lost
    def __carbonloss(self, dt, gas_rate, P, i):
        utau = u_tauP(self.XU2[i]/L, gas_rate, P, rho_steel)
        Ai = 2.0*np.pi*R*self.dx[i]
        mass = self.alpha_steel1[i]*alpha_carbon*Ai*k_eff_C(alpha_carbon,self.xCbulk, self.s_pore, utau)*rho_steel*(self.xeqC - self.xCbulk)*dt
        mass += self.alpha_slag1[i]*alpha_mgo*Ai*k_eff_mgo(gas_rate, self.diff_Mgo_in_slag, self.delta_slag,utau)*rho_slag*max(0.0,self.xeqmgo - self.xmgobulk)*(alpha_carbon*rho_carbon/rho_mgo)*dt
        self.lossC[i] += mass
        return mass

    # Loss of MgO for each cell. The unit here is m refractory lost
    def __mgoloss(self, dt, gas_rate, P, i):
        utau = u_tauP(self.XU2[i]/L, gas_rate, P, rho_steel)
        Ai = 2.0*np.pi*R*self.dx[i]
        mass = (alpha_mgo*rho_mgo/rho_carbon)*self.alpha_steel1[i]*alpha_carbon*Ai*k_eff_C(alpha_carbon,self.xCbulk, self.s_pore, utau)*rho_steel*(self.xeqC - self.xCbulk)*dt
        mass += self.alpha_slag1[i]*alpha_mgo*Ai*k_eff_mgo(gas_rate, self.diff_Mgo_in_slag, self.delta_slag,utau)*rho_slag*max(0.0,self.xeqmgo - self.xmgobulk)*dt
        self.lossMgO[i] += mass

        return mass

    def mass_wall(self):
        mass = 0.0
        for i in range(0, NI):
            for j in range(0, NJ):
                for k in range(0, NK):
                    V = 2.0*np.pi*self.dx[i]*self.rm[i, j]*self.dy[i, j] +\
                            self.dy[i, j]**2*np.pi*self.dx[i]
                    mass += V*rho_wall[j]*Cp_wall[j]
        return mass

    def mass_bottom(self):
        mass = 0.0
        for i in range(0, NM):
            Vslice = np.pi*R**2*self.db[i]  # Volume of cylinder
            mass += Vslice*rho_bottom[i]*Cp_bottom[i]
        return mass
    def dump_state(self, Campaign_number=0):
        hassg = hasattr(self,'sg')
        if hassg:
            sg = self.sg
            # Pickle is not able to dump the vtk structure, this is also not
            # important to keep. We then delete it and add it back after
            # pickling
            delattr(self, 'sg')
        with open(f'{str(self.ladle_number)}_{str(Campaign_number)}.bin', 'wb') as dumpfile:
            pickle.dump(self, dumpfile)
        if hassg:
            self.sg = sg

    @staticmethod
    def read_state(ladle_number, Campaign_number=0):
        with open(f'{ladle_number}_{Campaign_number}.bin', 'rb') as dumpfile:
            return pickle.load(dumpfile)
