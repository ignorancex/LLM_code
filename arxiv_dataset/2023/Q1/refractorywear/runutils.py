import numpy as np
import pandas as pd
import os
from oomodel import LadleModel, NI, NJ, NK, NM
import math

def _post_process(ladle, Campaign_number):
    pass

def _pre_process(ladle):
    pass

def run_(heat_number, ladle_number=0, use_number=0, Campaign_number=0, source='', wt1=1.0, post_process=_post_process, pre_process=_pre_process):
    tside = np.zeros((NI,NJ,NK))
    tbot  = np.zeros((NM))
    hext = 10
    lammgoct = 2
    lamdolt  = 5
    lamalt   = 15
    laminst  = 10
    lamstsht = 1
    tol = 10.0
    maxiter = 25
    res = 1000.0
    exist = os.path.exists(f'{ladle_number}_{Campaign_number}.bin')
    if (exist and use_number > 1):
        ladle = LadleModel.read_state(ladle_number, Campaign_number)
        time1 = np.datetime64(ladle.timep_data[0])
        ladle.setDataFrame(heat_number, ladle_number, use_number, source) # read in data for case
        time2 = np.datetime64(ladle.timep_data[0])
        delta = np.timedelta64(time2-time1)/np.timedelta64(1,'h')
        if (delta > 60.0):
            print("Relining of wall bricks 22-40")
            ladle.yerode[23:40] = 0.0
    else:
        print(f'Create new instance for use {use_number}')
        ladle = LadleModel()
        ladle.setDataFrame(heat_number, ladle_number, use_number, source) # read in data for case
        ladle.setInitialCondition(1450.0, 500.0) # Set initial condition
        ladle.mass_slag  = 500.0 # Need some slag initially.
        ladle.mass_steel  = min(max(110.0*1000.0,ladle.mass_steel),150.0*1000.0) # It does not make sense when the data says there is less than 110 tons
        ladle.tmax = 60*90
        try:
            ladle.run(60, 273.15+200, 1.0)
        except:
            ladle.reset_state("db")
            ladle.setInitialCondition(1450.0, 500.0) # Set initial condition
            ladle.mass_slag  = 500.0 # Need some slag initially.
            ladle.mass_steel  = min(max(110.0*1000.0,ladle.mass_steel),155.0*1000.0) # It does not make sense when the data says there is less than 110 tons
            ladle.tmax = 60*90
            ladle.run(10, 273.15+200, 1.0)
    ladle.setHbottom(hext)
    ladle.setHext(hext)
    tsteel = ladle.Tsteelin
    it = 0
    h = use_number
#     wt1 = 1.0
    ladle.lamwalltune[:] = 1.0
    if wt1 > 0.0:
        ladle.tmax = 60.0*60
        ladle.mass_steel = 0.0
        ladle.mass_slag  = 0.0
        ladle.isempty = True
        ladle.islidon = False
        ladle.run(60, 273.15+20,0.0)
        tside[:,:,:] = ladle.Tside
        tbot[:]  = ladle.Tbottom
        ladle.isempty = False
        ladle.islidon = True  # Puts on the lid

    ladle.lamwalltune[6] = lamstsht
    ladle.lamwalltune[5] = laminst
    ladle.lamwalltune[4] = lamalt
    ladle.lamwalltune[3] = lamdolt
    ladle.lamwalltune[0:2] = lammgoct # Otherwise = 1.0
#     ladle.lamwalltune[:] = ladle.lamwalltune[:]*2.0
    tstep = 30
    while abs(res) > tol:
        it += 1
        if it == math.floor(maxiter/3.0):
            tstep = tstep/4.0
        if it == math.floor(maxiter/2.0):
            tstep = tstep/4.0
        ladle.reset_state(source)
        ladle.Tsteelin = tsteel
        ladle.mass_slag  = 500.0
        ladle.mass_steel  = min(max(110.0*1000.0, ladle.mass_steel), 150.0*1000.0)
        try:
            ladle.run(tstep, 273.15+50, 0.85)
        except:
            if tstep > 1:
                tstep = tstep/2.0
                ladle.Tside[:,:,:] = tside
                ladle.Tbottom[:]   = tbot
                ladle.TsideT0[:,:,:] = tside
                ladle.TbottomT0[:]   = tbot
                print("had to reduce time step")
                continue
            else:
                print(f'Reduction of timestep did not help for {ladle.heat_number}, exiting')
                ladle.Tside[:,:,:] = tside
                ladle.Tbottom[:]   = tbot
                ladle.TsideT0[:,:,:] = tside
                ladle.TbottomT0[:]   = tbot
                ladle.dump_state(Campaign_number)
                break
        res = ladle.calc_residual0()
        if it > 10:
            print(f'it {it}, res {res}')
        tsteel -= res/1.5
        if abs(res) >= tol and it < maxiter:
            ladle.Tside[:,:,:] = tside
            ladle.Tbottom[:]   = tbot
            ladle.TsideT0[:,:,:] = tside
            ladle.TbottomT0[:]   = tbot
            continue
        if abs(res) < tol or it >=maxiter:
            try:
                ladle.erode_wall()
            except:
                raise HTTPException(status_code = 500,
                                    detail="Ladle wall refractory too thin, please reline")
                pass
            averageWallTemp = round(np.average(ladle.Tside[:,0,0]-273.15) , 2)
        if (it > maxiter):
             print(f'Maximum number of iterations exceeded for {ladle.heat_number}, with exit residual of {res}')
             ladle.dump_state(Campaign_number)
             break
    ladle.lamwalltune[:] = 1.0
    post_process(ladle, Campaign_number)
    ladle.dump_state(Campaign_number)
    return ladle.yerode
#     ladle.save_db()

