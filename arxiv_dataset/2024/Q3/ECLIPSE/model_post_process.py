"""

This file post process the solution of the model.
It fills out the simPostTree with the solutions and the simCompTree with comparable datas for clinicalData comparison.

"""
import jax
import jax.numpy as jnp
import state_eq
import matplotlib.pyplot as plt
import pandas as pd

def sim_post(y, initTree, simPost):

    """ Calculates transient results for compartments that have analytical expressions 
        for pressure, flow and volume. 

    Args:
        - y: [dict(), dict()] -> results from ODE solver, y[0]=Pressures, y[1]=Volumes
        - initTree: pytree() -> initTree instance from model_main
        - simPost: pytree() -> contains every output keys to fill up

    Returns:
        Composed dict() -> Flows, pressures and volumes of each compartment of the lpm

        To access to a quantity use simPost.Quantity[key].
        For example pressure: simPost.Pressure['Pao'].
    """
    T = initTree.timeTree['T']
    P0=initTree.P0
    V0=initTree.V0
    ECMO = initTree.ECMO
    CRRT = initTree.CRRT
    LVAD = initTree.LVAD
    paramsModel = initTree.paramsModel

    simPost.Pressures['Pecmopump'] = y[0]['Pecmotupo']-y[0]['Pecmotudp']
    simPost.Pressures['Pcrrtpump'] = y[0]['Pcrrttupf']-y[0]['Pcrrttuin']

    LVADdrainAccess=str(list(LVAD['access']['drain'].keys())[0])
    LVADreturnAccess=str(list(LVAD['access']['return'].keys())[0])
    simPost.Pressures['Plvadpump'] = y[0]['P'+LVADreturnAccess]-y[0]['P'+LVADdrainAccess]
    
    ###### Calculate flows in each compartment #######
    ## Vessels ##
    vQ = jax.vmap(state_eq.Q, in_axes=(0, 0, None), out_axes=0)
    simPost.Flows['Qao'] = y[2]['Qao']

    simPost.Flows['Qsart'] = y[2]['Qsart']
    simPost.Flows['Qsvn'] = vQ(y[0]['Psvn'], y[0]['Pra'], paramsModel['Rsvn'])

    simPost.Flows['Qpas'] = y[2]['Qpas']
    simPost.Flows['Qpart'] = y[2]['Qpart']
    simPost.Flows['Qpvn'] = vQ(y[0]['Ppvn'], y[0]['Pla'], paramsModel['Rpvn'])

    ## Heart cavities ##
    vQ_valves = jax.vmap(state_eq.Q_valves, in_axes=(0, 0, None), out_axes=0) 
    simPost.Flows['Qra'] = vQ_valves(y[0]['Pra'], y[0]['Prv'], paramsModel['CQtri'])
    simPost.Flows['Qrv'] = vQ_valves(y[0]['Prv'], y[0]['Ppas'], paramsModel['CQpa'])
    simPost.Flows['Qla'] = vQ_valves(y[0]['Pla'], y[0]['Plv'], paramsModel['CQmi'])
    simPost.Flows['Qlv'] = vQ_valves(y[0]['Plv'], y[0]['Pao'], paramsModel['CQao'])

    ## ECMO ##
    vQ_ECMO = jax.vmap(state_eq.Q_ECMO, in_axes=(0, None, None), out_axes=1)
    Qecmodrain, dQecmotudp, Qecmopump, dQecmotupo, Qecmooxy, dQecmotuor, Qecmoreturn = jnp.where(jnp.equal(ECMO['status'], 0.0),0.,
                                                                              vQ_ECMO(y, paramsModel, ECMO))
    simPost.Flows['Qecmodrain'] = Qecmodrain
    simPost.Flows['Qecmotudp'] = y[2]['Qecmotudp']
    simPost.Flows['Qecmopump'] = Qecmopump
    simPost.Flows['Qecmotupo'] = y[2]['Qecmotupo']
    simPost.Flows['Qecmooxy'] = Qecmooxy
    simPost.Flows['Qecmotuor'] = y[2]['Qecmotuor']
    simPost.Flows['Qecmoreturn'] = Qecmoreturn

    ## CRRT ##
    vQ_CRRT = jax.vmap(state_eq.Q_CRRT, in_axes=(0, None, None), out_axes=1)
    dQcrrttuin, Qcrrtpump, dQcrrttupf, Qcrrtfil, dQcrrttuout = jnp.where(jnp.equal(CRRT['status'], 0.0), 0.,
                                                                      vQ_CRRT(y, paramsModel, CRRT))
    simPost.Flows['Qcrrttuin'] = y[2]['Qcrrttuin']
    simPost.Flows['Qcrrtpump'] = Qcrrtpump
    simPost.Flows['Qcrrttupf']= y[2]['Qcrrttupf']
    simPost.Flows['Qcrrtfil'] = Qcrrtfil
    simPost.Flows['Qcrrttuout'] = y[2]['Qcrrttuout']

    ## LVAD ##    
    simPost.Flows['Qlvadpump'] = y[2]['Qlvadpump']

    # -------------
    pressures=['Pao', 'Psart', 'Psvn', 'Ppas', 'Ppart', 'Ppvn', 
               'Pecmodrain', 'Pecmotudp', 'Pecmotupo', 'Pecmooxy', 'Pecmotuor', 'Pecmoreturn',
               'Pcrrttuin', 'Pcrrttupf', 'Pcrrtfil', 'Pcrrttuout',
               'Pra', 'Prv', 'Pla', 'Plv']
    
    vesselsVolumes=['Vao', 'Vsart', 'Vsvn', 'Vpas', 'Vpart', 'Vpvn']
    cardiacVolumes=['Vra', 'Vrv', 'Vla', 'Vlv']

    ecCircuitVolumes=['Vecmodrain', 'Vecmotudp', 'Vecmopump', 'Vecmotupo', 'Vecmooxy', 'Vecmotuor', 'Vecmoreturn',
                      'Vcrrttuin', 'Vcrrttupf', 'Vcrrtpump', 'Vcrrtfil', 'Vcrrttuout',
                      'Vlvadpump']
    
    # Updating remaining pressure values
    for p in pressures:
        simPost.Pressures[p]=y[0][p]

    # Update remaining volume values
    basicVolumes=[*vesselsVolumes, *ecCircuitVolumes]

    # Put init pressures in initTree
    vol=0
    for v in basicVolumes:
        if v[-4:]!='pump':
            simPost.Volumes[v]=(y[0]['P'+v[1:]]-P0['P'+v[1:]])*paramsModel['C'+v[1:]]+V0[v]
            if v not in ecCircuitVolumes:
                vol=vol+simPost.Volumes[v] 
        else:
            simPost.Volumes[v]=simPost.Pressures['P'+v[1:]]
            vol=vol+simPost.Volumes[v]

    for v in cardiacVolumes:
            simPost.Volumes[v]=y[1][v]
            vol=vol+simPost.Volumes[v]
            
    return simPost

def create_Outputs(simPostTree, simCompTree):

    """ Calculates metrics that can be compared to clinical data. Serves as input to parameter
        identification process. 

    Args:
        - simPostTree: pytree() -> A tree containing the solution data from lpm
        - simCompTree: pytree() -> A tree containing comparison keys to fill up

    Returns:
        Scalar values that can be compared to clinical data and used for parameter
        identification.

        To access to a value use simCompTree.results['value']
        For example, to access SP, use simCompTree.results['SP'].

    """
    
    # Outputs which are compared to the clinical data
    #1: Arterial Pressure (equivalent location: aorta)
    simCompTree.results['SP'] = jnp.max(simPostTree.Pressures['Pao'])
    simCompTree.results['DP'] = jnp.min(simPostTree.Pressures['Pao'])
    simCompTree.results['MAP'] = jnp.mean(simPostTree.Pressures['Pao'])
    #simCompTree.results['MAP'] = 2/3 * jnp.min(simPostTree.Pressures['Pao']) + 1/3 * jnp.max(simPostTree.Pressures['Pao'])

    #2: Volumes - Left Heart
    simCompTree.results['ESVLV'] = jnp.min(simPostTree.Volumes['Vlv'])
    simCompTree.results['EDVLV'] = jnp.max(simPostTree.Volumes['Vlv'])
    simCompTree.results['ESVLA'] = jnp.min(simPostTree.Volumes['Vla'])
    simCompTree.results['EDVLA'] = jnp.max(simPostTree.Volumes['Vla'])

    #3: Volumes - Right Heart
    simCompTree.results['ESVRV'] = jnp.min(simPostTree.Volumes['Vrv'])
    simCompTree.results['EDVRV'] = jnp.max(simPostTree.Volumes['Vrv'])
    simCompTree.results['ESVRA'] = jnp.min(simPostTree.Volumes['Vra'])
    simCompTree.results['EDVRA'] = jnp.max(simPostTree.Volumes['Vra'])

    #4: Pulmonary Artery Pressure (PAK/Swan-Ganz Katheter)
    #simCompTree.results['CO'] = jnp.mean(simPostTree.Flows['Qlv'])*60/1000 # in (l/min)
    simCompTree.results['CO'] = jnp.mean(simPostTree.Flows['Qpas'])*60/1000 # in (l/min)
    simCompTree.results['SPAP'] = jnp.max(simPostTree.Pressures['Ppas'])
    simCompTree.results['DPAP'] = jnp.min(simPostTree.Pressures['Ppas'])
    simCompTree.results['MPAP'] = jnp.mean(simPostTree.Pressures['Ppas'])   # Mean Pulmonary Artery Pressure
    #simCompTree.results['MPAP'] = 2/3 * jnp.min(simPostTree.Pressures['Ppas']) + 1/3 * jnp.max(simPostTree.Pressures['Ppas'])

    # Pulmonary Capillary Wedge Pressure (PCWP)
    # Surrogate for Ppvn, Pla and EDLVP (with decreasing correlation)
    #simCompTree.results['PCWP'] = simPostTree.Pressures['Plv'][jnp.argmax(simPostTree.Volumes['Vlv'])]
    #simCompTree.results['PCWP'] = jnp.mean(simPostTree.Pressures['Pla'])
    simCompTree.results['PCWP'] = jnp.mean(simPostTree.Pressures['Ppvn'])

    #5: Pump in (l/min)
    simCompTree.results['PF'] = jnp.mean(simPostTree.Flows['Qecmopump'])*60/1000
    
    return simCompTree

def saveSimulationResultsToCSVFile(initTree,simPostTree):
        alldata = jnp.array(initTree.timeTree['T'])
        columnNames = ['t']
        ncols = 1
        for key in simPostTree.Volumes:
            columnNames.append(key)
            alldata = jnp.concatenate((alldata,jnp.array(simPostTree.Volumes[key])))
            ncols=ncols+1
        for key in simPostTree.Pressures:
            columnNames.append(key)
            alldata = jnp.concatenate((alldata,jnp.array(simPostTree.Pressures[key])))
            ncols=ncols+1
        for key in simPostTree.Flows:
            columnNames.append(key)
            alldata = jnp.concatenate((alldata,jnp.array(simPostTree.Flows[key])))
            ncols=ncols+1
        df=pd.DataFrame(jnp.reshape(alldata,(len(alldata)//ncols,ncols),order="F"),columns=columnNames)
        df.to_csv('output.csv', index=False)