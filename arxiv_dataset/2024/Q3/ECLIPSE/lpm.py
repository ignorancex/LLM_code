"""

Lumped parameter model of cardiovascular system.

This is the main code for the modeling.
It computes pressures/flows from a systems of ODEs.
First, it calculates the initiate values of each compartment (pressure, volume)
Then, it calculates the pressure derivative and volume derivative (by calculating the flow in a first place)
Finally, it resolves the system of equations.

"""

import jax
import jax.numpy as jnp
from jax import jit
import diffrax
import state_eq

def model_solve(timeTree, paramsModel, cardiacCyc, P0, V0, ECMO, CRRT, LVAD):

    """ Manages the solving of the model system equations.
    It first gets the initial values (volume, pressure) of each compartment from model_init()
    It then gets the pressure and volume differential from model_ODE.
    Finally it launches the solving and calculates the solution.

    Args:
        Every input parameters come from the initTree
        - timeTree: dict() -> time tree
        - paramsModel: dict() -> parameters of the lpm
        - cardiacCyc: dict() -> parameters of the current cardiac cycle 
        - V0: dict() -> initial volumes of the model 
        - ECMO: dict() -> extracorporeal circuit ECMO properties
        - CRRT: dict() -> extracorporeal circuit CRRT properties

    Returns:
        The solution of the model. 
        To access the pressures and flows from model_main: 
        - solutionODE.ys[0]=dict() of pressures
        - solutionODE.ys[1]=dict() of volumes
        - solutionODE.ys[2]=dict() of flows
    """
    initialVolumes={'Vra': V0['Vra'], 'Vrv': V0['Vrv'], 'Vla': V0['Vla'], 'Vlv': V0['Vlv']}
    
    initialFlows={'Qao': 0, 'Qsart': 0, 'Qpas': 0, 'Qpart': 0,
                  'Qecmotudp': 0, 'Qecmotupo': 0, 'Qecmotuor': 0,
                  'Qcrrttuin': 0, 'Qcrrttupf': 0, 'Qcrrttuout': 0,
                  'Qlvadpump': 0}

    initialQuantities=[P0, initialVolumes, dict(sorted(initialFlows.items()))]

    sol = diffrax.diffeqsolve(terms=diffrax.ODETerm(model_ODE), 
                                    solver=diffrax.Dopri8(),
                                    t0=timeTree['tSet']['tmin'], t1=timeTree['tSet']['tmax'], 
                                    dt0=timeTree['tSet']['h'],
                                    y0=initialQuantities,
                                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8, dtmin=1e-5, pcoeff=0.1, icoeff=1.0, dcoeff=0),
                                    args=[paramsModel, cardiacCyc, ECMO, CRRT, LVAD], 
                                    max_steps = 16**7,
                                    saveat=diffrax.SaveAt(ts=timeTree['T']),
                                    )       

    return sol

def model_ODE(t, y, args):

    """ Calculates the differential (dP_i/dt and dV_i/dt) of the quantities (pressures and volumes).
    CVS + Lungs mechanics
    Args:
        - t: [array] -> time of computation
        - y: list(dict()) -> pressures and volumes (quantities of the model)
        - args: list() -> arguments sent from the model_solve()

    Returns:
        [dict(), dict()] -> Pressures and volumes differential which is an input of the ODE solver.
    """
    
    paramsModel = args[0]
    cardiacCyc = args[1]
    ECMO=args[2]
    CRRT=args[3]
    LVAD=args[4]

    ###### Calculate atrial and ventricular activity #######
    ea=state_eq.act_atrium(t,cardiacCyc['Tcyc'],cardiacCyc['Tpwb_atr'],cardiacCyc['Tpww_atr'])
    ev=state_eq.act_ventricle(t,cardiacCyc['Tcyc'],cardiacCyc['Ts1_ven'],cardiacCyc['Ts2_ven'])

    ###### Calculate flows in each compartment #######
    # Aorta + Systemic vessels
    dQao=(y[0]['Pao']-y[0]['Psart']-paramsModel['Rao']*y[2]['Qao'])/paramsModel['Lao']
    dQsart=(y[0]['Psart']-y[0]['Psvn']-(paramsModel['Rsart']+paramsModel['Rmc'])*y[2]['Qsart'])/paramsModel['Lsart']
    Qsvn=state_eq.Q(y[0]['Psvn'],y[0]['Pra'],paramsModel['Rsvn'])
    
    # Right heart
    Qra=state_eq.Q_valves(y[0]['Pra'], y[0]['Prv'], paramsModel['CQtri'])
    Qrv=state_eq.Q_valves(y[0]['Prv'], y[0]['Ppas'], paramsModel['CQpa'])

    # Pulmonary circulation
    dQpas=(y[0]['Ppas']-y[0]['Ppart']-paramsModel['Rpas']*y[2]['Qpas'])/paramsModel['Lpas']
    dQpart=(y[0]['Ppart']-y[0]['Ppvn']-(paramsModel['Rpart']+paramsModel['Rpmc'])*y[2]['Qpart'])/paramsModel['Lpart']
    Qpvn=state_eq.Q(y[0]['Ppvn'], y[0]['Pla'], paramsModel['Rpvn'])

    # Left heart
    Qla=state_eq.Q_valves(y[0]['Pla'], y[0]['Plv'], paramsModel['CQmi'])
    Qlv=state_eq.Q_valves(y[0]['Plv'], y[0]['Pao'], paramsModel['CQao'])
   
    # Extracorporeal circuits
    Qecmodrain, dQecmotudp, Qecmopump, dQecmotupo, Qecmooxy, dQecmotuor, Qecmoreturn= jnp.where(
        jnp.equal(ECMO['status'], 0.0), 0., 
        state_eq.Q_ECMO(y, paramsModel, ECMO))
    
    dQlvadpump = jnp.where(
        jnp.equal(LVAD['status'], 0.0), 0., 
        state_eq.Q_LVAD(y, LVAD))
    
    dQcrrttuin, Qcrrtpump, dQcrrttupf, Qcrrtfil, dQcrrttuout = jnp.where(
        jnp.equal(CRRT['status'], 0.0), 0.,
        state_eq.Q_CRRT(y, paramsModel, CRRT))
    
    ###### Build system of ODEs (dP/dt and dV/dt) #######
    # Get the access and the flows of extracorporeal circuits to connect them to the CVS
    access={'ECMO': ECMO['access'], 'CRRT': CRRT['access'], 'LVAD': LVAD['access']}
    flows={'ECMO': {'drain': Qecmodrain, 'return': Qecmoreturn},
           'CRRT': {'drain': y[2]['Qcrrttuin'], 'return': y[2]['Qcrrttuout']},
           'LVAD': {'drain': y[2]['Qlvadpump'], 'return': y[2]['Qlvadpump']}}
    
    ### ---------------------- Cardiovascular system -----------------------###
    # Vessels 
    dPao=state_eq.dP('ao', Qlv, y[2]['Qao'], paramsModel['Cao'], access, flows)                  # Circ0: Aorta  
    dPsart=state_eq.dP('sart', y[2]['Qao'], y[2]['Qsart'], paramsModel['Csart'], access, flows)  # Circ1: Syst. Art.
    dPsvn=state_eq.dP('svn', y[2]['Qsart'], Qsvn, paramsModel['Csvn'], access, flows)            # Circ2: Syst. Ven.

    # New dP for pulmonary
    dPpas=state_eq.dP('pas', Qrv, y[2]['Qpas'],paramsModel['Cpas'], access, flows) 
    dPpart=state_eq.dP('part', y[2]['Qpas'], y[2]['Qpart'],paramsModel['Cpart'], access, flows) 
    dPpvn=state_eq.dP('pvn', y[2]['Qpart'], Qpvn, paramsModel['Cpvn'], access, flows) 
    # End Vessels
    
    # Heart cavities
    dPla=(state_eq.P_atrium(ea,paramsModel['Emaxla'],
                    paramsModel['Edla'],y[1]['Vla'])-y[0]['Pla'])/8E-5                  # Left heart: Left Atrium P
    dPlv=(state_eq.P_ventricle(ev,paramsModel['Emaxlv'],
                paramsModel['LV_Pd_beta'],paramsModel['LV_Pd_kappa'],
                paramsModel['LV_Pd_alpha'],y[1]['Vlv'])-y[0]['Plv'])/8E-5               # Left heart: Left Ventricle P
    dPra = (state_eq.P_atrium(ea,paramsModel['Emaxra'],
                paramsModel['Edra'],y[1]['Vra'])-y[0]['Pra'])/8E-5                      # Right heart: Right Atrium P
    dPrv = (state_eq.P_ventricle(ev,paramsModel['Emaxrv'],
                paramsModel['RV_Pd_beta'],paramsModel['RV_Pd_kappa'],
                paramsModel['RV_Pd_alpha'],y[1]['Vrv'])-y[0]['Prv'])/8E-5                # Right heart: Right Ventricle P
   
    dVla=state_eq.dV('la', Qpvn, Qla, access, flows)                                   # Left heart: Left Atrium V
    dVlv=state_eq.dV('lv', Qla, Qlv, access, flows)                                    # Left heart: Left Ventricle V
    dVra=state_eq.dV('ra', Qsvn, Qra, access, flows)                                   # Right heart: Right Atrium V
    dVrv=state_eq.dV('rv', Qra, Qrv, access, flows)                                    # Right heart: Right Ventricle V
    
    # End Heart cavities
    ### ---------------------- End Cardiovascular System ----------------------###

    ### ---------------------- Extracorporeal Circuits ------------------------###
    # ECMO
    dPecmodrain=state_eq.dP('ecmodrain', Qecmodrain, y[2]['Qecmotudp'], paramsModel['Cecmodrain'], access, flows)     # ECMO: Drain cannula     
    dPecmotudp=state_eq.dP('ecmotudp', y[2]['Qecmotudp'], Qecmopump, paramsModel['Cecmotudp'], access, flows)             # ECMO: Tubing Drain-Pump
    dPecmotupo=state_eq.dP('ecmotupo', Qecmopump, y[2]['Qecmotupo'], paramsModel['Cecmotupo'], access, flows)             # ECMO: Tubing Pump-Oxy
    dPecmooxy=state_eq.dP('ecmooxy', y[2]['Qecmotupo'], Qecmooxy, paramsModel['Cecmooxy'], access, flows)                 # ECMO: Oxygenator       
    dPecmotuor=state_eq.dP('ecmotuor', Qecmooxy, y[2]['Qecmotuor'], paramsModel['Cecmotuor'], access, flows)              # ECMO: Tubing Oxy-Return
    dPecmoreturn=state_eq.dP('ecmoreturn', y[2]['Qecmotuor'], Qecmoreturn, paramsModel['Cecmoreturn'], access, flows) # ECMO: Return cannula    
    # ECMO end ##

    # CRRT
    dPcrrttuin=state_eq.dP('crrttuin', y[2]['Qcrrttuin'], Qcrrtpump, paramsModel['Ccrrttuin'], access, flows)    # CRRT: Drain tube/Tube in
    dPcrrttupf=state_eq.dP('crrttupf', Qcrrtpump, y[2]['Qcrrttupf'], paramsModel['Ccrrttupf'], access, flows)    # CRRT: Tubing pump/filter
    dPcrrtfil=state_eq.dP('crrtfil', y[2]['Qcrrttupf'], Qcrrtfil, paramsModel['Ccrrtfil'], access, flows)        # CRRT: Filter
    dPcrrttuout=state_eq.dP('crrttuout', Qcrrtfil, y[2]['Qcrrttuout'], paramsModel['Ccrrttuout'], access, flows) # CRRT: Return tube/Tube out
    # CRRT end ##
    
    ### ---------------------- End Extracorporeal Circuits ---------------------###
    
    dtPressures = {'Pao': dPao, 'Psart': dPsart, 'Psvn': dPsvn, 'Ppas': dPpas, 'Ppart': dPpart, 'Ppvn': dPpvn,
                   'Pecmodrain': dPecmodrain, 'Pecmotudp': dPecmotudp,'Pecmotupo': dPecmotupo, 'Pecmooxy': dPecmooxy, 'Pecmotuor': dPecmotuor, 'Pecmoreturn': dPecmoreturn,
                   'Pcrrttuin': dPcrrttuin, 'Pcrrttupf': dPcrrttupf,'Pcrrtfil': dPcrrtfil, 'Pcrrttuout' : dPcrrttuout,
                   'Pra': dPra, 'Prv': dPrv,  'Pla': dPla, 'Plv': dPlv}

    dtVolumes = {'Vra': dVra, 'Vrv': dVrv, 'Vla': dVla, 'Vlv': dVlv}
    
    dtFlows={'Qao': dQao, 'Qsart': dQsart, 'Qpas': dQpas, 'Qpart': dQpart,
             'Qecmotudp': dQecmotudp, 'Qecmotupo': dQecmotupo, 'Qecmotuor': dQecmotuor,
             'Qcrrttuin': dQcrrttuin, 'Qcrrttupf': dQcrrttupf, 'Qcrrttuout': dQcrrttuout,
             'Qlvadpump': dQlvadpump}
    
    return [dtPressures, dtVolumes, dtFlows]