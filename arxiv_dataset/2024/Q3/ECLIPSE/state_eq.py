# -*- coding: utf-8 -*-
"""

This code defines the state equations for the passive and active compartments

"""

import jax
import jax.numpy as jnp
import numpy as np

def smooth_transition(x, width=0.01, offsetX=0., lowerBound=0., upperBound=1.):
    """ Create a smooth transition from 0 to 1 using a hyperbolic tangent function.
    
    Args:
        - x: float -> the input value around which to center the transition.
        - width: float -> controls the sharpness of the transition.
        - offsetX: float -> the offset of the transition on the x-axis.
        - lowerBound: float -> the lower bound of the transition on the y-axis.
        - upperBound: float -> the upper bound of the transition on the y-axis.
       
    Returns: 
        float -> the smooth step transition value between 0 and 1.
    """
    dy = upperBound - lowerBound

    return 0.5 * dy * (1 + jnp.tanh((x - offsetX) / width)) + lowerBound

def act_ventricle(t, Tcyc, Ts1,Ts2):

    """ Calculate the ventricle activation for the specified time t.
        
    Args:
        - t: [array] -> the time where the calculation is done
        - Tcyc: float -> cardiac cycle parameter
        - Ts1: float -> cardiac cycle parameter
        - Ts2: float -> cardiac cycle parameter
       

    Returns: 
        [array] -> the ventricle activation.
     """

    return jnp.piecewise(t,
            [jnp.less(jnp.mod(t, Tcyc), Ts1), jnp.logical_and(jnp.greater_equal(jnp.mod(t, Tcyc), Ts1), jnp.less(jnp.mod(t, Tcyc), Ts2))],
            [lambda t: 1/2*(1-jnp.cos((jnp.mod(t, Tcyc))/Ts1*jnp.pi)), 
             lambda t: 1/2*(1+jnp.cos(((jnp.mod(t, Tcyc))-Ts1)/(Ts2-Ts1)*jnp.pi)), 
             0.]) 

def act_atrium(t, Tcyc, Tpwb,Tpww):

    """ Calculate the atrium activation for the specified time t.
        
    Args:
        - t: [array] -> the time where the calculation is done
        - Tcyc: float -> cardiac cycle parameter
        - Tpwb: float -> cardiac cycle parameter
        - Tpww : float -> cardiac cycle parameter
       

    Returns: 
        [array] -> the atrium activation.
     """

    return jnp.piecewise(t,
        [jnp.less(jnp.mod(t, Tcyc), Tpwb), jnp.logical_and(jnp.greater_equal(jnp.mod(t, Tcyc), Tpwb), jnp.less(jnp.mod(t, Tcyc), Tpwb+Tpww))],
        [0., 
         lambda t: 1/2*(1-jnp.cos(((jnp.mod(t, Tcyc))-Tpwb)/Tpww*2*jnp.pi)), 
         0.]) 
 
def Q(Pin, Pout, R):

    """ Calculate the flow for the specified upstream and outstream pressures through a resistance.
        
    Args:
        - Pin: [array] -> upstream pressure
        - Pout: [array] -> outstream pressure
        - R: float -> parameter model, resistance
       
    Returns: 
        [array] -> flow.
     """
    
    return (Pin-Pout)/R
    
def P(V, C):

    """ Calculate the pressure for the specified volume and compliance based on pressure-volume relationship.
        
    Args:
        - V: float -> parameter model, volume
        - C: float -> parameter model, compliance
       
    Returns: 
        float -> pressure.
    """

    return V/C

def Q_valves(P1, P2, CQi, epsilon=1E-8):
    """ Calculate the flow of the valve for the specified upstream and outstream pressures.
        
    Args:
        - P1: [array] -> upstream pressure
        - P2: [array] -> outstream pressure
        - CQi: float -> parameter model, flow parameter/resistance for the valve
        - epsilon: float -> small penality as gradient of sqrt(|x|) at x = 0 does not exist
       
    Returns: 
        [array] -> flow.
     """
    
    transSign = smooth_transition((P1 - P2), width=0.01, offsetX=0.0, lowerBound=-1.0, upperBound=1.0)
    transState = smooth_transition((P1 - P2), width=0.01, offsetX=0.0)

    return jnp.sqrt((P1 - P2) * transSign + epsilon) * CQi * transState

def P_atrium(eact,Es,Ed,V):

    """ Calculate the atrium pressure.
        
    Args:
        - eact: float -> parameter to calculate the elastance
        - Es: float -> parameter to calculate the elastance
        - Ed: float -> parameter to calculate the elastance
        - V: float -> parameter model, volume
       
    Returns: 
        float -> pressure.
     """
    
    elastance=Ed+(Es-Ed)*eact

    return elastance*(V)

def P_ventricle(eact,Es,beta,kappa,alpha,V):

    """ Calculate the ventricle pressure.
        
    Args:
        - eact: float -> parameter to calculate the pressure
        - Es: float -> parameter to calculate the pressure
        - beta: float -> parameter to calculate the pressure
        - kappa: float -> parameter to calculate the pressure
        - V: float -> parameter model, volume
       
    Returns: 
        float -> pressure.
     """
    
    return (V)*eact*Es+(1-eact)*(alpha*jnp.exp(kappa*(V))+beta)
 
def dP(compartment, Qin, Qout, C, access, flows):

    """ Calculate the pressure differential dP/dt depending on which circuit (ECMO, CRRT or both) is active.
        
    Args:
        - compartment: string -> the compartment where the calculation of dP is needed
        - Qin: [array] -> flow going into the compartment
        - Qout: [array] -> flow going outside the compartment
        - C: float -> the compliance parameter
        - access: dict() -> access of the extracorporeal circuits
        - flows: dict() -> flows of the extracorporeal circuits drain and return cannulae or tubing

    Returns: 
        [array] -> dP/dt: (Qin+Qreturn)/(Qout+Qdrain)/C, Qreturn and Qdrain are calculated depending 
        on the activationStatus and the access of the circuits.
    """
    isAnECMOdrainAccess=int(compartment==list(access['ECMO']['drain'].keys())[0])
    isAnECMOreturnAccess=int(compartment==list(access['ECMO']['return'].keys())[0])
    isACRRTdrainAccess=int(compartment==list(access['CRRT']['drain'].keys())[0])
    isACRRTreturnAccess=int(compartment==list(access['CRRT']['return'].keys())[0])
    isALVADdrainAccess=int(compartment==list(access['LVAD']['drain'].keys())[0])
    isALVADreturnAccess=int(compartment==list(access['LVAD']['return'].keys())[0])

    Q_drain=isAnECMOdrainAccess*flows['ECMO']['drain']+isACRRTdrainAccess*flows['CRRT']['drain']+isALVADdrainAccess*flows['LVAD']['drain']
    Q_return=isAnECMOreturnAccess*flows['ECMO']['return']+isACRRTreturnAccess*flows['CRRT']['return']+isALVADreturnAccess*flows['LVAD']['return']

    return ((Qin+Q_return)-(Qout+Q_drain))/C

def dV(compartment, Qin, Qout, access, flows):

    """ Calculate the volume differential dV/dt depending on which circuit (ECMO, CRRT or both) 
    is active and if the compartment is an access.
        
    Args:
        - compartment: String -> the compartment where the calculation of dV is needed
        - Qin: [array] -> Flow going into the compartment
        - Qout: [array] -> Flow going outside the compartment
        - access: dict() -> Access of the extracorporeal circuits
        - flows: dict() -> Flows of the extracorporeal circuits drain and return cannulae or tubing

    Returns: 
        [array] -> dV/dt: (Qin+Qreturn)/(Qout+Qdrain), Qreturn and Qdrain are calculated depending 
        on the activationStatus and the access of the circuits.
    """
    
    isAnECMOdrainAccess=int(compartment==list(access['ECMO']['drain'].keys())[0])
    isAnECMOreturnAccess=int(compartment==list(access['ECMO']['return'].keys())[0])
    isACRRTdrainAccess=int(compartment==list(access['CRRT']['drain'].keys())[0])
    isACRRTreturnAccess=int(compartment==list(access['CRRT']['return'].keys())[0])
    isALVADdrainAccess=int(compartment==list(access['LVAD']['drain'].keys())[0])
    isALVADreturnAccess=int(compartment==list(access['LVAD']['return'].keys())[0])
    Q_drain=isAnECMOdrainAccess*flows['ECMO']['drain']+isACRRTdrainAccess*flows['CRRT']['drain']+isALVADdrainAccess*flows['LVAD']['drain']
    Q_return=isAnECMOreturnAccess*flows['ECMO']['return']+isACRRTreturnAccess*flows['CRRT']['return']+isALVADreturnAccess*flows['LVAD']['return']

    return ((Qin+Q_return)-(Qout+Q_drain))
    
def diameter_fitting(D, params):

    """ Function that makes the fitting universal for each diameter given in the Getinge data sheets. 
    Calculates the value of the coefficient R in sqrt(1/R*dp) for every diameter.

    Args:
        - D: [array] -> diameter of the cannula
        - params: [array] -> parameters of the cannula

    Returns: 
        [array] -> coefficient R for the unified cannula model.
    """
        
    a, b, c, d = params

    return a * D**3 + b * D**2 + c * D + d

def unified_cannula_model(dp, paramCannula, epsilon=1E-8):
    
    """ Unified function that fits the experimental Getinge data sheets. 
    Calculates the flow for specific cannula type and diameter.

    Args:
        - dp: [array] -> pressure difference between upstream and downstream compartment
        - paramCannula: pytree -> diameter and parameters of cannula
        - epsilon: float -> small penality as gradient of sqrt(|x|) at x = 0 does not exist

    Returns: 
        [array] -> flow of cannula in ml/s.
    """

    D = paramCannula['diameter']
    params = paramCannula['params']

    R = diameter_fitting(D, params)

    transSign = smooth_transition(dp, width=0.1, lowerBound=-1.0, upperBound=1.0)

    Q = jnp.sqrt(1/R * dp * transSign + epsilon) * transSign

    return Q

def Q_Cannula(Pin, Pout, paramCannula):

    """ Calculates the flow in cannula depending on the parameters stored in 'paramCannula'.

    Args:
        - Pin: [array] -> upstream pressure 
        - Pout: [array] -> downstream pressure
        - paramCannula: pytree -> diameter and parameters of cannula

    Returns: 
        [array] -> flow of cannula in ml/s.
    """

    Pdrop = Pin - Pout

    # * 16/67 to convert l/min in ml/s
    Q = unified_cannula_model(Pdrop, paramCannula) * 16.67

    return Q

def ECMO_RBP_flow(Pin, Pout, K, rpm, epsilon=1E-8):

    """ Calculates the flow of a RBP in ECMO system depending on the parameters stored in 'paramPump'.
        Equation based on Boes19 paper. Curve fitting for H = f(Q). Inverse of that implemented here.

    Args:
        - Pin: [array] -> upstream pressure 
        - Pout: [array] -> downstream pressure
        - K: pytree -> coefficients of pump equation
        - rpm: [array] -> the rpm of the pump
        - epsilon: float -> small penality as gradient of sqrt(|x|) at x = 0 does not exist

    Returns: 
        [array] -> flow of cannula in mL/s.
    """

    H = Pout-Pin
    qinf = K[4]*rpm
    threshold = K[0]*(rpm**2) - K[1]*rpm*qinf - K[2]*(qinf**2)

    # Definition of Pump Flow for H Smaller than Threshold
    # signQ1 to ensure that root is real
    exprSignQ1 = K[1]**2*rpm**2 + 4.0*K[0]*K[2]*rpm**2 - 4.0*K[2]*H
    signQ1 = smooth_transition(exprSignQ1, width=0.01, offsetX=0., lowerBound=-1., upperBound=1.)
    Q1 = (-(0.5*(-jnp.sqrt(signQ1 * exprSignQ1 + epsilon) + K[1]*rpm)) / K[2]) * 16.67
    
    # Definition of Pump Flow for H Greater than Threshold
    exprSignQ2 = 4.0*K[3]*H - 4.0*K[2]*H + K[1]**2*rpm**2 + 4.0*K[2]*K[3]*qinf**2 + 4.0*K[0]*K[2]*rpm**2 - 4.0*K[0]*K[3]*rpm**2 + 4.0*K[1]*K[3]*qinf*rpm
    signQ2 = smooth_transition(exprSignQ2, width=0.01, offsetX=0., lowerBound=-1., upperBound=1.)
    Q2 = (-(0.5*(2.0*K[3]*qinf + K[1]*rpm - jnp.sqrt(signQ2 * exprSignQ2 + epsilon))) / (K[2] - 1.0*K[3])) * 16.67
    
    smooth_factor = smooth_transition(H, width=0.01, offsetX=threshold)

    return (1 - smooth_factor) * Q1 + smooth_factor * Q2

def Q_pump(Pin, Pout, paramPump):

    """ Calculates the flow in the ECMO pump depending on the rpm. Roller Pump or Rotary Blood 
        Pump (RBP) possible.

    Args:
        - Pin: [array] -> upstream pressure 
        - Pout: [array] -> downstream pressure
        - paramPump: list() -> the rpm of the pump - paramPump[0, 1, 2, 3, 4, 5] corresponds respectively to [l1, l2, p1, p2, p3, H]

    Returns: 
        [array] -> flow of pump Qpump in mL/s.
    """

    pumpType=np.array(list(paramPump.keys()))[0]

    # ECMO System with Roller Pump
    if pumpType=='roller':
            
        return paramPump[pumpType]

    # ECMO System with Rotary Blood Pump like Rotaflow or DP3
    else:

        return ECMO_RBP_flow(Pin, Pout, paramPump[pumpType], paramPump['rpm'])

def R_L_tubing(paramsModel, tubeType):
    """ Calculates the resistance and inertance of ECLS tubings.

    Args:
        - paramsModel: list() -> parameters of the lpm. 
        - tubeType: string -> type of tubing (e.g. ecmotudp, crrttuin, crrttupf, ...)

    Returns: 
        [array] -> flow of tubing in ml/s.
    """

    L = paramsModel['L'+tubeType]*0.01 # cm to m
    D = paramsModel['D'+tubeType]*0.01 # cm to m

    # Resistance by Hagen Poiseuille law
    R = ((128*L*paramsModel['muB'])/(jnp.pi*(D**4)))/(10**6 * 133.32)

    # Inertance
    I = (paramsModel['rohB']*L)/(133.32*(jnp.pi*(D/2)**2))

    return jnp.array([R, I])

def dQ_tubing(Pin, Pout, paramsModel, tubeType):
    L = paramsModel['L'+tubeType]*0.01 # cm to m
    D = paramsModel['D'+tubeType]*0.01 # cm to m
    deltaP=Pin-Pout
    return (133.3*deltaP*(jnp.pi*(D/2)**2))/(paramsModel['rohB']*L)
   
    
def Q_ECMO(y, paramsModel, ECMO):

    """ Calculates the flows within the compartments of the extracorporeal circuit ECMO.

    Args:
        - y: [array] -> solution of ODE at current time step t.
        - paramsModel: list() -> parameters of the lpm. 
        - ECMO : dict() -> ECMO properties from the initTree

    Returns:
        [array] -> Flows of external circuit 
        [Qecmodrain, Qecmotudp, Qecmopump, Qecmotupo, Qecmooxy, Qecmotuor, Qecmoreturn].

    """

    drainAccess=list(ECMO['access']['drain'].keys())[0]
    returnAccess=list(ECMO['access']['return'].keys())[0]

    Rtudp, Ltudp = R_L_tubing(paramsModel, 'ecmotudp')
    Rtupo, Ltupo = R_L_tubing(paramsModel, 'ecmotupo')
    Rtuor, Ltuor = R_L_tubing(paramsModel, 'ecmotuor')

    Qecmodrain = Q_Cannula(y[0]['P'+drainAccess], y[0]['Pecmodrain'], ECMO['cannula']['drain'])
    dQecmotudp = (y[0]['Pecmodrain'] - y[0]['Pecmotudp'] - (Rtudp * y[2]['Qecmotudp'])) / Ltudp
    Qecmopump = Q_pump(y[0]['Pecmotudp'], y[0]['Pecmotupo'], ECMO['pump'])
    dQecmotupo = (y[0]['Pecmotupo'] - y[0]['Pecmooxy'] - Rtupo * y[2]['Qecmotupo']) / Ltupo
    Qecmooxy= Q(y[0]['Pecmooxy'], y[0]['Pecmotuor'], paramsModel['Recmooxy']) 
    dQecmotuor = (y[0]['Pecmotuor'] - y[0]['Pecmoreturn'] - (Rtuor * y[2]['Qecmotuor'])) / Ltuor
    Qecmoreturn = Q_Cannula(y[0]['Pecmoreturn'], y[0]['P'+returnAccess], ECMO['cannula']['return'])

    return jnp.array([Qecmodrain, dQecmotudp, Qecmopump, dQecmotupo, Qecmooxy, dQecmotuor, Qecmoreturn])

def Q_CRRT(y, paramsModel, CRRT):

    """ Calculates the flows/derivatives within the compartments of the extracorporeal circuit CRRT.

    Args:
        - y: [array] -> solution of ODE at current time step t.
        - paramsModel: list() -> parameters of the lpm. 
        - CRRT : dict() -> CRRT properties from the initTree

    Returns:
        [array] -> Flows of external CRRT circuit. 
        [dQcrrttuin, Qcrrtpump, dQcrrttupf, Qcrrtfil, dQcrrttuout].

    """

    drainAccess=str(list(CRRT['access']['drain'].keys())[0])
    returnAccess=str(list(CRRT['access']['return'].keys())[0])

    Rtuin, Ltuin = R_L_tubing(paramsModel, 'crrttuin')
    Rtupf, Ltupf = R_L_tubing(paramsModel, 'crrttupf')
    Rtuout, Ltuout = R_L_tubing(paramsModel, 'crrttuout')

    dQcrrttuin = (y[0]['P'+drainAccess] - y[0]['Pcrrttuin'] - Rtuin*y[2]['Qcrrttuin'])/Ltuin
    Qcrrtpump = Q_pump(y[0]['Pcrrttuin'], y[0]['Pcrrttupf'], CRRT['pump'])
    dQcrrttupf = (y[0]['Pcrrttupf'] - y[0]['Pcrrtfil'] - Rtupf*y[2]['Qcrrttupf'])/Ltupf
    Qcrrtfil= Q(y[0]['Pcrrtfil'], y[0]['Pcrrttuout'], paramsModel['Rcrrtfil'])
    dQcrrttuout = (y[0]['Pcrrttuout'] - y[0]['P'+returnAccess] - Rtuout*y[2]['Qcrrttuout'])/Ltuout

    return jnp.array([dQcrrttuin, Qcrrtpump, dQcrrttupf, Qcrrtfil, dQcrrttuout])

def LVAD_flow(Pin, Pout, Q, param):
    """ Calculates the flow derivative of the LVAD pump. Based on dynamic pump model including 
        periphery presented in Böes19.

    Args:
        - Pin: [array] -> upstream pressure 
        - Pout: [array] -> downstream pressure
        - Q: [array] -> solution of pump flow at current time step t.
        - param: list() -> parameters of the LVAD.

    Returns:
        [array] -> Flow derivative of LVAD.

    """

    H=Pout-Pin
  
    qinf=param['coeff']['kinf']*param['rpm']
    transitionQqInf = smooth_transition(Q, offsetX=qinf)
    transitionQzero = smooth_transition(Q, lowerBound=-1.0, upperBound=1.0)

    dQ = 1 / (param['coeff']['L'] + param['coeff']['Lper']) * \
        (param['coeff']['a']*param['rpm']**2 - param['coeff']['R1']*param['rpm']*Q \
         - param['coeff']['R2']*Q**2 - H + param['coeff']['Rrec']*(Q-qinf)**2 * (1-transitionQqInf) \
         - param['coeff']['Rper'] * Q**2 * transitionQzero)
    
    return dQ

def Q_LVAD(y, LVAD):
    """ Calculates the flow derivative of the LVAD pump including losses due to its periphery.

    Args:
        - y: [array] -> solution of ODE system at current time step t.
        - LVAD: list() -> parameters of the LVAD.

    Returns:
        [array] -> Flow derivative of LVAD.

    """
    drainAccess=str(list(LVAD['access']['drain'].keys())[0])
    returnAccess=str(list(LVAD['access']['return'].keys())[0])

    dQlvadpump = LVAD_flow(y[0]['P'+drainAccess], y[0]['P'+returnAccess], y[2]['Qlvadpump'], LVAD)

    return jnp.array(dQlvadpump)    