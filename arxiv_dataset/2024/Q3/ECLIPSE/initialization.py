"""

This file initializes all the model initial configuration. 
Manages interpolation of pump curve, pytrees loading and creation of patient data specific file. 

"""
import numpy as np
import sensitivity_analysis
import model_main
import jax
import jax.numpy as jnp
import os
import file_manager
import collections
import state_eq
import sys
import re
import warnings
import sensitivity_analysis

######### GLOBAL parameters : Patient, pytrees,parameters, V0 #######
def pytrees_init(patientId, patientFitted, runtimeTree, GSABoundsSpreading=10.0):

    """ Initializes the pytrees containing configuration and data for the simulation to be run.

    Args:
        - patientId: integer -> id of the patient used to load the clinicalData in the initTree
        - patientFitted: boolean -> specify if the model parameters should be patient-specific
        - runtimeTree: dict() -> runtime variables for the simulation
        - GSABoundsSpreading: float -> percentage of spreading to calculate bounds for sensitivity analysis

    Returns:
        Datas and trees depending on the run specified
            's' or sensitivity_analysis returns GSATree, initTree, simPostTree, simCompTree

            'f' or fitting/optimization returns clinicalData, params_dict_x0, dict_bounds, x0_batch, 
            initTree, simPostTree, simCompTree

            'p' or production returns initTree, simPostTree, simCompTree.
    """

    initTree = model_main.parameter_pytree(runtimeTree, patientId, patientFitted)
    simPostTree = model_main.simPost_pytree()
    simCompTree = model_main.simComp_pytree()
    
    clinicalData=initTree.clinicalData

    match runtimeTree['runType']:
        case 's':
            
            # Get the parameters of the model (stored in initTree)
            paramsModel = initTree.paramsModel

            excludeKeys = ['rpm', 'drain', 'return']
            
            # CRRT['access']={'drain': {'svn': 1}, 'return': {'ra': 1}}
            
            # Basic parameters
            # Get the sensitivity input parameters that we want to analyse
            paramNamesGSA, paramGroupGSA = sensitivity_analysis.read_param_file_GSA( \
                file_manager.find_path('sensitivity', runtimeTree['sensitivityInput']))
           
            # Intersect paramsModel and paramsGSA to load the initial values for GSA
            paramValuesGSA = {k: paramsModel[k] for k in paramNamesGSA if k not in excludeKeys}
        
            # Generate bounds based on the sensitivity bound percentage around initial values
            lb = [(x - (GSABoundsSpreading/100)*x) for x in paramValuesGSA.values()]
            ub = [(x + (GSABoundsSpreading/100)*x) for x in paramValuesGSA.values()]

            names = list(paramValuesGSA.keys())
            bounds = [[lb[i], ub[i]] for i in range(len(names))] 
            
            if runtimeTree['includeECLSGSA'] == True:
                # [drain, return] for CRRT
                lbCRRT = [12, 12]
                ubCRRT = [15, 16]
            
                # Add crrt connections
                names.append(excludeKeys[1])    # append 'drain'
                names.append(excludeKeys[2])    # append 'return'
                names.append(excludeKeys[0])    # append 'rpm'
                
                bounds.append([lbCRRT[0], ubCRRT[0]])   # append bounds for 'drain'
                bounds.append([lbCRRT[1], ubCRRT[1]])   # append bounds for 'return'
                #bounds.append([1000, 5000])             # append bounds for 'rpm'

                ECMOrpm = initTree.ECMO['pump']['rpm']
                bounds.append([(ECMOrpm-(GSABoundsSpreading/100)*ECMOrpm), \
                               (ECMOrpm+(GSABoundsSpreading/100)*ECMOrpm)]) # append bounds for 'rpm'

                if paramGroupGSA is not None:
                    paramGroupGSA.append(excludeKeys[1])    # append 'drain'
                    paramGroupGSA.append(excludeKeys[2])    # append 'return'
                    paramGroupGSA.append(excludeKeys[0])    # append 'rpm'

                initTree.CRRT.update({'status': 1.0})

            # Transform the params into an OrderedDict()
            paramGSAOrdered = collections.OrderedDict(sorted(zip(names, bounds)))
            paramGroupGSAOrdered = None
            if paramGroupGSA is not None: paramGroupGSAOrdered = collections.OrderedDict(sorted(zip(names, paramGroupGSA)))

            # Output results for GSA (data comparable to clinicalData)
            simulatedData = simCompTree
            resultsTmp = dict()
            for key in simulatedData.results.keys():
                if key in clinicalData:
                    resultsTmp[key] = simulatedData.results[key]

            resultsTmp=collections.OrderedDict(sorted(resultsTmp.items()))

            # Store information of the GSA into a pytree
            GSATree = sensitivity_analysis.GSA_pytree(clinicalData, resultsTmp, paramGSAOrdered, \
                                            GSABoundsSpreading, runtimeTree['checkConvergenceGSA'], \
                                            runtimeTree['includeECLSGSA'])
            
            return GSATree, initTree, simPostTree, simCompTree, paramGroupGSAOrdered
         
        case 'f':

            # Function to generate random values between lb and ub/batch job
            sample = lambda key, paramOpt, paramOpt_dict_x0: dict(zip(paramOpt['param'], 
            jax.random.uniform(key, jnp.array(list(paramOpt_dict_x0.values())).shape, 
            minval=paramOpt['bounds'][:,1], maxval=paramOpt['bounds'][:,2])))

            # --------
            nSeeds = 8
            # --------
            # Getting the initial values x0 for optimization
            paramOpt = np.genfromtxt(file_manager.find_path('optimization'), dtype=[('param','U15'),('bounds',float,3)])
            paramOpt_dict_x0 = {k:d for k,d in zip(paramOpt['param'], paramOpt['bounds'][:,0])}
            
            # Generating the bounds
            lb = {k:d for k,d in zip(paramOpt['param'], paramOpt['bounds'][:,1])}
            ub = {k:d for k,d in zip(paramOpt['param'], paramOpt['bounds'][:,2])}
            paramOpt_dict_bounds = (lb, ub)
        
            # Create random initial values for batch job
            x0_batch = dict()
            key = jax.random.PRNGKey(0)
            subkeys = jax.random.split(key, nSeeds)
            x0_batch = jax.vmap(sample, in_axes=(0, None, None))(subkeys, paramOpt, paramOpt_dict_x0)

            return clinicalData, paramOpt_dict_x0, paramOpt_dict_bounds, x0_batch, initTree, simPostTree, simCompTree
    
        case 'p':
            return initTree, simPostTree, simCompTree

def get_patient_specific_params(params, patientFittingPath):

    """
    Retrieves patient-specific parameters from a fitting result file and updates the given parameters dictionary.

    Args:
        - params:  dict() -> The dictionary of parameters to be updated.
        - patientFittingPath: string -> The path to the patient fitting result file.

    Returns:
        - params: dict() -> The updated parameters dictionary.

    Raises:
        ValueError: If the parameters section is not found in the file.

    """

    # Open patient fitting result file and get the fitted parameters
    with open(patientFittingPath, 'r') as file:
        content = file.read()

    parameters_pattern = r'######### Parameters #########\n# Parameter Value\n(.*?)(?=\n#|\Z)'
    match = re.search(parameters_pattern, content, re.DOTALL)
    
    if match is None:
        raise ValueError("Parameters section not found in the file. Please check the file structure.")
    
    parameters_section = match.group(1)

    patientFittingParams = {}
    for line in parameters_section.strip().split('\n'):  # Skipping the header line
        parts = line.split()
        if len(parts) == 2:  # Ensure there are exactly 2 parts: parameter name and value
            param_name, value = parts
            patientFittingParams[param_name] = float(value)

    # Update the default parameter dictionary with the patient-specific values
    params.update(patientFittingParams)
    
    return params

def load_params_model(parameterModelFilename, patientId, patientFitted):

    """ Load the default model parameters and apply patient-specific values for given patientId.

    Args:
        - parameterModelFilename: string -> the path of the default model parameters .txt file
        - patientId: integer -> the current patient id
        - patientFitted: boolean -> specify if the model parameters should be patient fitted
    Return:
        - params: dict() -> model parameters.
    """
        
    params = {}

    # Load the default parameter_model
    paramsRaw = np.genfromtxt(parameterModelFilename, dtype=[('name', 'U12'),('value',float)])
    params={k:d for k,d in zip(paramsRaw['name'], paramsRaw['value'])}

    if patientFitted == True:
        patientFittingPath=file_manager.optimizationFolder+'Results/result_optimization_patient_'+str(patientId)+'.txt'
        if os.path.exists(patientFittingPath):
            params = get_patient_specific_params(params, patientFittingPath)
        else:
            warnings.warn(f'You must do a fitting for patient {patientId} before using fitted parameters. (Default config is used)')
       
    return params

def load_cardiac_cyc(paramsModel):
    
    """ Loads every infos about the cardiac cycle.

    Args:
        - paramsModel: dict() -> the clinical datas of the current patient

    Return:
        dict() -> cardiac cycle.
    """

    # Duration of Cardiac dynamics
    Tcyc = 60/paramsModel['bpm']
    # Timings of Atrial Contraction
    Tpwb_atr=round(0.9*Tcyc,4)
    Tpww_atr=round(0.09*Tcyc,4)

    #Timings of Venctricular Contraction
    Ts1_ven=round(0.3*Tcyc,4)
    Ts2_ven=round(0.45*Tcyc,4)

    return {'Tcyc': Tcyc, 'Tpwb_atr': Tpwb_atr, 'Tpww_atr': Tpww_atr,
                   'Ts1_ven': Ts1_ven, 'Ts2_ven': Ts2_ven}

def load_P0(paramsModel, cardiacCyc, V0, filename):

    ea=state_eq.act_atrium(0., cardiacCyc['Tcyc'], cardiacCyc['Tpwb_atr'], cardiacCyc['Tpww_atr'])
    ev=state_eq.act_ventricle(0., cardiacCyc['Tcyc'], cardiacCyc['Ts1_ven'], cardiacCyc['Ts2_ven'])

    # No heart cavities (special pressure) and no pump (no pressure)
    # Pressures of interest
    pressureECLS=['ecmodrain', 'ecmotudp', 'ecmotupo', 'ecmooxy', 'ecmotuor', 'ecmoreturn', 
                  'crrttuin', 'crrttupf', 'crrtfil', 'crrttuout']
    
    # Initialization of pressures for the specified compartments
    ## ------------------
    initPressuresVessels=np.genfromtxt(filename, dtype=[('name', 'U12'),('value',float)])
    initPressuresVessels={k:d for k,d in zip(initPressuresVessels['name'], initPressuresVessels['value'])}

    initPressuresHeartCavity={'Pra': state_eq.P_atrium(ea,paramsModel['Emaxra'],paramsModel['Edra'],V0['Vra']), 
                'Prv': state_eq.P_ventricle(ev,paramsModel['Emaxrv'],paramsModel['RV_Pd_beta'],paramsModel['RV_Pd_kappa'],paramsModel['RV_Pd_alpha'],V0['Vrv']), 
                'Pla': state_eq.P_atrium(ea, paramsModel['Emaxla'], paramsModel['Edla'], V0['Vla']),
                'Plv': state_eq.P_ventricle(ev, paramsModel['Emaxlv'], paramsModel['LV_Pd_beta'], paramsModel['LV_Pd_kappa'], paramsModel['LV_Pd_alpha'], V0['Vlv'])
                                    }
    initPressuresECLS={'P'+c: 0 for c in pressureECLS}
    
    # Grouping the two pressures dict into a single one
    initPressures={**initPressuresVessels, **initPressuresHeartCavity, **initPressuresECLS}

    return initPressures

def load_V0(paramsModel, clinicalData, ECMO, V0FileNames):

    """ Create dictionary with initial volumes of the model for the CVS and ECLS compartments.
        CVS is based on total blood volume and common ratios for each compartment based on literature.
        ECLS is based on cannula and tubing dimensions and priming volumes for filters based on data sheets.

    Args:
        - paramsModel: dict() -> model parameters
        - clinicalData: dict() -> clinical data of the current patient
        - V0Filename: string -> path of the .txt file containing the initial volumes

    Return:
        dict() -> V0/Initial volumes.
    """

     #### CVS ####
    # Caluclate Body mass index and total blood volume
    BMI=clinicalData['weight']/(clinicalData['height']/100)**2
    TBV=((90-5*clinicalData['sex'])-0.4*clinicalData['age'])/np.sqrt(BMI/22)*clinicalData['weight']
   
    # Read ratios and calculate volumes of each CVS compartment based ond TBV
    V0Ratios = np.genfromtxt(V0FileNames[0], dtype=[('name', 'U12'),('value',float)])
    initVolumesCVS={k:d for k,d in zip(V0Ratios['name'], V0Ratios['value']*TBV)}

    #### ECLS ####
    volumeTubesECLS=['ecmotudp', 'ecmotupo', 'ecmotuor', 
                     'crrttuin', 'crrttupf', 'crrttuout']
    
    initVolumeCannulaeECLS={'Vecmo'+can: ECMO['cannula'][can]['length']*jnp.pi*
                         (ECMO['cannula'][can]['diameter']/(2*3*10))**2 for can in ['drain', 'return']}

    initVolumeTubesECLS={'V'+tube: paramsModel['L'+tube]*jnp.pi*
                            (paramsModel['D'+tube]/2)**2 for tube in volumeTubesECLS}
    
    initVolumes={**initVolumesCVS, **initVolumeCannulaeECLS, **initVolumeTubesECLS}

    # Read initial volumes for filters
    V0Tmp = np.genfromtxt(V0FileNames[1], dtype=[('name', 'U12'),('value',float)])
    initVolumes.update({k:d for k,d in zip(V0Tmp['name'], V0Tmp['value'])})
    
    print('Total blood volume', round(TBV, 0))
   
    return initVolumes

def read_table_bmi(filename):
    table = []
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            age_min, age_max, height, weight, bmi = line.strip().split('\t')
            table.append((int(age_min), int(age_max), float(height), float(weight), float(bmi)))
    return table

def get_height_weight(age, table):
    for age_min, age_max, height, weight, _ in table:
        if age_min <= age <= age_max:
            return height, weight
    return None, None

def load_clinicalData(patientFilename):

    """ Loads the clinical data of the patient from the specified .txt file.

    Args:
        - patientFilename: string -> path of the .txt file containing the patient datas

    Return:
        dict() -> Clinical datas.
    """

    clinicalDataRaw = np.genfromtxt(patientFilename, dtype=[('patientdata','U25'),('value',float)])
    clinicalData = {k:d for k,d in zip(clinicalDataRaw['patientdata'], clinicalDataRaw['value'])}


    age = clinicalData.get('age', None)
    sex = clinicalData.get('sex', None)

    if age is not None:
        height = clinicalData.get('height', None)
        weight = clinicalData.get('weight', None)
        
        if height is None or weight is None:
            print('Height and weight not given. Using BMI table.')
            if sex == 0:
                gender = 'male'
                table = read_table_bmi('parameters/bmi_male_ger21.txt')
            elif sex == 1:
                gender = 'female'
                table = read_table_bmi('parameters/bmi_female_ger21.txt')
            else:
                print("Unknown sex")
                table = []
            
            if table:
                height, weight = get_height_weight(age, table)
                if height is not None and weight is not None:
                    clinicalData['height'] = height
                    clinicalData['weight'] = weight
                else:
                    print("Could not find height and weight for the given age")

            print(f"{age} years old {gender} patient: {height} cm and {weight} kg.")

        else:
            if sex == 0:
                gender = 'male'
            elif sex == 1:
                gender = 'female'

            print(f"{age} years old {gender} patient: {height} cm and {weight} kg.")
    else:
        print("Age not found in clinicalData")

    return clinicalData

######### END GLOBAL ################################
######### CANNULAE ########
def identify_params_can(canType, diameter):

    """ Loads the cannulae parameters depending on the type stored in canType and the model in canModel.

    Args:
        - canType: string -> arterial or venous
        - diameter: int -> diameter of cannula

    Returns: 
        pytree -> paramsCan containing parameters and diameter of cannula.
    """

    availableCannulae = {'arterial': {15: 23, 17: 23, 19: 23, 21: 23, 23: 23}, 'venous': {19: 38, 21: 55, 23: 55, 25: 55, 29: 55}}

    if diameter not in np.array(list(availableCannulae[canType].keys())):
        sys.exit("Diameter " + str(diameter) + " not available. Check getinge data sheet for HLS cannulae again.")
        
    paramsCanRaw = np.genfromtxt(file_manager.find_path('cannula'), dtype=[('canType', 'U8'),('coefs',float,4)])
    paramsCan={}

    paramsCan.update({'diameter': diameter})
    paramsCan.update({'length': availableCannulae[canType][diameter]})

    for i in range(len(paramsCanRaw['canType'])):
        if paramsCanRaw['canType'][i]==canType:
            paramsCan.update({'params': paramsCanRaw['coefs'][i]})

    return paramsCan

######### END CANNULAE #####

######### EC CIRCUITS ########
def convert_index_to_access(index):

    """ Convert an index (integer) into an access name thanks to the access_list file.

    Args:
        - index: integer -> the index to convert

    Returns: 
        string -> name of the compartment associate with the index.
    """

    compartmentDataRaw = np.genfromtxt(file_manager.find_path('accesslist'), dtype=[('compartment','U20'), ('index',int)])
    compartmentData = {k:d for k,d in zip(compartmentDataRaw['index'], compartmentDataRaw['compartment'])}

    index=int(round(index,0))

    return compartmentData[index]
     
def load_ECMO(clinicalData, paramsModel):

    """ Loads the extracorporeal circuit ECMO properties for the model.

    Args:
        - clinicalData: dict() -> the clinical datas of the patient
        - paramsModel: dict() -> parameters of the lpm

    Returns: 
        dict() -> All the informations about the ECMO circuit

        Format: {'status': 1, 

                'pump': {'centrifugal': [params], 'rpm': rpm},

                'cannula': {'drain': [paramsDrainCan], 'return': [paramsReturnCan]}, #TODO: CHANGE HEADER HERE

                'access': {'drain': {'drainAccess': 1}, 'return': {'returnAccess': 1}}
                }.
    """

    ecmo = {}

    # Status
    activationStatus=clinicalData['ECMOactive']
    ecmo.update({'status': activationStatus})

    if activationStatus == 0:
        print("ECMO OFF!")
    else:
        print("ECMO ON!")   

    # Pump
    rpm = clinicalData['ECMOrpm']
    pump = clinicalData['ECMOpump']
    
    newPumpCoef = np.genfromtxt(file_manager.find_path('pump'), dtype=[('pumpname','U20'), ('coefs',float, 8)])
    newPumpCoef = {k:d for k,d in zip(newPumpCoef['pumpname'], newPumpCoef['coefs'])}
    
    # DP3
    if pump == 0:
        pumpParams=newPumpCoef['DP3']

    # Rotaflow    
    elif pump == 1:
        pumpParams=newPumpCoef['Rotaflow']

    else:
        sys.exit("Unknown Pump Type # {} given. Check the pump type in patient data again.".format(int(pump)))

    ecmo.update({'pump': {'centrifugal': pumpParams}})
    ecmo['pump'].update({'rpm': rpm})
    
    # Oxygenator
    if 'ECMOoxy' in clinicalData:
        oxy = clinicalData['ECMOoxy']

        OxyCoef = np.genfromtxt(file_manager.find_path('oxy'), dtype=[('oxytype','U20'), ('coef', 'f8')])
        OxyCoef = {k:d for k,d in zip(OxyCoef['oxytype'], OxyCoef['coef'])}

        # Quadrox-i Adult
        if oxy == 0:
            oxyParams = {'Recmooxy': OxyCoef['Quadroxiadult']}

        # Quadrox-i Small Adult    
        elif oxy == 1:
            oxyParams = {'Recmooxy': OxyCoef['Quadroxismalladult']}
        
        # Nautilus MC3
        elif oxy == 2:
            oxyParams = {'Recmooxy': OxyCoef['NautilusMC3']}

        else:
            sys.exit("Unknown Oxy Type # {} given. Check the oxy type in patient data again.".format(int(oxy)))

        paramsModel.update(oxyParams)

    else:
        warnings.warn(f'No oxygenator type specified. Default Quadrox i-Adult used now!')

    # Cannula
    ecmoType=clinicalData['ECMOtype']

    if ecmoType == 1:
        # VV ECMO
        drainCanParams=identify_params_can('venous', clinicalData['ECMOdrainD'])
        returnCanParams=identify_params_can('venous', clinicalData['ECMOreturnD'])

    if ecmoType == 2:
        # VA ECMO
        drainCanParams=identify_params_can('venous', clinicalData['ECMOdrainD'])
        returnCanParams=identify_params_can('arterial', clinicalData['ECMOreturnD'])

    ecmo.update({'cannula': {'drain': drainCanParams,
                             'return': returnCanParams}})

    # Access
    drainAccessIndex=clinicalData['ECMOdrainAccess']
    returnAccessIndex=clinicalData['ECMOreturnAccess']
    ecmo.update({'access': {'drain': {str(convert_index_to_access(drainAccessIndex)): 1},
                            'return': {str(convert_index_to_access(returnAccessIndex)): 1}}})

    return ecmo, paramsModel

def load_LVAD(clinicalData):

    lvad={}
    
    # Get Status
    activationStatus=clinicalData['LVADactive']
    lvad.update({'status': activationStatus})

    if activationStatus == 0:
        print("LVAD OFF!")
    else:
        print("LVAD ON!") 

    # Get Pump
    lvad.update({'rpm': clinicalData['LVADrpm']})
    pump=int(clinicalData['LVADpump'])
    
    readAllPumps= np.genfromtxt(file_manager.find_path('pump'), dtype=[('pumpname','U20'), ('coefs',float, 8)])
    readSpecificPump = readAllPumps[pump]
    lvad.update({readSpecificPump[0]: 999})
    lvad.update({'coeff': {'a':  readSpecificPump[1][0], 'R1': readSpecificPump[1][1] * (60/1E3), 'R2': readSpecificPump[1][2]* (60*60/1E6), 
    'Rrec': readSpecificPump[1][3] * (60*60/1E6), 'kinf': readSpecificPump[1][4] * (1E3/60), 'L': readSpecificPump[1][5] / 1E3, 
    'Rper': readSpecificPump[1][6] * (60*60/1E6), 'Lper': readSpecificPump[1][7] / 1E3}})
    
    # Access
    drainAccessIndex=clinicalData['LVADdrainAccess']
    returnAccessIndex=clinicalData['LVADreturnAccess']
    lvad.update({'access': {'drain': {str(convert_index_to_access(drainAccessIndex)): 1},
                            'return': {str(convert_index_to_access(returnAccessIndex)): 1}}})
    
    return lvad

def load_CRRT(clinicalData):

    """ Loads the extracorporeal circuit CRRT properties for the model.

    Args:
        - clinicalData: dict() -> the clinical datas of the patient

    Returns: 
        Dict() -> All the informations about the CRRT circuit

        Format: {'status': 1, 

                'pump': {'roller': flow (mL/s)},

                'cannula': {'drain': [paramsDrainCan], 'return': [paramsReturnCan]},

                'access': {'drain': {'drainAccess': 1}, 'return': {'returnAccess': 1}}
                }.
    """

    crrt={}
 
    # Status
    activationStatus=clinicalData['CRRTactive']
    crrt.update({'status': activationStatus})

    if activationStatus == 0:
        print("CRRT OFF!")
    else:
        print("CRRT ON!") 

    # Pump
    rpm=clinicalData['CRRTflow']
    crrt.update({'pump': {'roller': int(rpm)/60}})

    # Access
    drainAccessIndex=clinicalData['CRRTdrainAccess']
    returnAccessIndex=clinicalData['CRRTreturnAccess']
    crrt.update({'access': {'drain': {str(convert_index_to_access(drainAccessIndex)): 1},
                            'return': {str(convert_index_to_access(returnAccessIndex)): 1}}})

    return crrt

######### END EC CIRCUITS ######
def parseRunTimeVariables():

    runtimeTree = {}
    dictData = dict(np.genfromtxt('runtime.txt',dtype=str))

    print("\n-------------------------")
    print("-------------------------")

    if 'duration' in dictData.keys():
        runtimeTree['duration'] = int(dictData['duration'])
        print("Simulation time: ", dictData['duration'], "s")
    else:
        runtimeTree['duration'] = 20.0
        print("No simulation time given in runtime variables input file. Default set: ", runtimeTree['duration'], "s")

    if 'step' in dictData.keys():
        runtimeTree['step'] = float(dictData['step'])
        print("Time step size: ", dictData['step'], "s")
    else:
        runtimeTree['step'] = 1E-4
        print("No time step size given in runtime variables input file. Default set: ", runtimeTree['step'], "s")

    if 'runType' in dictData.keys():
        if dictData['runType'] in ['p' ,'f', 's']:
            runtimeTree['runType'] = dictData['runType']
            print("runType: ",dictData['runType'])
        else:
            sys.exit("Unknown RunType given in runtime variables input file.")
    else:
        sys.exit("No RunType given in runtime variables input file.")

    if runtimeTree['runType'] == 's':

        if 'sensitivityInput' in dictData.keys():
            runtimeTree['sensitivityInput'] = dictData['sensitivityInput']
            print("Sensitivity input file: ", dictData['sensitivityInput'])
        else:
            sys.exit("No Input File for Sensitivity Analysis Specified!")

        if 'sampleSize' in dictData.keys():
            runtimeTree['sampleSize'] = int(dictData['sampleSize'])
            print("Sample size for GSA: ", runtimeTree['sampleSize'])
        else:
            runtimeTree['sampleSize'] = 2**8
            print("No sample size given. Set to default ", runtimeTree['sampleSize'])
        
        if 'checkConvergenceGSA' in dictData.keys():

            if int(dictData['checkConvergenceGSA']) == 1:
                runtimeTree['checkConvergenceGSA'] = True
            else:
                runtimeTree['checkConvergenceGSA'] = False
        else:
            runtimeTree['checkConvergenceGSA'] = False

        if 'includeECLSGSA' in dictData.keys():

            if int(dictData['includeECLSGSA']) == 1:
                runtimeTree['includeECLSGSA'] = True
            else:
                runtimeTree['includeECLSGSA'] = False
        else:
            runtimeTree['includeECLSGSA'] = False

        if 'perturbation' in dictData.keys():
            runtimeTree['perturbation'] = int(dictData['perturbation'])
            print("Perturbation for GSA: ", runtimeTree['perturbation'], " %")
        else:
            runtimeTree['perturbation'] = 10
            print("No perturbation given. Set to default ", runtimeTree['perturbation'], " %")

        if runtimeTree['checkConvergenceGSA']: print("\n !! GSA will be executed on different sample sizes to check convergence !!\n")

    if runtimeTree['runType'] == 'f':
        if 'exploreParameterSpace' in dictData.keys():

            if int(dictData['exploreParameterSpace']) == 1:
                runtimeTree['exploreParameterSpace'] = True
            else:
                runtimeTree['exploreParameterSpace'] = False
        else:
            runtimeTree['exploreParameterSpace'] = False
            
    if 'patientId' in dictData.keys():
        runtimeTree['patientId'] = int(dictData['patientId'])
        print("patientId: ", dictData['patientId'])
    else:
        sys.exit("No PatientId given in runtime variables input file.")
    
    if 'saveSolutionToCSV' in dictData.keys():
        runtimeTree['saveSolutionToCSV'] = int(dictData['saveSolutionToCSV'])
    else:
        runtimeTree['saveSolutionToCSV'] = 0
    print("saveSolutionToCSV: ", runtimeTree['saveSolutionToCSV'])   

    if 'patientFitted' in dictData.keys():
        patientFitted = int(dictData['patientFitted'])

        if patientFitted == 1:
            runtimeTree['patientFitted'] = True

        else:
            runtimeTree['patientFitted'] = False
    else:
        runtimeTree['patientFitted'] = False
    print("patientFitted: ", runtimeTree['patientFitted']) 

    if 'analyzeECMObehaviour' in dictData.keys():
        runtimeTree['analyzeECMObehaviour'] = dictData['analyzeECMObehaviour'].astype(int)
    else:
        runtimeTree['analyzeECMObehaviour'] = 0
    print("analyzeECMOBehavior: ", dictData['analyzeECMObehaviour'])
    
    print("-------------------------")
    print("-------------------------\n")

    return runtimeTree   