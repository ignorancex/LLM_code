"""

This file codes the 3 pytrees classes, launch the model and create outputs for the results.

"""

import equinox
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_structure
import initialization
import lpm
import model_post_process
import file_manager

###### ------------------------------------------------------ ######
class parameter_pytree(equinox.Module):

    """ Initializes the initTree of the model (a pytree) which contains every information (time, clinicalData, parameters, cardiac cycle,
    initial volumes and the properties of the extracorporeal circuits).

    Args:
        - patientId: int -> id of the patient used to load the clinicalData of the tree.
        - patientFitted: boolean -> if it is on True, the parameters of the model are fitted to the patient 
        which has the specified id.

    Returns:
        An instance of the class (an initTree object).
    """
    timeTree: dict()
    clinicalData: dict()
    paramsModel: dict()
    cardiacCyc: dict()
    P0: dict()
    V0: dict()
    ECMO:dict()
    CRRT:dict()
    LVAD:dict()

    def __init__(self, runtimeTree, patientId=1, patientFitted=False):

        # Patient id
        file_manager.patientId=str(patientId)
        
        # Clinical data
        self.clinicalData = initialization.load_clinicalData(file_manager.find_path('patient'))
        self.clinicalData.update({'patientId': patientId})

        # Cardiac cycle
        self.cardiacCyc = initialization.load_cardiac_cyc(self.clinicalData)

        # Initialization of simulation time and time step size
        tSet = {'tmin': 0, 'tmax': runtimeTree['duration'], 'h': runtimeTree['step']}
        T = jnp.linspace(tSet['tmin'], tSet['tmax'], int(tSet['tmax']/tSet['h']))

        # Definition of time ranges for the output and plotting
        numberCycles = 2
        ub = jnp.size(T) - 1
        lb = ub - int(numberCycles*self.cardiacCyc['Tcyc']/tSet['h'])
        T = jnp.linspace(tSet['tmax']-tSet['h']*(ub-lb), tSet['tmax'], ub-lb)
        T0 = jnp.linspace(0, numberCycles*self.cardiacCyc['Tcyc'], ub-lb) # Plotting time starting from zero

        self.timeTree = {'tSet': tSet, 'T': T, 'T0': T0, 'ub': ub, 'lb': lb}

        # Initialization of patient, parameters, initVolumes and circuits
        self.paramsModel = initialization.load_params_model(file_manager.find_path('model'), patientId, patientFitted)

        # Initialize extracorporeal circuits
        self.ECMO, self.paramsModel = initialization.load_ECMO(self.clinicalData, self.paramsModel)
        self.CRRT = initialization.load_CRRT(self.clinicalData) 
        self.LVAD = initialization.load_LVAD(self.clinicalData)
    
        # Initialize volumes
        self.V0 = initialization.load_V0(self.paramsModel, self.clinicalData, self.ECMO, [file_manager.find_path('ratiovolumes'), file_manager.find_path('volumes')])
        self.P0 = initialization.load_P0(self.paramsModel, self.cardiacCyc, self.V0, file_manager.find_path('pressures'))
    
    def update(self, update):
        """ Update the parameters (paramsModel) of the initTree.

        Args:
            update: dict(), contains the new parameters

        Returns:
            An instance of the class (an initTree object) with updated parameters
        """

        keys, vals = zip(*update.items())
        return equinox.tree_at(lambda tree: [tree.paramsModel[key] for key in keys], self, vals)
    
class simPost_pytree(equinox.Module):

    """ Initializes the simPostTree of the model (a pytree) which contains every output information (flows, pressures, volumes)
    of every compartment. Post stands for post-processing.

    Args:
        X

    Returns:
        An instance of the class (a simPostTree object).
    """
    Flows: dict()
    Pressures: dict() 
    Volumes: dict()

    def __init__(self):
        
        keysFlow = ['Qao', 'Qsart', 'Qsvn', 'Qra', 'Qrv', 'Qpas', 'Qpart', 'Qpvn', 'Qla', 'Qlv',
                    'Qecmodrain', 'Qecmotudp', 'Qecmopump', 'Qecmotupo', 'Qecmooxy', 'Qecmotuor', 'Qecmoreturn',
                    'Qcrrttuin', 'Qcrrtpump', 'Qcrrttupf','Qcrrtfil', 'Qcrrttuout',
                    'Qlvadpump']
        
        keysPressures = ['Pao','Psart','Psvn', 'Pra', 'Prv', 'Ppas', 'Ppart', 'Ppvn', 'Pla', 'Plv',
                         'Pecmodrain', 'Pecmotudp', 'Pecmopump','Pecmotupo', 'Pecmooxy', 'Pecmotuor', 'Pecmoreturn', 
                         'Pcrrttuin', 'Pcrrtpump', 'Pcrrttupf','Pcrrtfil', 'Pcrrttuout',
                         'Plvadpump']
        
        keysVolumes = ['Vao', 'Vsart', 'Vsvn', 'Vra', 'Vrv', 'Vpas', 'Vpart', 'Vpvn', 'Vla', 'Vlv',
                       'Vecmodrain', 'Vecmotudp', 'Vecmopump', 'Vecmotupo', 'Vecmooxy', 'Vecmotuor', 'Vecmoreturn', 
                       'Vcrrttuin', 'Vcrrtpump', 'Vcrrttupf', 'Vcrrtfil', 'Vcrrttuout',
                       'Vlvadpump']
        
        self.Flows = dict.fromkeys(keysFlow, 0)
        self.Pressures = dict.fromkeys(keysPressures, 0)
        self.Volumes = dict.fromkeys(keysVolumes, 0)

class simComp_pytree(equinox.Module):

    """ Initializes the simCompTree of the model (a pytree) which contains redefined output information to be comparable to clinical datas.
        Comp stands for comparison.

    Args:
        X

    Returns:
        An instance of the class (a simCompTree object).
    """
    
    results: dict()

    def __init__(self):
        
        outputs = ['SP', 'DP', 'MAP', 'ESVLV', 'EDVLV', 'ESVLA', 'EDVLA',
                        'ESVRV', 'EDVRV', 'ESVRA', 'EDVRA', 
                        'CO', 'SPAP', 'DPAP', 'MPAP', 'PCWP', 'PF']
        
        self.results = dict.fromkeys(outputs, 0)

@jit
def modeling(initTree, simPostTree, simCompTree):

    """ Launches the modeling and post-process the output results.

    Args:
        - initTree: Initial parameters of the model, type: dict()
        - simPostTree: Instance of class simPost_pytree -> Contains solution keys for post processing
        - simCompTree: Instance of class simComp_pytree -> Contains comparable keys for comparison with clinicalData
        
    Returns:
        pytree, pytree -> simCompTree and simPostTree filled with values (solution + post-processing).
    """

    # Launch the model and get the results
    solutionODE= lpm.model_solve(initTree.timeTree,
                                  initTree.paramsModel, initTree.cardiacCyc, 
                                  initTree.P0, initTree.V0,
                                  initTree.ECMO, initTree.CRRT, initTree.LVAD)

    # Fills up the simPostTree with every compartment properties (pressures, volumes, flows)
    simPostTree = model_post_process.sim_post(solutionODE.ys, initTree, simPostTree)
    
    # Fills up the simCompTree to make output results comparable to clinical datas
    simCompTree = model_post_process.create_Outputs(simPostTree, simCompTree)

    return simCompTree, simPostTree