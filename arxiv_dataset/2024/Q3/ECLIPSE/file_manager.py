"""

This file manages the paths of the model.

"""

parameterFolder = 'Parameters/'
sensitivityFolder = 'Sensitivity/' # for the sensitivity_input.txt
optimizationFolder = 'Optimization/' # for the parameter_opt.txt
patientDatasFolder = 'Patients/'
accessListFolder = 'Access/'

patientId = ''

def find_path(entity, filePath=None):
    
    """ Returns the path of the file containing datas of the entity.

    Args:
        - entity: string() -> 'cannula', 'model', 'volumes', 'pump', 'patient', 'sensitivity', 'optimization', 'accesslist'.
        - filePath: string() -> path of the file that is being read.

    Returns:
        - string() -> path of the file.
    """
    
    match entity:
        case 'cannula':
            return parameterFolder+'parameter_cannula.txt'
        case 'model':
            return parameterFolder+'parameter_model.txt'
        case 'pump':
            return parameterFolder+'parameter_pump.txt'
        case 'oxy':
            return parameterFolder+'parameter_oxy.txt'
        case 'pressures':
            return parameterFolder+'initial_pressures.txt'
        case 'ratiovolumes':
            return parameterFolder+'model_CVS_init_volumes_ratios.txt'
        case 'volumes':
            return parameterFolder+'model_ECLS_init_volumes.txt'
        case 'patient':
            return patientDatasFolder+'patient_' + patientId + '_data.txt'
        case 'sensitivity':
            return sensitivityFolder + filePath + '.txt'
        case 'optimization':
            return optimizationFolder+'parameter_opt.txt'
        case 'accesslist':
            return accessListFolder+'access_list.txt'