"""
Author: Renato Durrer
Created: 25.03.2019

File for measurement functions.

"""
from Labber import ScriptTools, LogFile
import numpy as np
import time
import os
from scipy.signal import argrelmax


def calibrate_QPC(PG1, PG2, V_QPC, QPC_C, Measurement, path_out, std=None,
                  mean=0, n_pts=71, pinch_off=-1.35, start=-0.85, wait=False):
    """
    Recalibrates QPC. It is not checked whether or not QPC signal is pinched off at V_QPC['value']

    Parameters
    ----------
    PG1 : dict,
        {'name': str, 'value': float}
        name and value of the PG1 for which QPC gets calibrated

    PG2 :  dict,
        {'name': str, 'value': float}
        name and value of the PG2 for which QPC gets calibrated

    V_QPC : dict,
        {'name': str, 'pinch_off': float}
        name: Name of QPC gate
        pinch_off: minimum voltage for QPC gate

    QPC_C : str,
        Name of QPC log channel

    Measurement : ScriptTools.MeasurementObject
        Measurement object with which measurements are performed.

    path_out : os.path object
        path pointing to the output file of the measurements

    std : float, optional
          default: None
          standard deviation of QPC signal

    mean : float, optional
           default: 0
           offset of QPC signal

    n_pts : int, optional
            default: 71
            Number of points with which the QPC calibration measurement is performed

    pinch_off : float, optional
                default: -1.25
                Voltage for which the QPC is pinched off (given the gate
                voltage configuration defined in the measurement object)

    start : float, optional
            default: -0.8
            Starting point for QPC calibration measurement

    wait : bool, optional
           default: False
           If True, after calibration program stops for 100s
           in order to let the system relax.

    Returns
    -------
    float,
        operation point of the QPC
    """
    print('\nQPC Calibration in Progress...')
    V_QPC['pinch_off'] = pinch_off

    # fix the Plunger Gate voltages
    Measurement.updateValue(PG1['name'], PG1['value'], itemType='SINGLE')
    Measurement.updateValue(PG2['name'], PG2['value'], itemType='SINGLE')
    Measurement.updateValue(V_QPC['name'], V_QPC['pinch_off'], itemType='STOP')
    # If standard deviation not known, measure it
    if std is None:
        Measurement.updateValue(V_QPC['name'], V_QPC['pinch_off'], itemType='START')
        Measurement.updateValue(V_QPC['name'], 51, itemType='N_PTS')
        Measurement.performMeasurement(return_data=False)
        file = LogFile(path_out)
        I_pinch_off = file.getData(QPC_C)[0]
        std = np.sqrt(np.var(I_pinch_off))
        mean = np.mean(I_pinch_off)
        print('std: {}\n'
              'mean: {}'.format(std, mean))

    # define QPC Gate sweeping
    Measurement.updateValue(V_QPC['name'], start, itemType='START')
    Measurement.updateValue(V_QPC['name'], n_pts, itemType='N_PTS')
    Measurement.performMeasurement(return_data=False)

    # read the file and calculate gradient
    file = LogFile(path_out)
    I_QPC = file.getData(QPC_C)[0]
    V_QPC['msms'] = file.getData(V_QPC['name'])[0]
    gradient = -1 * np.gradient(I_QPC)
    # get the maximum
    maxs = argrelmax(gradient, order=4)
    loc = -1
    # check wheter maximas are actually relevant
    for k in range(len(maxs[0])):
        if I_QPC[maxs[0][-(k + 1)]] > 15 * std + mean:
            loc = maxs[0][-(k + 1)]
            break
    # if no maximum relevant, raise an exception
    if loc == -1:
        raise Exception('Calibration failed, couldnt find the sweet spot.')
    sweet_spot = V_QPC['msms'][loc]

    # update QPC Gate Voltage
    Measurement.updateValue(V_QPC['name'], sweet_spot, itemType='SINGLE')
    print('QPC Calibration Sucessful.\n'
          'Sweet Spot found at: {}V\n'.format(sweet_spot))

    if wait:
        (path, file) = os.path.split(path_out)
        Measurement.setOutputFile(os.path.join(path , 'set_gate.hdf5'))
        Measurement.performMeasurement(return_data=False)
        Measurement.setOutputFile(path_out)

        print('Waiting 160s for QPC to relax.')
        time.sleep(160)

    return sweet_spot


def measure(PG1, PG2, Measurement, path, gate_names=None, calibrate=False, config=None,
            return_qpc_voltage=False, wait=False):
    """
    Performs measurements given a gate configuration config and two gates (PG1 & PG2) to sweep.

    Parameters
    ----------
    PG1 : dict, {'name': str, 'mean': float, 'start': float, 'stop': float}
        dict containing information for Plunger Gate 1

    PG2: dict, {'name': str, 'mean': float, 'start': float, 'stop': float}
        dict containing information for Plunger Gate 2

    Measurement : ScriptTools.MeasurementObject() obejct
        Measurement Object used for the measurement

    path : os.path object
        path to the output file (without the .hdf5 ending)

    calibrate : bool, optional
                default: False
                If True, the QPC is calibrated before the measurement.

    config : dict, optional
             {'Gate_Name': operation_voltage (float),...}
             dict containing the set of gate voltages

    return_qpc_voltage : bool, optional
                         If True and calibrate True as well, then
                         found calibration value for QPC Gate is returned
    Returns
    -------
    void, if return_qpc_voltage == False
    float - QPC calibration point, if return_qpc_voltage == True
    """
    # Set new configuration
    sweet_spot = None
    if config is not None:
        for name, value in config.items():
            Measurement.updateValue(name, value, itemType='SINGLE')
    if calibrate:
        if gate_names == None:
            raise Exception('You need to specify the names for the gates'
                            'and pass it as an argument gate_names. It must include'
                            'the names of QPC_G and I_QPC.')
        # Define configuration for QPC Calibration
        PG1_cal = {'name': PG1['name'], 'value': PG1['mean']}
        PG2_cal = {'name': PG2['name'], 'value': PG2['mean']}
        V_QPC = {'name': gate_names['QPC_G']}
        I_QPC = gate_names['I_QPC']
        # Calibrate QPC
        Measurement.setOutputFile(path + '_cal.hdf5')
        sweet_spot = calibrate_QPC(PG1_cal, PG2_cal, V_QPC, I_QPC, Measurement,
                                   path + '_cal.hdf5', std=1e-12, mean=-5e-12, wait=wait)
        Measurement.updateValue(V_QPC['name'], sweet_spot, itemType='SINGLE')

    # reset the Plunger Gate sweep ranges
    Measurement.updateValue(PG1['name'], PG1['start'], itemType='START')
    Measurement.updateValue(PG1['name'], PG1['stop'], itemType='STOP')
    if 'n_pts' in PG1.keys():
        Measurement.updateValue(PG1['name'], PG1['n_pts'], itemType='N_PTS')
    elif 'step' in PG1.keys():
        Measurement.updateValue(PG1['name'], PG1['step'], itemType='STEP')
    else:
        raise Exception('You need to either define n_pts or step for PG1.')

    Measurement.updateValue(PG2['name'], PG2['start'], itemType='START')
    Measurement.updateValue(PG2['name'], PG2['stop'], itemType='STOP')
    if 'n_pts' in PG2.keys():
        Measurement.updateValue(PG2['name'], PG2['n_pts'], itemType='N_PTS')
    elif 'step' in PG2.keys():
        Measurement.updateValue(PG2['name'], PG2['step'], itemType='STEP')
    else:
        raise Exception('You need to either define n_pts or step for PG2.')

    # Perform Measurement
    Measurement.setOutputFile(path + '.hdf5')
    Measurement.performMeasurement()

    # return QPC operating point
    if return_qpc_voltage:
        return sweet_spot
