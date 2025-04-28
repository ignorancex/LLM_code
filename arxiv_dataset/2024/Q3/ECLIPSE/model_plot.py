"""

This file manages the plotting after solving the model.

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

###### Plotting overview of results for comparison #######

def LV_PV_Loop(axs, Vlv, Plv):

    """ Plots the LV-PV Loop.

    Args:
        - axs: [array] -> axes for the plotting
        - Vlv: [array] -> volume of the left ventricle through time
        - Plv: [array] -> pressure of the left ventricle through time

    Return:
        X.
    """

    # Plot pV-loop
    axs[0, 0].title.set_text('PV Loops')
    axs[0, 0].plot(Vlv, Plv)
    axs[0, 0].set_title('LV PV Loop')
    axs[0, 0].set_xlabel("V [ml]")
    axs[0, 0].set_ylabel("P [mmHg]")

def Pao_Plv(axs, T, Pao, Plv, Pla, Pra, Prv):

    """ Plots the aortic and left ventricle pressure.

    Args:
        - axs: [array] -> axes for the plotting
        - T: [array] -> x axis of the plot/time 
        - Pao: [array] -> pressure of the aorta through time
        - Plv: [array] -> pressure of the left ventricle through time

    Return:
        X.
    """

    # Plot pressue of aorta and left ventricle
    axs[0, 1].title.set_text('AO and LV Pressures [mmHg]')
    axs[0, 1].plot(T, Pao, T, Plv)
    axs[0, 1].legend(['Ao','LV', 'LA', 'RA', 'RV'],loc='upper left')
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("P [mmHg]")

def Q_Pulm(axs, T, Qpa, Qpp, Qps):

    """ Plots the pulmonary flows.

    Args:
        - axs: [array] -> axes for the plotting
        - T: [array] -> x axis of the plot/time 
        - Qpa: [array] -> flow of the pulmonary artery through time
        - Qpart: [array] -> flow of pulmonary arteries through time

    Return:
        X.
    """

    # Plot flows of pulmonary arteries
    axs[1, 0].title.set_text('Pulmonary flows')
    axs[1, 0].plot(T, Qpa, T, Qpp, T, Qps)
    axs[1, 0].legend(['artery','periph. vessels', 'shunt'],loc='upper left')
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("Q [mL/s]")

def mean_Flows(axs, Qs, Vs):

    """ Plots the means flows of the model.

    Args:
        - axs: [array] -> axes for the plotting
        - Qs: dict() -> the flows of the simPostTree
        - Vs: [array] -> the volumes of the simPostTree (to calculate the EF)

    Return:
        X.
    """

    # Create bar plot with mean flows of certain compartments
    CompartmentsPlot = {'Qlv': 'CO', 'Qsart': 'sart', 'Qsvn': 'svn', 'Qra': 'ra', 'Qrv': 'rv', 'Qecmopump': 'EPump', 'Qcrrtpump': 'CPump'}
    meanFlows= {k: jnp.mean(Qs[k]*60/1000) for k in CompartmentsPlot.keys()}

    EDV=jnp.max(Vs['Vlv'])
    ESV=jnp.min(Vs['Vlv'])
    EF=str(jnp.round(100*(EDV-ESV)/EDV))

    axs[1, 1].title.set_text('Mean cardiac output')
    axs[1, 1].bar(list(CompartmentsPlot.values()), jnp.array(list(meanFlows.values())))
    axs[1, 1].text(0, 1.75, "EF: " + EF + "%", size=12, ha='center', va='center', weight='bold')
    axs[1, 1].set_ylabel("Q [l/min]")

def plot_results(initTree, simPostTree):

    """ Manages the plotting for the model. Plotting based on the ListOfPlots.txt file.

    Args:
        - initTree: [pytree] -> Initial parameters of the model
        - simPostTree: [pytree] -> Contains solution keys for post processing

    Return:
        - /
    """

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral' 
    plt.rcParams['font.size'] = 16 

    cm = 1/2.54

    # Check if folder Plots exist and if not create a new one
    if not os.path.exists('Plots'):
        os.makedirs('Plots')

    T=initTree.timeTree['T']
    paramsModel=initTree.paramsModel

    compartments=['ao', 'sart', 'svn', 'ra', 'rv', 'la', 'lv', 'pas', 'part', 'pvn']

    for c in compartments:
        plt.plot(T, simPostTree.Pressures['P'+c], label=c+' pressure')

    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [mmHg]')

    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig('Plots/AllPressures.png')

    listOfPlots = np.genfromtxt('ListOfPlots.txt',dtype=str)
    for plot in listOfPlots:
        match plot:
            case 'heartvolumes':

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                axs[0].plot(T, simPostTree.Volumes['Vlv'], label='Vlv')
                axs[0].plot(T, simPostTree.Volumes['Vla'], label='Vla')
                axs[0].legend()
                axs[0].set_title('Left Ventricle')
                axs[0].set_xlabel('Time in s')
                axs[0].set_ylabel('Volume in mL')
                axs[0].set_ylim([0, 140])

                axs[1].plot(T, simPostTree.Volumes['Vrv'], label='Vrv')
                axs[1].plot(T, simPostTree.Volumes['Vra'], label='Vra')
                axs[1].legend()
                axs[1].set_title('Right Ventricle')
                axs[1].set_xlabel('Time in s')
                axs[1].set_ylabel('Volume in mL')
                axs[1].set_ylim([0, 140])

                print("\nEDVLV: {:.2f} mL".format(jnp.max(simPostTree.Volumes['Vlv'])))
                print("ESVLV: {:.2f} mL".format(jnp.min(simPostTree.Volumes['Vlv'])))
                print("EDVRV: {:.2f} mL".format(jnp.max(simPostTree.Volumes['Vrv'])))
                print("ESVRV: {:.2f} mL".format(jnp.min(simPostTree.Volumes['Vrv'])))
                
                plt.savefig('Plots/heartvolumes.png')

            case 'Pressures':
                plt.figure()
                plt.title('CVS pressures evolution')
                compartments=['ao', 'sart', 'svn', 'ra', 'rv', 'la', 'lv', 'pas', 'part', 'pvn']
                n=1
                for c in compartments:
                    plt.plot(n*T, simPostTree.Pressures['P'+c], label=c+' pressure')
                    n=n+1
                plt.xlabel('Time in s')
                plt.ylabel('Pressure in mmHg')
                plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig('Plots/CVS_pressures.png')

                if initTree.ECMO['status'] == 1:
                    plt.figure()
                    plt.title('ECMO pressures')

                    ecmocompartments=['ecmodrain', 'ecmotudp', 'ecmopump', 'ecmooxy', 'ecmotuor', 'ecmoreturn']

                    for cecmo in ecmocompartments:
                        plt.plot(T, (simPostTree.Pressures['P'+cecmo]), label=cecmo)

                    plt.xlabel('Time in s')
                    plt.ylabel('Pressure in mmHg')
                    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                    plt.tight_layout()
                    plt.savefig('Plots/ECMO_pressure.png')

            case 'CVSpressuresMean':
                plt.figure()
                plt.title('CVS pressures Mean evolution')
                compartments=['ao', 'sart', 'svn', 'ra', 'rv', 'la', 'lv', 'pas', 'part', 'pvn']
                n=1
                for c in compartments:
                    plt.plot(n*T, jnp.mean(simPostTree.Pressures['P'+c])*jnp.ones(len(T)), label=c+' pressure')
                    n=n+1
                plt.xlabel('Time [s]')
                plt.ylabel('Pressure [mmHg]')
                plt.legend()
                plt.savefig('Plots/CVSpressuresMean.png')

            case 'pvloops':
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # LV pV Loop
                axs[0].plot(simPostTree.Volumes['Vlv'], simPostTree.Pressures['Plv'])
                axs[0].set_title('LV pV Loop')
                axs[0].set_xlabel('V in mL')
                axs[0].set_ylabel('P in mmHg')

                # RV pV Loop
                axs[1].plot(simPostTree.Volumes['Vrv'], simPostTree.Pressures['Prv'])
                axs[1].set_title('RV pV Loop')
                axs[1].set_xlabel('V in mL')
                axs[1].set_ylabel('P in mmHg')

                plt.tight_layout()
                plt.savefig('Plots/pvloops.png')

            case 'paoplv':
                plt.figure()
                plt.title('Left ventricle and aortic pressures')
                plt.plot(T, simPostTree.Pressures['Pao'], label='Aortic')
                plt.plot(T, simPostTree.Pressures['Plv'], label='Left ventricle')
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel('Pressure [mmHg]')
                plt.savefig('Plots/paoplv.png')
                
            case 'qpulm':
                # TODO: pmc not available anymore!
                plt.figure()
                plt.title('Pulmonary flows')
                plt.plot(T, simPostTree.Flows['Qpart'], label='Pulmonary artery')
                plt.plot(T, simPostTree.Flows['Qpas'], label='Pulmonary aortic sinus')
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel('Flow [mL/s]')
                plt.savefig('Plots/qpulm.png')
                
            case 'meanflows':
                plt.figure()
                CompartmentsPlot = {'Qlv': 'CO', 'Qao': 'ao', 'Qsart': 'sart', 'Qsvn': 'svn', 'Qra': 'ra', 'Qrv': 'rv', 'Qecmopump': 'EPump', 'Qcrrtpump': 'CPump'}
                meanFlows= {k: jnp.mean(simPostTree.Flows[k]*60/1000) for k in CompartmentsPlot.keys()}

                EDV=jnp.max(simPostTree.Volumes['Vlv'])
                ESV=jnp.min(simPostTree.Volumes['Vlv'])
                EF=str(jnp.round(100*(EDV-ESV)/EDV))

                plt.title('Mean cardiac output') 
                plt.bar(list(CompartmentsPlot.values()), jnp.array(list(meanFlows.values())))
                plt.text(0, 1.75, "EF: " + EF + "%", size=12, ha='center', va='center', weight='bold')
                plt.ylabel("Q [l/min]")
                plt.savefig('Plots/meanflows.png')

            case 'checkPressures':
                
                print('\nPressure right atrium should be 2: ', jnp.round(jnp.mean(simPostTree.Pressures['Pra'])))
                print('Pressure before part should be 14: ', jnp.round(jnp.mean(simPostTree.Pressures['Ppart'])))
                print('Pressure left atrium should be 5 : ', jnp.round(jnp.mean(simPostTree.Pressures['Pla'])))
                print('Pressure after left ventricle should be 100: ', jnp.round(jnp.mean(simPostTree.Pressures['Pao'])))
                print('Pressure before sart should be 30: ', jnp.round(jnp.mean(simPostTree.Pressures['Psart'])))
                print('Pressure after sart/before svn should be 10: ', jnp.round(jnp.mean(simPostTree.Pressures['Psvn'])))

            case 'LVAD':
                fig, axs = plt.subplots(2, 2,figsize=(15,15))
                axs[0, 0].title.set_text('LV-pV Loop')
                axs[0, 0].plot(simPostTree.Volumes['Vlv'], simPostTree.Pressures['Plv'])
                axs[0, 0].set_xlim(40, 150)
                axs[0, 0].set_ylim(0, 140)
                axs[0, 0].set_xlabel("Volume in mL")
                axs[0, 0].set_ylabel("Pressure in mmHg")

                axs[0, 1].plot(T, simPostTree.Pressures['Pao'], label='Aorta')
                axs[0, 1].plot(T, simPostTree.Pressures['Plv'], label='LV')
                axs[0, 1].set_ylim(0, 140)
                axs[0, 1].set_xlabel("Time in s")
                axs[0, 1].set_ylabel("Pressure in mmHg")
                axs[0, 1].legend()

                axs[1, 0].plot(T, simPostTree.Flows['Qlv']/16.67, label='LV')
                axs[1, 0].plot(T, simPostTree.Flows['Qlvadpump']/16.67, label='LVAD')
                axs[1, 0].set_xlabel("Time in s")
                axs[1, 0].set_ylabel("Flow in L/min")
                axs[1, 0].legend()

                CompartmentsPlot = {'Qlv': 'CO', 'Qlvadpump': 'LVAD'}
                meanFlows= {k: jnp.mean(simPostTree.Flows[k]*60/1000) for k in CompartmentsPlot.keys()}

                EDV=jnp.max(simPostTree.Volumes['Vlv'])
                ESV=jnp.min(simPostTree.Volumes['Vlv'])
                EF=str(jnp.round(100*(EDV-ESV)/EDV))

                axs[1, 1].title.set_text('Mean Cardiac Output')
                axs[1, 1].bar(list(CompartmentsPlot.values()), jnp.array(list(meanFlows.values())))
                axs[1, 1].text(0, 1.75, "EF: " + EF + "%", size=12, ha='center', va='center', weight='bold')
                axs[1, 1].set_ylabel("Flow in L/min")
                plt.savefig('Plots/LVAD.png')
                
            case 'Flows':
                plt.figure()

                print('\nCVS: Flow in ...')
                for c in compartments:
                    plt.plot(T, simPostTree.Flows['Q'+c]*0.06, label=c+' flow')
                    print(c, 'value =', \
                          jnp.round(jnp.mean(simPostTree.Flows['Q'+c])*0.06, decimals = 4) \
                            , 'L/min')
                    
                plt.xlabel('Time in s')
                plt.ylabel('Flow in L/min')
                plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('Plots/CVS_flows.png')

                if initTree.ECMO['status'] == 1:
                    plt.figure()

                    ecmocompartments=['ecmodrain', 'ecmotudp', 'ecmopump', 'ecmooxy', 'ecmotuor', 'ecmoreturn']

                    print('\nECMO: Flow in ...')
                    for cecmo in ecmocompartments:
                        plt.plot(T, (simPostTree.Flows['Q'+cecmo])*0.06, label=cecmo)
                        print(cecmo, 'value =', \
                              jnp.round(jnp.mean(simPostTree.Flows['Q'+cecmo])*0.06, decimals = 4) \
                                , 'L/min')

                    plt.xlabel('Time in s')
                    plt.ylabel('Flow in L/min')
                    plt.grid(True)
                    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                    plt.tight_layout()
                    plt.savefig('Plots/ECMO_flows.png')


                if initTree.CRRT['status'] == 1:
                    plt.figure()

                    compartmentscrrt=['crrttuin', 'crrttupf', 'crrtfil', 'crrttuout']
                
                    print('\nCRRT: Flow in ...')
                    for ccrrt in compartmentscrrt:
                        plt.plot(T, (simPostTree.Flows['Q'+ccrrt])*0.06, label=ccrrt+' flow')
                        print(ccrrt, 'value =', \
                              jnp.round(jnp.mean(simPostTree.Flows['Q'+ccrrt])*0.06, decimals = 4) \
                                , 'L/min')
                        
                    plt.xlabel('Time in s')
                    plt.ylabel('Flow in L/min')
                    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig('Plots/CRRT_flows.png')

                if initTree.LVAD['status'] == 1:
                    plt.figure()

                    compartmentslvad=['lvadpump']

                    print('\nLVAD: Flow in ...')
                    for clvad in compartmentslvad:
                        plt.plot(T, (simPostTree.Flows['Q'+clvad])*0.06, label=clvad+' flow')
                        print(clvad, 'value =', \
                                jnp.round(jnp.mean(simPostTree.Flows['Q'+clvad])*0.06, decimals = 4) \
                                , 'L/min')
                        
                    plt.xlabel('Time in s')
                    plt.ylabel('Flow in L/min')
                    plt.grid(True)
                    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                    plt.tight_layout()
                    
                    plt.savefig('Plots/LVAD_flows.png')

            case 'valveflows':
                plt.figure(figsize=(12*cm, 8*cm))

                compartments=['la', 'lv', 'rv', 'ra']

                print('\nValves: Flow out of ...')
                for c in compartments:
                    plt.plot(T, simPostTree.Flows['Q'+c]*0.06, label=c+' flow')
                    print(c, 'value =', \
                          jnp.round(jnp.mean(simPostTree.Flows['Q'+c])*0.06, decimals = 4), \
                                    'L/min')
                    
                plt.xlabel('Time in s')
                plt.ylabel('Flow in L/min)')
                plt.grid(True)
                plt.ylim(0, 80)
                plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig('Plots/Valve_flows.png')