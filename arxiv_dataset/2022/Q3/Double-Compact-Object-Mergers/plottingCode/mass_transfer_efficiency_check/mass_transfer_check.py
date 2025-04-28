import h5py as h5
import numpy as np


def setCOMPASDCOmask(fdata='None', types='BBH', withinHubbleTime=True, optimistic=False):
        #We do not want all the formation channels just the ones that form BBHs
        fDCO    = fdata['doubleCompactObjects']
        if types == 'BBH':
            maskTypes = (fDCO['stellarType1'][...].squeeze() == 14) &\
                        (fDCO['stellarType2'][...].squeeze() == 14)
        elif types == 'BNS':
            maskTypes = (fDCO['stellarType1'][...].squeeze() == 13) &\
                        (fDCO['stellarType2'][...].squeeze() == 13)
        elif types == 'BHNS':
            maskTypes = ((fDCO['stellarType1'][...].squeeze() == 14) &\
                        (fDCO['stellarType2'][...].squeeze() == 13)) |\
                        ((fDCO['stellarType1'][...].squeeze() == 13) &\
                        (fDCO['stellarType2'][...].squeeze() == 14))
        elif types == 'ALL':
            maskTypes = ((fDCO['stellarType1'][...].squeeze() == 14) |\
                        (fDCO['stellarType1'][...].squeeze() == 13)) &\
                        ((fDCO['stellarType2'][...].squeeze() == 13) |\
                        (fDCO['stellarType2'][...].squeeze() == 14))
        else:
            raise ValueError('types=%s not in BBH, BNS, BHNS' %(types))         
        if withinHubbleTime == True:
            maskHubble = (fDCO['mergesInHubbleTimeFlag'][...].squeeze()==True)
        else:
            #Array where all are true
            maskHubble = np.ones(len(fDCO['mergesInHubbleTimeFlag'][...].squeeze()), dtype=bool)
                          
        if optimistic == True:
            #we do not care about the optimistic flag (both False and True allowed)
            #Array where all are true
            maskOptimistic = np.ones(len(fDCO['optimisticCEFlag'][...].squeeze()), dtype=bool)
        else:
            #optimistic scenario not allowed (pessimistic) hence the flag must be false
            #This removes systems with CEE from HG donors (no core envelope separation)
            maskOptimistic = fDCO['optimisticCEFlag'][...].squeeze() == False
                          
        #we never want in first timestep after CEE, because 
        #we define it as a system that should not have survived the CEE
        maskNoRLOFafterCEE =  (fDCO['RLOFSecondaryAfterCEE'][...].squeeze()==False)
                          

        DCOmask = maskTypes & maskHubble & maskOptimistic & maskNoRLOFafterCEE

        return DCOmask


def print_sum_and_flag(keys=['systems'], DCOkeys=['M1', 'M2'], which='B'):
    # we are going to check with successfull mergers in B: 

    path_B = '/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/massTransferEfficiencyFixed_0_25/COMPASOutput.h5'
    fdata_B = h5.File(path_B)
    DCOmaskBHNS = setCOMPASDCOmask(fdata=fdata_B, types='BHNS', withinHubbleTime=True, optimistic=False) 
    BHNS_seeds_B = fdata_B['doubleCompactObjects']['seed'][...].squeeze()[DCOmaskBHNS]
    # metallicities = fdata_B['doubleCompactObjects']['Metallicity1'][...].squeeze()
    print()
    print('------')
    print(which)

    if which=='B':
        path = '/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/massTransferEfficiencyFixed_0_25/COMPASOutput.h5' # change this line!    
        
    elif which=='C':
        path = '/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/massTransferEfficiencyFixed_0_5/COMPASOutput.h5' # change this line! 

    elif which=='D':
        path = '/n/holystore01/LABS/berger_lab/Lab/fbroekgaarden/DATA/all_dco_legacy_CEbug_fix/massTransferEfficiencyFixed_0_75/COMPASOutput.h5' 
    
    fewseeds = np.asarray(BHNS_seeds_B)

    fdata = h5.File(path)
    mask_BHNS = np.in1d(fdata['systems']['SEED'][...].squeeze(), fewseeds) 
    print('normalization =', float(np.sum(mask_BHNS))/(

        len(BHNS_seeds_B)))
    
    
    for key in keys:
        fraction = float(np.sum(fdata['systems'][key][...].squeeze()[mask_BHNS]==1))/(len(BHNS_seeds_B)
        print(key, fraction, fdata['systems'][key][...].squeeze()[mask_BHNS])
        
        
        
    mask_BHNS = np.in1d(fdata['doubleCompactObjects']['seed'][...].squeeze(), fewseeds) 
    
 

    for key in DCOkeys:
        fraction = float(np.sum(fdata['doubleCompactObjects'][key][...].squeeze()[mask_BHNS]))/(len(BHNS_seeds_B))
        print(key,  fraction, fdata['doubleCompactObjects'][key][...].squeeze()[mask_BHNS])



    DCOmaskBHNS = setCOMPASDCOmask(fdata=fdata, types='BHNS', withinHubbleTime=True, optimistic=False) 
    DCOmaskBHBH = setCOMPASDCOmask(fdata=fdata, types='BBH',  withinHubbleTime=True, optimistic=False)
    DCOmaskBNS  = setCOMPASDCOmask(fdata=fdata, types='BNS',  withinHubbleTime=True, optimistic=False)


    DCOs = ['BHNS', 'BBH', 'BNS']
    for i, mask in enumerate([DCOmaskBHNS, DCOmaskBHBH, DCOmaskBNS]):


        mask_DCO =  np.in1d(fdata['doubleCompactObjects']['seed'][...].squeeze()[mask], fewseeds) 

        fraction = float(np.sum(mask_DCO))/len(BHNS_seeds_B)
        print(DCOs[i],  fraction)


print_sum_and_flag(keys=['disbound', 'stellar_merger'], DCOkeys=['mergesInHubbleTimeFlag'], which='B')
print_sum_and_flag(keys=['disbound', 'stellar_merger'], DCOkeys=['mergesInHubbleTimeFlag'], which='C')
print_sum_and_flag(keys=['disbound', 'stellar_merger'], DCOkeys=['mergesInHubbleTimeFlag'], which='D')






