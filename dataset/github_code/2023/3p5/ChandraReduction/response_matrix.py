import numpy as np
from astropy.io import fits

def construct_response(fits_file_dir, fits_arf_dir = None, min_val = 1.e-6):
    '''
    Returns
    -------
    cin_min, cin_max, cout_min, cout_max, det_res
    '''
    
    in_header = 1
    out_header = 2

    with fits.open(fits_file_dir) as rmf:

        # Calculate the detector response, using the arf and rmf

        # Get the input and output energy arrays           
        cin_min = rmf[in_header].data['ENERG_LO'] # [keV]
        cin_max = rmf[in_header].data['ENERG_HI'] # [keV]

        cout_min = rmf[out_header].data['E_MIN'] # [keV]
        cout_max = rmf[out_header].data['E_MAX'] # [keV]
        
        cout_de = cout_max - cout_min # [keV]
        
    with fits.open(fits_arf_dir) as arf:
        effA = arf[1].data['SPECRESP']
        
    
    with fits.open(fits_file_dir) as rmf:
            
        det_res = np.zeros((len(cout_min),len(cin_min)))

        for i in range(len(cin_min)):

            # Reconstruct sparse matrix
            det_col = np.zeros(len(cout_min))
            jtot = 0
            for j in range(rmf[in_header].data['N_GRP'][i]): # Number of groups
                fc = rmf[in_header].data['F_CHAN'][i][j]-1 # First channel but it's 1-indexed
                sl = rmf[in_header].data['N_CHAN'][i][j] # Channels in this group
                
#                 print(jtot, sl)
                det_col[fc:fc+sl] = rmf[in_header].data['MATRIX'][i][jtot:jtot+sl]
                jtot += sl
        
            tocut = np.where(det_col < min_val)
            det_col[tocut] = 0.
            
            det_res[:,i] = det_col * effA[i]

    return cin_min, cin_max, cout_min, cout_max, det_res
