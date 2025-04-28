import numpy as np
from scipy.interpolate import CubicSpline,interp1d
from scipy.optimize import fixed_point
import numpy as np
from astropy.io import ascii 
#adapted from https://docs.galpy.org/en/v1.7.2/_modules/galpy/potential/AdiabaticContractionWrapperPotential.html

# Main function
def main(argv):
    print("main")
    global NUM
    sim =argv[1]
    infile = argv[2]
    table= ascii.read(infile)

    #load in variables
    ri_loaded = table['ri']
    mbi = table['mbi']
    mhi = table['mhi']
    mbf = table['mbf']
    rf_loaded = table['rbf']


    # Create spline functions
    mhi_s = CubicSpline(ri_loaded, mhi)
    mbi_s = CubicSpline(ri_loaded, mbi)
    mbf_s = CubicSpline(rf_loaded, mbf)


    sim_split = sim.split('_')

    # For Auriga
    r_200_auriga_array = {'6':211.83,'16': 241.53,'24':239.57,'27':253.81,'21':238.65,'23': 245.27}
    rv = r_200_auriga_array[sim_split[1]]
    rv_index = np.argmin(np.abs(ri_loaded-rv))

    #A, W parameters
    A, w= 0.61, 0.5 #Auriga 
    # A,w = 0.36,0.23 #FIRE
    #A,w =0.4,0.45 #TNG-50
    # A,w = 0.25,0.45 #vintergatan


    # func_r_mean= lambda ri: A*rv*(ri/rv)**w
    def func_r_mean(ri):
        res = A*rv*(ri/rv)**w
        return res

    index_min = np.argmin(np.abs(ri_loaded-0.05))
    func_r_contract= lambda rf: ri_loaded[index_min:rv_index]*(mbi_s(func_r_mean(ri_loaded[index_min:rv_index]))+mhi_s(func_r_mean(ri_loaded[index_min:rv_index])))\
        /((mhi_s(func_r_mean(ri_loaded[index_min:rv_index]))+mbf_s(func_r_mean(rf))))
    try:
        rf= fixed_point(func_r_contract,ri_loaded[index_min:rv_index])
    except RuntimeWarning as e:
        print("Warning: RuntimeWarning - invalid value encountered in power.")
        print("Specific details:", e)
    

    func_M_DM = interp1d(rf,mhi[index_min:rv_index],bounds_error=False,fill_value=(mhi[index_min],mhi[rv_index]))
        

    #Save solution
    np.save('/work2/09036/abdelh/frontera/contra/rf_arrays/rf_'+sim+'_galpy_A_W_more_bins_500.npy',np.array(rf))
    np.save('/work2/09036/abdelh/frontera/contra/mhf_arrays/mhf_'+sim+'_galpy_A_W_more_bins_500.npy',np.array(func_M_DM(rf)))
 
if __name__ == "__main__":
    import sys
    main(sys.argv)

