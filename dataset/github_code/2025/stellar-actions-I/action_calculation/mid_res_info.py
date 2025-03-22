#imports
import numpy as np

from functions.data_reading import read_oldstars, read_stars, read_darkmatter
from functions.coordinate_conversion import cart_to_cylind, cart_to_cylind_v


def mid_res_info(i,j,ID_list=None):
    ''' Using Chuhan's simulation filename and coordinate shift rn
    input parameters:
    test: also shifting vz by subtracting COM vz velocity
    i: snapshot number 
    j: initial stars (0), newborn stars (1), dark matter (3)
    ID_list- it could be a cohort of certain age or even the entire stars' ID list or even a single star. ID_list can be list or array.

    returns C_cyl,V_cyl,M,ID,A
    '''
    print(f'Starting snapshot {i} Myr')
    # fileName = '/scratch/jh2/ax8338/midResRunNew/simulations/snapdir_{:03d}/'.format(i, i)
    fileName = '/scratch/jh2/cz4203/final_simulation/snapshots/snapdir_{:03d}/'.format(i, i)
    if j==0:
        C, ID, M, V, A, Pot = read_oldstars(fileName)
    elif j==1:
         C, ID, M, V, A, Pot = read_stars(fileName)
    elif j==2:
        C, ID, V, M ,Pot = read_darkmatter(fileName)
    
    #apply the shift
    # shift = np.array((0,0,-68.3)) #pc
    shift = np.array((0,0,-1.79*i+62.25))
    shift_v = np.array((0,0,-1.75))
    C-=shift
    V-=shift_v
    #applying mid-resolution mask on everything before matching IDs
    if j==1:
        mid_mask = (M>20)&(M<500)
        C=C[mid_mask]
        ID=ID[mid_mask]
        M=M[mid_mask]
        V=V[mid_mask]
        A=A[mid_mask]
        Pot=Pot[mid_mask]

    C_cyl_all=cart_to_cylind(C)
    print(C_cyl_all.shape)
    ID_list=np.array(ID_list)
    print(ID_list.shape)
    print(np.isin(ID,ID_list).sum())
    #matching IDs and getting all parameters
    C_cyl=C_cyl_all[np.isin(ID,ID_list)]
    print(f'C_cyl_shape{C_cyl.shape}')
    V_cyl=cart_to_cylind_v(V[np.isin(ID,ID_list)],C[np.isin(ID,ID_list)])
    M=M[np.isin(ID,ID_list)]
    A=A[np.isin(ID,ID_list)]
    ID=ID[np.isin(ID,ID_list)]
    
    # if j==2 or (np.shape(ID_list)[0]<1e3):
    sort_mask = np.argsort(ID)
    print(f'ID_shape{ID.shape}')
    print(f'Data of the stars in ID_list read and converted to cylindrical coordinates for i={i} i.e. {i}Myr')
    
    print(f'ID_list shape {ID_list.shape}')
    print(f'Outputting C_cyl, V_cyl, M, ID,A for i={i} i.e. {i}Myr')
    return C_cyl[sort_mask],V_cyl[sort_mask],M[sort_mask],ID[sort_mask],A[sort_mask]
