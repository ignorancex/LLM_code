import numpy as np
from PyEMD import EEMD
import multiprocessing
from astropy.io import fits
from scipy.interpolate import interp1d

def __split(cube, parts):
    nz, ny, nx = cube.shape
    y_mark = []
    cubes = []
    for i in range(parts+1):
        temp = int(ny*(i/parts))
        y_mark.append(temp)  # y[0, y1, y2, y3, ..., max_y]
    for i in range(parts):
        a = y_mark[i]
        b = y_mark[i+1]
        array3d = cube[:, a:b, :]
        cubes.append(array3d)

    return cubes


def __eemd_cal(cube):
    nz, ny, nx = cube.shape
    IMFs_list = []

    eemd = EEMD()

    for i in range(ny):
        for j in range(nx):
            IMFs = eemd.eemd(cube[:,i,j])
            IMFs_list.append(IMFs)

    return IMFs_list




def eemd_decomposition(cube, num_processor=1):
    '''
    Calculate the intrinsic mode functions (IMFs) of the data using the EEMD algorithm.
    Parameters:
    cube: a 3D data cube
    num_processor: Number of processors that can be used in this calculation
    Return:
    A list of IMFs
    '''

    cube = np.nan_to_num(cube)

    cubes = __split(cube, num_processor)
    IMFs_lists = []
    process = multiprocessing.Pool(processes=num_processor)
    IMFs_lists = process.map(__eemd_cal, cubes)
    process.close()
    process.join()

    list_IMFs = []
    for i in range(num_processor):
        list_IMFs += IMFs_lists[i]

    return list_IMFs




def reconstruct_inverted_cube(list_IMFs, cube_shape, imfn=0, n_imf_min=4):
    '''
    Use the IMF of interest and invert it to reconstruct 3D data cube
    Parameters:
    list_IMFs: List of intrinsic mode functions (IMFs)
    cube_shape: the shape of the 3D data cube to be reconstructed, the shape must be the same as the data shape of decomposition
    imfn: imfn: which IMF will be used to reconstruct data cube, e.g., imfn=0 for IMF0, imfn=1 for IMF1, etc.
    n_imf_min: If the number of IMFs obtained from decomposing a signal is less than the value determined by n_imf_min, the reconstructed data at the location of the signal is replaced by 0
    Return:
    A 3D data cube
    '''

    nz, ny, nx = cube_shape

    inverted_IMFs = []
    zeros = np.zeros(nz)
    for i in range(len(list_IMFs)):
        if list_IMFs[i].shape[0] > n_imf_min:
            IMF = list_IMFs[i][imfn]

            u_x = [0, ]
            u_y = [IMF[0], ]
            for k in range(1, nz-1):
                if (np.sign(IMF[k]-IMF[k-1]) == 1) and (np.sign(IMF[k]-IMF[k+1]) == 1):
                    u_x.append(k)
                    u_y.append(IMF[k])
            u_x.append(nz-1)
            u_y.append(IMF[-1])

            u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)

            q_u = zeros
            for k in range(0, nz):
                q_u[k] = u_p(k)

            inverted_IMFs.append(q_u - IMF)
        else:
            inverted_IMFs.append(zeros)


    recons_cube = np.zeros(cube_shape)
    for i in range(len(inverted_IMFs)):
        recons_cube[:, int(i/nx), i%nx] = inverted_IMFs[i]

    return recons_cube






if __name__ == "__main__":
    file = './THOR_HI_38.3deg_C-D_ft_VGPS.fits'
    hdu = fits.open(file)
    data = hdu[0].data

    list_IMFs = eemd_decomposition(cube=data, num_processor=48)
    # np.save(file + '_list_IMFs.npy', np.array(list_IMFs, dtype=object))
    recons_cube = reconstruct_inverted_cube(list_IMFs=list_IMFs, cube_shape=data.shape, imfn=0, n_imf_min=4)

    hdu[0].data = recons_cube
    hdu.writeto(file + '_invertedIMF' + '.fits', overwrite=True)
