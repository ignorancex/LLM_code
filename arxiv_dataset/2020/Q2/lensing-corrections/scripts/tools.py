import numpy as np
import os

package_path = os.path.dirname(os.path.abspath(__file__))+'/'
dpath = package_path + '../PostBornEma/'


def loadfftlogdata():
    data = np.loadtxt(dpath+'FFT_k-1Pk.dat',skiprows=2)

    #----------------------#
    n         = data[:,0]
    Re_c_n    = data[:,1]
    Im_c_n    = data[:,2]
    Re_nu_n   = data[:,3]
    Im_nu_n   = data[:,4]
    #----------------------#

    c_n  = Re_c_n  + 1j * Im_c_n
    nu_n = Re_nu_n + 1j * Im_nu_n


    # for cl psipsi
    data =  np.load(dpath+'Il_nu_t_new1-ell-nn-tt.npy')
    ell       = data[:,0]
    nn        = data[:,1]
    tt        = data[:,2]
    #----------------------#

    data =  np.load(dpath+'Il_nu_t_new1-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_0 = Re_I + 1j * Im_I


    #for cl psi delta
    data =  np.load(dpath+'Il_nu_t_nd2-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_2 = Re_I + 1j * Im_I


    # for cl delta delta
    data =  np.load(dpath+'Il_nu_t_nd4-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_4 = Re_I + 1j * Im_I

    del Re_I, Im_I, data


    #shrink
    ell_ = np.unique(ell)
    nu_n_= np.unique(nu_n) 
    t_   = np.unique(tt)



    #make 3D arrays 
    ell = ell.reshape(ell_.size, nu_n_.size, t_.size)
    nn  = nn.reshape(ell_.size, nu_n_.size, t_.size)
    tt  = tt.reshape(ell_.size, nu_n_.size, t_.size)
    I_0 = I_0.reshape(ell_.size, nu_n_.size, t_.size)
    I_2 = I_2.reshape(ell_.size, nu_n_.size, t_.size)
    I_4 = I_4.reshape(ell_.size, nu_n_.size, t_.size)

    cn2 = c_n.copy()
    cn2[:-1] *=2

    #insert a new axis for r and multiply with c_n 
    I0_lcrt = np.expand_dims(I_0, 2)*cn2.reshape(1, -1, 1, 1)
    I2_lcrt = np.expand_dims(I_2, 2)*cn2.reshape(1, -1, 1, 1)
    I4_lcrt = np.expand_dims(I_4, 2)*cn2.reshape(1, -1, 1, 1)

    #change t and c
    I0_ltrc = np.swapaxes(I0_lcrt, 1, 3)
    I2_ltrc = np.swapaxes(I2_lcrt, 1, 3)
    I4_ltrc = np.swapaxes(I4_lcrt, 1, 3)


    del I_0, I_2, I_4

    return ell_, t_, nu_n_, I0_ltrc, I2_ltrc, I4_ltrc

def loadggwts():
    #get Gaussian quadrature weights
    data = np.loadtxt(dpath+'GG_weights.dat',skiprows=2)
    t1   = data[:,0]
    w1   = data[:,1]
    return t1, w1




def loadfftlogdata_highell():
    data = np.loadtxt(dpath+'FFT_k-1Pk.dat',skiprows=2)

    #----------------------#
    n         = data[:,0]
    Re_c_n    = data[:,1]
    Im_c_n    = data[:,2]
    Re_nu_n   = data[:,3]
    Im_nu_n   = data[:,4]
    #----------------------#

    c_n  = Re_c_n  + 1j * Im_c_n
    nu_n = Re_nu_n + 1j * Im_nu_n


    # for cl psipsi
    data =  np.load(dpath+'Il_nu_t_nd0_tmin0p96_lmin1000_lmax5000-ell-nn-tt.npy')
    ell       = data[:,0]
    nn        = data[:,1]
    tt        = data[:,2]
    #----------------------#

    data =  np.load(dpath+'Il_nu_t_nd0_tmin0p96_lmin1000_lmax5000-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_0 = Re_I + 1j * Im_I


    #for cl psi delta
    data =  np.load(dpath+'Il_nu_t_nd2_tmin0p96_lmin1000_lmax5000-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_2 = Re_I + 1j * Im_I


    # for cl delta delta
    data =  np.load(dpath+'Il_nu_t_nd4_tmin0p96_lmin1000_lmax5000-ReI-ImI.npy')
    Re_I      = data[:,0]
    Im_I      = data[:,1]
    #----------------------#

    I_4 = Re_I + 1j * Im_I

    del Re_I, Im_I, data


    #shrink
    ell_ = np.unique(ell)
    nu_n_= np.unique(nu_n) 
    t_   = np.unique(tt)



    #make 3D arrays 
    ell = ell.reshape(ell_.size, nu_n_.size, t_.size)
    nn  = nn.reshape(ell_.size, nu_n_.size, t_.size)
    tt  = tt.reshape(ell_.size, nu_n_.size, t_.size)
    I_0 = I_0.reshape(ell_.size, nu_n_.size, t_.size)
    I_2 = I_2.reshape(ell_.size, nu_n_.size, t_.size)
    I_4 = I_4.reshape(ell_.size, nu_n_.size, t_.size)

    cn2 = c_n.copy()
    cn2[:-1] *=2

    #insert a new axis for r and multiply with c_n 
    I0_lcrt = np.expand_dims(I_0, 2)*cn2.reshape(1, -1, 1, 1)
    I2_lcrt = np.expand_dims(I_2, 2)*cn2.reshape(1, -1, 1, 1)
    I4_lcrt = np.expand_dims(I_4, 2)*cn2.reshape(1, -1, 1, 1)

    #change t and c
    I0_ltrc = np.swapaxes(I0_lcrt, 1, 3)
    I2_ltrc = np.swapaxes(I2_lcrt, 1, 3)
    I4_ltrc = np.swapaxes(I4_lcrt, 1, 3)


    del I_0, I_2, I_4

    return ell_, t_, nu_n_, I0_ltrc, I2_ltrc, I4_ltrc

def loadggwts_highell():
    #get Gaussian quadrature weights
    data = np.loadtxt(dpath+'GG_weights_highell_tmin0p96.dat',skiprows=2)
    t1   = data[:,0]
    w1   = data[:,1]
    return t1, w1
