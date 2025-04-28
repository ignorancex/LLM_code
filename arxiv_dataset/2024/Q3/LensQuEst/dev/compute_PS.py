###
N_runs = 1000
# number of pixels for the flat map
nX = 400 # 1200
nY = 400 #1200
template_fname = 'Cls_%dx%d_point_sources_bigger.pkl'%(nX,nY)
psfile = 'point_sources_bigger.png'
mask_file = 'mask_simple400x400.png'
mean_field = 'mask_simple400x400_point_sources_bigger.pkl'
psapod = 1.5
run=False
###

##### 
import warnings
warnings.filterwarnings("ignore")
#####

import pickle
import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

#to get latex to work, shoulldn't be necessary for most ppl
os.environ['PATH'] = "%s:/usr/local/cuda-11.2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/anaconda/bin:/home/delon/texlive/bin/x86_64-linux:/home/delon/.local/bin:/home/delon/bin"%os.environ['PATH']


# In[3]:


from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *


# In[128]:

final_keys = ['(Auto QE)-(Power Spectrum(hat NL))', 
              '(Auto QE)-(Power Spectrum(hat NL)) [M]',
              '(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]']
Cls_tot = dict(zip(final_keys, np.array([np.array([]), np.array([]), np.array([])])))

print("Map properties")


# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


# In[129]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: cmb.flensedTT(l) + cmb.fdetectorNoise(l)

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


print("CMB lensing power spectrum")
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)


#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


from scipy.ndimage import gaussian_filter 
from scipy.fft import fft2

mask = rgb2gray(plt.imread(mask_file))
apodized_mask = gaussian_filter(mask, 3)
point_sources = rgb2gray(plt.imread(psfile))
point_sources = gaussian_filter(point_sources, psapod) 
apodized_mask += point_sources
mask = 1-mask
apodized_mask = 1 - apodized_mask

fig, axs = plt.subplots(ncols=2, figsize=(16,8))
fig.subplots_adjust(wspace=0, hspace=0)

im0 = axs[0].imshow(mask, vmin=0, vmax=1)
im1 = axs[1].imshow(apodized_mask, vmin=0, vmax=1)


for ax,s in zip(axs, ['Raw', 'Apodized']):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.95, .95,
        s=s,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
            c='white',
       fontsize=30)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)
fig.suptitle('Mask', y=.9)

plt.savefig('figures/mask%dx%d_point_sources.pdf'%(nX,nY), bbox_inches='tight')


from tqdm import trange
import pickle
mean_field = pickle.load(open(mean_field, 'rb'))

if(run):
    for run_n in trange(N_runs):
        # # Testing on simulated CMB Map
        # Basically the next cells do the following
        # 1. Generates a gaussain random field with the same power spectrum as the unlensed CMB. This will be a "realization" of the CMB
        # 2. Generates a gaussain random field with teh same power spectrum as "kappa" in a universe? TODO wjat. This will be a specific realization of "kappa", e.g. we fix kappa
        # 3. Lens our simulated CMB with our simulated Kapppa 
        # 4. Generated gaussian random field with same power spectrum as we expect from detector noise and add this to our lensed CMB
        #generate GRF with the same power spectrum as the unlensed CMB
        cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
        cmb0 = baseMap.inverseFourier(cmb0Fourier)

        kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
        kCmb = baseMap.inverseFourier(kCmbFourier)

        lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
        lensedCmbFourier = baseMap.fourier(lensedCmb)


        fgFourier = baseMap.genGRF(cmb.fForeground, test=False)
        lensedCmbFourier = lensedCmbFourier + fgFourier
        lensedCmb = baseMap.inverseFourier(lensedCmbFourier)


        noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
        totalCmbFourier = lensedCmbFourier + noiseFourier
        totalCmb = baseMap.inverseFourier(totalCmbFourier)

        totalMaskedCmb = apodized_mask*totalCmb
        totalMaskedCmbFourier = baseMap.fourier(totalMaskedCmb)

        data = {}
        Cls = {}
        c_keys = ['Standard QE', 'AFC Eq(7)',
                          'Standard QE Masked', 'AFC Eq(7) Masked',
                          'Standard QE Masked MFS', 'AFC Eq(7) Masked MFS']

        fsky = np.sum(apodized_mask)/(nX*nY)

        c1 = fsky
        c2 = fsky**2

        funcs = dict(zip(c_keys, 
                         [baseMap.computeQuadEstKappaNorm, 
                          baseMap.computeQuadEstKappaAutoCorrectionMap,
                          baseMap.computeQuadEstKappaNorm, 
                          baseMap.computeQuadEstKappaAutoCorrectionMap,
                          baseMap.computeQuadEstKappaNorm, 
                          baseMap.computeQuadEstKappaAutoCorrectionMap]))

        inp_data = dict(zip(c_keys, 
                         [totalCmbFourier, 
                          totalCmbFourier,
                          totalMaskedCmbFourier, 
                          totalMaskedCmbFourier,
                          totalMaskedCmbFourier, 
                          totalMaskedCmbFourier]))
        for key in funcs:
            data[key] = funcs[key](cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, 
                                                               dataFourier=inp_data[key], test=False)
            if(key=='Standard QE Masked MFS'):
                data[key] -= mean_field
            Cls[key] = [0,0,0]
            Cls[key][0], Cls[key][1], Cls[key][2] = baseMap.powerSpectrum(dataFourier=data[key])

            if('Masked' in key):
                if('AFC' in key): #scale noise by 1/fsky^2
                    Cls[key][1] /= c2
                else: #scale power specturm by 1/fsky
                    Cls[key][1] /= c1
                    Cls[key][2] /= c1


        c_keys += final_keys

        if(run_n==0):
            Cls_tot['(Auto QE)-(Power Spectrum(hat NL))'] = np.array([[Cls['Standard QE'][0], #binning same 
                                                        Cls['Standard QE'][1] - Cls['AFC Eq(7)'][1], #subtract power spectrums 
                                                        Cls['Standard QE'][2]]])

            Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M]'] = np.array([[Cls['Standard QE Masked'][0], #binning same 
                                                        Cls['Standard QE Masked'][1] - Cls['AFC Eq(7) Masked'][1], #subtract power spectrums 
                                                        Cls['Standard QE Masked'][2]]])
            Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]'] = np.array([[Cls['Standard QE Masked MFS'][0], #binning same 
                                                        Cls['Standard QE Masked MFS'][1] - Cls['AFC Eq(7) Masked MFS'][1], #subtract power spectrums 
                                                        Cls['Standard QE Masked MFS'][2]]])

            continue



        else:
            Cls_tot['(Auto QE)-(Power Spectrum(hat NL))'] = np.vstack(([[Cls['Standard QE'][0], #binning same 
                                                        Cls['Standard QE'][1] - Cls['AFC Eq(7)'][1], #subtract power spectrums 
                                                        Cls['Standard QE'][2]]], Cls_tot['(Auto QE)-(Power Spectrum(hat NL))'] ))

            Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M]'] = np.vstack(([[Cls['Standard QE Masked'][0], #binning same 
                                                        Cls['Standard QE Masked'][1] - Cls['AFC Eq(7) Masked'][1], #subtract power spectrums 
                                                        Cls['Standard QE Masked'][2]]], Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M]'] ))
            Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]'] = np.vstack(([[Cls['Standard QE Masked MFS'][0], #binning same 
                                                        Cls['Standard QE Masked MFS'][1] - Cls['AFC Eq(7) Masked MFS'][1], #subtract power spectrums 
                                                        Cls['Standard QE Masked MFS'][2]]], Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]'] ))
#         print(np.shape(Cls_tot['(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]'] ))
        f = open(template_fname, 'wb') 
        pickle.dump(Cls_tot, f)
else:
    f = open(template_fname, 'rb') 
    Cls_tot = pickle.load(f) 


for i in range(N_runs-1):
    for key in final_keys:
        assert(all(Cls_tot[key][i][0] == Cls_tot[key][i+1][0]))
lCen =  Cls_tot[final_keys[0]][0][0]
Cls = {}
sCls = {}
for key in final_keys:
    Cls[key] = np.sum(np.transpose(Cls_tot[key], axes=[1,2,0])[1], axis=1)/N_runs
    sCls[key] = np.sqrt(np.sum(np.square(np.transpose(Cls_tot[key], axes=[1,2,0])[2]), axis=1))/(N_runs)
    
fig, axs = plt.subplots(nrows=1, figsize=(15,8), sharey=True)
axs = [axs]
fig.subplots_adjust(wspace=0, hspace=0)

ell = baseMap.l.flatten()
factor = 1. # lCen**2
labels ={'Standard QE': r'${\big<\hat \kappa \hat\kappa \big>}$',
        'AFC Eq(7)': r'Computed noise $\hat N$',
        '(Auto QE)-(Power Spectrum(hat NL))': r'${\big<\hat \kappa \hat\kappa \big>- \hat N}$',
        'Standard QE Masked MFS': r'${\big<\hat \kappa \hat\kappa \big>}/f_{\rm sky}$ \texttt{[M+MFS]}',
        'AFC Eq(7) Masked MFS': r'Computed noise $\hat N$ \texttt{[M+MFS]}',
        '(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]': r'${\big<\hat \kappa \hat\kappa \big>/f_{\rm sky}- \hat N}/f_{\rm sky}^2$ \texttt{[M+MFS]}',
        'Standard QE Masked': r'${\big<\hat \kappa \hat\kappa \big>}/f_{\rm sky}$ \texttt{[M]}',
        'AFC Eq(7) Masked': r'Computed noise $\hat N$ \texttt{[M]}',
        '(Auto QE)-(Power Spectrum(hat NL)) [M]': r'${\big<\hat \kappa \hat\kappa \big>/f_{\rm sky}- \hat N}/f_{\rm sky}^2$ \texttt{[M]}'
}

col = {'Standard QE': 'r',
        'Standard QE Masked MFS': 'b',
        'AFC Eq(7)': 'r',
        'AFC Eq(7) Masked MFS': 'b',
        '(Auto QE)-(Power Spectrum(hat NL))': 'r',
        '(Auto QE)-(Power Spectrum(hat NL)) [M+MFS]': 'b',
        'Standard QE Masked': 'g',
        'AFC Eq(7) Masked': 'g',
        '(Auto QE)-(Power Spectrum(hat NL)) [M]': 'g',

}

theory=[p2d_cmblens.fPinterp, fNqCmb_fft]
theory_l=[r'$\big<\kappa\kappa\big>$']#, r'Projected noise $N$']
theory_s=['black', 'lightgrey']
for f,l,sty in zip(theory, theory_l, theory_s):
    L = np.logspace(np.log10(1.), np.log10(np.max(ell)), 201, 10.)
    ClExpected = np.array(list(map(f, L)))
    for ax in axs:
        ax.plot(L, factor*ClExpected, sty, label=l)
       
       
bnds = {'x':[],'y':[]}
for key in final_keys:    
    Cl = Cls[key]
    sCl = sCls[key]
    Ipos = np.where(Cl>=1e-10)
    Ineg = np.where(Cl<0.)

    ax = axs[0]
    fctr = factor
#     print(list(zip(Cl, sCl)))
    ax.errorbar(lCen[Ipos], factor*(Cl[Ipos]), yerr=fctr*sCl[Ipos], c=col[key], alpha=.75, fmt=':', capsize=3, label=labels[key], capthick=1)
    tmp_data = {
       'x': lCen[Ipos],
       'y1': [y - e for y, e in zip(factor*(Cl[Ipos]), sCl[Ipos])],
       'y2': [y + e for y, e in zip(factor*(Cl[Ipos]), sCl[Ipos])]}
    ax.fill_between(**tmp_data, color=col[key], alpha=.25)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell$')

    bnds['x'] += [np.min(lCen), np.max(lCen)]
    bnds['y'] += [np.min(Cl[Ipos]), factor*np.max(Cl[Ipos])]


fig.suptitle(r'Lensed CMB + Foregrounds + White Detector Noise + Point Sources \texttt{(CMB-S3)}', y=0.95)

for ax in axs:
    ax.set_xlim(np.min(bnds['x']), np.max(bnds['x']))
    ax.set_ylim(1e-10, 1e-4)
#     print(np.max(bnds['y']))


    l=ax.legend(frameon=False, loc='upper left')
    s = r'\texttt{[M]} = Masked'
    s += '\n'
    s += r'\texttt{[MFS]} = Mean Field Subtracted'
    ax.text(.90, .95,
       s=s,
       transform=ax.transAxes,
       horizontalalignment='right',
       verticalalignment='top',
      fontsize=18)





plt.savefig('figures/Cl_compare%dx%d_more_data_point_sources_bigger.pdf'%(nX, nY), bbox_inches='tight')
plt.savefig('figures/Cl_compare%dx%d_more_data_point_sources_bigger.png'%(nX, nY), bbox_inches='tight')
