import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import coplot.plots as pl
import coplot.plot_settings as pls
import cmbnncs.simulator as simulator
import cmbnncs.utils as utils
import cmbnncs.spherical as spherical
import loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import healpy as hp
import math
import pymaster as nmt


def change_randn_num(randn_num):
    randn_num_change = randn_num.split('.')
    randn_num_change = randn_num_change[0]+randn_num_change[1]
    return randn_num_change

def change_randn_nums(randn_nums):
    rdns = ''
    rdns_list = []
    for rdn in randn_nums:
        rdns = rdns + change_randn_num(rdn)
        rdns_list.append(change_randn_num(rdn))
    return rdns, rdns_list

def mse(true, predict):
    '''mean square error'''
    return np.mean( (predict-true)**2 )


def cl2dl(Cl, ell_start, ell_in=None, get_ell=True):
    '''
    ell_start: 0 or 2, which should depend on Dl
    ell_in: the ell of Cl (as the input of this function)
    '''
    if ell_start==0:
        lmax_cl = len(Cl) - 1
    elif ell_start==2:
        lmax_cl = len(Cl) + 1
    
    ell = np.arange(lmax_cl + 1)
    if ell_in is not None:
        if ell_start==2:
            ell[2:] = ell_in
    
    factor = ell * (ell + 1.) / 2. / np.pi
    if ell_start==0:
        Dl = np.zeros_like(Cl)
        Dl[2:] = Cl[2:] * factor[2:]
        ell_2 = ell
    elif ell_start==2:
        Dl = Cl * factor[2:]
        ell_2 = ell[2:]
    if get_ell:
        return ell_2, Dl
    else:
        return Dl

# The function defined below will compute the power spectrum between two
# NmtFields f_a and f_b, using the coupling matrix stored in the
# NmtWorkspace wsp and subtracting the deprojection bias clb.
# Note that the most expensive operations in the MASTER algorithm are
# the computation of the coupling matrix and the deprojection bias. Since
# these two objects are precomputed, this function should be pretty fast!
def compute_master(f_a, f_b, wsp, clb):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled, cl_bias=clb)
    return cl_decoupled

def namaster_dl_TT_QQ_UU(cmb_t, mask, bl=None, nside=512, aposize=1, nlb=10,
                         cl_th=None, cls_th=None, cmb_t_th=None, sim_n=2):
    '''Calculate Cl * ell*(ell+1)/2/np.pi of TT, QQ, and UU.
    
    cmb_t : 1-D array with shape (nside**2*12,), the (recovered) CMB I, Q, or U map.
    mask : 1-D array with shape (nside**2*12,), the mask file used to the CMB map.
    bl : 1-D array with shape (3*nside,), the beam file used to the CMB map, the multipoles starts from 0 to 3*nside-1, so, lmax=3*nside - 1
    aposize : float or None, apodization scale in degrees.
    nlb : int, the bin size (\delta_\ell) of multipoles, it can be set to ~ 1/fsky
    cl_th : 1-D array, the theoretical TT, QQ, or UU power spectrum, where ell start from 0.
    cls_th : 6-D array with shape (6, M), the theoretical Cls and ell start from 0.
             cls_th[:4, :] correspongding to TT, EE, BB, and TE power spectra, respectively, and cls_th[4:, :] is 0.
    cmb_t_th : 1-D array with shape (nside**2*12,), the simulated CMB map based on the theoretical power spectrum.
    sim_n : int, the number of simulation.
    '''
    if aposize is not None:
        mask = nmt.mask_apodization(mask, aposize=aposize, apotype="Smooth")
    if cmb_t_th is None:
        f_t = nmt.NmtField(mask, [cmb_t], templates=None, beam=bl)
    else:
        f_t = nmt.NmtField(mask, [cmb_t], templates=[[cmb_t-cmb_t_th]], beam=bl)
    #method 1
    # b = nmt.NmtBin.from_nside_linear(nside, nlb=nlb, is_Dell=True) #nlb=\delta_ell ~ 1/fsky
    # dl_TT = nmt.compute_full_master(f_t, f_t, b)[0]
    #method 2
    b = nmt.NmtBin.from_nside_linear(nside, nlb=nlb, is_Dell=False) #nlb=\delta_ell ~ 1/fsky
    if cl_th is None:
        cl_bias = None
    else:
        cl_00_th = cl_th.reshape(1, -1)
        cl_bias = nmt.deprojection_bias(f_t, f_t, cl_00_th)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_t, f_t, b)
    cl_master = compute_master(f_t, f_t, w, cl_bias)
    ell = b.get_effective_ells()
    
    #get error
    if cl_th is not None:
        cl_mean = np.zeros_like(cl_master)
        cl_std = np.zeros_like(cl_master)
        for i in np.arange(sim_n):
            print("Simulating %s/%s"%(i+1, sim_n))
            t, q, u = hp.synfast(cls_th, nside, pol=True, new=True, verbose=False, pixwin=False)
            f0_sim = nmt.NmtField(mask, [t], templates=[[cmb_t-cmb_t_th]])
            cl = compute_master(f0_sim, f0_sim, w, cl_bias)
            cl_mean += cl
            cl_std += cl*cl
        cl_mean /= sim_n
        cl_std = np.sqrt(cl_std / sim_n - cl_mean*cl_mean)
        
        factor = ell*(ell+1)/2/np.pi
        dl_std = factor * cl_std
        
    ell, dl_master = cl2dl(cl_master[0], ell_start=2, ell_in=ell)
    hp.mollview(mask, title='Mask')
    if cl_th is None:
        return ell, dl_master
    else:
        return ell, dl_master, dl_std[0]

def namaster_dl_EE_BB(cmb_qu, mask, bl=None, nside=512, aposize=1, nlb=10):
    '''
    cmb_qu : 2-D array with shape (2, nside**2*12), CMB Q and U maps.
    mask : 1-D array with shape (nside**2*12,), the mask file used to the Q and U maps.
    bl : 1-D array with shape (3*nside,), the beam file used to the CMB map, the multipoles starts from 0 to 3*nside-1, so, lmax=3*nside - 1
    aposize : float or None, apodization scale in degrees.
    nlb : int, the bin size (\delta_\ell) of multipoles, it can be set to ~ 1/fsky
    '''
    if aposize is not None:
        mask = nmt.mask_apodization(mask, aposize=aposize, apotype="Smooth")
    f_qu = nmt.NmtField(mask, cmb_qu, beam=bl)
    b = nmt.NmtBin.from_nside_linear(nside, nlb=nlb, is_Dell=True) #nlb=10, \delta_ell ~ 1/fsky
    dl_22 = nmt.compute_full_master(f_qu, f_qu, b)
    ell = b.get_effective_ells()
    hp.mollview(mask, title='Mask')
    #dl_22[0]: EE, dl_22[3]: BB
    return ell, dl_22


class PlotCMBFull(object):
    def __init__(self, cmb, cmb_ML, randn_num='', map_type='I', fig_type='test', 
                 map_n=0, input_freqs=[100,143,217,353], out_freq=143, extra_suffix=''):
        """
        map_type: 'I', 'Q' or 'U'
        fig_type: 'test' or 'obs'
        """
        self.cmb = cmb
        self.cmb_ML = cmb_ML
        self.randn_num = randn_num
        self.map_type = map_type
        self.fig_type = fig_type
        self.map_n = map_n
        self.input_freqs = input_freqs
        self.freq_num = len(input_freqs)
        self.out_freq = out_freq
        self.ell = None
        self.extra_suffix = extra_suffix
    
    @property
    def minmax(self):
        if self.map_type=='I':
            return 500
        else:
            return 10
    
    @property
    def nside(self):
        return int(np.sqrt(len(self.cmb)/12))
    
    @property
    def lmax(self):
        if self.nside==512:
            self.xlim_max = 1500
        elif self.nside==256:
            self.xlim_max = 760
        return 3*self.nside - 1
    
    @property
    def randn_marker(self):
        return change_randn_num(self.randn_num)
    
    @property
    def fig_prefix(self):
        if self.fig_type=='obs':
            return 'plkcmb'
        elif self.fig_type=='test':
            return 'simcmb'
    
    def bl_plk(self):
        beams = loader.get_planck_beams(nside=self.nside, relative_dir='obs_data')
        return beams[str(self.out_freq)][:self.lmax+1]
    
    def bl_fwhm(self, fwhm):
        bl = hp.gauss_beam(fwhm*np.pi/10800., lmax=self.lmax)
        return bl[:self.lmax+1]
    
    def bl(self, fwhm=None):
        if fwhm is None:
            print("Using Planck beam file !!!")
            return self.bl_plk()
        else:
            return self.bl_fwhm(fwhm)
    
    @property
    def bin_lengh(self):
        return 30
    
    @property
    def bin_n(self):
        return int(math.ceil( (self.lmax-1)/float(self.bin_lengh) ))
    
    def get_plk_fwhm(self):
        """
        The recovered CMB map has beam with fwhm=9.43 (for output with 100GHz), while the Planck CMB has 5 arcmin beam.
        The generated beam map is used to calculate residual and MSE of CMB map.
        
        Note
        ----
        Note that this procedure is not right!!! The right way is remove the beam from the CMB map, and then add 9.43 arcmin beam,
        but it is not feasible. Therefore, this operation is only an approximate method, since the area where the beams work is much 
        smaller than that of a pixel when nside=256
        """
        if self.out_freq == 100:
            self.plk_fwhm = 9.43
        elif self.out_freq == 143:
            self.plk_fwhm = 7.27
        elif self.out_freq == 217:
            self.plk_fwhm = 5.01
        elif self.out_freq == 70:
            self.plk_fwhm = 13.31
        elif self.out_freq == 353:
            self.plk_fwhm = 4.86

    @property
    def residual_map(self):
        return self.cmb_ML - self.cmb
    
    def mask_plk(self):
        print("Using Planck mask !!!")
        if self.map_type=='I':
            self.mask = np.load('obs_data/mask/COM_Mask_CMB-common-Mask-Int_%s_R3.00.npy'%self.nside)
        else:
            self.mask = np.load('obs_data/mask/COM_Mask_CMB-common-Mask-Pol_%s_R3.00.npy'%self.nside)
        self.fsky = np.count_nonzero(self.mask) / float(len(self.mask))

    def mask_manual(self):
        self.mask = np.ones(self.nside**2*12)
        self.fsky = np.count_nonzero(self.mask) / float(len(self.mask))
        
    def plot_cmb(self, savefig=False, root='figures', hold=False):
        if self.fig_type=='obs':
            title = 'Planck CMB'
        elif self.fig_type=='test':
            title = 'Simulated CMB'
        matplotlib.rcParams.update({'font.size': 16})
        hp.mollview(self.cmb, cmap='jet', min=-self.minmax, max=self.minmax, title=title, hold=hold)
        if savefig:
            utils.mkdir(root)
            plt.savefig(root + '/%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n), bbox_inches='tight')

    def plot_cmb_ML(self, savefig=False, root='figures', hold=False):
        matplotlib.rcParams.update({'font.size': 16})
        hp.mollview(self.cmb_ML, cmap='jet', min=-self.minmax, max=self.minmax, title='Recovered CMB', hold=hold)
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root + '/ML_%s_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root + '/ML_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker), bbox_inches='tight')

    def plot_residual(self, savefig=False, root='figures', hold=False):
        matplotlib.rcParams.update({'font.size': 16})
        hp.mollview(self.residual_map, cmap='jet', min=-self.minmax/10., max=self.minmax/10., title='Residual', hold=hold)
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root+'/residual_%s_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root+'/residual_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker), bbox_inches='tight')

    def get_dl(self, fwhm=None, aposize=1, nlb=None, bin_residual=True):
        '''
        aposize : float or None
        nlb : int or None
        '''
        if nlb is None:
            self.nlb = math.ceil(1/self.fsky)
        else:
            self.nlb = nlb
        
        self.get_plk_fwhm()
        if self.fig_type=='obs':
            self.ell, self.dl = namaster_dl_TT_QQ_UU(self.cmb, self.mask, bl=self.bl(fwhm=5.0), nside=self.nside, aposize=aposize, nlb=self.nlb)
            self.ell, self.dl_ML = namaster_dl_TT_QQ_UU(self.cmb_ML, self.mask, bl=self.bl(fwhm=self.plk_fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb)
        else:
            self.ell, self.dl = namaster_dl_TT_QQ_UU(self.cmb, self.mask, bl=self.bl(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb, cl_th=None)
            self.ell, self.dl_ML = namaster_dl_TT_QQ_UU(self.cmb_ML, self.mask, bl=self.bl(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb, cl_th=None)

        self.mse_map = mse(self.cmb, self.cmb_ML)
        self.mse_dl = mse(self.dl, self.dl_ML)##
        print('mseSpectra:%s'%self.mse_dl)
        
        #different from sim_tt
        self.dl_diff = self.dl_ML - self.dl
        
        if bin_residual:
            self.ell_bined = [np.mean(self.ell[i*self.bin_lengh:(i+1)*self.bin_lengh]) for i in range(self.bin_n)]
            self.dl_diff_bined = [self.dl_diff[i*self.bin_lengh:(i+1)*self.bin_lengh] for i in range(self.bin_n)]
            self.dl_diff_bined_best = [np.mean(self.dl_diff_bined[i]) for i in range(self.bin_n)]
            self.dl_diff_bined_err = [np.std(self.dl_diff_bined[i]) for i in range(self.bin_n)]
        
    def plot_dl(self, savefig=False, root='figures', one_panel=True, 
                show_title=False, title_str=None, show_mse=False, 
                fwhm=None, aposize=1, nlb=None, bin_residual=True):
        if self.ell is None:
            self.get_dl(fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=bin_residual)
        
        if one_panel:
            fig_spectra = plt.figure(figsize=(6*1.2, 4.5*1.2))
            fig_spectra.subplots_adjust(hspace=0)
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ticks_size = 12
            fontsize = 16
        else:
            gs = gridspec.GridSpec(3, 2, height_ratios=[5.5, 3, 1])
            ticks_size = 12
            fontsize = 18
        
        if one_panel:
            ax_0 = plt.subplot(gs[0])
        else:
            ax_0 = plt.subplot(gs[3])
        ax_0 = pls.PlotSettings().setting(ax=ax_0,labels=[r'$\ell$', r'$D_\ell^{TT}[\mu k^2]$'],
                                          ticks_size=ticks_size,show_xticks=False,minor_locator_N=8,major_locator_N=5)
        if self.fig_type=='obs':
            ax_0.plot(self.ell, self.dl, label='Planck CMB')
        elif self.fig_type=='test':
            ax_0.plot(self.ell, self.dl, label='Simulated CMB')
            
        if self.map_type=='I':
            ax_0.plot(self.ell, self.dl_ML, label='Recovered CMB')
            ax_0.set_xlim(0, self.xlim_max)
            ax_0.set_ylim(10, 7100)
        else:
            ax_0.plot(self.ell, self.dl_ML, label='Recovered CMB')
            ax_0.set_xlim(0, self.xlim_max)
        
        if show_mse:
            ax_0.text(self.lmax*0.6, max(self.dl)*0.52, r'$MSE_{CMB}:%.2f$'%self.mse_map, fontsize=fontsize)
            ax_0.text(self.lmax*0.6, max(self.dl)*0.35, r'$MSE_{D_\ell}:%.2f$'%self.mse_dl, fontsize=fontsize)
        
        ax_0.legend(fontsize=fontsize)
        
        if show_title:
            if self.freq_num==1:
                if title_str is None:
                    plt.title('%s frequency: %s'%(self.freq_num, self.input_freqs), fontsize=fontsize)
                else:
                    plt.title(title_str, fontsize=fontsize)
            else:
                if title_str is None:
                    plt.title('%s frequencies: %s'%(self.freq_num, self.input_freqs), fontsize=fontsize)
                else:
                    plt.title(title_str, fontsize=fontsize)
        
        if one_panel:
            ax_1 = plt.subplot(gs[1])
        else:
            ax_1 = plt.subplot(gs[5])
        ax_1 = pls.PlotSettings().setting(ax=ax_1,labels=[r'$\ell$', r'$\Delta D_\ell^{TT}[\mu k^2]$'],
                                          ticks_size=ticks_size,minor_locator_N=8,major_locator_N=5)
        ax_1.plot([0, max(self.ell)], [0,0], '--', color=pl.fiducial_colors[9])
        if bin_residual:
            ax_1.errorbar(self.ell_bined, self.dl_diff_bined_best, yerr=self.dl_diff_bined_err, fmt='.')
        else:
            ax_1.plot(self.ell, self.dl_diff, color=pl.fiducial_colors[8])
            
        if not savefig:
            plt.plot([768,768], [-280,280])
            plt.text(768-50, 20, '768')
            plt.plot([1000,1000], [-280,280])
            plt.text(1000-50, 20, '1000')
        ax_1.set_xlim(0, self.xlim_max)
        
        if self.map_type=='I':
            ax_1.set_ylim(-100, 100)
        else:
            ax_1.set_ylim(-3, 3)
            
        if savefig:
            if self.extra_suffix:
                pl.savefig(root, 'spectra_%s_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.extra_suffix), fig_spectra)
            else:
                pl.savefig(root, 'spectra_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker), fig_spectra)

    def plot_all(self, savefig=False, root='figures', fwhm=None, aposize=1, nlb=None, bin_residual=True):
        if self.ell is None:
            self.get_dl(fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=bin_residual)
        
        fig = plt.figure(figsize=(6*1.2*2, 4.5*1.2*2))
        fig.subplots_adjust(wspace=0.21, hspace=0)
        
        pls.PlotSettings().setting(location=(2,2,1),set_labels=False)
        self.plot_cmb(hold=True)
        
        pls.PlotSettings().setting(location=(2,2,2),set_labels=False)
        self.plot_cmb_ML(hold=True)
        
        pls.PlotSettings().setting(location=(2,2,3),set_labels=False)
        self.plot_residual(hold=True)
        self.plot_dl(one_panel=False, show_title=True, show_mse=True, fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=bin_residual)
        
        if savefig:
            if self.extra_suffix:
                pl.savefig(root, '%s_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.extra_suffix), fig)
            else:
                pl.savefig(root, '%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker), fig)

    def _get_miniPatch(self, Map):
        '''
        select a 3*3 deg^2 patch
        '''
        ps = spherical.PixelSize(nside=self.nside)
        patch_size = int(3/ps.pixel_length)
        map_blocks = spherical.Cut(Map).block_all()
        patch_0 = map_blocks[0][:patch_size, :patch_size]
        patch_1 = map_blocks[4][:patch_size, :patch_size]
        start_pix = (self.nside-patch_size)//2
        patch_2 = map_blocks[4][start_pix:start_pix+patch_size, start_pix:start_pix+patch_size]
        patch_3 = map_blocks[4][-patch_size:, -patch_size:]
        patch_4 = map_blocks[11][-patch_size:, -patch_size:]
        return [patch_0, patch_1, patch_2, patch_3, patch_4]
    
    def get_miniPatch(self):
        self.cmb_miniBatches = self._get_miniPatch(self.cmb)
        self.cmb_ML_miniBatches = self._get_miniPatch(self.cmb_ML)
        self.residual_map_miniBatches = self._get_miniPatch(self.residual_map)
    
    def plot_miniPatch(self, savefig=False, root='figures'):
        self.get_miniPatch()
        fig = plt.figure(figsize=(3*5, 3*3))
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.15,hspace=0.25)
        for row in range(3):
            for column in range(5):
                pls.PlotSettings().setting(location=(3,5,row*5+column+1),set_labels=False,minor_locator_N=1)
                # plt.subplot(3,5,row*5+column+1)
                if row==0:
                    im = plt.imshow(self.cmb_miniBatches[column], cmap='jet', vmin=-500, vmax=500)
                    if self.fig_type=='obs':
                        plt.title('Planck CMB', fontsize=16)
                    elif self.fig_type=='test':
                        plt.title('Simulated CMB', fontsize=16)
                elif row==1:
                    im = plt.imshow(self.cmb_ML_miniBatches[column], cmap='jet', vmin=-500, vmax=500)
                    plt.title('Recovered CMB', fontsize=16)
                    if column==4:
                        cbar_ax = fig.add_axes([1.01, 0.358, 0.01, 0.641])
                        plt.colorbar(im, cax=cbar_ax)
                elif row==2:
                    im = plt.imshow(self.residual_map_miniBatches[column], cmap='jet', vmin=-50, vmax=50)
                    plt.title('Residual', fontsize=16)
                    if column==4:
                        cbar_ax = fig.add_axes([1.01, 0., 0.01, 0.287])
                        plt.colorbar(im, cax=cbar_ax)
        
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root + '/miniPatch_%s_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root + '/miniPatch_%s_%s_%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker), bbox_inches='tight')


#%%
class PlotCMBBlock(object):
    def __init__(self, cmb, cmb_ML, randn_num='', map_type='I', fig_type='test', 
                 map_n=0, input_freqs=[100,143,217,353], out_freq=143, block_n=0, extra_suffix=''):
        """
        map_type: 'I', 'Q' or 'U'
        fig_type: 'test' or 'obs'
        """
        self.cmb = cmb
        self.cmb_ML = cmb_ML
        self.randn_num = randn_num
        self.map_type = map_type
        self.fig_type = fig_type
        self.map_n = map_n
        self.input_freqs = input_freqs
        self.freq_num = len(input_freqs)
        self.out_freq = out_freq
        self.block_n = block_n
        self.extra_suffix = extra_suffix
        self.ell = None
    
    @property
    def minmax(self):
        if self.map_type=='I':
            return 500
        else:
            return 10
    
    @property
    def dl_type(self):
        if self.map_type=='I':
            return 'TT'
        else:
            return '%s%s'%(self.map_type, self.map_type)
    
    @property
    def nside(self):
        return int(len(self.cmb))
    
    @property
    def lmax(self):
        if self.nside==512:
            self.xlim_max = 1500
        elif self.nside==256:
            self.xlim_max = 760
        return 3*self.nside - 1
    
    @property
    def randn_marker(self):
        return change_randn_num(self.randn_num)
    
    @property
    def fig_prefix(self):
        if self.fig_type=='obs':
            return 'plkcmb'
        elif self.fig_type=='test':
            return 'simcmb'

    def bl_plk(self):
        beams = loader.get_planck_beams(nside=self.nside, relative_dir='obs_data')
        return beams[str(self.out_freq)][:self.lmax+1]
    
    def bl_fwhm(self, fwhm):
        bl = hp.gauss_beam(fwhm*np.pi/10800., lmax=self.lmax)
        return bl[:self.lmax+1]
    
    def bl(self, fwhm=None):
        if fwhm is None:
            print("Using Planck beam file !!!")
            return self.bl_plk()
        else:
            return self.bl_fwhm(fwhm)

    @property
    def bin_lengh(self):
        return 6 #6*nlb = 30, let nlb=5
    
    @property
    def bin_n(self):
        return int(math.ceil( (self.lmax-1)/float(self.bin_lengh) ))
    
    @property
    def residual_map(self):
        if self.fig_type=='obs':
            return self.cmb_ML - self.cmb_beam
        else:
            return self.cmb_ML - self.cmb
    
    def mask_plk(self):
        if self.map_type=='I':
            mask = np.load('obs_data/mask/COM_Mask_CMB-common-Mask-Int_%s_R3.00.npy'%self.nside)
        else:
            mask = np.load('obs_data/mask/COM_Mask_CMB-common-Mask-Pol_%s_R3.00.npy'%self.nside)
        mask_0 = spherical.Cut(mask).block(self.block_n)
        self.mask = spherical.Block2Full(mask_0, self.block_n).full()
        self.fsky = np.count_nonzero(self.mask) / float(len(self.mask))
        
    def mask_manual(self):
        mask_0 = np.ones((self.nside, self.nside))
        self.mask = spherical.Block2Full(mask_0, self.block_n).full()
        self.fsky = np.count_nonzero(self.mask) / float(len(self.mask))
        
    def plot_cmb(self, savefig=False, root='figures', hold=False, one_panel=True):
        if self.fig_type=='obs':
            title = 'Planck CMB'
        elif self.fig_type=='test':
            title = 'Simulated CMB'
        if one_panel:
            plt.figure()#
        matplotlib.rcParams.update({'font.size': 16})
        plt.imshow(self.cmb, cmap='jet', vmin=-self.minmax, vmax=self.minmax)
        plt.colorbar()
        plt.title(title, fontsize=16)
        if savefig:
            utils.mkdir(root)
            if self.use_mask:
                plt.savefig(root + '/%s_%s_%s_block%s_mask.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.block_n), bbox_inches='tight')
            else:
                plt.savefig(root + '/%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.block_n), bbox_inches='tight')

    def plot_cmb_ML(self, savefig=False, root='figures', hold=False, one_panel=True):
        if one_panel:
            plt.figure()#
        matplotlib.rcParams.update({'font.size': 16})
        plt.imshow(self.cmb_ML, cmap='jet', vmin=-self.minmax, vmax=self.minmax)
        plt.colorbar()
        plt.title('Recovered CMB', fontsize=16)
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root + '/ML_%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root + '/ML_%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n), bbox_inches='tight')

    def plot_residual(self, savefig=False, root='figures', hold=False, one_panel=True):
        if one_panel:
            plt.figure()#
        matplotlib.rcParams.update({'font.size': 16})
        if self.map_type=='I':
            plt.imshow(self.residual_map, cmap='jet', vmin=-self.minmax/50., vmax=self.minmax/50.)
        else:
            plt.imshow(self.residual_map, cmap='jet', vmin=-self.minmax/50., vmax=self.minmax/50.)
        plt.colorbar()
        plt.title('Residual', fontsize=16)
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root+'/residual_%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root+'/residual_%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n), bbox_inches='tight')
    
    def get_dl(self, fwhm=None, aposize=1, nlb=None, bin_residual=True):
        '''
        aposize : float or None
        nlb : int or None
        '''
        if nlb is None:
            self.nlb = math.ceil(1/self.fsky)
        else:
            self.nlb = nlb
        self.cmb_sp = spherical.Block2Full(self.cmb, self.block_n).full()
        self.cmb_ML_sp = spherical.Block2Full(self.cmb_ML, self.block_n).full()
        
        self.ell, self.dl = namaster_dl_TT_QQ_UU(self.cmb_sp, self.mask, bl=self.bl(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb)
        self.ell, self.dl_ML = namaster_dl_TT_QQ_UU(self.cmb_ML_sp, self.mask, bl=self.bl(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb)
        
        self.mse_map = mse(self.cmb, self.cmb_ML)
        self.mse_dl = mse(self.dl, self.dl_ML)##
        print('mseSpectra:%s'%self.mse_dl)
        self.dl_diff = self.dl_ML - self.dl
        
        if bin_residual:
            self.ell_bined = [np.mean(self.ell[i*self.bin_lengh:(i+1)*self.bin_lengh]) for i in range(self.bin_n)]
            self.dl_diff_bined = [self.dl_diff[i*self.bin_lengh:(i+1)*self.bin_lengh] for i in range(self.bin_n)]
            self.dl_diff_bined_best = [np.mean(self.dl_diff_bined[i]) for i in range(self.bin_n)]
            self.dl_diff_bined_err = [np.std(self.dl_diff_bined[i]) for i in range(self.bin_n)]

    def plot_dl(self, savefig=False, root='figures', one_panel=True, 
                show_title=False, title_str=None, show_mse=False, 
                fwhm=None, aposize=1, nlb=None, bin_residual=True):
        if self.ell is None:
            self.get_dl(fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=bin_residual)
        
        if one_panel:
            fig_spectra = plt.figure(figsize=(6*1.2, 4.5*1.2))
            fig_spectra.subplots_adjust(hspace=0)
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ticks_size = 12 + 4
            fontsize = 16 + 2
        else:
            gs = gridspec.GridSpec(3, 2, height_ratios=[5.5, 3, 1])
            ticks_size = 12
            fontsize = 18
        
        if one_panel:
            ax_0 = plt.subplot(gs[0])
        else:
            ax_0 = plt.subplot(gs[3])
        ax_0 = pls.PlotSettings().setting(ax=ax_0,labels=[r'$\ell$', r'$D_\ell^{%s}[\mu k^2]$'%self.dl_type],
                                          ticks_size=ticks_size,show_xticks=False,minor_locator_N=8,major_locator_N=5)
        if self.fig_type=='obs':
            ax_0.plot(self.ell, self.dl, label='Planck CMB')
        elif self.fig_type=='test':
            ax_0.plot(self.ell, self.dl, label='Simulated CMB')
        
        if self.map_type=='I':
            ax_0.plot(self.ell, self.dl_ML, label='Recovered CMB')
            ax_0.set_xlim(0, self.xlim_max)
            ax_0.set_ylim(10, 7100)
            # if self.fig_type=='obs':
            #     ax_0.set_ylim(10, 8000)
            # else:
            #     ax_0.set_ylim(10, 7500)
        else:
            ax_0.plot(self.ell, self.dl_ML, 'r', label='Recovered CMB') ## *2 !!!
            ax_0.set_xlim(0, self.xlim_max)
            # ax_0.set_ylim(0.001, 2.6)
            
        if show_mse:
            ax_0.text(self.lmax*0.62, max(self.dl)*0.4, r'$MSE_{CMB}:%.2f$'%self.mse_map, fontsize=fontsize)
            ax_0.text(self.lmax*0.62, max(self.dl)*0.3, r'$MSE_{D_\ell}:%.2f$'%self.mse_dl, fontsize=fontsize)
        
        ax_0.legend(fontsize=fontsize)
        
        if show_title:
            if self.freq_num==1:
                if title_str is None:
                    plt.title('%s frequency: %s'%(self.freq_num, self.input_freqs), fontsize=fontsize)
                else:
                    plt.title(title_str, fontsize=fontsize)
            else:
                if title_str is None:
                    plt.title('%s frequencies: %s'%(self.freq_num, self.input_freqs), fontsize=fontsize)
                else:
                    plt.title(title_str, fontsize=fontsize)
        
        if one_panel:
            ax_1 = plt.subplot(gs[1])
        else:
            ax_1 = plt.subplot(gs[5])
        ax_1 = pls.PlotSettings().setting(ax=ax_1,labels=[r'$\ell$', r'$\Delta D_\ell^{%s}[\mu k^2]$'%self.dl_type],
                                          ticks_size=ticks_size,minor_locator_N=8,major_locator_N=5)
        ax_1.plot([0, max(self.ell)], [0,0], '--', color=pl.fiducial_colors[9])
        if bin_residual:
            ax_1.errorbar(self.ell_bined, self.dl_diff_bined_best, yerr=self.dl_diff_bined_err, fmt='.')
        else:
            ax_1.plot(self.ell, self.dl_diff, color=pl.fiducial_colors[8])
        if not savefig:
            plt.plot([768,768], [-280,280])###
            plt.plot([1000,1000], [-280,280])###
            plt.plot([1250,1250], [-280,280])###
            plt.plot([1300,1300], [-280,280])###
        ax_1.set_xlim(0, self.xlim_max)
        
        if self.map_type=='I':
            ax_1.set_ylim(-100, 100)
        else:
            ax_1.set_ylim(-0.5, 0.5)
        
        if savefig:
            if self.extra_suffix:
                pl.savefig(root, 'spectra_%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), fig_spectra)
            else:
                pl.savefig(root, 'spectra_%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n), fig_spectra)

    def plot_all(self, savefig=False, root='figures', fwhm=None, aposize=1, nlb=None, bin_residual=True):
        if self.ell is None:
            self.get_dl(fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=bin_residual)
        
        fig = plt.figure(figsize=(6*1.2*2, 4.5*1.2*2))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        
        pls.PlotSettings().setting(location=(2,2,1),set_labels=False)
        self.plot_cmb(hold=True, one_panel=False)
        
        pls.PlotSettings().setting(location=(2,2,2),set_labels=False)
        self.plot_cmb_ML(hold=True, one_panel=False)
        
        pls.PlotSettings().setting(location=(2,2,3),set_labels=False)
        self.plot_residual(hold=True, one_panel=False)
        
        # pls.PlotSettings().setting(location=(2,2,4),set_labels=False)
        self.plot_dl(one_panel=False, show_title=True, show_mse=True, fwhm=fwhm, aposize=aposize, nlb=nlb)#True
        # plt.suptitle(self.case_labels[str(self.case)], fontsize=22)
        
        if savefig:
            if self.extra_suffix:
                pl.savefig(root, '%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), fig)
            else:
                pl.savefig(root, '%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n), fig)

    def _get_miniPatch(self, Map):
        '''
        select a 3*3 deg^2 patch
        '''
        ps = spherical.PixelSize(nside=self.nside)
        patch_size = int(3/ps.pixel_length)
        start_pix = (self.nside-patch_size)//2
        patch_0 = Map[start_pix:start_pix+patch_size, :patch_size]
        patch_1 = Map[start_pix:start_pix+patch_size, start_pix:start_pix+patch_size]
        patch_2 = Map[start_pix:start_pix+patch_size, -patch_size:]
        return [patch_0, patch_1, patch_2]
    
    def get_miniPatch(self):
        self.cmb_miniBatches = self._get_miniPatch(self.cmb)
        self.cmb_ML_miniBatches = self._get_miniPatch(self.cmb_ML)
        self.residual_map_miniBatches = self._get_miniPatch(self.residual_map)
    
    def plot_miniPatch(self, savefig=False, root='figures'):
        self.get_miniPatch()
        fig = plt.figure(figsize=(3*3, 3*3))
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.15,hspace=0.25)
        for row in range(3):
            for column in range(3):
                pls.PlotSettings().setting(location=(3,3,row*3+column+1),set_labels=False,minor_locator_N=1)
                # plt.subplot(3,3,row*3+column+1)
                if row==0:
                    im = plt.imshow(self.cmb_miniBatches[column], cmap='jet', vmin=-500, vmax=500)
                    if self.fig_type=='obs':
                        plt.title('Planck CMB', fontsize=16)
                    elif self.fig_type=='test':
                        plt.title('Simulated CMB', fontsize=16)
                elif row==1:
                    im = plt.imshow(self.cmb_ML_miniBatches[column], cmap='jet', vmin=-500, vmax=500)
                    plt.title('Recovered CMB', fontsize=16)
                    if column==2:
                        cbar_ax = fig.add_axes([1.01, 0.358, 0.015, 0.641])
                        plt.colorbar(im, cax=cbar_ax)
                elif row==2:
                    im = plt.imshow(self.residual_map_miniBatches[column], cmap='jet', vmin=-10, vmax=10)
                    plt.title('Residual', fontsize=16)
                    if column==2:
                        cbar_ax = fig.add_axes([1.01, 0., 0.015, 0.287])
                        plt.colorbar(im, cax=cbar_ax)
        
        if savefig:
            utils.mkdir(root)
            if self.extra_suffix:
                plt.savefig(root + '/miniPatch_%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), bbox_inches='tight')
            else:
                plt.savefig(root + '/miniPatch_%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,self.map_type,self.map_n,self.randn_marker,self.block_n), bbox_inches='tight')
    
    
#%%
class PlotCMB_EEBB(object):
    def __init__(self, cmb_qu, cmb_ML_qu, map_n=0, nside=512, block_n=0, randn_marker='',extra_suffix=''):
        self.cmb_qu = cmb_qu
        self.cmb_ML_qu = cmb_ML_qu
        self.map_n = map_n
        self.nside = nside
        self.block_n = block_n
        self.randn_marker = randn_marker
        self.extra_suffix = extra_suffix
        self.ell = None
        
    @property
    def lmax(self):
        if self.nside==512:
            self.xlim_max = 1500
        elif self.nside==256:
            self.xlim_max = 760
        return 3*self.nside - 1

    @property
    def bin_lengh(self):
        return 6 #6*nlb = 30, let nlb=5

    @property
    def bin_n(self):
        return int(math.ceil( (self.lmax-1)/float(self.bin_lengh) ))
    
    @property
    def fig_prefix(self):
        return 'simcmb'

    def mask_manual(self):
        mask_0 = np.ones((self.nside, self.nside))
        self.mask = spherical.Block2Full(mask_0, self.block_n).full()
        self.fsky = np.count_nonzero(self.mask) / float(len(self.mask))
    
    def bl_fwhm(self, fwhm):
        bl = hp.gauss_beam(fwhm*np.pi/10800., lmax=self.lmax)
        return bl[:self.lmax+1]
    
    def get_dl(self, fwhm=None, aposize=1, nlb=None, bin_residual=True):
        '''
        aposize : float or None
        nlb : int or None
        '''
        # self.get_fiducial_dls()
        if nlb is None:
            self.nlb = math.ceil(1/self.fsky)
        else:
            self.nlb = nlb
        
        self.ell, self.dl = namaster_dl_EE_BB(self.cmb_qu, self.mask, bl=self.bl_fwhm(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb)
        self.ell, self.dl_ML = namaster_dl_EE_BB(self.cmb_ML_qu, self.mask, bl=self.bl_fwhm(fwhm=fwhm), nside=self.nside, aposize=aposize, nlb=self.nlb)
        self.dl_EE, self.dl_BB = self.dl[0], self.dl[3]
        self.dl_ML_EE, self.dl_ML_BB = self.dl_ML[0], self.dl_ML[3]
        
        self.diff_EE = self.dl_ML_EE - self.dl_EE
        self.diff_BB = self.dl_ML_BB - self.dl_BB
        # print(self.diff_BB)
        
        #bined ell & dl residual
        if bin_residual:
            self.ell_bined = [np.mean(self.ell[i*self.bin_lengh:(i+1)*self.bin_lengh]) for i in range(self.bin_n)]
            #residual of EE
            self.diff_EE_bined = [self.diff_EE[i*self.bin_lengh:(i+1)*self.bin_lengh] for i in range(self.bin_n)]
            self.diff_EE_bined_best = [np.mean(self.diff_EE_bined[i]) for i in range(self.bin_n)]
            self.diff_EE_bined_err = [np.std(self.diff_EE_bined[i]) for i in range(self.bin_n)]
            #residual of BB
            self.diff_BB_bined = [self.diff_BB[i*self.bin_lengh:(i+1)*self.bin_lengh] for i in range(self.bin_n)]
            self.diff_BB_bined_best = [np.mean(self.diff_BB_bined[i]) for i in range(self.bin_n)]
            self.diff_BB_bined_err = [np.std(self.diff_BB_bined[i]) for i in range(self.bin_n)]
    
    
    def plot_dl(self, savefig=False, root='figures', dl_type='', bin_residual=True):        
        '''
        dl_type: EE or BB
        '''
        fig_spectra = plt.figure(figsize=(6*1.*2, 4.5*1.))
        fig_spectra.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.23)
        # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ticks_size = 12 #+ 4
        fontsize = 16 #+ 2
        
        ax_0 = pls.PlotSettings().setting(location=[1,2,1],labels=[r'$\ell$', r'$D_\ell^{%s}[\mu k^2]$'%dl_type],
                                          ticks_size=ticks_size,show_xticks=False,minor_locator_N=8,major_locator_N=5)
        
        ax_0.loglog(self.ell, eval('self.dl_%s'%dl_type), label='Simulated CMB')
        ax_0.loglog(self.ell, eval('self.dl_ML_%s'%dl_type), label='Recovered CMB')
        
        
        ax_0.set_xlim(0, self.xlim_max)
        # if dl_type=='EE':
        #     ax_0.set_ylim(-0.05, 0.05)
        # elif dl_type=='BB':
        #     ax_0.set_ylim(-0.003, 0.003)
        
        ax_0.legend(loc=2, fontsize=fontsize)
        
        ax_1 = pls.PlotSettings().setting(location=[1,2,2],labels=[r'$\ell$', r'$\Delta D_\ell^{%s}[\mu k^2]$'%dl_type],
                                          ticks_size=ticks_size,minor_locator_N=8,major_locator_N=5)
        
        ax_1.plot([0, max(self.ell)], [0,0], '--', color=pl.fiducial_colors[9])
        if bin_residual:
            ax_1.errorbar(self.ell_bined, eval('self.diff_%s_bined_best'%dl_type), yerr=eval('self.diff_%s_bined_err'%dl_type), fmt='.')
        else:
            ax_1.plot(self.ell, eval('self.diff_%s'%dl_type), color=pl.fiducial_colors[8])
        ax_1.set_xlim(0, self.xlim_max)
        if dl_type=='EE':
            # pass
            ax_1.set_ylim(-0.9, 0.1)
            # ax_1.set_ylim(-2e-5, 2e-5) #test, plot CL
        elif dl_type=='BB':
            ax_1.set_ylim(-0.04, 0.04)
        
        if not savefig:
            plt.plot([768,768], [-280,280])###
            plt.plot([1000,1000], [-280,280])###
            plt.plot([1250,1250], [-280,280])###
            plt.plot([1300,1300], [-280,280])###
        
        if savefig:
            if self.extra_suffix:
                pl.savefig(root+'/pdf', 'spectra_%s_%s_%s_%s_block%s_%s.pdf'%(self.fig_prefix,dl_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), fig_spectra)
                pl.savefig(root+'/jpg', 'spectra_%s_%s_%s_%s_block%s_%s.jpg'%(self.fig_prefix,dl_type,self.map_n,self.randn_marker,self.block_n,self.extra_suffix), fig_spectra)
            else:
                pl.savefig(root+'/pdf', 'spectra_%s_%s_%s_%s_block%s.pdf'%(self.fig_prefix,dl_type,self.map_n,self.randn_marker,self.block_n), fig_spectra)
                pl.savefig(root+'/jpg', 'spectra_%s_%s_%s_%s_block%s.jpg'%(self.fig_prefix,dl_type,self.map_n,self.randn_marker,self.block_n), fig_spectra)

    def plot_all(self, savefig=False, root='figures', fwhm=None, aposize=1, nlb=None, bin_residual=True):
        if self.ell is None:
            self.get_dl(fwhm=fwhm, aposize=aposize, nlb=nlb, bin_residual=True)
        self.plot_dl(savefig=savefig, root=root, dl_type='EE') #EE
        self.plot_dl(savefig=savefig, root=root, dl_type='BB') #BB


#%% RMS of the residual maps
def mask_latitude(Map, nside=256, degree=30, inclusive=False, start_southPole=True):
    '''
    mask the map according to latitude
    
    :param start_southPole: if True, start from the south pole, otherwise, start from the north pole
    '''
    npix = hp.nside2npix(nside)
    if start_southPole:
        theta, phi = hp.pix2ang(nside=nside, ipix=npix-1)
    else:
        theta, phi = hp.pix2ang(nside=nside, ipix=0)
    idx_list = hp.query_disc(nside=nside, vec=hp.ang2vec(theta=theta, phi=phi), radius=degree/180.*np.pi, inclusive=inclusive)
    mask = np.zeros(npix)
    mask[idx_list] = 1
    map_mask = Map * mask
    return map_mask, idx_list

def get_RMS(Map, nside=256, degree_bin=10, inclusive=False):
    
    rms_num = 180//degree_bin
    rms_all = []
    for i in range(rms_num):
        mask_1, idx_1 = mask_latitude(Map, nside=nside, degree=degree_bin*i, inclusive=inclusive)
        mask_2, idx_2 = mask_latitude(Map, nside=nside, degree=degree_bin*(i+1), inclusive=inclusive)
        diff = mask_2 - mask_1
        pix_num = len(idx_2) - len(idx_1)
        rms_all.append( np.sqrt(sum(diff**2)/pix_num) )# RMS, this is right !!!
    rms_all = np.array(rms_all)
    degs = np.arange(-90, 90, degree_bin) + degree_bin/2.
    return degs, rms_all


#%% calcualte cosmic variance
def cosmic_variance(ell, get_std=True):
    '''
    sigma^2 = (delta_C_ell/C_ell)^2 = 2/(2*ell + 1)
    '''
    cv = 2/(2*ell + 1)
    if get_std:
        return np.sqrt(cv)
    else:
        return cv

