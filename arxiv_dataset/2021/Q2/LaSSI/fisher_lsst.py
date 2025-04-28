import universe
reload(universe)
from universe import *

import parameters_fisher
reload(parameters_fisher)
from parameters_fisher import *

import projection_kernel
reload(projection_kernel)
from projection_kernel import *

#import pn_2d
#reload(pn_2d)
#from pn_2d import *
#
#import covp_2d
#reload(covp_2d)
#from covp_2d import *


##################################################################################

class FisherLsst(object):
   
   def __init__(self, cosmoPar, galaxyBiasPar, shearMultBiasPar, photoZPar, photoZSPar=None, nBins=2, nL=20, fsky=1., fNk=lambda l:0., magBias=False, name=None, nProc=3, save=True):   #, fullCross=True
      self.save = save
      self.nProc = nProc
      
      # CMB lensing noise
      self.fNk = fNk
      
      # sky fraction
      self.fsky = fsky
      
      # ell bins
      self.nL = nL
      self.lMin = 20.   # as in DESC SRD v1
      self.lMax = 1.e3  #3.e3  #1.e3
      self.L, self.dL, self.Nmodes, self.Le = generateEllBins(self.lMin, self.lMax, self.nL, self.fsky)
      
      # number of bins
      self.nBins = nBins
      self.nG = self.nBins
      self.nS = self.nBins
      #
      self.nKK = 1
      self.nKG = self.nG
      self.nKS = self.nS
      self.nGG = self.nG * (self.nG+1) / 2   # not just gg from same z-bin
      self.nGS = self.nG * self.nS # not just higher z s than g
      self.nSS = self.nS * (self.nS+1) / 2
      #
      self.n2pt = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS + self.nSS
      print "Tomographic bins: "+str(self.nBins)
      print "2-pt functions: "+str(self.n2pt)
      
      # size of data vector
      self.nData = self.n2pt * self.nL
      print "Data vector has "+str(self.nData)+" elements"

      # include known magnification bias or not
      self.magBias = magBias

      # Improving the conditioning of the cov matrix
      # Relative unit for shear
      self.sUnit = 10.#20.
      # Relative unit
      self.kUnit = 3.
      # ell scaling so that ell^alpha * C_ell is relatively independent of ell
      self.alpha = 1.

      
      # output file names
      self.name = "lsst_kgs_nBins"+str(self.nBins)+"_nL"+str(self.nL)
      if self.magBias:
         self.name += "_magbias"
#      if fullCross:
#         self.name += "_fullcross"
      if photoZPar.outliers<>0.:
         self.name += "_outliers"+str(photoZPar.outliers)
      else:
         self.name += "_gphotoz"
      if name is not None:
         self.name += "_"+name
      if photoZSPar is not None:
         self.name += "_diffGS"
      print "Ouput file name:", self.name
      
      # create folders for output, if needed
      if not os.path.exists("./output"):
         os.makedirs("./output")
      if not os.path.exists("./output/cov"):
         os.makedirs("./output/cov")
      if not os.path.exists("./output/dDatadPar"):
         os.makedirs("./output/dDatadPar")
      if not os.path.exists("./output/fisher"):
         os.makedirs("./output/fisher")


      # create folders for figures
      if not os.path.exists("./figures"):
         os.makedirs("./figures")
      self.figurePath = "./figures/"+self.name
      if not os.path.exists(self.figurePath):
         os.makedirs(self.figurePath)
      print "Figures folder:", self.figurePath



      ##################################################################################

      # cosmology parameters
      self.cosmoPar = cosmoPar
      
      # nuisance parameters
      self.galaxyBiasPar = galaxyBiasPar
      self.shearMultBiasPar = shearMultBiasPar
      self.photoZPar = photoZPar
      self.photoZSPar = photoZSPar  # optional param, to allow different bins for g and s
      # combined nuisance parameters
      self.nuisancePar = self.galaxyBiasPar.copy()
      self.nuisancePar.addParams(self.shearMultBiasPar)
      self.nuisancePar.addParams(self.photoZPar)
      if self.photoZSPar is not None:
         self.nuisancePar.addParams(self.photoZSPar)
      
      # all parameters
      self.fullPar = self.cosmoPar.copy()
      self.fullPar.addParams(self.nuisancePar)
      print "Params: "+str(self.fullPar.nPar)+" total = "+str(self.cosmoPar.nPar)+" cosmo + "+str(self.nuisancePar.nPar)+" nuisance"

      ##################################################################################
      # Fiducial data vector and covariance
      
      tStartFisher = time()
      
      print "Run CLASS",
      tStart = time()
      self.u = Universe(self.cosmoPar.paramsClassy)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"

      print "Tracer and shear bins",
      tStart = time()
      self.w_k, self.w_g, self.w_s, self.zBounds = self.generateTomoBins(self.u, self.nuisancePar.fiducial)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"
      
      print "Masks to select specific ells or 2-point functions",
      tStart = time()
      # 0 for data to keep, 1 for data to discard
      self.lMaxMask = self.generatelMaxMask(kMaxG=0.3)
      #
      self.noNullMask = self.generateNoNullMask()
      self.gsOnlyMask = self.generateGSOnlyMask()
      self.gOnlyMask = self.generateGOnlyMask()
      self.sOnlyMask = self.generateSOnlyMask()
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"
      
      print "Power spectra",
      tStart = time()
      self.p_kk, self.p_kg, self.p_ks, self.p_gg, self.p_gs, self.p_ss, self.p_kk_shot, self.p_gg_shot, self.p_ss_shot = self.generatePowerSpectra(self.u, self.w_k, self.w_g, self.w_s, save=self.save)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"

      print "Data vector",
      tStart = time()
      self.dataVector, self.shotNoiseVector = self.generateDataVector(self.p_kk, self.p_kg, self.p_ks, self.p_gg, self.p_gs, self.p_ss, self.p_kk_shot, self.p_gg_shot, self.p_ss_shot)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"

      print "Covariance matrix",
      tStart = time()
#      self.covMat, self.invCov = self.generateCov(self.p_kk, self.p_kg, self.p_ks, self.p_gg, self.p_gs, self.p_ss, self.p_kk_shot, self.p_gg_shot, self.p_ss_shot, save=self.save)
      self.covMat = self.generateCov(self.p_kk, self.p_kg, self.p_ks, self.p_gg, self.p_gs, self.p_ss, self.p_kk_shot, self.p_gg_shot, self.p_ss_shot, save=self.save)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"
      
      print "Derivatives of the data vector"
      if self.save:
         self.saveDerivativeDataVector()
      self.loadDerivativeDataVector()
      
      print "Fisher matrices",
      tStart = time()
      self.generateAllFisher(save=self.save)
      tStop = time()
      print "("+str(np.round(tStop-tStart,1))+" sec)"

      tStopFisher = time()
      print "Full calculation took "+str(np.round((tStopFisher-tStartFisher)/60.,1))+" min"

   
   ##################################################################################


   def generatelMaxMask(self, kMaxG=0.3):
      '''Creates a mask to discard the clustering modes
      with ell >= kMax * chi - 0.5,
      where chi is the mean comoving distance to the bin,
      and kMax=0.3 h/Mpc,
      as in the DESC SRD 2018.
      mask is 1 for modes to mask out, 0 otherwise.
      Should be called after the tomo bins have been generated.
      '''
      #tStart = time()
      lMaxMask = np.zeros(self.nData)
      iData = 0

      # compute correspponding lMax in each tomo bin
      Z = np.array([self.w_g[iBin].zMean() for iBin in range(self.nBins)])
      Chi = self.u.bg.comoving_distance(Z)
      LMaxBins = (kMaxG * Chi - 0.5)

      # check for kg
      LMaxCheckKG = np.array([self.L > LMaxBins[iBin] for iBin in range(self.nBins)]).flatten()
      # check for gg
      LMaxCheckGG = np.array([self.L > min(LMaxBins[iBin1], LMaxBins[iBin2]) for iBin1 in range(self.nBins) for iBin2 in range(iBin1, self.nBins)]).flatten()
      # check for gs
      LMaxCheckGS = np.array([self.L > LMaxBins[iBin1] for iBin1 in range(self.nBins) for iBin2 in range(self.nBins)]).flatten()

      # kk: no masking needed
      # kg
      lMaxMask[self.nKK*self.nL:(self.nKK+self.nKG)*self.nL] = LMaxCheckKG
      # ks: no masking needed
      # gg:
      lMaxMask[(self.nKK+self.nKG+self.nKS)*self.nL:(self.nKK+self.nKG+self.nKS+self.nGG)*self.nL] = LMaxCheckGG
      # gs:
      lMaxMask[(self.nKK+self.nKG+self.nKS+self.nGG)*self.nL:(self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL] = LMaxCheckGS
      # ss: no masking needed

      #tStop = time()
      #print "took", tStop-tStart, "sec"
      return lMaxMask


#   def generatelMaxMask(self, kMaxG=0.3):
#      '''Creates a mask to discard the clustering modes
#      with ell >= kMax * chi - 0.5,
#      where chi is the mean comoving distance to the bin,
#      and kMax=0.3 h/Mpc,
#      as in the DESC SRD 2018.
#      mask is 1 for modes to mask out, 0 otherwise.
#      Should be called after the tomo bins have been generated.
#      '''
#      lMaxMask = np.zeros(self.nData)
#      iData = 0
#      
#      # kk
#      iData += self.nKK # ie +=1
#      
#      # kg
#      for iBin1 in range(self.nBins):
#         z1 = self.w_g[iBin1].zMean()
#         chi1 = self.u.bg.comoving_distance(z1)
#         lMax = kMaxG * chi1 - 0.5
#         lMaxMask[iData*self.nL:(iData+1)*self.nL] = self.L > lMax
#         iData += 1
#         
#      # ks
#      iData += self.nKS
#      
#      # gg
#      for iBin1 in range(self.nBins):
#         z1 = self.w_g[iBin1].zMean()
#         chi1 = self.u.bg.comoving_distance(z1)
#         for iBin2 in range(iBin1, self.nBins):
#            z2 = self.w_g[iBin2].zMean()
#            chi2 = self.u.bg.comoving_distance(z2)
#            chi = min(chi1, chi2)
#            lMax = kMaxG * chi - 0.5
#            lMaxMask[iData*self.nL:(iData+1)*self.nL] = self.L > lMax
#            iData += 1
#      # gs
#      for iBin1 in range(self.nBins):
#         z1 = self.w_g[iBin1].zMean()
#         chi1 = self.u.bg.comoving_distance(z1)
#         for iBin2 in range(self.nBins):
#            lMax = kMaxG * chi1 - 0.5
#            lMaxMask[iData*self.nL:(iData+1)*self.nL] = self.L > lMax
#            iData += 1
#      return lMaxMask



   def generateNoNullMask(self):
      '''Creates a mask to discard all the spectra that would be null
      if photo-z were perfect.
      I.e. discard gg crosses, and gs where z_g>z_s.
      '''
      noNullMask = np.zeros(self.nData)
      iData = 0
      # kk, kg, ks
      iData += self.nKK + self.nKG + self.nKS
      # gg
      for iBin1 in range(self.nBins):
         for iBin2 in range(iBin1, self.nBins):
            if (iBin2<>iBin1):
               noNullMask[iData*self.nL:(iData+1)*self.nL] = 1
            iData += 1
      # gs
      for iBin1 in range(self.nBins):
         for iBin2 in range(self.nBins):
            if (iBin2<iBin1):
               noNullMask[iData*self.nL:(iData+1)*self.nL] = 1
            iData += 1
      return noNullMask



   def generateGSOnlyMask(self):
      '''Creates a mask to keep only gg, gs, ss:
      0 for kk, kg, ks; 1 for gg, gs and ss.
      '''
      gsOnlyMask = np.zeros(self.nData)
      gsOnlyMask[:(self.nKK+self.nKG+self.nKS)*self.nL] = 1
      return gsOnlyMask



   def generateKOnlyMask(self):
      '''Creates a mask to keep only CMB lensing:
      0 for kk, 1 for kg, ks, gg, gs and ss.
      '''
      gOnlyMask = np.zeros(self.nData)
      gOnlyMask[(self.nKK+self.nKG+self.nKS+self.nGG)*self.nL: (self.nKK+self.nKG+self.nKS+self.nGG+self.nGS+self.nSS)*self.nL] = 1
      return gOnlyMask

   def generateGOnlyMask(self):
      '''Creates a mask to keep only clustering:
      1 for kk, kg, ks, 0 for gg, 1 for gs and ss.
      '''
      gOnlyMask = np.zeros(self.nData)
      gOnlyMask[:(self.nKK+self.nKG+self.nKS)*self.nL] = 1
      gOnlyMask[(self.nKK+self.nKG+self.nKS+self.nGG)*self.nL: (self.nKK+self.nKG+self.nKS+self.nGG+self.nGS+self.nSS)*self.nL] = 1
      return gOnlyMask

   def generateSOnlyMask(self):
      '''Creates a mask to discard the keep only cosmic shear:
      1 for kk, kg, ks, gg, gs, 0 for ss.
      '''
      sOnlyMask = np.zeros(self.nData)
      sOnlyMask[:(self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL] = 1
      return sOnlyMask



   ##################################################################################



   def cij(self, cijPar):
      """Extract the contamination matrix c_ij
      from the photo-z parameters, enforcing the constraint
      c_ii = 1 - sum_{j\neq i} c_{ij}
      The input given should be photoZPar[2*self.nBins:]
      """
      cij = np.zeros((self.nBins, self.nBins))
      par = cijPar
      
      iPar = 0
      for i in range(self.nBins):
         # left of the diagonal
         for j in range(i):
            cij[i,j] = par[iPar]
            iPar += 1
         # right of the diagonal
         for j in range(i+1, self.nBins):
            cij[i,j] = par[iPar]
            iPar += 1
         # diagonal element
         cij[i,i] = 1. - np.sum(cij[i,:])   # ok to include c[i,i] in the sum: it is 0

      return cij



   def dndzPhotoZ(self, u, photoZPar, test=False):
      """Compute the true redshift distributions dn/dz_t of the tomographic bins,
      given the Gaussian photo-z error and potentially the outlier contamination matrix,
      encoded in photoZPar.
      Outputs a dictionary dndz_t of functions dndz_t[iBin] for each bin.
      Computation uses arrays rather than functions and quad, for speed.
      """
   
      # LSST source sample
      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(u, name='glsst')
      # split it into bins
      zBounds = w_glsst.splitBins(self.nBins)
      if test:
         print "Redshift bounds for the bins:", zBounds
      # every tomo bin spans the whole redshift range
      zMin = 1./w_glsst.aMax-1.  #max(zMinP - 5.*sz, 1./w_glsst.aMax-1.)   # 1./w_glsst.aMax-1.
      zMax = 1./w_glsst.aMin-1.  #min(zMaxP + 5.*sz, 1./w_glsst.aMin-1.)   # 1./w_glsst.aMin-1.

      # extract the outlier contamination matrix, if needed
      if len(photoZPar)==self.nBins*(self.nBins+1):
         cij = self.cij(photoZPar[2*self.nBins:])

      # Gaussian photo-z
      dndzG = {}
      tStart = time()
      # Gaussian photo-z bias and uncertainty for all bins
      dz = photoZPar[:self.nBins]
      ddz = dz[:,None,None]
      sz = photoZPar[self.nBins:2*self.nBins] * (1.+0.5*(zBounds[:-1]+zBounds[1:]))
      ssz = sz[:,None,None]
      
      # axes: [iBin, z, zp]
      z = np.linspace(zMin, zMax, 1001)
      zz = z[None,:,None]
      zp = z.copy()
      zzp = zp[None,None,:]
      
      # integrand
      # sharp bins in zp
      integrand = np.exp(-0.5*(zz-zzp-ddz)**2/ssz**2) / np.sqrt(2.*np.pi*ssz**2)
      if test:
         print integrand.shape
      integrand *= (zBounds[:-1,None,None]<zzp) * (zzp<zBounds[1:,None,None])
      integrand *= w_glsst.dndz(zzp)
      
      # do the integrals
      # axes: [iBin, z]
      result = np.trapz(integrand, zp, axis=-1)
      if test:
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         for iBin in range(self.nBins):
            ax.plot(z, result[iBin,:])
         #
         ax.set_title('dn/dz: Gaussian photo-z')
         plt.show()
         
      
      # interpolate
      for iBin in range(self.nBins):
         dndzG[iBin] = interp1d(z, result[iBin,:], kind='linear', bounds_error=False, fill_value=0.)
      tStop = time()
      if test:
         print "-- getting dn/dz took", tStop-tStart, "sec"


      dndz_t = {}
      # If outliers are not included
      if len(photoZPar)==2*self.nBins:
         dndz_t = lambda iBin, z: dndzG[iBin](z)
      # If outliers are included
      elif len(photoZPar)==self.nBins*(self.nBins+1):
         dndz_t = lambda iBin, z: np.sum([cij[iBin, j] * dndzG[j](z) for j in range(self.nBins)], axis=0)
      else:
         print "Error: PhotoZPar does not have the right size"

      if test:
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         for iBin in range(self.nBins):
            ax.plot(z, dndz_t(iBin,z))
         #
         ax.set_title('dn/dz: including outliers')
         plt.show()


      return dndz_t






   def generateTomoBins(self, u, nuisancePar, save=True, doS=True, test=False):
      '''The option doS=False is only used to save time when sampling dn/dz,
      since the s bins take much longer to generate (by a factor ~100).
      '''

      # CMB lensing kernel
      tStart = time()
      w_k = WeightLensSingle(u, z_source=1100., name="cmblens")
      tStop = time()
      if test:
         print "CMB lensing kernel took", tStop-tStart, "sec"

      # split the nuisance parameters
      galaxyBiasPar = nuisancePar[:self.galaxyBiasPar.nPar]
      shearMultBiasPar = nuisancePar[self.galaxyBiasPar.nPar:self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar]
      photoZPar = nuisancePar[self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar:self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar+self.photoZPar.nPar]

      # option to allow different photo-z params for g and s
      if self.photoZSPar is not None:
         photoZSPar = nuisancePar[self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar+self.photoZPar.nPar:]
      
      # LSST source sample
      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(u, name='glsst')
      # split it into bins
      zBounds = w_glsst.splitBins(self.nBins)
      # every tomo bin spans the whole redshift range
      zMin = 1./w_glsst.aMax-1.  #max(zMinP - 5.*sz, 1./w_glsst.aMax-1.)   # 1./w_glsst.aMax-1.
      zMax = 1./w_glsst.aMin-1.  #min(zMaxP + 5.*sz, 1./w_glsst.aMin-1.)   # 1./w_glsst.aMin-1.
      
      
      ##########################################################################
      
      # generate the corresponding tracer and shear bins
      w_g = np.empty(self.nBins, dtype=object)
      w_s = np.empty(self.nBins, dtype=object)
      # Only needed if magnification bias is included
      w_g_nomagbias = np.empty(self.nBins, dtype=object)

      ##########################################################################
      # tomo bins for g and s

      # dn/dz for g
      dnGdz_t = self.dndzPhotoZ(u, photoZPar, test=test)
      # dn/dz for s
      # if same bins for g and s
      if self.photoZSPar is None:
         dnSdz_t = dnGdz_t
      else:
         dnSdz_t = self.dndzPhotoZ(u, photoZSPar, test=test)

      if test:
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         for iBin in range(self.nBins):
            z = np.linspace(zMin, zMax, 1001)
            ax.plot(z, dnGdz_t(iBin, z), '-', label=r'Clustering')
            ax.plot(z, dnSdz_t(iBin, z), '--', label=r'Shear')
         #
         ax.legend(loc=1, fontsize='x-small', labelspacing=0.)
         ax.set_title('dn/dz: including outliers')
         plt.show()


      ##########################################################################
      # projection kernels for power spectra
      
      for iBin in range(self.nBins):

         # Shear bin
         tStart = time()
         w_s[iBin] = WeightLensCustomFast(u,
                                      lambda z, iBin=iBin: dnSdz_t(iBin, z), # dn/dz_true
                                      m=lambda z, iBin=iBin: shearMultBiasPar[iBin], # multiplicative shear bias
                                      zMinG=zMin,
                                      zMaxG=zMax,
                                      name='s'+str(iBin),
                                      nProc=self.nProc)
         tStop = time()
         if test:
            print "-- shear bin took", tStop-tStart, "sec"

         # Clustering bin
         w_g[iBin] = WeightTracerCustom(u,
                                        lambda z, iBin=iBin: galaxyBiasPar[iBin] * w_glsst.b(z), # galaxy bias
                                        lambda z, iBin=iBin: dnGdz_t(iBin, z), # dn/dz_true
                                        zMin=zMin,
                                        zMax=zMax,
                                        name='g'+str(iBin))
         tStop = time()
         if test:
            print "-- clustering bin "+str(iBin)+" took", tStop-tStart, "sec"
            print "mean z = "+str(w_g[iBin].zMean())

         # add magnification bias, if requested
         if self.magBias:
            alpha = w_glsst.magnificationBias(w_g[iBin].zMean())
            #print "bin "+str(iBin)+": mag bias alpha="+str(alpha)
            w_g_nomagbias[iBin] = WeightTracerCustom(u,
                                           lambda z, iBin=iBin: galaxyBiasPar[iBin] * w_glsst.b(z), # galaxy bias
                                           lambda z, iBin=iBin: dnGdz_t(iBin, z), # dn/dz_true
                                           zMin=zMin,
                                           zMax=zMax,
                                           name='g'+str(iBin))
            w_g[iBin].f = lambda a, iBin=iBin: w_g_nomagbias[iBin].f(a) + 2.*(alpha-1.)*w_s[iBin].f(a)

         if test:
            z = np.linspace(zMin, zMax, 1001)
            plt.plot(z, w_s[iBin].f(z), ':', label=r'shear')
            plt.plot(z, w_g[iBin].f(z), '-', label=r'g, no mag bias')
            if self.magBias:
               plt.plot(z, w_g_nomagbias[iBin].f(z), '--', label=r'g, mag bias')
            plt.legend(loc=1, fontsize='x-small', labelspacing=0.)
            plt.title(r'clustering bin '+str(iBin))
            plt.show()
      
      if doS:
         return w_k, w_g, w_s, zBounds
      else:
         return w_k, w_g, zBounds



#   def generateTomoBins(self, u, nuisancePar, save=True, doS=True, test=False):
#      '''The option doS=False is only used to save time when sampling dn/dz,
#      since the s bins take much longer to generate (by a factor ~100).
#      !!! I am using loose mean redshifts for tomo bins at several places.
#      '''
#      # CMB lensing kernel
#      tStart = time()
#      w_k = WeightLensSingle(u, z_source=1100., name="cmblens")
#      tStop = time()
#      if test:
#         print "CMB lensing kernel took", tStop-tStart, "sec"
#
#      # split the nuisance parameters
#      galaxyBiasPar = nuisancePar[:self.galaxyBiasPar.nPar]
#      shearMultBiasPar = nuisancePar[self.galaxyBiasPar.nPar:self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar]
#      photoZPar = nuisancePar[self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar:self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar+self.photoZPar.nPar]
#      # option to allow different photo-z params for g and s
#      if self.photoZSPar is not None:
#         photoZSPar = nuisancePar[self.galaxyBiasPar.nPar+self.shearMultBiasPar.nPar+self.photoZPar.nPar:]
#      
#      # LSST source sample
#      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(u, name='glsst')
#      # split it into bins
#      zBounds = w_glsst.splitBins(self.nBins)
#      # every tomo bin spans the whole redshift range
#      zMin = 1./w_glsst.aMax-1.  #max(zMinP - 5.*sz, 1./w_glsst.aMax-1.)   # 1./w_glsst.aMax-1.
#      zMax = 1./w_glsst.aMin-1.  #min(zMaxP + 5.*sz, 1./w_glsst.aMin-1.)   # 1./w_glsst.aMin-1.
#      
#      
#      ##########################################################################
#      
#      # generate the corresponding tracer and shear bins
#      w_g = np.empty(self.nBins, dtype=object)
#      w_s = np.empty(self.nBins, dtype=object)
#
#      # extract the outlier contamination matrix, if needed
#      if len(photoZPar)==self.nBins*(self.nBins+1):
#         cij = self.cij(photoZPar[2*self.nBins:])
#      # same for the s bins, if different
#      if self.photoZSPar is not None:
#         if len(photoZSPar)==self.nBins*(self.nBins+1):
#            cSij = self.cij(photoZSPar[2*self.nBins:])
#
#      ##########################################################################
#      # tomo bin for g
#
#      # Gaussian photo-z
#      dndzG = {}
#      tStart = time()
#      # Gaussian photo-z bias and uncertainty for all bins
#      dz = photoZPar[:self.nBins]
#      ddz = dz[:,None,None]
#      sz = photoZPar[self.nBins:2*self.nBins] * (1.+0.5*(zBounds[:-1]+zBounds[1:]))
#      ssz = sz[:,None,None]
#      
#      # axes: [iBin, z, zp]
#      z = np.linspace(zMin, zMax, 501)
#      zz = z[None,:,None]
#      zp = z.copy()
#      zzp = zp[None,None,:]
#      
#      # integrand
#      # sharp bins in zp
#      integrand = np.exp(-0.5*(zz-zzp-ddz)**2/ssz**2) / np.sqrt(2.*np.pi*ssz**2)
#      integrand *= (zBounds[:-1,None,None]<zzp) * (zzp<zBounds[1:,None,None])
#      integrand *= w_glsst.dndz(zzp)
#      
#      # integrals
#      # axes: [iBin, z]
#      result = np.trapz(integrand, zp, axis=-1)
#      
#      # interpolate
#      for iBin in range(self.nBins):
#         dndzG[iBin] = interp1d(z, result[iBin,:], kind='linear', bounds_error=False, fill_value=0.)
#      tStop = time()
#      if test:
#         print "-- getting dn/dz took", tStop-tStart, "sec"
#
#
#
#      # Create g bin,
#      # taking into account potential outliers 
#      dndz_t = {}
#      for iBin in range(self.nBins):
#         # If outliers are not included
#         if len(photoZPar)==2*self.nBins:
#            dndz_t[iBin] = lambda z: dndzG[iBin](z)
#         # If outliers are included
#         elif len(photoZPar)==self.nBins*(self.nBins+1):
#            dndz_t[iBin] = lambda z: np.sum(cij[:,iBin] * dndzG[iBin](z))
#         else:
#            print "Error: PhotoZPar does not have the right size"
#
#         tStart = time()
#         # tracer bin
#         w_g[iBin] = WeightTracerCustom(u,
#                                        lambda z: galaxyBiasPar[iBin] * w_glsst.b(z), # galaxy bias
#                                        dndz_t[iBin], # dn/dz_true
#                                        zMin=zMin,
#                                        zMax=zMax,
#                                        name='g'+str(iBin))
#         tStop = time()
#         if test:
#            print "-- clustering bin took", tStop-tStart, "sec"
#
#         # add magnification bias, if requested
#         if self.magBias:
#            alpha = w_glsst.magnificationBias(0.5*(zMinP+zMaxP))
#            print "bin "+str(iBin)+": mag bias alpha="+str(alpha)
#            w_g_nomagbias[iBin] = WeightTracerCustom(u,
#                                           lambda z: galaxyBiasPar[iBin] * w_glsst.b(z), # galaxy bias
#                                           dndz_t[iBin], # dn/dz_true
#                                           zMin=zMin,
#                                           zMax=zMax,
#                                           name='g'+str(iBin))
#            w_g[iBin].f = lambda a: w_g_nomagbias[iBin].f(a) + 2.*(alpha-1.)*w_s[iBin].f(a)
#
#      
#      
#         ##########################################################################
#         # tomo bin for s
#          
#      if doS:
#         
#         for iBin in range(self.nBins):
#
#            tStart = time()
#            # if same bins for g and s
#            if self.photoZSPar is None:
#               w_s[iBin] = WeightLensCustomFast(u,
#                                            dndz_t[iBin], # dn/dz_true
#                                            m=lambda z: shearMultBiasPar[iBin], # multiplicative shear bias
#                                            zMinG=zMin,
#                                            zMaxG=zMax,
#                                            name='s'+str(iBin),
#                                            nProc=self.nProc)
#
##            # if different bins for g and s
##            else:
##               # photo-z bias and uncertainty for this bin:
##               dz = photoZSPar[iBin]
##               sz = photoZSPar[self.nBins+iBin] * (1.+0.5*(zMinP+zMaxP))
##
##               # Gaussian photo-z error
##               p_z_given_zp = lambda zp,z: np.exp(-0.5*(z-zp-dz)**2/sz**2) / np.sqrt(2.*np.pi*sz**2)
##
##               # If outliers are not included
##               if len(photoZSPar)==2*self.nBins:
##                  f = lambda zp,z: w_glsst.dndz(zp) * p_z_given_zp(zp,z)
##                  dndz_tForInterp = lambda z: integrate.quad(f, zMinP, zMaxP, args=(z), epsabs=0., epsrel=1.e-3)[0]
##
##               # If outliers are included
##               elif len(photoZSPar)==self.nBins*(self.nBins+1):
##
##                  def dndzp_outliers(zp):
##                     ''' This is the dn/dz_p, such that for bin i:
##                     n_i^new = n_i * (1-sum_{j \neq i} c_ij) + sum_{j \neq i} c_ji n_j.
##                     '''
##                     result = w_glsst.dndz(zp)
##                     # if zp is in the bin
##                     if zp>=zBounds[iBin] and zp<zBounds[iBin+1]:
##                        result *= 1. - np.sum([cSij[iBin*(self.nBins-1)+j] for j in range(self.nBins-1)])
##                     # if zp is in another bin
##                     else:
##                        # find which bin this is
##                        jBin = np.where(np.array([(zp>=zBounds[j])*(zp<zBounds[j+1]) for j in range(self.nBins)])==1)[0][0]
##                        # since the diagonal c_ii is not encoded, make sure to skip it if iBin > jBin
##                        i = iBin - (iBin>jBin)
##                        result *= cSij[jBin*(self.nBins-1)+i]
##                     return result
##
##                  f = lambda zp,z: dndzp_outliers(zp) * p_z_given_zp(zp,z)
##                  dndz_tForInterp = lambda z: integrate.quad(f, zMin, zMax, args=(z), epsabs=0., epsrel=1.e-3)[0]
##               else:
##                  print "Error: PhotoZPar does not have the right size"
##               tStop = time()
##               if test:
##                  print "-- before dn/dz took", tStop-tStart, "sec"
##               # interpolate it for speed (for lensing kernel calculation)
##               tStart = time()
##               Z = np.linspace(zMin, zMax, 501)
##               with sharedmem.MapReduce(np=self.nProc) as pool:
##                  F = np.array(pool.map(dndz_tForInterp, Z))
##               dnSdz_t = interp1d(Z, F, kind='linear', bounds_error=False, fill_value=0.)
##               tStop = time()
##               if test:
##                  print "-- getting dn/dz took", tStop-tStart, "sec"
##               w_s[iBin] = WeightLensCustom(u,
##                                            dnSdz_t, # dn/dz_true
##                                            m=lambda z: shearMultBiasPar[iBin], # multiplicative shear bias
##                                            zMinG=zMin,
##                                            zMaxG=zMax,
##                                            name='s'+str(iBin),
##                                            nProc=self.nProc)
##
#            tStop = time()
#            if test:
#               print "-- shear bin took", tStop-tStart, "sec"
#      
#      if doS:
#         return w_k, w_g, w_s, zBounds
#      else:
#         return w_k, w_g, zBounds
      

   ##################################################################################

   def generatePowerSpectra(self, u, w_k, w_g, w_s, name=None, save=True):
      if name is None:
         name = "_"+self.name
      
      # Generate all power spectra
      # common for all power spectra
      # axes: [ell, a]
      #nA = 501
      nA = 1001
      aMin = 1./(1. + 6.)
      aMax = 1./(1.+1.e-5)
      a = np.linspace(aMin, aMax, nA)
      z = 1./a-1.
      chi = u.bg.comoving_distance(z)
      #
      integrand = 3.e5/( u.hubble(z) * a**2 )
      fp3d = np.vectorize(u.fPinterp)
      f = lambda l: fp3d((l + 0.5)/chi, z)
      integrand = integrand[None,:] * np.array(map(f, self.L))
      #integrand *= fp3d((self.L[:,None] + 0.5)/chi[None,:], z[None,:]+0.*self.L[:,None])
      #
      # specific for each power spectrum
      def fp2d(w1, w2=None):
         if w2 is None:
            return np.trapz(integrand * w1.f(a)[None,:]**2 / chi[None,:]**2, a, axis=-1)
         else:
            return np.trapz(integrand * w1.f(a)[None,:] * w2.f(a)[None,:] / chi[None,:]**2, a, axis=-1)
   
      # kk
      p_kk = fp2d(w_k)
      p_kk_shot = self.fNk(self.L)
      # kg
      p_kg = {}
      for iBin in range(self.nBins):
         p_kg[iBin] = fp2d(w_k, w_g[iBin])
      # ks
      p_ks = {}
      for iBin in range(self.nBins):
         p_ks[iBin] = fp2d(w_k, w_s[iBin])
      # gg
      p_gg = {}
      p_gg_shot = {}
      for iBin1 in range(self.nBins):
         p_gg[iBin1, iBin1] = fp2d(w_g[iBin1])
         p_gg_shot[iBin1, iBin1] = 1. / w_g[iBin1].ngal
         for iBin2 in range(iBin1+1, self.nBins):
               p_gg[iBin1, iBin2] = fp2d(w_g[iBin1], w_g[iBin2])
               p_gg[iBin2, iBin1] = p_gg[iBin1, iBin2]
               p_gg_shot[iBin1, iBin2] = 0. * self.L
      # gs
      p_gs = {}
      for iBin1 in range(self.nBins):
         for iBin2 in range(self.nBins):
               p_gs[iBin1, iBin2] = fp2d(w_g[iBin1], w_s[iBin2])
      # ss
      p_ss = {}
      p_ss_shot = {}
      for iBin1 in range(self.nBins):
         p_ss[iBin1, iBin1] = fp2d(w_s[iBin1])
         p_ss_shot[iBin1, iBin1]  = 0.26**2 / w_s[iBin1].ngal
         for iBin2 in range(iBin1+1, self.nBins):
               p_ss[iBin1, iBin2] = fp2d(w_s[iBin1], w_s[iBin2])
               p_ss[iBin2, iBin1] = p_ss[iBin1, iBin2]
               p_ss_shot[iBin1, iBin2] = 0. * self.L

      return p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot


   def savePowerSpectra(self):
      '''Saves the power spectra to files.
      Not needed for the Fisher forecast, but can be useful.
      '''
      
      # kk
      data = np.zeros((self.nL, 1+self.nKK))
      data[:,0] = self.L.copy()
      data[:,1] = self.p_kk.copy()
      np.savetxt('./output/p2d/'+self.name+'_p_kk.txt', data)

      # kg
      data = np.zeros((self.nL, 1+self.nKG))
      data[:,0] = self.L.copy()
      for iBin in range(self.nBins):
         data[:, 1+iBin] = self.p_kg[iBin]
      np.savetxt('./output/p2d/'+self.name+'_p_kg.txt', data)
         
      # ks
      data = np.zeros((self.nL, 1+self.nKS))
      data[:,0] = self.L.copy()
      for iBin in range(self.nBins):
         data[:, 1+iBin] = self.p_ks[iBin]
      np.savetxt('./output/p2d/'+self.name+'_p_ks.txt', data)
         
      # gg
      data = np.zeros((self.nL, 1+self.nGG))
      data[:,0] = self.L.copy()
      i = 0
      for iBin in range(self.nBins):
         for jBin in range(iBin+1, self.nBins):
            data[:, 1+i] = self.p_gg[iBin, jBin]
            i += 1
      np.savetxt('./output/p2d/'+self.name+'_p_gg.txt', data)
         
      # gs
      data = np.zeros((self.nL, 1+self.nGS))
      data[:,0] = self.L.copy()
      for iBin in range(self.nBins):
         for jBin in range(self.nBins):
            data[:, 1+iBin*self.nBins+jBin] = self.p_gs[iBin, jBin]
      np.savetxt('./output/p2d/'+self.name+'_p_gs.txt', data)
         
      # ss
      data = np.zeros((self.nL, 1+self.nSS))
      data[:,0] = self.L.copy()
      i = 0
      for iBin in range(self.nBins):
         for jBin in range(iBin+1, self.nBins):
            data[:, 1+i] = self.p_ss[iBin, jBin]
            i += 1
      np.savetxt('./output/p2d/'+self.name+'_p_ss.txt', data)

#   def generatePowerSpectra(self, u, w_k, w_g, w_s, name=None, save=True):
#      if name is None:
#         name = "_"+self.name
#   
#      # common for all power spectra
#      # axes: [ell, a]
#      nA = 501
#      aMin = 1./(1. + 6.)
#      aMax = 1./(1.+1.e-5)
#      a = np.linspace(aMin, aMax, nA)
#      z = 1./a-1.
#      chi = u.bg.comoving_distance(z)
#      #
#      integrand = 3.e5/( u.hubble(z) * a**2 )
#      fp3d = np.vectorize(u.fPinterp)
#      f = lambda l: fp3d((l + 0.5)/chi, z)
#      integrand = integrand[None,:] * np.array(map(f, self.L))
#      #integrand *= fp3d((self.L[:,None] + 0.5)/chi[None,:], z[None,:]+0.*self.L[:,None])
#   
#      # kk
#      p2d_kk = np.trapz(integrand * w_k.f(a)[None,:]**2 / chi[None,:]**2, a, axis=-1)
#      # kg
#      p2d_kg = {}
#      for iBin in range(self.nBins):
#         p2d_kg[iBin] = np.trapz(integrand * w_k.f(a)[None,:] * w_g[iBin].f(a)[None,:] / chi[None,:]**2, a, axis=-1)
#      # ks
#      p2d_ks = {}
#      for iBin in range(self.nBins):
#         p2d_ks[iBin] = np.trapz(integrand * w_k.f(a)[None,:] * w_s[iBin].f(a)[None,:] / chi[None,:]**2, a, axis=-1)
#      # gg
#      p2d_gg = {}
#      for iBin1 in range(self.nBins):
#         p2d_gg[iBin1, iBin1] = np.trapz(integrand * w_g[iBin1].f(a)[None,:]**2 / chi[None,:]**2, a, axis=-1)
#         for iBin2 in range(iBin1+1, self.nBins):
#               p2d_gg[iBin1, iBin2] = np.trapz(integrand * w_g[iBin1].f(a)[None,:] * w_g[iBin2].f(a)[None,:] / chi[None,:]**2, a, axis=-1)
#               p2d_gg[iBin2, iBin1] = p2d_gg[iBin1, iBin2]
#      # gs
#      p2d_gs = {}
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#               p2d_gs[iBin1, iBin2] = np.trapz(integrand * w_g[iBin1].f(a)[None,:] * w_s[iBin2].f(a)[None,:] / chi[None,:]**2, a, axis=-1)
#      # ss
#      p2d_ss = {}
#      for iBin1 in range(self.nBins):
#         p2d_ss[iBin1, iBin1] = np.trapz(integrand * w_s[iBin1].f(a)[None,:]**2 / chi[None,:]**2, a, axis=-1)
#         for iBin2 in range(iBin1+1, self.nBins):
#               p2d_ss[iBin1, iBin2] = np.trapz(integrand * w_s[iBin1].f(a)[None,:] * w_s[iBin2].f(a)[None,:] / chi[None,:]**2, a, axis=-1)
#               p2d_ss[iBin2, iBin1] = p2d_ss[iBin1, iBin2]
#
#
#      return p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss


#   def generatePowerSpectra(self, u, w_k, w_g, w_s, name=None, save=True):
#      if name is None:
#         name = "_"+self.name
#      
#      # kk
#      #cmb = CMB(beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=3.e3, lMaxP=5.e3, atm=False, name="cmbs4")
#      #cmbLensRec = CMBLensRec(cmb, save=False, nProc=3)
#      p2d_kk = P2d(u, u, w_k, fPnoise=self.fNk, doT=False, name=name, L=self.L, nProc=1, save=save)
#      
#      # kg
#      p2d_kg = np.empty((self.nBins), dtype=object)
#      for iBin in range(self.nBins):
#         # auto-correlation: same bin
#         p2d_kg[iBin] = P2d(u, u, w_k, w_g[iBin], fPnoise=lambda l:0., doT=False, name=name, L=self.L, nProc=1, save=save)
#      
#      # ks
#      p2d_ks = np.empty((self.nBins), dtype=object)
#      for iBin in range(self.nBins):
#         # auto-correlation: same bin
#         p2d_ks[iBin] = P2d(u, u, w_k, w_s[iBin], fPnoise=lambda l:0., doT=False, name=name, L=self.L, nProc=1, save=save)
#
#      # gg: do not impose same bin
#      p2d_gg = np.empty((self.nBins, self.nBins), dtype=object)
#      for iBin1 in range(self.nBins):
#         # auto-correlation: same bin
#         p2d_gg[iBin1, iBin1] = P2d(u, u, w_g[iBin1], fPnoise=lambda l:1./w_g[iBin1].ngal, doT=False, name=name, L=self.L, nProc=1, save=save)
#         # cross-correlation: different bins
#         for iBin2 in range(iBin1+1, self.nBins):
#            p2d_gg[iBin1, iBin2] = P2d(u, u, w_g[iBin1], w_g[iBin2], doT=False, name=name, L=self.L, nProc=1, save=save)
#            # so that the order doesn't matter
#            p2d_gg[iBin2, iBin1] = p2d_gg[iBin1, iBin2]
#
#      # gs: do not impose higher z s than g
#      p2d_gs = np.empty((self.nBins, self.nBins), dtype=object)
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#            p2d_gs[iBin1, iBin2] = P2d(u, u, w_g[iBin1], w_s[iBin2], doT=False, name=name, L=self.L, nProc=1, save=save)
#      
#      # ss
#      p2d_ss = np.empty((self.nBins, self.nBins), dtype=object)
#      for iBin1 in range(self.nBins):
#         # auto-correlation: same bin
#         p2d_ss[iBin1, iBin1] = P2d(u, u, w_s[iBin1], fPnoise=lambda l:0.26**2/w_s[iBin1].ngal, doT=False, name=name, L=self.L, nProc=1, save=save)
#         # cross correlation: different bins
#         for iBin2 in range(iBin1+1, self.nBins):
#            p2d_ss[iBin1, iBin2] = P2d(u, u, w_s[iBin1], w_s[iBin2], doT=False, name=name, L=self.L, nProc=1, save=save)
#            # so that the order doesn't matter
#            p2d_ss[iBin2, iBin1] = p2d_ss[iBin1, iBin2]
#
#      return p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss


   ##################################################################################


   def generateDataVector(self, p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot):
      '''The data vector is made of the various power spectra,
      with a choice of units that makes the covariance matrix for gg and ss more similar,
      and with an ell^alpha factor that makes the covariance matrix more ell-independent.
      The "shotNoiseVector" is the vector of shot noises and shape noises, useful to see whether
      we are cosmic variance or shot noise limited
      '''
      # Generate data vector, and shot noise vector
      dataVector = np.zeros(self.nData)
      shotNoiseVector = np.zeros(self.nData)
      iData = 0
      # kk
      dataVector[iData*self.nL:(iData+1)*self.nL] = p_kk * self.kUnit**2 * self.L**self.alpha
      shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = p_kk_shot * self.kUnit**2 * self.L**self.alpha
      iData += 1
      # kg
      for iBin1 in range(self.nBins):
         dataVector[iData*self.nL:(iData+1)*self.nL] = p_kg[iBin1] * self.kUnit * self.L**self.alpha
         iData += 1
      # ks
      for iBin1 in range(self.nBins):
         dataVector[iData*self.nL:(iData+1)*self.nL] = p_ks[iBin1] * self.kUnit * self.sUnit * self.L**self.alpha
         iData += 1
      # gg
      for iBin1 in range(self.nBins):
         for iBin2 in range(iBin1, self.nBins):
            dataVector[iData*self.nL:(iData+1)*self.nL] = p_gg[iBin1, iBin2] * self.L**self.alpha
            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = p_gg_shot[iBin1, iBin2] * self.L**self.alpha
            iData += 1
      # gs
      for iBin1 in range(self.nBins):
         for iBin2 in range(self.nBins):
            dataVector[iData*self.nL:(iData+1)*self.nL] = p_gs[iBin1, iBin2] * self.sUnit * self.L**self.alpha
            iData += 1
      # ss
      for iBin1 in range(self.nBins):
         for iBin2 in range(iBin1, self.nBins):
            dataVector[iData*self.nL:(iData+1)*self.nL] = p_ss[iBin1, iBin2] * self.sUnit**2 * self.L**self.alpha
            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = p_ss_shot[iBin1, iBin2] * self.sUnit**2 * self.L**self.alpha
            iData += 1

      return dataVector, shotNoiseVector



#   def generateDataVector(self, p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss):
#      '''The data vector is made of the various power spectra,
#      with a choice of units that makes the covariance matrix for gg and ss more similar,
#      and with an ell^alpha factor that makes the covariance matrix more ell-independent.
#      The "shotNoiseVector" is the vector of shot noises and shape noises, useful to see whether
#      we are cosmic variance or shot noise limited
#      '''
#      dataVector = np.zeros(self.nData)
#      shotNoiseVector = np.zeros(self.nData)
#      iData = 0
#      
#      # kk
#      dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_kk.fPinterp, self.L))
#      dataVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#      #
#      shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_kk.fPnoise, self.L))
#      shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#      #
#      iData += 1
#      # kg
#      for iBin1 in range(self.nBins):
#         dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_kg[iBin1].fPinterp, self.L))
#         dataVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#         #
#         shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_kg[iBin1].fPnoise, self.L))
#         shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#         #
#         iData += 1
#      # ks
#      for iBin1 in range(self.nBins):
#         dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_ks[iBin1].fPinterp, self.L))
#         dataVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit * self.L**self.alpha
#         #
#         shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_ks[iBin1].fPnoise, self.L))
#         shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit * self.L**self.alpha
#         #
#         iData += 1
#      # gg
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_gg[iBin1, iBin2].fPinterp, self.L))
#            dataVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#            #
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_gg[iBin1, iBin2].fPnoise, self.L))
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.L**self.alpha
#            #
#            iData += 1
#      # gs
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#            dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_gs[iBin1, iBin2].fPinterp, self.L))
#            dataVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit * self.L**self.alpha
#            #
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_gs[iBin1, iBin2].fPnoise, self.L))
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit * self.L**self.alpha
#            #
#            iData += 1
#      # ss
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            dataVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_ss[iBin1, iBin2].fPinterp, self.L))
#            dataVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit**2 * self.L**self.alpha
#            #
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] = np.array(map(p2d_ss[iBin1, iBin2].fPnoise, self.L))
#            shotNoiseVector[iData*self.nL:(iData+1)*self.nL] *= self.sUnit**2 * self.L**self.alpha
#            #
#            iData += 1
#
#      return dataVector, shotNoiseVector


   ##################################################################################

   def generateCov(self, p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot, save=True):
      covMat = np.zeros((self.nData, self.nData))
      # below, i1 and i2 define the row and column of the nL*nL blocks for each pair of 2-point function
      # i1, i2 \in [0, n2pt]
   
      if not save:
         print "read cov"
         path = './output/cov/'+self.name+'_cov.npy'
         covMat = np.load(path)
#         print "read inv cov"
#         path = './output/cov/'+self.name+'_invcov.npy'
#         invCov = np.load(path)

      else:
         # include the shot noises
         p_kk += p_kk_shot
         for iBin in range(self.nBins):
            p_gg[iBin, iBin] += p_gg_shot[iBin, iBin]
            p_ss[iBin, iBin] += p_ss_shot[iBin, iBin]
         # generic Gaussian cov
         cov = lambda Pac, Pbd, Pad, Pbc, Npairs: np.diagflat((Pac * Pbd + Pad * Pbc) / Npairs)
         
         # "kk-kk"
         i1 = 0
         i2 = 0
         covBlock = cov(p_kk, p_kk, p_kk, p_kk, self.Nmodes)
         covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**4 * self.L**(2*self.alpha) * covBlock
         
         # "kk-kg"
         i1 = 0
         i2 = self.nKK
         # considering kg[i1]
         for iBin1 in range(self.nBins):
            covBlock = cov(p_kk, p_kg[iBin1], p_kg[iBin1], p_kk, self.Nmodes)
            covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**3 * self.L**(2*self.alpha) * covBlock
            # move to next column
            i2 += 1
         
         # "kk-ks"
         i1 = 0
         i2 = self.nKK + self.nKG
         # considering ks[i1]
         for iBin1 in range(self.nBins):
            covBlock = cov(p_kk, p_ks[iBin1], p_ks[iBin1], p_kk, self.Nmodes)
            covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**3 * self.sUnit * self.L**(2*self.alpha) * covBlock
            # move to next column
            i2 += 1

         # "kk-gg"
         i1 = 0
         i2 = self.nKK + self.nKG + self.nKS
         # considering gg[i1, j1]
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               covBlock = cov(p_kg[iBin1], p_kg[iBin2], p_kg[iBin2], p_kg[iBin1], self.Nmodes)
               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1

         # "kk-gs"
         i1 = 0
         i2 = self.nKK + self.nKG + self.nKS + self.nGG
         # considering gs[i1, j1]
         for iBin1 in range(self.nBins):
            for iBin2 in range(self.nBins):
               covBlock = cov(p_kg[iBin1], p_ks[iBin2], p_ks[iBin2], p_kg[iBin1], self.Nmodes)
               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.sUnit * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1

         # "kk-ss"
         i1 = 0
         i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
         # considering ss[i1, j1]
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               covBlock = cov(p_ks[iBin1], p_ks[iBin2], p_ks[iBin2], p_ks[iBin1], self.Nmodes)
               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.sUnit**2 * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1

         # "kg-kg"
         i1 = self.nKK
         # considering kg[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK
            # considering kg[j1]
            for jBin1 in range(self.nBins):
               # compute only upper diagonal
               if i2>=i1:
                  covBlock = cov(p_kk, p_gg[iBin1, jBin1], p_kg[jBin1], p_kg[iBin1], self.Nmodes)
                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1
            # move to next row
            i1 += 1
         
         # "kg-ks"
         i1 = self.nKK
         # considering kg[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG
            # considering ks[j1]
            for jBin1 in range(self.nBins):
               # compute only upper diagonal
               if i2>=i1:
                  covBlock = cov(p_kk, p_gs[iBin1, jBin1], p_ks[jBin1], p_kg[iBin1], self.Nmodes)
                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.sUnit * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1
            # move to next row
            i1 += 1

         # "kg-gg"
         i1 = self.nKK
         # considering kg[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS
            # considering gg[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(jBin1, self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     covBlock = cov(p_kg[jBin1], p_gg[iBin1, jBin2], p_kg[jBin2], p_gg[iBin1, jBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1

         # "kg-gs"
         i1 = self.nKK
         # considering kg[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS + self.nGG
            # considering gs[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     covBlock = cov(p_kg[jBin1], p_gs[iBin1, jBin2], p_ks[jBin2], p_gg[iBin1, jBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.sUnit * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1

         # "kg-ss"
         # considering kg[i1]
         i1 = self.nKK
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
            # considering ss[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(jBin1, self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     covBlock = cov(p_ks[jBin1], p_gs[iBin1, jBin2], p_ks[jBin2], p_gs[iBin1, jBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.sUnit**2 * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1

         # "ks-ks"
         i1 = self.nKK + self.nKG
         # considering ks[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG
            # considering ks[j1]
            for jBin1 in range(self.nBins):
               # compute only upper diagonal
               if i2>=i1:
                  covBlock = cov(p_kk, p_ss[iBin1, jBin1], p_ks[jBin1], p_ks[iBin1], self.Nmodes)
                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit**2 * self.sUnit**2 * self.L**(2*self.alpha) * covBlock
               # move to next column
               i2 += 1
            # move to next row
            i1 += 1

         # "ks-gg"
         i1 = self.nKK + self.nKG
         # considering ks[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS
            # considering gg[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(jBin1, self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     # watch the order for gs
                     covBlock = cov(p_kg[jBin1], p_gs[jBin2, iBin1], p_kg[jBin2], p_gs[jBin1, iBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.sUnit * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1

         # "ks-gs"
         i1 = self.nKK + self.nKG
         # considering ks[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS + self.nGG
            # considering gs[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     # watch the order for gs
                     covBlock = cov(p_kg[jBin1], p_ss[iBin1, jBin2], p_ks[jBin2], p_gs[jBin1, iBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.sUnit**2 * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1

         # "ks-ss"
         i1 = self.nKK + self.nKG
         # considering ks[i1]
         for iBin1 in range(self.nBins):
            i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
            # considering ss[j1, j2]
            for jBin1 in range(self.nBins):
               for jBin2 in range(jBin1, self.nBins):
                  # compute only upper diagonal
                  if i2>=i1:
                     covBlock = cov(p_ks[jBin1], p_ss[iBin1, jBin2], p_ks[jBin2], p_ss[iBin1, jBin1], self.Nmodes)
                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.kUnit * self.sUnit**3 * self.L**(2*self.alpha) * covBlock
                  # move to next column
                  i2 += 1
            # move to next row
            i1 += 1
         
         #print "gg-gg"
         # considering gg[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               # considering gg[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS
               for jBin1 in range(self.nBins):
                  for jBin2 in range(jBin1, self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        covBlock = cov(p_gg[iBin1,jBin1], p_gg[iBin2,jBin2], p_gg[iBin1,jBin2], p_gg[iBin2,jBin1], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         #print "gg-gs"
         # considering gg[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               # considering gs[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS + self.nGG
               for jBin1 in range(self.nBins):
                  for jBin2 in range(self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        covBlock = cov(p_gg[iBin1,jBin1], p_gs[iBin2,jBin2], p_gs[iBin1,jBin2], p_gg[iBin2,jBin1], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) *  covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         #print "gg-ss"
         # considering gg[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               # considering ss[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
               for jBin1 in range(self.nBins):
                  for jBin2 in range(jBin1, self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        covBlock = cov(p_gs[iBin1,jBin1], p_gs[iBin2,jBin2], p_gs[iBin1,jBin2], p_gs[iBin2,jBin1], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         #print "gs-gs"
         # considering gs[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS + self.nGG
         for iBin1 in range(self.nBins):
            for iBin2 in range(self.nBins):
               # considering gs[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS + self.nGG
               for jBin1 in range(self.nBins):
                  for jBin2 in range(self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        # watch the order for gs
                        covBlock = cov(p_gg[iBin1,jBin1], p_ss[iBin2,jBin2], p_gs[iBin1,jBin2], p_gs[jBin1,iBin2], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         #print "gs-ss"
         # considering gs[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS + self.nGG
         for iBin1 in range(self.nBins):
            for iBin2 in range(self.nBins):
               # considering ss[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
               for jBin1 in range(self.nBins):
                  for jBin2 in range(jBin1, self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        covBlock = cov(p_gs[iBin1,jBin1], p_ss[iBin2,jBin2], p_gs[iBin1,jBin2], p_ss[iBin2,jBin1], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**3 * self.L**(2*self.alpha) * covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         #print "ss-ss"
         # considering ss[i1,i2]
         i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               # considering ss[j1,j2]
               i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
               for jBin1 in range(self.nBins):
                  for jBin2 in range(jBin1, self.nBins):
                     # compute only upper diagonal
                     if i2>=i1:
                        covBlock = cov(p_ss[iBin1,jBin1], p_ss[iBin2,jBin2], p_ss[iBin1,jBin2], p_ss[iBin2,jBin1], self.Nmodes)
                        covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**4 * self.L**(2*self.alpha) * covBlock
                     # move to next column
                     i2 += 1
               # move to next row
               i1 += 1

         # fill lower diagonal by symmetry
         # here i1 and i2 don't index the matrix blocks, but the matrix elements
         for i1 in range(self.nData):
            for i2 in range(i1):
               covMat[i1, i2] = covMat[i2, i1]


         # save the cov matrix
         path = './output/cov/'+self.name+'_cov.npy'
         print "Save cov mat to:", path
         np.save(path, covMat)

         # save the inverse cov matrix
#         invCov = invertMatrixSvdTruncated(covMat, epsilon=1.e-8, keepLow=True)
#         invCov = np.linalg.inv(covMat)
#         path = './output/cov/'+self.name+'_invcov.npy'
#         np.save(path, invCov)


      return covMat#, invCov











#   def generateCov(self, p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss):
#      covMat = np.zeros((self.nData, self.nData))
#      # below, i1 and i2 define the row and column of the nL*nL blocks for each pair of 2-point function
#      # i1, i2 \in [0, n2pt]
#      
#      # "kk-kk"
#      i1 = 0
#      i2 = 0
#      covBlock = CovP2d(p2d_kk, p2d_kk, p2d_kk, p2d_kk, self.Nmodes)
#      covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#      
#      # "kk-kg"
#      i1 = 0
#      i2 = self.nKK
#      # considering kg[i1]
#      for iBin1 in range(self.nBins):
#         covBlock = CovP2d(p2d_kk, p2d_kg[iBin1], p2d_kg[iBin1], p2d_kk, self.Nmodes)
#         covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#         # move to next column
#         i2 += 1
#      
#      # "kk-ks"
#      i1 = 0
#      i2 = self.nKK + self.nKG
#      # considering ks[i1]
#      for iBin1 in range(self.nBins):
#         covBlock = CovP2d(p2d_kk, p2d_ks[iBin1], p2d_ks[iBin1], p2d_kk, self.Nmodes)
#         covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) * covBlock.covMat
#         # move to next column
#         i2 += 1
#
#      # "kk-gg"
#      i1 = 0
#      i2 = self.nKK + self.nKG + self.nKS
#      # considering gg[i1, j1]
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            covBlock = CovP2d(p2d_kg[iBin1], p2d_kg[iBin2], p2d_kg[iBin2], p2d_kg[iBin1], self.Nmodes)
#            covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#
#      # "kk-gs"
#      i1 = 0
#      i2 = self.nKK + self.nKG + self.nKS + self.nGG
#      # considering gs[i1, j1]
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#            covBlock = CovP2d(p2d_kg[iBin1], p2d_ks[iBin2], p2d_ks[iBin2], p2d_kg[iBin1], self.Nmodes)
#            covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#
#      # "kk-ss"
#      i1 = 0
#      i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#      # considering ss[i1, j1]
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            covBlock = CovP2d(p2d_ks[iBin1], p2d_ks[iBin2], p2d_ks[iBin2], p2d_ks[iBin1], self.Nmodes)
#            covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#
#      # "kg-kg"
#      i1 = self.nKK
#      # considering kg[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK
#         # considering kg[j1]
#         for jBin1 in range(self.nBins):
#            # compute only upper diagonal
#            if i2>=i1:
#               covBlock = CovP2d(p2d_kk, p2d_gg[iBin1, jBin1], p2d_kg[jBin1], p2d_kg[iBin1], self.Nmodes)
#               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#         # move to next row
#         i1 += 1
#         
#      # "kg-ks"
#      i1 = self.nKK
#      # considering kg[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG
#         # considering ks[j1]
#         for jBin1 in range(self.nBins):
#            # compute only upper diagonal
#            if i2>=i1:
#               covBlock = CovP2d(p2d_kk, p2d_gs[iBin1, jBin1], p2d_ks[jBin1], p2d_kg[iBin1], self.Nmodes)
#               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "kg-gg"
#      i1 = self.nKK
#      # considering kg[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS
#         # considering gg[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(jBin1, self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  covBlock = CovP2d(p2d_kg[jBin1], p2d_gg[iBin1, jBin2], p2d_kg[jBin2], p2d_gg[iBin1, jBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "kg-gs"
#      i1 = self.nKK
#      # considering kg[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS + self.nGG
#         # considering gs[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  covBlock = CovP2d(p2d_kg[jBin1], p2d_gs[iBin1, jBin2], p2d_ks[jBin2], p2d_gg[iBin1, jBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "kg-ss"
#      # considering kg[i1]
#      i1 = self.nKK
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#         # considering ss[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(jBin1, self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  covBlock = CovP2d(p2d_ks[jBin1], p2d_gs[iBin1, jBin2], p2d_ks[jBin2], p2d_gs[iBin1, jBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "ks-ks"
#      i1 = self.nKK + self.nKG
#      # considering ks[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG
#         # considering ks[j1]
#         for jBin1 in range(self.nBins):
#            # compute only upper diagonal
#            if i2>=i1:
#               covBlock = CovP2d(p2d_kk, p2d_ss[iBin1, jBin1], p2d_ks[jBin1], p2d_ks[iBin1], self.Nmodes)
#               covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#            # move to next column
#            i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "ks-gg"
#      i1 = self.nKK + self.nKG
#      # considering ks[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS
#         # considering gg[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(jBin1, self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  # watch the order for gs
#                  covBlock = CovP2d(p2d_kg[jBin1], p2d_gs[jBin2, iBin1], p2d_kg[jBin2], p2d_gs[jBin1, iBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "ks-gs"
#      i1 = self.nKK + self.nKG
#      # considering ks[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS + self.nGG
#         # considering gs[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  # watch the order for gs
#                  covBlock = CovP2d(p2d_kg[jBin1], p2d_ss[iBin1, jBin2], p2d_ks[jBin2], p2d_gs[jBin1, iBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#
#      # "ks-ss"
#      i1 = self.nKK + self.nKG
#      # considering ks[i1]
#      for iBin1 in range(self.nBins):
#         i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#         # considering ss[j1, j2]
#         for jBin1 in range(self.nBins):
#            for jBin2 in range(jBin1, self.nBins):
#               # compute only upper diagonal
#               if i2>=i1:
#                  covBlock = CovP2d(p2d_ks[jBin1], p2d_ss[iBin1, jBin2], p2d_ks[jBin2], p2d_ss[iBin1, jBin1], self.Nmodes)
#                  covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**3 * self.L**(2*self.alpha) * covBlock.covMat
#               # move to next column
#               i2 += 1
#         # move to next row
#         i1 += 1
#      
#      #print "gg-gg"
#      # considering gg[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            # considering gg[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(jBin1, self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     covBlock = CovP2d(p2d_gg[iBin1,jBin1], p2d_gg[iBin2,jBin2], p2d_gg[iBin1,jBin2], p2d_gg[iBin2,jBin1], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.L**(2*self.alpha) * covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      #print "gg-gs"
#      # considering gg[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            # considering gs[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS + self.nGG
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     covBlock = CovP2d(p2d_gg[iBin1,jBin1], p2d_gs[iBin2,jBin2], p2d_gs[iBin1,jBin2], p2d_gg[iBin2,jBin1], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit * self.L**(2*self.alpha) *  covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      #print "gg-ss"
#      # considering gg[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            # considering ss[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(jBin1, self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     covBlock = CovP2d(p2d_gs[iBin1,jBin1], p2d_gs[iBin2,jBin2], p2d_gs[iBin1,jBin2], p2d_gs[iBin2,jBin1], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      #print "gs-gs"
#      # considering gs[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS + self.nGG
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#            # considering gs[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS + self.nGG
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     # watch the order for gs
#                     covBlock = CovP2d(p2d_gg[iBin1,jBin1], p2d_ss[iBin2,jBin2], p2d_gs[iBin1,jBin2], p2d_gs[jBin1,iBin2], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**2 * self.L**(2*self.alpha) * covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      #print "gs-ss"
#      # considering gs[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS + self.nGG
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(self.nBins):
#            # considering ss[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(jBin1, self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     covBlock = CovP2d(p2d_gs[iBin1,jBin1], p2d_ss[iBin2,jBin2], p2d_gs[iBin1,jBin2], p2d_ss[iBin2,jBin1], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**3 * self.L**(2*self.alpha) * covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      #print "ss-ss"
#      # considering ss[i1,i2]
#      i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#      for iBin1 in range(self.nBins):
#         for iBin2 in range(iBin1, self.nBins):
#            # considering ss[j1,j2]
#            i2 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#            for jBin1 in range(self.nBins):
#               for jBin2 in range(jBin1, self.nBins):
#                  # compute only upper diagonal
#                  if i2>=i1:
#                     covBlock = CovP2d(p2d_ss[iBin1,jBin1], p2d_ss[iBin2,jBin2], p2d_ss[iBin1,jBin2], p2d_ss[iBin2,jBin1], self.Nmodes)
#                     covMat[i1*self.nL:(i1+1)*self.nL, i2*self.nL:(i2+1)*self.nL] = self.sUnit**4 * self.L**(2*self.alpha) * covBlock.covMat
#                  # move to next column
#                  i2 += 1
#            # move to next row
#            i1 += 1
#
#      # fill lower diagonal by symmetry
#      # here i1 and i2 don't index the matrix blocks, but the matrix elements
#      for i1 in range(self.nData):
#         for i2 in range(i1):
#            covMat[i1, i2] = covMat[i2, i1]
#
#      return covMat


   ##################################################################################
   
   def printSnrPowerSpectra(self):
      path = self.figurePath+"/snr.txt"
      with open(path, 'w') as f:
         f.write("SNR\n\n")

         ###########################################################
         # kk
         
         f.write("KK\n")
         I = range(0*self.nL, self.nKK*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total kk: "+str(snr)+"\n\n")
 

         ###########################################################
         # kg
         
         f.write("KG\n")
         f.write("all\n")
         i1 = self.nKK
         for iBin1 in range(self.nBins):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+": "+str(snr)+"\n")
            i1 += 1
         # total
         I = range(self.nKK*self.nL, (self.nKK+self.nKG)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total kg: "+str(snr)+"\n\n")


         ###########################################################
         # ks

         f.write("KS\n")
         f.write("all\n")
         i1 = self.nKK + self.nKG
         for iBin1 in range(self.nBins):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+": "+str(snr)+"\n")
            i1 += 1
         # total
         I = range((self.nKK+self.nKG)*self.nL, (self.nKK+self.nKG+self.nKS)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total ks: "+str(snr)+"\n\n")


         ###########################################################
         # gg
         
         # gg: auto
         f.write("GG\n")
         f.write("auto\n")
         i1 = self.nKK + self.nKG + self.nKS
         Itotal = []
         for iBin1 in range(self.nBins):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+","+str(iBin1)+": "+str(snr)+"\n")
            i1 += self.nBins - iBin1
            Itotal += I
         # gg: total auto
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=Itotal)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=Itotal)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total auto: "+str(snr)+"\n")
         
         
         # gg: cross i,i+1
         f.write("cross i,i+1\n")
         i1 = self.nKK + self.nKG + self.nKS + 1
         Itotal = []
         for iBin1 in range(self.nBins-1):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+","+str(iBin1+1)+": "+str(snr)+"\n")
            i1 += self.nBins - iBin1
            Itotal += I
         # gg: total i,i+1
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=Itotal)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=Itotal)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total i,i+1: "+str(snr)+"\n")


         # gg: cross i,i+2
         f.write("cross i,i+2\n")
         i1 = self.nKK + self.nKG + self.nKS + 2
         Itotal = []
         for iBin1 in range(self.nBins-2):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+","+str(iBin1+2)+": "+str(snr)+"\n")
            i1 += self.nBins - iBin1
            Itotal += I
         # gg: total i,i+2
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=Itotal)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=Itotal)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total i,i+2: "+str(snr)+"\n")
         
         # gg: all
         f.write("all\n")
         i1 = self.nKK + self.nKG + self.nKS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               I = range(i1*self.nL, (i1+1)*self.nL)
               d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
               cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
      #         invCov = np.linalg.inv(cov)
      #         snr = np.dot(d.transpose(), np.dot(invCov, d))
               snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
               snr = np.sqrt(snr)
               f.write("   "+str(iBin1)+","+str(iBin2)+": "+str(snr)+"\n")
               i1 += 1
         # gg: total
         I = range((self.nKK+self.nKG+self.nKS)*self.nL, (self.nKK+self.nKG+self.nKS+self.nGG)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total gg: "+str(snr)+"\n\n")

         ###########################################################
         # gs

         # gs: all
         f.write("GS\n")
         f.write("all\n")
         i1 = self.nKK + self.nKG + self.nKS + self.nGG
         for iBin1 in range(self.nBins):
            for iBin2 in range(self.nBins):
               I = range(i1*self.nL, (i1+1)*self.nL)
               d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
               cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
      #         invCov = np.linalg.inv(cov)
      #         snr = np.dot(d.transpose(), np.dot(invCov, d))
               snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
               snr = np.sqrt(snr)
               f.write("   "+str(iBin1)+","+str(iBin2)+": "+str(snr)+"\n")
               i1 += 1

         # gs, no null crosses
         I = range((self.nKK+self.nKG+self.nKS+self.nGG)*self.nL, (self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask + self.noNullMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask + self.noNullMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total gs, no null crosses: "+str(snr)+"\n\n")

         # gs: total
         I = range((self.nKK+self.nKG+self.nKS+self.nGG)*self.nL, (self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total gs: "+str(snr)+"\n\n")


         ###########################################################
         # ss
         
         f.write("SS\n")
         
         # ss: auto
         f.write("auto\n")
         i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
         Itotal = []
         for iBin1 in range(self.nBins):
            I = range(i1*self.nL, (i1+1)*self.nL)
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
   #         invCov = np.linalg.inv(cov)
   #         snr = np.dot(d.transpose(), np.dot(invCov, d))
            snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
            snr = np.sqrt(snr)
            f.write("   "+str(iBin1)+","+str(iBin1)+": "+str(snr)+"\n")
            i1 += self.nBins - iBin1
            Itotal += I
         # ss: total auto
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=Itotal)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=Itotal)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total auto: "+str(snr)+"\n")

         # ss: all
         f.write("all\n")
         i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
         for iBin1 in range(self.nBins):
            for iBin2 in range(iBin1, self.nBins):
               I = range(i1*self.nL, (i1+1)*self.nL)
               d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
               cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
      #         invCov = np.linalg.inv(cov)
      #         snr = np.dot(d.transpose(), np.dot(invCov, d))
               snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
               snr = np.sqrt(snr)
               f.write("   "+str(iBin1)+","+str(iBin2)+": "+str(snr)+"\n")
               i1 += 1
         # ss: total
         I = range((self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL, (self.nKK+self.nKG+self.nKS+self.nGG+self.nGS+self.nSS)*self.nL)
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("total ss: "+str(snr)+"\n\n")

         ###########################################################
         # Combinations

         # gg, gs, ss
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask + self.gsOnlyMask)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask + self.gsOnlyMask)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("Total gg, gs, ss: "+str(snr)+"\n\n")

         # gg, gs, ss, no null crosses
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask + self.gsOnlyMask + self.noNullMask)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask + self.gsOnlyMask + self.noNullMask)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("Total gg, gs, ss, no null crosses: "+str(snr)+"\n\n")

         # All, no null crosses
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask + self.noNullMask)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask + self.noNullMask)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("Total, no null crosses: "+str(snr)+"\n\n")

         # All
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask)
#         invCov = np.linalg.inv(cov)
#         snr = np.dot(d.transpose(), np.dot(invCov, d))
         snr = np.dot(d.transpose(), np.linalg.solve(cov, d))
         snr = np.sqrt(snr)
         f.write("Total: "+str(snr)+"\n\n")


   
   ##################################################################################


   def saveDerivativeDataVector(self):
      # Derivatives of the data vector:
      # matrix of size self.params.nPar x self.nData
      derivative = np.zeros((self.fullPar.nPar, self.nData))
      
      
      def derivativeWrtCosmoPar(iPar):
         print "Derivative wrt "+self.cosmoPar.names[iPar],
         tStart = time()
         # high
         name = self.name+self.cosmoPar.names[iPar]+"high"
         cosmoParClassy = self.cosmoPar.paramsClassy.copy()
#         print cosmoParClassy
#         print "#"
         cosmoParClassy[self.cosmoPar.names[iPar]] = self.cosmoPar.paramsClassyHigh[self.cosmoPar.names[iPar]]
#         print cosmoParClassy
         u = Universe(cosmoParClassy)
         w_k, w_g, w_s, zBounds = self.generateTomoBins(u, self.nuisancePar.fiducial)
         p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot = self.generatePowerSpectra(u, w_k, w_g, w_s, name=name, save=True)
         dataVectorHigh, shotNoiseVectorHigh = self.generateDataVector(p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot)
         # low
         name = self.name+self.cosmoPar.names[iPar]+"low"
         cosmoParClassy = self.cosmoPar.paramsClassy.copy()
         cosmoParClassy[self.cosmoPar.names[iPar]] = self.cosmoPar.paramsClassyLow[self.cosmoPar.names[iPar]]
         u = Universe(cosmoParClassy)
         w_k, w_g, w_s, zBounds = self.generateTomoBins(u, self.nuisancePar.fiducial)
         p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_shot, p_gg_shot, p_ss_shot = self.generatePowerSpectra(u, w_k, w_g, w_s, name=name, save=True)
         dataVectorLow, shotNoiseVectorLow = self.generateDataVector(p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_kk_shot, p_gg_shot, p_ss_shot)
         # derivative
         result = (dataVectorHigh-dataVectorLow) / (self.cosmoPar.high[iPar]-self.cosmoPar.low[iPar])
#         derivative[iPar,:] = (dataVectorHigh-dataVectorLow) / (self.cosmoPar.high[iPar]-self.cosmoPar.low[iPar])
#         derivative[iPar,:] = (dataVectorHigh-self.dataVector) / (self.cosmoPar.high[iPar]-self.cosmoPar.fiducial[iPar])

#         print "all zero?"
#         print np.mean(dataVectorHigh-self.dataVector) / np.std(self.dataVector)
#         print self.cosmoPar.high[iPar]-self.cosmoPar.fiducial[iPar]

         # check that all went well
         if not all(np.isfinite(derivative[iPar,:])):
            print "########"
            print "problem with "+self.cosmoPar.names[iPar]
            print "high value = "+str(self.cosmoPar.high[iPar])
            print "low value = "+str(self.cosmoPar.fiducial[iPar])
         tStop = time()
         print "("+str(np.round(tStop-tStart,1))+" sec)"
         
         return result
      
      
#      with sharedmem.MapReduce(np=self.nProc) as pool:
#         result = pool.map(derivativeWrtCosmoPar, range(self.cosmoPar.nPar))
#         derivative[:self.cosmoPar.nPar,:] = result.copy()
      derivative[:self.cosmoPar.nPar,:] = np.array(map(derivativeWrtCosmoPar, range(self.cosmoPar.nPar)))
      
      
      
      # Nuisance parameters
      def derivativeWrtNuisancePar(iPar):
         print "Derivative wrt "+self.nuisancePar.names[iPar],
         tStart = time()
         params = self.nuisancePar.fiducial.copy()
         # high
         name = "_"+self.name+self.nuisancePar.names[iPar]+"high"
         params[iPar] = self.nuisancePar.high[iPar]
         w_k, w_g, w_s, zBounds = self.generateTomoBins(self.u, params)
         p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_shot, p_gg_shot, p_ss_shot = self.generatePowerSpectra(self.u, w_k, w_g, w_s, name=name, save=True)
         dataVectorHigh, shotNoiseVectorHigh = self.generateDataVector(p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_shot, p_gg_shot, p_ss_shot)
         # low
         name = self.name+self.nuisancePar.names[iPar]+"low"
         params[iPar] = self.nuisancePar.low[iPar]
         w_k, w_g, w_s, zBounds = self.generateTomoBins(self.u, params)
         p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_shot, p_gg_shot, p_ss_shot = self.generatePowerSpectra(self.u, w_k, w_g, w_s, name=name, save=True)
         dataVectorLow, shotNoiseVectorLow = self.generateDataVector(p_kk, p_kg, p_ks, p_gg, p_gs, p_ss, p_shot, p_gg_shot, p_ss_shot)
         # derivative
         result = (dataVectorHigh-dataVectorLow) / (self.nuisancePar.high[iPar]-self.nuisancePar.low[iPar])
#         derivative[self.cosmoPar.nPar+iPar,:] = (dataVectorHigh-dataVectorLow) / (self.nuisancePar.high[iPar]-self.nuisancePar.low[iPar])
#         derivative[self.cosmoPar.nPar+iPar,:] = (dataVectorHigh-self.dataVector) / (self.nuisancePar.high[iPar]-self.nuisancePar.fiducial[iPar])
         # check that all went well
         if not all(np.isfinite(derivative[self.cosmoPar.nPar+iPar,:])):
            print "########"
            print "problem with "+self.nuisancePar.names[iPar]
            print "high value = "+str(self.nuisancePar.high[iPar])
            print "low value = "+str(self.nuisancePar.fiducial[iPar])
         
         tStop = time()
         print "("+str(np.round(tStop-tStart,1))+" sec)"

         return result


#      with sharedmem.MapReduce(np=self.nProc) as pool:
#         result = pool.map(derivativeWrtCosmoPar, range(self.nuisancePar.nPar))
#         derivative[:self.nuisancePar.nPar,:] = result.copy()
      derivative[self.cosmoPar.nPar:self.cosmoPar.nPar+self.nuisancePar.nPar,:] = np.array(map(derivativeWrtNuisancePar, range(self.nuisancePar.nPar)))

      path = "./output/dDatadPar/dDatadPar_"+self.name
      np.savetxt(path, derivative)




##! Non-parallel version, working.
#   def saveDerivativeDataVector(self):
#      # Derivatives of the data vector:
#      # matrix of size self.params.nPar x self.nData
#      derivative = np.zeros((self.fullPar.nPar, self.nData))
#
#      for iPar in range(self.cosmoPar.nPar):
#         print "Derivative wrt "+self.cosmoPar.names[iPar],
#         tStart = time()
#         # high
#         name = self.name+self.cosmoPar.names[iPar]+"high"
#         cosmoParClassy = self.cosmoPar.paramsClassy.copy()
##         print cosmoParClassy
##         print "#"
#         cosmoParClassy[self.cosmoPar.names[iPar]] = self.cosmoPar.paramsClassyHigh[self.cosmoPar.names[iPar]]
##         print cosmoParClassy
#         u = Universe(cosmoParClassy)
#         w_k, w_g, w_s, zBounds = self.generateTomoBins(u, self.nuisancePar.fiducial)
#         p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss = self.generatePowerSpectra(u, w_k, w_g, w_s, name=name, save=True)
#         dataVectorHigh, shotNoiseVectorHigh = self.generateDataVector(p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss)
#         # low
#         name = self.name+self.cosmoPar.names[iPar]+"low"
#         cosmoParClassy = self.cosmoPar.paramsClassy.copy()
#         cosmoParClassy[self.cosmoPar.names[iPar]] = self.cosmoPar.paramsClassyLow[self.cosmoPar.names[iPar]]
#         u = Universe(cosmoParClassy)
#         w_k, w_g, w_s, zBounds = self.generateTomoBins(u, self.nuisancePar.fiducial)
#         p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss = self.generatePowerSpectra(u, w_k, w_g, w_s, name=name, save=True)
#         dataVectorLow, shotNoiseVectorLow = self.generateDataVector(p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss)
#         # derivative
#         derivative[iPar,:] = (dataVectorHigh-dataVectorLow) / (self.cosmoPar.high[iPar]-self.cosmoPar.low[iPar])
##         derivative[iPar,:] = (dataVectorHigh-self.dataVector) / (self.cosmoPar.high[iPar]-self.cosmoPar.fiducial[iPar])
#
##         print "all zero?"
##         print np.mean(dataVectorHigh-self.dataVector) / np.std(self.dataVector)
##         print self.cosmoPar.high[iPar]-self.cosmoPar.fiducial[iPar]
#
#         # check that all went well
#         if not all(np.isfinite(derivative[iPar,:])):
#            print "########"
#            print "problem with "+self.cosmoPar.names[iPar]
#            print "high value = "+str(self.cosmoPar.high[iPar])
#            print "low value = "+str(self.cosmoPar.fiducial[iPar])
#         tStop = time()
#         print "("+str(np.round(tStop-tStart,1))+" sec)"
#
#
#      # Nuisance parameters
#      for iPar in range(self.nuisancePar.nPar):
#         print "Derivative wrt "+self.nuisancePar.names[iPar],
#         tStart = time()
#         params = self.nuisancePar.fiducial.copy()
#         # high
#         name = "_"+self.name+self.nuisancePar.names[iPar]+"high"
#         params[iPar] = self.nuisancePar.high[iPar]
#         w_k, w_g, w_s, zBounds = self.generateTomoBins(self.u, params)
#         p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss = self.generatePowerSpectra(self.u, w_k, w_g, w_s, name=name, save=True)
#         dataVectorHigh, shotNoiseVectorHigh = self.generateDataVector(p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss)
#         # low
#         name = self.name+self.nuisancePar.names[iPar]+"low"
#         params[iPar] = self.nuisancePar.low[iPar]
#         w_k, w_g, w_s, zBounds = self.generateTomoBins(self.u, params)
#         p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss = self.generatePowerSpectra(self.u, w_k, w_g, w_s, name=name, save=True)
#         dataVectorLow, shotNoiseVectorLow = self.generateDataVector(p2d_kk, p2d_kg, p2d_ks, p2d_gg, p2d_gs, p2d_ss)
#         # derivative
#         derivative[self.cosmoPar.nPar+iPar,:] = (dataVectorHigh-dataVectorLow) / (self.nuisancePar.high[iPar]-self.nuisancePar.low[iPar])
##         derivative[self.cosmoPar.nPar+iPar,:] = (dataVectorHigh-self.dataVector) / (self.nuisancePar.high[iPar]-self.nuisancePar.fiducial[iPar])
#         # check that all went well
#         if not all(np.isfinite(derivative[self.cosmoPar.nPar+iPar,:])):
#            print "########"
#            print "problem with "+self.nuisancePar.names[iPar]
#            print "high value = "+str(self.nuisancePar.high[iPar])
#            print "low value = "+str(self.nuisancePar.fiducial[iPar])
#
#         tStop = time()
#         print "("+str(np.round(tStop-tStart,1))+" sec)"
#
#      path = "./output/dDatadPar/dDatadPar_"+self.name
#      np.savetxt(path, derivative)

   def loadDerivativeDataVector(self, name=None):
      if name is None:
         name = self.name
      path = "./output/dDatadPar/dDatadPar_"+name
      self.derivativeDataVector = np.genfromtxt(path)


   ##################################################################################
   

#   def generateFisherSparse(self, mask=None):
#      if mask is None:
#         mask=self.lMaxMask
#      fisherData = np.zeros((self.fullPar.nPar, self.fullPar.nPar))
#      # extract unmasked cov elements, and invert
#      cov = extractMaskedMat(self.covMat, mask=mask)
#      # invert the matrix once
#      invCov = scipy.sparse.linalg.inv(cov)
#      # Fisher from the data
#      for i in range(self.fullPar.nPar):
#         for j in range(self.fullPar.nPar):
#            di = extractMaskedVec(self.derivativeDataVector[i,:], mask=mask)
#            dj = extractMaskedVec(self.derivativeDataVector[j,:], mask=mask)
#            fisherData[i,j] = np.dot(di.transpose(), np.dot(invCov, dj))
#      return fisherData



   def generateFisher(self, mask=None, deriv=None):
      '''Here I compute the explicit matrix inverse once.
      This is way faster than doing scipy.linalg.solve every time,
      and a bit faster than doing scipy.linalg.lu_factor once then lu_solve each time.
      '''
      if mask is None:
         mask=self.lMaxMask
      if deriv is None:
         deriv = self.derivativeDataVector
      fisherData = np.zeros((self.fullPar.nPar, self.fullPar.nPar))
      # extract unmasked cov elements, and invert
      cov = extractMaskedMat(self.covMat, mask=mask)
      # invert the matrix once
      invCov = np.linalg.inv(cov)
      # Fisher from the data
      for i in range(self.fullPar.nPar):
         for j in range(self.fullPar.nPar):
            di = extractMaskedVec(deriv[i,:], mask=mask)
            dj = extractMaskedVec(deriv[j,:], mask=mask)
            fisherData[i,j] = np.dot(di.transpose(), np.dot(invCov, dj))
      return fisherData


#   def generateFisherLUOnce(self, mask=None):
#      '''A bit slower than np.linalg.inv in practice.
#      '''
#      if mask is None:
#         mask=self.lMaxMask
#      fisherData = np.zeros((self.fullPar.nPar, self.fullPar.nPar))
#      # extract unmasked cov elements, and invert
#      cov = extractMaskedMat(self.covMat, mask=mask)
#      # solve the LU factorization, to "invert" many times fast
#      # without ever computing the inverse
#      lu, piv = scipy.linalg.lu_factor(cov)
#      # Fisher from the data
#      for i in range(self.fullPar.nPar):
#         for j in range(self.fullPar.nPar):
#            di = extractMaskedVec(self.derivativeDataVector[i,:], mask=mask)
#            dj = extractMaskedVec(self.derivativeDataVector[j,:], mask=mask)
#            fisherData[i,j] = np.dot(di.transpose(), scipy.linalg.lu_solve((lu,piv), dj))
#      return fisherData





   def generateAllFisher(self, save=False):
      if save:
         print("Generating all Fisher matrices")
      
         tStart = time()
         fisher = self.generateFisher(mask=self.lMaxMask)
         np.savetxt("./output/fisher/"+self.name+"_fisherdata_gks.txt", fisher)
         tStop = time()
         print("- GKS took "+str((tStop-tStart)/60.)+" min")

         tStart = time()
         fisher = self.generateFisher(mask=self.lMaxMask+self.gsOnlyMask)
         np.savetxt("./output/fisher/"+self.name+"_fisherdata_gs.txt", fisher)
         tStop = time()
         print("- GS took "+str((tStop-tStart)/60.)+" min")

         tStart = time()
         fisher = self.generateFisher(mask=self.lMaxMask+self.gsOnlyMask+self.noNullMask)
         np.savetxt("./output/fisher/"+self.name+"_fisherdata_gsnonull.txt", fisher)
         tStop = time()
         print("- GS no null took "+str((tStop-tStart)/60.)+" min")

      print("Reading all Fisher matrices")
      self.fisherDataGks = np.genfromtxt("./output/fisher/"+self.name+"_fisherdata_gks.txt")
      self.fisherDataGs = np.genfromtxt("./output/fisher/"+self.name+"_fisherdata_gs.txt")
      self.fisherDataGsnonull = np.genfromtxt("./output/fisher/"+self.name+"_fisherdata_gsnonull.txt")



   ##################################################################################
   ##################################################################################

   def plotEllBins(self, show=False):

      Z = np.linspace(1.e-4, 4., 201)

      def fLMax(z, kMaxG=0.3):
         '''Enforce that k < kMaxG = 0.3h/Mpc
         for clustering, following DESC SRD v1,
         where nonlin gal bias is a 10% effect on clustering.
         '''
         return kMaxG * self.u.bg.comoving_distance(z) - 0.5

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot(Z, fLMax(Z, kMaxG=0.3), label=r'$k_\text{max}=0.3$ h/Mpc')
      #ax.plot(Z, fLMax(Z, kMaxG=0.5))
      #
      # show the actual ell and z for the various bins
      for iBin in range(self.nBins):
         zBin = self.w_g[iBin].zMean()
         lMaxG = fLMax(zBin, kMaxG=0.3)
         for iL in range(self.nL):
            if self.L[iL]<lMaxG:
               ax.plot(zBin, self.L[iL], 'ko')
            else:
               ax.plot(zBin, self.L[iL], 'ko', alpha=0.3)
      #
      ax.legend(loc=1)
      ax.set_xlim((0., 4.))
      ax.set_ylim((0., 1100.))
      ax.set_xlabel(r'$z_\text{mean}$')
      ax.set_ylabel(r'$\ell$')
      #
      fig.savefig(self.figurePath+"/ell_bins.pdf", bbox_inches='tight')
      if show:
         plt.show()
      fig.clf()


#   def checkConditionNumbers(self, mask=None):
#      if mask is None:
#         mask=self.lMaxMask
#      print "Cov matrix"
#      tStart = time()
#      cov = extractMaskedMat(self.covMat, mask=mask)
#      tStop = time()
#      print "extracting cov mat took", tStop-tStart, "sec"
#      tStart = time()
#      condNumber = np.linalg.cond(cov)
#      tStop = time()
#      print "computing condition number took", tStop-tStart, "sec"
#      numPrecision =  np.finfo(cov.dtype).eps
#      print "inverse condition number:", 1. / condNumber
#      print "float numerical precision:", numPrecision
#      if 1. / condNumber > numPrecision:
#         print "--> OK"
#      else:
#         print "--> Not OK"
#
#      print "Fisher matrix"
#      tStart = time()
#      fisherData, fisherPosterior = self.generateFisher(mask=mask)
#      tStop = time()
#      print "generating Fisher matrix took", tStop-tStart, "sec"
#      tStart = time()
#      condNumber = np.linalg.cond(fisherPosterior)
#      tStop = time()
#      print "computing condition number took", tStop-tStart, "sec"
#      numPrecision =  np.finfo(fisherPosterior.dtype).eps
#      print "inverse condition number:", 1. / condNumber
#      print "number numerical precision:", numPrecision
#      if 1. / condNumber > numPrecision:
#         print "--> OK"
#      else:
#         print "--> Not OK"


   def plotDndz(self, show=False):

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # full LSST source sample
      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(self.u, name='glsst')
      zMin = 1./w_glsst.aMax-1.
      zMax = 1./w_glsst.aMin-1.
      Z = np.linspace(zMin, zMax, 1001)
      dndzFull = w_glsst.dndz(Z)
      dndzFull /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
      ax.plot(Z, dndzFull, 'k', label=r'LSST galaxies')
      #
      # binned with photo-z uncertainties
      dnGdz_t = self.dndzPhotoZ(self.u, self.photoZPar.fiducial, test=False)
      for iBin in range(self.nBins):
         # evaluate dn/dz
         dndz = dnGdz_t(iBin, Z)
         dndz /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
         # plot it
         ax.fill_between(Z, 0., dndz, facecolor=plt.cm.autumn(1.*iBin/self.nBins), edgecolor='', alpha=0.7)
      #
      # CMB lensing kernel
      W = np.array(map(self.w_k.f, 1./(1.+Z)))
      H = self.u.hubble(Z) / 3.e5   # H/c in (h Mpc^-1)
      ax.plot(Z, W/H * 0.5 * np.max(dndzFull)/np.max(W/H), 'b', label=r'$W_{\kappa_\text{CMB}}$')
      #
      ax.legend(loc=1, fontsize='x-small')
      ax.set_ylim((0., 23.))
      ax.set_xlim((0.,4.))
      ax.set_xlabel(r'$z$')
      ax.set_ylabel(r'$dN / d\Omega\; dz$ [arcmin$^{-2}$]')
      #
      fig.savefig(self.figurePath+"/dndz.pdf")
      if show:
         plt.show()
      fig.clf()
#      plt.show()


   
#   def plotDndz(self):
#      fig=plt.figure(0)
#      ax=fig.add_subplot(111)
#      #
#      # full LSST source sample
#      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(self.u, name='glsst')
#      zMin = 1./w_glsst.aMax-1.
#      zMax = 1./w_glsst.aMin-1.
#      Z = np.linspace(zMin, zMax, 501)
#      dndz = w_glsst.dndz(Z)
#      dndz /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
#      ax.plot(Z, dndz, 'k')
#      #
#      # binned with photo-z uncertainties
#      for iBin in range(self.nBins):
#         # redshift range for that bin
#         zMin = 1./self.w_g[iBin].aMax-1.
#         zMax = 1./self.w_g[iBin].aMin-1.
#         Z = np.linspace(zMin, zMax, 501)
#         # evaluate dn/dz
#         dndz = np.array(map(self.w_g[iBin].dndz, Z))
#         dndz /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
#         # plot it
#         ax.fill_between(Z, 0., dndz, facecolor=plt.cm.autumn(1.*iBin/self.nBins), edgecolor='', alpha=0.7)
#      #
#      # CMB lensing kernel
#      W = np.array(map(self.w_k.f, 1./(1.+Z)))
#      H = self.u.hubble(Z) / 3.e5   # H/c in (h Mpc^-1)
#      ax.plot(Z, W/H, 'k', label=r'$W_{\kappa_\text{CMB}}$')
#      #
#      ax.set_ylim((0., 23.))
#      ax.set_xlim((0.,4.))
#      ax.set_xlabel(r'$z$')
#      ax.set_ylabel(r'$dN / d\Omega\; dz$ [arcmin$^{-2}$]')
#      #
#      fig.savefig(self.figurePath+"/dndz.pdf")
#      fig.clf()
##      plt.show()
#

   ##################################################################################
   
   
   def SampleDndz(self, photoZPar, nSamples=10, path=None, log=False):
      '''Make a plot with samples of dn/dz,
      determined by the Fisher matrix in the photoZPar.
      The photoZPar is extracted from the input fullPar.
      '''
      # Initialize the plot
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # full LSST source sample
      w_glsst = WeightTracerLSSTSourcesDESCSRDV1(self.u, name='glsst')
      zMin = 1./w_glsst.aMax-1.
      zMax = 1./w_glsst.aMin-1.
      Z = np.linspace(zMin, zMax, 1001)
      dndz = w_glsst.dndz(Z)
      dndz /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
      ax.plot(Z, dndz, 'k')
      
      # draw samples
      tStart = time()
      # new photo-z par, to store the sample
      newPhotoZPar = photoZPar.copy()
      for iSample in range(nSamples):
   
         # generate one sample of the photoZPar
         mean = photoZPar.fiducial
         cov = np.linalg.inv(photoZPar.fisher)
         newPhotoZPar.fiducial = np.random.multivariate_normal(mean, cov, size=1)[0]
   
         # generate the corresponding nuisancePar
         newNuisancePar = self.galaxyBiasPar.copy()
         newNuisancePar.addParams(self.shearMultBiasPar)
         newNuisancePar.addParams(newPhotoZPar)
         
         # generate the corresponding dn/dz for the particular sample
         w_k, w_g, zBounds = self.generateTomoBins(self.u, newNuisancePar.fiducial, doS=False)


   
         # add it to the plot
         for iBin in range(self.nBins):
            # redshift range for that bin
            zMin = 1./self.w_g[iBin].aMax-1.
            zMax = 1./self.w_g[iBin].aMin-1.
            Z = np.linspace(zMin, zMax, 1001)
            # evaluate dn/dz
            dndz = np.array(map(w_g[iBin].dndz, Z))
            dndz /= (180.*60./np.pi)**2 # convert from 1/sr to 1/arcmin^2
            # plot it
#            ax.fill_between(Z, 0., dndz, facecolor=plt.cm.autumn(1.*iBin/self.nBins), edgecolor='', alpha=0.7)
            ax.plot(Z, dndz, c=plt.cm.autumn(1.*iBin/self.nBins), lw=1, alpha=0.3)

      tStop = time()
      print "took "+str((tStop-tStart)/60.)+" min"
      #
      ax.set_xlabel(r'$z$')
      ax.set_ylabel(r'$dN / d\Omega\; dz$ [arcmin$^{-2}$]')
      if log:
         ax.set_yscale('log', nonposy='clip')
      
      # save figure if requested
      if path is not None:
         fig.savefig(path)
         fig.clf()
      else:
         plt.show()





   ##################################################################################



   def plotDiagCov(self, mask=None, show=False):
      '''Useful to see how to improve the condition number
      of the cov mat.
      '''
      if mask is None:
         mask=self.lMaxMask
      # set to zero the masked data elements
      maskMat = np.outer(1-mask,1-mask)
      cov = self.covMat * maskMat

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.axvline((self.nKK)*self.nL, c='gray')
      ax.axvline((self.nKK+self.nKG)*self.nL, c='gray')
      ax.axvline((self.nKK+self.nKG+self.nKS)*self.nL, c='gray')
      ax.axvline((self.nKK+self.nKG+self.nKS+self.nGG)*self.nL, c='gray')
      ax.axvline((self.nKK+self.nKG+self.nKS+self.nGG+self.nGS)*self.nL, c='gray')
      # 
      ax.semilogy(np.sqrt(np.diag(cov)))
      #
      ax.set_ylabel(r'$\sqrt{\text{Cov}_{i,i}}$')
      ax.set_xlabel(r'Data vector index $i$')

      fig.savefig(self.figurePath+"/diag_cov.pdf", bbox_inches='tight')
      if show:
         plt.show()
      fig.clf()




   def plotCovMat(self, mask=None, show=False):
      if mask is None:
         mask=self.lMaxMask
      fig=plt.figure(0, figsize=(12,8))
      ax=fig.add_subplot(111)
      #
      # compute correlation matrix
      corMat = np.zeros_like(self.covMat)
      for i in range(self.nData):
         for j in range(self.nData):
            corMat[i,j] = self.covMat[i,j] / np.sqrt(self.covMat[i,i] * self.covMat[j,j])
      
      # set to zero the masked data elements
      maskMat = np.outer(1-mask,1-mask)
      corMat *= maskMat
      
      upperDiag = np.triu(np.ones(self.nData))
      plt.imshow(corMat * upperDiag, interpolation='nearest', norm=LogNorm(vmin=1.e-4, vmax=1), cmap=cmaps.viridis_r)
      #
      ax.plot(np.arange(self.nData+1)-0.5, np.arange(self.nData+1)-0.5, 'k', lw=1)
      #
      # 2-pt function delimiters
      for i in range(1, self.nGG+self.nGS+self.nSS):
         ax.axhline(self.nL*i-0.5, xmin=(self.nL*i-0.5)/self.nData, c='gray', lw=0.25, ls='-')
         ax.axvline(self.nL*i-0.5, ymin=1.-(self.nL*i-0.5)/self.nData, c='gray', lw=0.25, ls='-')
      #
      # block delimiters
      ax.axhline(self.nL*self.nGG-0.5, xmin=(self.nL*self.nGG-0.5)/self.nData, c='k', lw=1.5)
      ax.axhline(self.nL*(self.nGG+self.nGS)-0.5, xmin=(self.nL*(self.nGG+self.nGS)-0.5)/self.nData, c='k', lw=1.5)
      #
      ax.axvline(self.nL*self.nGG-0.5, ymin=1.-(self.nL*self.nGG-0.5)/self.nData, c='k', lw=1.5)
      ax.axvline(self.nL*(self.nGG+self.nGS)-0.5, ymin=1.-(self.nL*(self.nGG+self.nGS)-0.5)/self.nData, c='k', lw=1.5)
      #
      plt.colorbar()
      ax.set_xlim((-0.5, (self.nData-1)+0.5))
      ax.set_ylim((-0.5, (self.nData-1)+0.5))
      ax.invert_yaxis()
      #ax.xaxis.tick_top()
      ax.xaxis.set_ticks([])
      ax.yaxis.set_ticks([])
      #ax.grid(True)
      #ax.set_title(r'Full cor: '+infile)
      #
      fig.savefig(self.figurePath+"/cor_mat.pdf", bbox_inches='tight', format='pdf', dpi=2400)
      if show:
         plt.show()
      fig.clf()


   def plotInvCovMat(self, show=False):
      fig=plt.figure(0, figsize=(12,8))
      ax=fig.add_subplot(111)
      #
      # extract the unmasked cov elements
      cov = extractMaskedMat(self.covMat, mask=self.lMaxMask)
      invCov = np.linalg.inv(cov)
      #
      upperDiag = np.triu(np.ones(len(cov[:,0])))
#      plt.imshow(self.invCov * upperDiag, interpolation='nearest', norm=LogNorm(vmin=1.e-4, vmax=1), cmap=cmaps.viridis_r)
      plt.imshow(np.abs(invCov) * upperDiag, interpolation='nearest', norm=LogNorm(), cmap=plt.cm.bwr)
      #
      ax.plot(np.arange(self.nData+1)-0.5, np.arange(self.nData+1)-0.5, 'k', lw=1)
      #
      # 2-pt function delimiters
      for i in range(1, self.nGG+self.nGS+self.nSS):
         ax.axhline(self.nL*i-0.5, xmin=(self.nL*i-0.5)/self.nData, c='gray', lw=0.25, ls='-')
         ax.axvline(self.nL*i-0.5, ymin=1.-(self.nL*i-0.5)/self.nData, c='gray', lw=0.25, ls='-')
      #
      # block delimiters
      ax.axhline(self.nL*self.nGG-0.5, xmin=(self.nL*self.nGG-0.5)/self.nData, c='k', lw=1.5)
      ax.axhline(self.nL*(self.nGG+self.nGS)-0.5, xmin=(self.nL*(self.nGG+self.nGS)-0.5)/self.nData, c='k', lw=1.5)
      #
      ax.axvline(self.nL*self.nGG-0.5, ymin=1.-(self.nL*self.nGG-0.5)/self.nData, c='k', lw=1.5)
      ax.axvline(self.nL*(self.nGG+self.nGS)-0.5, ymin=1.-(self.nL*(self.nGG+self.nGS)-0.5)/self.nData, c='k', lw=1.5)
      #
      plt.colorbar()
      ax.set_xlim((-0.5, (self.nData-1)+0.5))
      ax.set_ylim((-0.5, (self.nData-1)+0.5))
      ax.invert_yaxis()
      #ax.xaxis.tick_top()
      ax.xaxis.set_ticks([])
      ax.yaxis.set_ticks([])
      #
      fig.savefig(self.figurePath+"/invcov_mat.pdf", bbox_inches='tight', format='pdf', dpi=2400)
      if show:
         plt.show()
      fig.clf()





   def plotPowerSpectra(self, show=False):

      # kk
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = 0
      I = range(i1*self.nL, (i1+1)*self.nL)
      L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
      d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.kUnit**2
      shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.kUnit**2
      cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
      std = np.sqrt(np.diag(cov)) / self.kUnit**2
      #
      ax.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color='b')
      ax.plot(L, shot, ls='--', lw=1, color='grey')
      #
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_xlim((90., 2.5e3))
      ax.set_ylim((6.e-6, 2.5e-5))
      ax.set_title(r'CMB lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell C_\ell^{\kappa_\text{CMB} \kappa_\text{CMB}}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/p2d_kk.pdf", bbox_inches='tight')
      if show:
         plt.show()
      fig.clf()


      # kg
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      #
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.kUnit
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.kUnit
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov)) / self.kUnit
         #
         color = Colors[iBin1]
         ax.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color=color)
         ax.plot([], [], c=color, label=r'$\langle \kappa_\text{CMB}\, g_{'+str(iBin1)+r'}  \rangle$')
         ax.plot(L, shot, ls='--', lw=1, color='grey')  # same color for all tomo bins, since they have the same n_gal
         # advance counter in data vector
         i1 += 1
      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_title(r'Galaxy $\times$ CMB lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell C_\ell^{g \kappa_\text{CMB}}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/p2d_kg.pdf")
      if show:
         plt.show()
      fig.clf()
      

      # ks
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      #
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.kUnit / self.sUnit
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.kUnit / self.sUnit
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov)) / self.kUnit / self.sUnit
         #
         color = Colors[iBin1]
         ax.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color=color)
         ax.plot([], [], c=color, label=r'$\langle \kappa_\text{CMB}\, \kappa_{g_{'+str(iBin1)+r'}}  \rangle$')
         ax.plot(L, shot, ls='--', lw=1, color='grey')  # same color for all tomo bins, since they have the same n_gal
         # advance counter in data vector
         i1 += 1
      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_title(r'CMB lensing $\times$ galaxy lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell C_\ell^{\kappa_\text{CMB} \kappa_g}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/p2d_ks.pdf")
      if show:
         plt.show()
      fig.clf()


      # gg: panels
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      #
      fig=plt.figure(0)
      gs = gridspec.GridSpec(3, 1)#, height_ratios=[1, 1, 1])
      gs.update(hspace=0.)
      
      # auto
      ax0=plt.subplot(gs[0])
      i1 = self.nKK + self.nKG + self.nKS
      for iBin1 in [0,1,2,3]:
         color = Colors[iBin1]
         ax0.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax0.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color=color)
         ax0.plot(L, shot, ls='--', lw=1, color='grey')  # same color for all tomo bins, since they have the same n_gal
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax0.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax0.set_xlim((90., 2.5e3))
      ax0.set_ylim((1.e-5, 1e-3))
      ax0.set_xscale('log')
      ax0.set_yscale('log', nonposy='clip')
      plt.setp(ax0.get_xticklabels(), visible=False)
      #
      ax0.set_title(r'Clustering')
      ax0.set_ylabel(r'$\ell\; C_\ell^{g_ig_i}$', fontsize=18)

      # cross i,i+1
      ax1=plt.subplot(gs[1])
      i1 = self.nKK + self.nKG + self.nKS + 1
      for iBin1 in [4,5,6,7]:
         color = Colors[iBin1]
         ax1.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins-1):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax1.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color=color)
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax1.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax1.set_xlim((90., 2.5e3))
      ax1.set_xscale('log')
      ax1.set_yscale('log', nonposy='clip')
      plt.setp(ax1.get_xticklabels(), visible=False)
      #
      ax1.set_ylabel(r'$\ell\; C_\ell^{g_ig_{i+1}}$', fontsize=18)

      # cross i,i+2
      ax2=plt.subplot(gs[2])
      i1 = self.nKK + self.nKG + self.nKS + 2
      for iBin1 in [8,9]:
         color = Colors[iBin1]
         ax2.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins-2):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax2.errorbar(L, d, yerr=std, ls='-', lw=2, elinewidth=1.5, marker='.', markersize=2, color=color)
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax2.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      #
      ax2.set_xlim((90., 2.5e3))
      ax2.set_xscale('log')
      ax2.set_yscale('log', nonposy='clip')
      #
      ax2.set_ylabel(r'$\ell\; C_\ell^{g_ig_{i+2}}$', fontsize=18)
      ax2.set_xlabel(r'$\ell$')
      #
      fig.savefig(self.figurePath+"/p2d_gg.pdf")
      if show:
         plt.show()
      fig.clf()



      # gs
      Colors = plt.cm.winter(1.*np.arange(self.nBins)/(self.nBins-1.))
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG + self.nKS + self.nGG
      for iBin1 in range(self.nBins):
         color = Colors[iBin1]
         ax.plot([], [], c=color, label=r'$\langle g_{'+str(iBin1)+r'}\, \kappa_{g_j} \rangle$')

#         # show only gs for the highest s bin 
#         i1 += self.nBins - 1
#         #
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov)) / self.sUnit
#         #
#         ax.errorbar(L*(1.+0.01*i1/self.nGS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)# label=r'$\ell \langle g_{'+str(iBin1)+'} \gamma_{'+str(iBin2)+r'}\rangle$')
#         # move to next row
#         i1 += 1

         for iBin2 in range(self.nBins):  # show all the cross-correlations
            I = range(i1*self.nL, (i1+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
            std = np.sqrt(np.diag(cov)) / self.sUnit
            #
            ax.errorbar(L*(1.+0.01*i1/self.nGS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)# label=r'$\ell \langle g_{'+str(iBin1)+'} \gamma_{'+str(iBin2)+r'}\rangle$')
            # move to next row
            i1 += 1

      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell \; C_\ell^{g\kappa_g}$')
      ax.set_title(r'Galaxy $\times$ galaxy lensing')
      #
      fig.savefig(self.figurePath+"/p2d_gs.pdf")
      if show:
         plt.show()
      fig.clf()


      # ss: all on same plot
      Colors = plt.cm.jet(1.*np.arange(self.nBins)/(self.nBins-1.))
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
      for iBin1 in range(self.nBins):
         # add entry to caption
         color = Colors[iBin1]
         ax.plot([], [], c=color, label=r'$\langle\kappa_{g_i} \kappa_{g_{i+'+str(iBin1)+r'}} \rangle $')


#         # show only the auto
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit**2
#         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.sUnit**2
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov)) / self.sUnit**2
#         #
#         ax.errorbar(L*(1.+0.01*i1/self.nSS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)#, label=r'$\gamma_{'+str(iBin1)+'} \gamma_{'+str(iBin2)+'}$')
##            ax.plot(L*(1.+0.01*i1/self.nSS), shot, ls='--', lw=1, color=color)   # different color for each bin
#         ax.plot(L*(1.+0.01*i1/self.nSS), shot, ls='--', lw=1, color='grey')   # same color for all bins
#         # move to next row
#         i1 += self.nBins - iBin1

         # show all the cross-correlations
         for iBin2 in range(iBin1, self.nBins):
            color = Colors[iBin2-iBin1]
            #color = Colors[iBin1]
            #
            I = range(i1*self.nL, (i1+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit**2
            shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.sUnit**2
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
            std = np.sqrt(np.diag(cov)) / self.sUnit**2
            #
            ax.errorbar(L*(1.+0.01*i1/self.nSS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)
            ax.plot(L*(1.+0.01*i1/self.nSS), shot, ls='--', lw=1, color='grey')   # same color for all bins
            # move to next row
            i1 += 1

      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell \; C_\ell^{\kappa_g\kappa_g}$')
      ax.set_title(r'Shear tomography')
      #
      fig.savefig(self.figurePath+"/p2d_ss.pdf")
      if show:
         plt.show()
      fig.clf()





   def plotUncertaintyPowerSpectra(self, show=False):

      # kk
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = 0
      I = range(i1*self.nL, (i1+1)*self.nL)
      L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
      d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.kUnit**2
      shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.kUnit**2
      cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
      std = np.sqrt(np.diag(cov)) / self.kUnit**2
      #
      ax.plot(L, std / d, '.-', lw=2, color='b')
      #
      ax.set_xlim((90., 2.5e3))
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_title(r'CMB lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\sigma \left( C_\ell^{\kappa_\text{CMB} \kappa_\text{CMB}} \right) / C_\ell^{\kappa_\text{CMB} \kappa_\text{CMB}}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/sp2d_kk.pdf", bbox_inches='tight')
      if show:
         plt.show()
      fig.clf()


      # kg
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      #
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.kUnit
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.kUnit
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov)) / self.kUnit
         #
         color = Colors[iBin1]
         ax.plot(L, std/d, '.-', lw=2, color=color, label=r'$\langle \kappa_\text{CMB}\, g_{'+str(iBin1)+r'}  \rangle$')
         # advance counter in data vector
         i1 += 1
      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_title(r'Galaxy $\times$ CMB lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\sigma \left( C_\ell^{g \kappa_\text{CMB}} \right) / C_\ell^{g \kappa_\text{CMB}}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/sp2d_kg.pdf")
      if show:
         plt.show()
      fig.clf()
      

      # ks
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      #
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) /self.kUnit / self.sUnit
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) /self.kUnit / self.sUnit
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov)) /self.kUnit / self.sUnit
         #
         color = Colors[iBin1]
         ax.plot(L, std/d, '.-', lw=2, color=color, label=r'$\langle \kappa_\text{CMB}\, \kappa_{g_{'+str(iBin1)+r'}}  \rangle$')
         ax.plot(L, shot, ls='--', lw=1, color='grey')  # same color for all tomo bins, since they have the same n_gal
         # advance counter in data vector
         i1 += 1
      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      #
      ax.set_title(r'CMB lensing $\times$ galaxy lensing')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\sigma \left( C_\ell^{\kappa_\text{CMB} \kappa_g} \right) / C_\ell^{\kappa_\text{CMB} \kappa_g}$', fontsize=18)
      #
      fig.savefig(self.figurePath+"/sp2d_ks.pdf")
      if show:
         plt.show()
      fig.clf()


      # gg: panels
      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
      # crappy hack, in case the user asks for few bins
      Colors = np.concatenate((Colors, Colors, Colors, Colors))
      #
      fig=plt.figure(0)
      gs = gridspec.GridSpec(3, 1)#, height_ratios=[1, 1, 1])
      gs.update(hspace=0.)
      
      # auto
      ax0=plt.subplot(gs[0])
      i1 = self.nKK + self.nKG + self.nKS
      for iBin1 in [0,1,2,3]:
         color = Colors[iBin1]
         ax0.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax0.plot(L, std / d, '.-', lw=2, color=color)
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax0.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax0.set_xscale('log')
      ax0.set_yscale('log', nonposy='clip')
      plt.setp(ax0.get_xticklabels(), visible=False)
      #
      ax0.set_title(r'Clustering: $\sigma\left( C_\ell \right) / C_\ell$')
      ax0.set_ylabel(r'$g_i g_i$', fontsize=18)

      # cross i,i+1
      ax1=plt.subplot(gs[1])
      i1 = self.nKK + self.nKG + self.nKS + 1
      for iBin1 in [4,5,6,7]:
         color = Colors[iBin1]
         ax1.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins-1):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax1.plot(L, std / d, '-', lw=2, color=color)
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax1.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax1.set_xscale('log')
      ax1.set_yscale('log', nonposy='clip')
      plt.setp(ax1.get_xticklabels(), visible=False)
      #
      ax1.set_ylabel(r'$g_i g_{i+1}$', fontsize=18)

      # cross i,i+2
      ax2=plt.subplot(gs[2])
      i1 = self.nKK + self.nKG + self.nKS + 2
      for iBin1 in [8,9]:
         color = Colors[iBin1]
         ax2.plot([], [], c=color, label=r'$i='+str(iBin1)+'$')
      for iBin1 in range(self.nBins-2):
         I = range(i1*self.nL, (i1+1)*self.nL)
         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
         std = np.sqrt(np.diag(cov))
         #
         color = Colors[iBin1]
         ax2.plot(L, std / d, '.-', lw=2, color=color)
         # advance counter in data vector
         i1 += self.nBins - iBin1
      #
      ax2.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      #
      ax2.set_xlim((90., 2.5e3))
      ax2.set_xscale('log')
      ax2.set_yscale('log', nonposy='clip')
      #
      ax2.set_ylabel(r'$g_i g_{i+2}$', fontsize=18)
      ax2.set_xlabel(r'$\ell$')
      #
      fig.savefig(self.figurePath+"/sp2d_gg.pdf")
      if show:
         plt.show()
      fig.clf()



      # gs
      Colors = plt.cm.winter(1.*np.arange(self.nBins)/(self.nBins-1.))
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG + self.nKS + self.nGG
      for iBin1 in range(self.nBins):
         color = Colors[iBin1]
         ax.plot([], [], c=color, label=r'$\langle g_{'+str(iBin1)+r'}\, \kappa_{g_j} \rangle$')

#         # show only gs for the highest s bin 
#         i1 += self.nBins - 1
#         #
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov)) / self.sUnit
#         #
#         ax.errorbar(L*(1.+0.01*i1/self.nGS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)# label=r'$\ell \langle g_{'+str(iBin1)+'} \gamma_{'+str(iBin2)+r'}\rangle$')
#         # move to next row
#         i1 += 1

         for iBin2 in range(self.nBins):  # show all the cross-correlations
            I = range(i1*self.nL, (i1+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
            std = np.sqrt(np.diag(cov)) / self.sUnit
            #
            ax.plot(L*(1.+0.01*i1/self.nGS), std / d, '.-', lw=1, color=color)
            # move to next row
            i1 += 1

      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\sigma\left( C_\ell^{g\kappa_g} \right) / C_\ell^{g\kappa_g}$')
      ax.set_title(r'Galaxy $\times$ galaxy lensing')
      #
      fig.savefig(self.figurePath+"/sp2d_gs.pdf")
      if show:
         plt.show()
      fig.clf()


      # ss: all on same plot
      Colors = plt.cm.jet(1.*np.arange(self.nBins)/(self.nBins-1.))
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
      for iBin1 in range(self.nBins):
         # add entry to caption
         color = Colors[iBin1]
         ax.plot([], [], c=color, label=r'$\langle\kappa_{g_i} \kappa_{g_{i+'+str(iBin1)+r'}} \rangle $')


#         # show only the auto
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit**2
#         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.sUnit**2
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov)) / self.sUnit**2
#         #
#         ax.errorbar(L*(1.+0.01*i1/self.nSS), d, yerr=std, ls='-', lw=1, elinewidth=1.5, marker='.', markersize=2, color=color)#, label=r'$\gamma_{'+str(iBin1)+'} \gamma_{'+str(iBin2)+'}$')
##            ax.plot(L*(1.+0.01*i1/self.nSS), shot, ls='--', lw=1, color=color)   # different color for each bin
#         ax.plot(L*(1.+0.01*i1/self.nSS), shot, ls='--', lw=1, color='grey')   # same color for all bins
#         # move to next row
#         i1 += self.nBins - iBin1

         # show all the cross-correlations
         for iBin2 in range(iBin1, self.nBins):
            color = Colors[iBin2-iBin1]
            #color = Colors[iBin1]
            #
            I = range(i1*self.nL, (i1+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I) / self.sUnit**2
            shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I) / self.sUnit**2
            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
            std = np.sqrt(np.diag(cov)) / self.sUnit**2
            #
            ax.plot(L*(1.+0.01*i1/self.nSS), std / d, '.-', lw=1, color=color)
            # move to next row
            i1 += 1

      #
      ax.set_xlim((90., 2.5e3))
      ax.legend(loc=1, fontsize='x-small', labelspacing=0., handlelength=0.4, borderaxespad=0.05)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\sigma\left( C_\ell^{\gamma\gamma} \right) / C_\ell^{\gamma\gamma}$')
      ax.set_title(r'Shear tomography')
      #
      fig.savefig(self.figurePath+"/sp2d_ss.pdf")
      if show:
         plt.show()
      fig.clf()





#   def plotUncertaintyPowerSpectra(self, show=False):
#
#      # kk
#      fig=plt.figure(0)
#      ax=fig.add_subplot(111)
#      #
#      i1 = 0
#      I = range(i1*self.nL, (i1+1)*self.nL)
#      L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#      d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#      shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I)
#      cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#      std = np.sqrt(np.diag(cov))
#      #
#      ax.plot(L, std / d, '.-', lw=2, color='b')
#      #
#      ax.set_xscale('log')
#      ax.set_yscale('log', nonposy='clip')
#      plt.setp(ax.get_xticklabels(), visible=False)
#      #
#      ax.set_title(r'CMB lensing')
#      ax.set_xlabel(r'$\ell$')
#      ax.set_ylabel(r'$\sigma \left( C_\ell^{\kappa_\text{CMB} \kappa_\text{CMB}} \right) / C_\ell^{\kappa_\text{CMB} \kappa_\text{CMB}}$', fontsize=18)
#      #
#      fig.savefig(self.figurePath+"/sp2d_kk.pdf")
#      if show:
#         plt.show()
#      fig.clf()
#
#
#      # kg
#      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
#      #
#      fig=plt.figure(0)
#      ax=fig.add_subplot(111)
#      #
#      i1 = self.nKK
#      for iBin1 in range(self.nBins):
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I)
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov))
#         #
#         color = Colors[iBin1]
#         ax.plot(L, std/d, '.-', lw=2, color=color)
#         # advance counter in data vector
#         i1 += 1
#      #
#      ax.set_xscale('log')
#      ax.set_yscale('log', nonposy='clip')
#      plt.setp(ax.get_xticklabels(), visible=False)
#      #
#      ax.set_title(r'Galaxy - CMB lensing')
#      ax.set_xlabel(r'$\ell$')
#      ax.set_ylabel(r'$\sigma \left( C_\ell^{g \kappa_\text{CMB}} \right) / C_\ell^{g \kappa_\text{CMB}}$', fontsize=18)
#      #
#      fig.savefig(self.figurePath+"/sp2d_kg.pdf")
#      if show:
#         plt.show()
#      fig.clf()
#      
#
#      # ks
#      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
#      #
#      fig=plt.figure(0)
#      ax=fig.add_subplot(111)
#      #
#      i1 = self.nKK + self.nKG
#      for iBin1 in range(self.nBins):
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#         shot = extractMaskedVec(self.shotNoiseVector, mask=self.lMaxMask, I=I)
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov))
#         #
#         color = Colors[iBin1]
#         ax.plot(L, std/d, '.-', lw=2, color=color, label=r'$\langle \kappa_\text{CMB} \gamma_{'+str(iBin1)+'} \rangle$')
#         # advance counter in data vector
#         i1 += 1
#      #
#      ax.set_xlim((90., 2.e3))
#      ax.legend(loc=1, fontsize='x-small', labelspacing=0.)
#      ax.set_xscale('log')
#      ax.set_yscale('log', nonposy='clip')
#      plt.setp(ax.get_xticklabels(), visible=False)
#      #
#      ax.set_title(r'CMB lensing - galaxy lensing')
#      ax.set_xlabel(r'$\ell$')
#      ax.set_ylabel(r'$\sigma \left( C_\ell^{\kappa_\text{CMB} \gamma} \right) / C_\ell^{\kappa_\text{CMB} \gamma}$', fontsize=18)
#      #
#      fig.savefig(self.figurePath+"/sp2d_ks.pdf")
#      if show:
#         plt.show()
#      fig.clf()
#
#
#      # gg: panels
#      Colors = plt.cm.autumn(1.*np.arange(self.nBins)/(self.nBins-1.))
#      #
#      fig=plt.figure(0)
#      gs = gridspec.GridSpec(3, 1)#, height_ratios=[1, 1, 1])
#      gs.update(hspace=0.)
#      
#      # auto
#      ax0=plt.subplot(gs[0])
#      i1 = self.nKK + self.nKG + self.nKS
#      for iBin1 in range(self.nBins):
##         d = self.dataVector[i1*self.nL:(i1+1)*self.nL]
##         std = np.sqrt(np.diag(self.covMat[i1*self.nL:(i1+1)*self.nL, i1*self.nL:(i1+1)*self.nL]))
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov))
#         #
#         color = Colors[iBin1]
#         ax0.plot(L, std / d, '.-', lw=2, color=color)
#         # advance counter in data vector
#         i1 += self.nBins - iBin1
#      #
#      ax0.set_xlim((90., 2.e3))
#      ax0.set_ylim((9.e-3, 2.5e-2))
#      ax0.set_xscale('log')
#      ax0.set_yscale('log', nonposy='clip')
#      plt.setp(ax0.get_xticklabels(), visible=False)
#      #
#      ax0.set_title(r'Clustering')
#      ax0.set_ylabel(r'$g_i g_i$', fontsize=18)
#
#      # cross i,i+1
#      ax1=plt.subplot(gs[1])
#      i1 = self.nKK + self.nKG + self.nKS + 1
#      for iBin1 in range(self.nBins-1):
##         d = self.dataVector[i1*self.nL:(i1+1)*self.nL]
##         std = np.sqrt(np.diag(self.covMat[i1*self.nL:(i1+1)*self.nL, i1*self.nL:(i1+1)*self.nL]))
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov))
#         #
#         color = Colors[iBin1]
#         ax1.plot(L, std / d, '-', lw=2, color=color)
#         # advance counter in data vector
#         i1 += self.nBins - iBin1
#      #
#      ax1.set_xscale('log')
#      ax1.set_yscale('log', nonposy='clip')
#      plt.setp(ax1.get_xticklabels(), visible=False)
#      #
#      ax1.set_ylabel(r'$g_i g_{i+1}$', fontsize=18)
#
#      # cross i,i+2
#      ax2=plt.subplot(gs[2])
#      i1 = self.nKK + self.nKG + self.nKS + 2
#      for iBin1 in range(self.nBins-2):
##         d = self.dataVector[i1*self.nL:(i1+1)*self.nL]
##         std = np.sqrt(np.diag(self.covMat[i1*self.nL:(i1+1)*self.nL, i1*self.nL:(i1+1)*self.nL]))
#         I = range(i1*self.nL, (i1+1)*self.nL)
#         L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#         d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#         cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#         std = np.sqrt(np.diag(cov))
#         #
#         color = Colors[iBin1]
#         ax2.plot(L, std / d, '.-', lw=2, color=color)
#         # advance counter in data vector
#         i1 += self.nBins - iBin1
#      #
#      ax2.set_xscale('log')
#      ax2.set_yscale('log', nonposy='clip')
#      #
#      ax2.set_ylabel(r'$g_i g_{i+2}$', fontsize=18)
#      ax2.set_xlabel(r'$\ell$')
#      #
#      fig.savefig(self.figurePath+"/sp2d_gg.pdf")
#      if show:
#         plt.show()
#      fig.clf()
#
#
#      # gs
#      Colors = plt.cm.winter(1.*np.arange(self.nBins)/(self.nBins-1.))
#      fig=plt.figure(1)
#      ax=fig.add_subplot(111)
#      #
#      i1 = self.nKK + self.nKG + self.nKS + self.nGG
#      for iBin1 in range(self.nBins):
#         color = Colors[iBin1]
#         for iBin2 in range(self.nBins):
##            d = self.dataVector[i1*self.nL:(i1+1)*self.nL]
##            std = np.sqrt(np.diag(self.covMat[i1*self.nL:(i1+1)*self.nL, i1*self.nL:(i1+1)*self.nL]))
#            I = range(i1*self.nL, (i1+1)*self.nL)
#            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#            std = np.sqrt(np.diag(cov))
#            #
#            ax.plot(L*(1.+0.01*i1/self.nGS), std / d, '.-', lw=1, color=color)
#            # move to next row
#            i1 += 1
#      #
##      ax.legend(loc=1)
#      ax.set_xscale('log')
#      ax.set_yscale('log', nonposy='clip')
#      ax.set_xlabel(r'$\ell$')
#      ax.set_ylabel(r'$\sigma\left( C_\ell^{g\gamma} \right) / C_\ell^{g\gamma}$')
#      ax.set_title(r'Galaxy $\times$ galaxy lensing')
#      #
#      fig.savefig(self.figurePath+"/sp2d_gs.pdf")
#      if show:
#         plt.show()
#      fig.clf()
#
#
#      # ss: all on same plot
#      Colors = plt.cm.jet(1.*np.arange(self.nBins)/(self.nBins-1.))
#      fig=plt.figure(2)
#      ax=fig.add_subplot(111)
#      #
#      i1 = self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
#      for iBin1 in range(self.nBins):
#         # add entry to caption
#         color = Colors[iBin1]
#         ax.plot([], [], c=color, label=r'$\langle\gamma_{i} \gamma_{i+'+str(iBin1)+r'} \rangle $')
#         for iBin2 in range(iBin1, self.nBins):
##            d = self.dataVector[i1*self.nL:(i1+1)*self.nL]
##            std = np.sqrt(np.diag(self.covMat[i1*self.nL:(i1+1)*self.nL, i1*self.nL:(i1+1)*self.nL]))
#            color = Colors[iBin2-iBin1]
#            #
#            I = range(i1*self.nL, (i1+1)*self.nL)
#            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
#            d = extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
#            cov = extractMaskedMat(self.covMat, mask=self.lMaxMask, I=I)
#            std = np.sqrt(np.diag(cov))
#            #
#            ax.plot(L*(1.+0.01*i1/self.nSS), std / d, '.-', lw=1, color=color)
#            # move to next row
#            i1 += 1
#      #
#      ax.set_xlim((90., 2.e3))
#      ax.legend(loc=1, labelspacing=0.05, handlelength=0.4, borderaxespad=0.01)
#      ax.set_xscale('log')
#      ax.set_yscale('log', nonposy='clip')
#      ax.set_xlabel(r'$\ell$')
#      ax.set_ylabel(r'$\sigma\left( C_\ell^{\gamma\gamma} \right) / C_\ell^{\gamma\gamma}$')
#      ax.set_title(r'Shear tomography')
#      #
#      fig.savefig(self.figurePath+"/sp2d_ss.pdf")
#      if show:
#         plt.show()
#      fig.clf()






   def plotDerivativeDataVectorCosmo(self, show=False):
      """Derivative of the data vector wrt cosmo parameters.
      """
#      # one color per cosmo param
#      Colors = plt.cm.jet(1.*np.arange(self.cosmoPar.nPar)/self.cosmoPar.nPar)
#      #
#      purple, darkmagenta, darkviolet
#      orange
#      lime, mediumspringgreen
#      darkolivegreen, darkgreen
#      r
#      royalblue, cornflowerblue
#      navy, midnightblue
#      gold, yellow
#      silver, gray, darkgray
#      saddlebrown, sienna, brown

#      fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
#      labelspacing: vertical spacing
#      handlelength=0.5
#      handletextpad=0.01
#      columnspacing=0.4

      Colors = ['purple', 'orange', 'lime', 'darkolivegreen', 'r', 'royalblue', 'navy', 'gold', 'silver', 'saddlebrown']
      

      # kk
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      #ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{\kappa_\text{CMB}\kappa_\text{CMB}} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_kk_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()


      # kg
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK, self.nKK+self.nKG):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      #ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{g\kappa_\text{CMB}} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_kg_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()


      # ks
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG, self.nKK+self.nKG+self.nKS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      #ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{\gamma\kappa_\text{CMB}} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_ks_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      
      # gg
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS, self.nKK+self.nKG+self.nKS+self.nGG):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      #ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{gg} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_gg_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      # gs
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in range(self.cosmoPar.nPar):
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS+self.nGG, self.nKK+self.nKG+self.nKS+self.nGG+self.nGS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*(i2pt-self.nGG)/self.nGS)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
#      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{gs} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_gs_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      # ss
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in range(self.cosmoPar.nPar):
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS+self.nGG+self.nGS, self.nKK+self.nKG+self.nKS+self.nGG+self.nGS+self.nSS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            dlnDdlnP = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            dlnDdlnP /= extractMaskedVec(self.dataVector, mask=self.lMaxMask, I=I)
            if self.cosmoPar.fiducial[iPar] <> 0.:
               dlnDdlnP *= self.cosmoPar.fiducial[iPar]
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*(i2pt-(self.nGG+self.nGS))/self.nSS)
            ax.plot(L, dlnDdlnP, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
#      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      ax.set_ylim((-6., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell^{ss} / d\ln \text{Param.}$')
      #
      fig.savefig(self.figurePath+"/dp2d_ss_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()



   ##################################################################################


   def plotErrorDerivativeDataVectorCosmo(self, relatErrorDeriv, show=False):
      """Plot the input relative difference of derivatives of the data vector wrt cosmo parameters.
      Used for convergence null test, when varying the step size.
      """

      Colors = ['purple', 'orange', 'lime', 'darkolivegreen', 'r', 'royalblue', 'navy', 'gold', 'silver', 'saddlebrown']
      

      # kk
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat. error on $dC_\ell^{\kappa_\text{CMB}\kappa_\text{CMB}} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_kk_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()


      # kg
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK, self.nKK+self.nKG):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat. error on $dC_\ell^{g\kappa_\text{CMB}} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_kg_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()


      # ks
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG, self.nKK+self.nKG+self.nKS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat. error on $dC_\ell^{\gamma\kappa_\text{CMB}} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_ks_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      
      # gg
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in [6]:  # Mnu
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS, self.nKK+self.nKG+self.nKS+self.nGG):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*i2pt/self.nGG)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat error on $dC_\ell^{gg} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_gg_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      # gs
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in range(self.cosmoPar.nPar):
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS+self.nGG, self.nKK+self.nKG+self.nKS+self.nGG+self.nGS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*(i2pt-self.nGG)/self.nGS)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
#      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat. error on $dC_\ell^{gs} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_gs_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()

      # ss
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      # for each cosmo parameter
      for iPar in range(self.cosmoPar.nPar)[::-1]:
#      for iPar in range(self.cosmoPar.nPar):
#      for iPar in [9]:  # curvature
         # plot all the 2pt functions
         for i2pt in range(self.nKK+self.nKG+self.nKS+self.nGG+self.nGS, self.nKK+self.nKG+self.nKS+self.nGG+self.nGS+self.nSS):
#            dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
            I = range(i2pt*self.nL, (i2pt+1)*self.nL)
            L = extractMaskedVec(self.L, mask=self.lMaxMask[I])
            relatErrordDdp = extractMaskedVec(self.derivativeDataVector[iPar, :], mask=self.lMaxMask, I=I)
            color = Colors[iPar]
            color = darkerLighter(color, amount=-0.5*(i2pt-(self.nGG+self.nGS))/self.nSS)
            ax.plot(L, relatErrordDdp, c=color, lw=3)
         ax.plot([],[], c=Colors[iPar], label=self.cosmoPar.namesLatex[iPar])
      #
      #ax.grid()
#      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
#      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.5, handletextpad=0.01, columnspacing=0.4, borderaxespad=0.01)
      ax.legend(loc=4, ncol=5, labelspacing=0.07, frameon=False, handlelength=0.7, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'Relat. error on $dC_\ell^{ss} / d\text{Param.}$')
      #
      fig.savefig(self.figurePath+"/relat_error_dp2d_ss_cosmo.pdf")
      if show:
         plt.show()
      fig.clf()






   ##################################################################################



   def plotSingleDerivative(self, ab, i2pt, iPar):
      """Derivative of the desired 2pt function wrt the desired parameter.
      """
      print "2-pt function:", ab, i2pt
      print "Parameter:", self.fullPar.names[iPar]
      
      if ab=='kk':
         i2pt += 0
      elif ab=='kg':
         i2pt += self.nKK
      elif ab=='ks':
         i2pt += self.nKK + self.nKG
      elif ab=='gg':
         i2pt += self.nKK + self.nKG + self.nKS
      elif ab=='gs':
         i2pt += self.nKK + self.nKG + self.nKS + self.nGG
      elif ab=='ss':
         i2pt += self.nKK + self.nKG + self.nKS + self.nGG + self.nGS
      else:
         return
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      dlnDdlnP = self.derivativeDataVector[iPar, i2pt*self.nL:(i2pt+1)*self.nL] / self.dataVector[i2pt*self.nL:(i2pt+1)*self.nL]
      if self.fullPar.fiducial[iPar] <> 0.:
         dlnDdlnP *= self.fullPar.fiducial[iPar]
#      color = Colors[iPar]
      ax.plot(self.L, dlnDdlnP, 'b', lw=3)
      ax.plot([],[], 'b', label=self.fullPar.namesLatex[iPar])
      #
      ax.grid()
      ax.legend(loc=4, ncol=5, labelspacing=0.05, frameon=False, handlelength=0.4, borderaxespad=0.01)
      ax.set_xscale('log', nonposx='clip')
#      ax.set_ylim((-4., 4.))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$d\ln C_\ell / d\ln \text{Param.}$')
#      ax.set_title()
      #
#      fig.savefig(self.figurePath+"/dp2d_gg_cosmo.pdf")
#      fig.clf()
      plt.show()



   ##################################################################################
   ##################################################################################



##!!!! older version, to be removed   
#   def photoZRequirements(self, mask=None, name=""):
#      '''Here the photo-z value is such that
#      sigma (delta z) = photoz
#      sigma (sigma z) = 1.5 * photoz
#      '''
#      if mask is None:
#         mask=self.lMaxMask
#      
#      # get the Fisher matrix
#      fisherData, fisherPosterior = self.generateFisher(mask=mask)
#      
#      # values of photo-z priors to try
#      nPhotoz = 21
#      Photoz = np.logspace(np.log10(1.e-5), np.log10(0.2), nPhotoz, 10.)
#      
#      # Posterior uncertainties on all parameters
#      sFull = np.zeros((self.fullPar.nPar, nPhotoz))
#      
#      # posterior uncertainties for various combinations,
#      # cosmology
#      sCosmoFull = np.zeros((len(self.cosmoPar.IFull), nPhotoz))
#      sCosmoLCDMMnuW0Wa = np.zeros((len(self.cosmoPar.ILCDMMnuW0Wa), nPhotoz))
#      sCosmoLCDMMnu = np.zeros((len(self.cosmoPar.ILCDMMnu), nPhotoz))
#      sCosmoLCDMMnuCurv = np.zeros((len(self.cosmoPar.ILCDMMnuCurv), nPhotoz))
#      sCosmoLCDMW0Wa = np.zeros((len(self.cosmoPar.ILCDMW0Wa), nPhotoz))
#      sCosmoLCDMW0WaCurv = np.zeros((len(self.cosmoPar.ILCDMW0WaCurv), nPhotoz))
#      # photo-z
#      sPhotozFull = np.zeros((self.photoZPar.nPar, nPhotoz))
#      sPhotozLCDMMnuW0Wa = np.zeros((self.photoZPar.nPar, nPhotoz))
#      sPhotozLCDMMnu = np.zeros((self.photoZPar.nPar, nPhotoz))
#      sPhotozLCDMMnuCurv = np.zeros((self.photoZPar.nPar, nPhotoz))
#      sPhotozLCDMW0Wa = np.zeros((self.photoZPar.nPar, nPhotoz))
#      sPhotozLCDMW0WaCurv = np.zeros((self.photoZPar.nPar, nPhotoz))
#      
#      for iPhotoz in range(nPhotoz):
#         photoz = Photoz[iPhotoz]
#         # update the photo-z Gaussian priors only,
#         # whether or not outliers are included
#         if self.photoZPar.outliers==0.:
#            newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=photoz, szStd=photoz*1.5)
#         else:
#            newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=photoz, szStd=photoz*1.5, outliers=0.1, outliersStd=0.05)
#         # update the full parameter object
#         newPar = self.cosmoPar.copy()
#         newPar.addParams(self.galaxyBiasPar)
#         newPar.addParams(self.shearMultBiasPar)
#         newPar.addParams(newPhotoZPar)
#         # get the new posterior Fisher matrix, including the prior
#         newPar.fisher += fisherData
#         
#         # Extract full uncertainties
#         sFull[:,iPhotoz] = newPar.paramUncertainties(marg=True)
#         
#         # Extract parameter combinations:
#         #
#         # Full: LCDM + Mnu + curv + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.IFull + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_full_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoFull = par.extractParams(range(len(self.cosmoPar.IFull)), marg=True)
#         sCosmoFull[:, iPhotoz] = parCosmoFull.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozFull = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozFull[:, iPhotoz] = parPhotozFull.paramUncertainties(marg=True)
#         #
#         # LCDM + Mnu
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnu + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_ldcmmnu_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnu = par.extractParams(range(len(self.cosmoPar.ILCDMMnu)), marg=True)
#         sCosmoLCDMMnu[:, iPhotoz] = parCosmoLCDMMnu.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnu = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozLCDMMnu[:, iPhotoz] = parPhotozLCDMMnu.paramUncertainties(marg=True)
#         #
#         # LCDM + Mnu + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnuW0Wa + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmmnuw0wa_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnuW0Wa = par.extractParams(range(len(self.cosmoPar.ILCDMMnuW0Wa)), marg=True)
#         sCosmoLCDMMnuW0Wa[:, iPhotoz] = parCosmoLCDMMnuW0Wa.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnuW0Wa = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozLCDMMnuW0Wa[:, iPhotoz] = parPhotozLCDMMnuW0Wa.paramUncertainties(marg=True)
#         #
#         # LCDM + Mnu + curv
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnuCurv + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmmnucurv_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnuCurv = par.extractParams(range(len(self.cosmoPar.ILCDMMnuCurv)), marg=True)
#         sCosmoLCDMMnuCurv[:, iPhotoz] = parCosmoLCDMMnuCurv.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnuCurv = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozLCDMMnuCurv[:, iPhotoz] = parPhotozLCDMMnuCurv.paramUncertainties(marg=True)
#         #
#         # LCDM + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMW0Wa + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmw0wa_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMW0Wa = par.extractParams(range(len(self.cosmoPar.ILCDMW0Wa)), marg=True)
#         sCosmoLCDMW0Wa[:, iPhotoz] = parCosmoLCDMW0Wa.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMW0Wa = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozLCDMW0Wa[:, iPhotoz] = parPhotozLCDMW0Wa.paramUncertainties(marg=True)
#         #
#         # LCDM + w0,wa + curvature
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMW0WaCurv + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmw0wacurv_gphotozprior_"+floatExpForm(photoz)+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMW0WaCurv = par.extractParams(range(len(self.cosmoPar.ILCDMW0WaCurv)), marg=True)
#         sCosmoLCDMW0WaCurv[:, iPhotoz] = parCosmoLCDMW0WaCurv.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMW0WaCurv = par.extractParams(range(-self.photoZPar.nPar, 0), marg=True)
#         sPhotozLCDMW0WaCurv[:, iPhotoz] = parPhotozLCDMW0WaCurv.paramUncertainties(marg=True)
#
#      
#      ##################################################################################
#      # Degradation of cosmo. par., depending on photo-z prior
#
#      def plotDegradation(sCosmo, parCosmo, path):
#         fig=plt.figure(0)
#         ax=fig.add_subplot(111)
#         #
#         # fiducial prior
#         ax.axvline(0.002, ls='--', lw=1, c='gray', label=r'Requirement')
#         #
#         for iPar in range(parCosmo.nPar):
##         for iPar in range(len(sCosmo)):
#            ax.plot(Photoz, sCosmo[iPar, :] / sCosmo[iPar, 0], label=parCosmo.namesLatex[iPar])
#         #
#         ax.set_xscale('log', nonposx='clip')
##         ax.legend(loc=2)
#         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
#         ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Perfect photo-z}$')
#         ax.set_xlabel(r'Gaussian photo-z prior')
#         #
#         fig.savefig(self.figurePath+path, bbox_inches='tight')
#         fig.clf()
#
#      # Full: LCDM + Mnu + curv + w0,wa
#      plotDegradation(sCosmoFull, parCosmoFull, "/gphotozreq_cosmo_deg_full_"+name+".pdf")
#      # LCDM + Mnu
#      plotDegradation(sCosmoLCDMMnu, parCosmoLCDMMnu, "/gphotozreq_cosmo_deg_lcdmmnu_"+name+".pdf")
#      # LCDM + Mnu + w0,wa
#      plotDegradation(sCosmoLCDMMnuW0Wa, parCosmoLCDMMnuW0Wa, "/gphotozreq_cosmo_deg_lcdmmnuw0wa_"+name+".pdf")
#      # LCDM + Mnu + curv
#      plotDegradation(sCosmoLCDMMnuCurv, parCosmoLCDMMnuCurv, "/gphotozreq_cosmo_deg_lcdmmnucurv_"+name+".pdf")
#      # LCDM + w0,wa
#      plotDegradation(sCosmoLCDMW0Wa, parCosmoLCDMW0Wa, "/gphotozreq_cosmo_deg_lcdmw0wa_"+name+".pdf")
#      # LCDM + w0,wa + curvature
#      plotDegradation(sCosmoLCDMW0WaCurv, parCosmoLCDMW0WaCurv, "/gphotozreq_cosmo_deg_lcdmw0wacurv_"+name+".pdf")
#
#
#      ##################################################################################
#      # Relative cosmo. par. uncertainty, as a function of photo-z prior
#      # !!! these plots don't end up very useful: the difference between uncertainties of 
#      # different parameters are orders of magnitude, while the evolution with photo-z prior
#      # is only tens of percent, so all the curves look flat
#
##      def relatError(sigma, par):
##         """Computes relative uncertainty, except if the fiducial parameter is zero.
##         In that case, return absolute_uncertainty.
##         """
##         result = np.zeros_like(sigma)
##         for iPar in range(par.nPar):
##            if par.fiducial[iPar]==0.:
##               result[iPar] = sigma[iPar]
##            else:
##               result[iPar] = sigma[iPar] / par.fiducial[iPar]
##         return result
##
##      def plotRelative(sP, par, path):
##         # compute relative uncertainty
##         sPOverP = relatError(sP, par)
##         
##         fig=plt.figure(0)
##         ax=fig.add_subplot(111)
##         #
##         # fiducial prior
##         ax.axvline(0.002, ls='--', c='gray', label=r'Requirement')
##         #
##         for iPar in range(par.nPar):
##            ax.plot(Photoz, sPOverP[iPar, :], label=par.namesLatex[iPar])
##         #
##         ax.set_xscale('log', nonposx='clip')
##         ax.set_yscale('log', nonposy='clip')
###         ax.legend(loc=2)
##         ax.legend(loc=2, labelspacing=0.1, frameon=True, handlelength=1.)
##         ax.set_ylabel(r'$\sigma_\text{Param} / \text{Param}$')
##         ax.set_xlabel(r'Gaussian photo-z prior')
##         #
##         fig.savefig(self.figurePath+path, bbox_inches='tight')
##         fig.clf()
##      
##      # Full: LCDM + Mnu + curv + w0,wa
##      plotRelative(sCosmoFull, parCosmoFull, "/gphotozreq_cosmo_rel_full_"+name+".pdf")
##      # LCDM + Mnu
##      plotRelative(sCosmoLCDMMnu, parCosmoLCDMMnu, "/gphotozreq_cosmo_rel_lcdmmnu_"+name+".pdf")
##      # LCDM + Mnu + w0,wa
##      plotRelative(sCosmoLCDMMnuW0Wa, parCosmoLCDMMnuW0Wa, "/gphotozreq_cosmo_rel_lcdmmnuw0wa_"+name+".pdf")
##      # LCDM + Mnu + curv
##      plotRelative(sCosmoLCDMMnuCurv, parCosmoLCDMMnuCurv, "/gphotozreq_cosmo_rel_lcdmmnucurv_"+name+".pdf")
##      # LCDM + w0,wa
##      plotRelative(sCosmoLCDMW0Wa, parCosmoLCDMW0Wa, "/gphotozreq_cosmo_rel_lcdmw0wa_"+name+".pdf")
##      # LCDM + w0,wa + curvature
##      plotRelative(sCosmoLCDMW0WaCurv, parCosmoLCDMW0WaCurv, "/gphotozreq_cosmo_rel_lcdmw0wacurv_"+name+".pdf")
#
#
#      ##################################################################################
#      # Comparing various param combinations
#
#      fig=plt.figure(10)
#      ax=fig.add_subplot(111)
#      #
#      # fiducial prior
#      ax.axvline(0.002, color='gray')
#      #
#      for iPar in range(parCosmoLCDMMnuW0Wa.nPar):
#         ax.plot(Photoz, sCosmoFull[iPar, :] / sCosmoLCDMMnuW0Wa[iPar, :], label=parCosmoFull.namesLatex[iPar])
#      #
#      ax.set_xscale('log', nonposx='clip')
##      ax.legend(loc=2)
#      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
#      ax.set_ylabel(r'$\sigma_\text{Param}^\text{Full} / \sigma_\text{Param}^\text{no curv.}$')
#      ax.set_xlabel(r'Gaussian photo-z prior')
#      #
#      fig.savefig(self.figurePath+"/gphotozreq_cosmo_full_over_lcdmmnuw0wa_"+name+".pdf", bbox_inches='tight')
#      fig.clf()
#
#      
#      ##################################################################################
#      # Photo-z posterior, as a function of photo-z prior
#
#      def plotPhotoZPosterior(sPhotoz, parPhotoz, path):
#         fig=plt.figure(1)
#         ax=fig.add_subplot(111)
#         #
#         # fiducial prior
#         ax.axvline(0.002, ls='--', lw=1, color='gray', label=r'Requirement')
#         ax.axhline(0.002, ls='--', lw=1, color='gray')
#         ax.plot(Photoz, Photoz, c='gray', ls=':', lw=1, label=r'Posterior=Prior')
#         ax.plot(Photoz, 1.5*Photoz, c='gray', ls=':')
#         #
#         # photo-z shifts
#         # add legend entry
#         color = 'r'
#         ax.plot([], [], color=color, label=r'$\delta z$')
#         darkLight = 0.
#         for iPar in range(self.nBins):
#            color = 'r'
#            color = darkerLighter(color, amount=darkLight)
#            darkLight += -0.75 * 1./self.nBins
#            ax.plot(Photoz, sPhotoz[iPar, :], color=color)
#         #
#         # photo-z scatter
#         # add legend entry
#         color = 'b'
#         ax.plot([], [], color=color, label=r'$\sigma_z / (1+z)$')
#         darkLight = 0.
#         for iPar in range(self.nBins, 2*self.nBins):
#            color = 'b'
#            color = darkerLighter(color, amount=darkLight)
#            darkLight += -0.75 * 1./self.nBins
#            ax.plot(Photoz, sPhotoz[iPar, :], color=color)
#         #
#         ax.set_xscale('log', nonposx='clip')
#         ax.set_yscale('log', nonposy='clip')
#         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
#         ax.set_ylabel(r'Gaussian photo-z posterior')
#         ax.set_xlabel(r'Gaussian photo-z prior')
#         #
#         fig.savefig(self.figurePath+path, bbox_inches='tight')
#         fig.clf()
#
#      # Full: LCDM + Mnu + curv + w0,wa
#      plotPhotoZPosterior(sPhotozFull, parPhotozFull, "/gphotozreq_photoz_full_"+name+".pdf")
#      # LCDM + Mnu
#      plotPhotoZPosterior(sPhotozLCDMMnu, parPhotozLCDMMnu, "/gphotozreq_photoz_lcdmmnu_"+name+".pdf")
#      # LCDM + Mnu + w0,wa
#      plotPhotoZPosterior(sPhotozLCDMMnuW0Wa, parPhotozLCDMMnuW0Wa, "/gphotozreq_photoz_lcdmmnuw0wa_"+name+".pdf")
#      # LCDM + Mnu + curv
#      plotPhotoZPosterior(sPhotozLCDMMnuCurv, parPhotozLCDMMnuCurv, "/gphotozreq_photoz_lcdmmnucurv_"+name+".pdf")
#      # LCDM + w0,wa
#      plotPhotoZPosterior(sPhotozLCDMW0Wa, parPhotozLCDMW0Wa, "/gphotozreq_photoz_lcdmw0wa_"+name+".pdf")
#      # LCDM + w0,wa + curvature
#      plotPhotoZPosterior(sPhotozLCDMW0WaCurv, parPhotozLCDMW0WaCurv, "/gphotozreq_photoz_lcdmw0wacurv_"+name+".pdf")
#
#
#   ##################################################################################
#   ##################################################################################
#   
#   
##!!!! older version, to be removed   
#   def photoZOutliersRequirements(self, mask=None, name="", Gphotoz='req'):
#      '''Here the priors for the Gaussian photo-z are fixed
#      at the level of the LSST requirements.
#      The outlier fractions are kept at 10%,
#      but the priors on the outlier fractions are varied.
#      !!! May not be super happy with having differing g and s bins
#      '''
#      if mask is None:
#         mask=self.lMaxMask
#      if Gphotoz=='req':
#         strGphotoz = "_reqGphotoz"
#      elif Gphotoz=='perfect':
#         strGphotoz = "_perfectGphotoz"
#      elif Gphotoz=='noprior':
#         strGphotoz = "_nopriorGphotoz"
#
#      # get the Fisher matrix
#      fisherData, fisherPosterior = self.generateFisher(mask=mask)
#      
#      # values of photo-z priors to try
#      nPhotoz = 21
#      Photoz = np.logspace(np.log10(1.e-4), np.log10(5.), nPhotoz, 10.)
#      
#      # Posterior uncertainties on all parameters
#      sFull = np.zeros((self.fullPar.nPar, nPhotoz))
#      
#      # posterior uncertainties for various combinations,
#      # cosmology
#      sCosmoFull = np.zeros((len(self.cosmoPar.IFull), nPhotoz))
#      sCosmoLCDMMnuW0Wa = np.zeros((len(self.cosmoPar.ILCDMMnuW0Wa), nPhotoz))
#      sCosmoLCDMMnu = np.zeros((len(self.cosmoPar.ILCDMMnu), nPhotoz))
#      sCosmoLCDMMnuCurv = np.zeros((len(self.cosmoPar.ILCDMMnuCurv), nPhotoz))
#      sCosmoLCDMW0Wa = np.zeros((len(self.cosmoPar.ILCDMW0Wa), nPhotoz))
#      sCosmoLCDMW0WaCurv = np.zeros((len(self.cosmoPar.ILCDMW0WaCurv), nPhotoz))
#      # photo-z
#      # keep only the cij, not the Gaussian photo-z params
#      nCij = self.photoZPar.nPar - 2*self.nBins
#      sPhotozFull = np.zeros((nCij, nPhotoz))
#      sPhotozLCDMMnuW0Wa = np.zeros((nCij, nPhotoz))
#      sPhotozLCDMMnu = np.zeros((nCij, nPhotoz))
#      sPhotozLCDMMnuCurv = np.zeros((nCij, nPhotoz))
#      sPhotozLCDMW0Wa = np.zeros((nCij, nPhotoz))
#      sPhotozLCDMW0WaCurv = np.zeros((nCij, nPhotoz))
#      
#      for iPhotoz in range(nPhotoz):
#         photoz = Photoz[iPhotoz]
#         # update the photo-z priors
#         
#         if Gphotoz=='req':
#            newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=0.002, szStd=0.003, outliers=0.1, outliersStd=photoz)
#         elif Gphotoz=='perfect':
#            newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=1.e-5, szStd=1.e-5, outliers=0.1, outliersStd=photoz)
#         elif Gphotoz=='noprior':
#            newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=1., szStd=1., outliers=0.1, outliersStd=photoz)
#         # if g and s bins are different
#         if self.photoZSPar is not None:
#            newPhotoZPar.addParams(newPhotoZPar)
#         
#         # update the full parameter object
#         newPar = self.cosmoPar.copy()
#         newPar.addParams(self.galaxyBiasPar)
#         newPar.addParams(self.shearMultBiasPar)
#         newPar.addParams(newPhotoZPar)
#         # if g and s bins are different
#         if self.photoZSPar is not None:
#            newPar.addParams(newPhotoZPar)
#         # get the new posterior Fisher matrix, including the prior
#         newPar.fisher += fisherData
#         
#         # Extract full uncertainties
#         sFull[:,iPhotoz] = newPar.paramUncertainties(marg=True)
#         
#         
#         ##################################################################################
#         # Extract parameter combinations:
#         
#         # Full: LCDM + Mnu + curv + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.IFull + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         # for the best/worse photo-z prior, save the forecast to file
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:            
#            par.printParams(path=self.figurePath+"/posterior_full_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoFull = par.extractParams(range(len(self.cosmoPar.IFull)), marg=True)
#         sCosmoFull[:, iPhotoz] = parCosmoFull.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozFull = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozFull[:, iPhotoz] = parPhotozFull.paramUncertainties(marg=True)
#
#         #print "!!!!!!!the extracted cij parameters have nPar = ", parPhotozFull.nPar
#
#
#         
#         # LCDM + Mnu
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnu + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:            
#            par.printParams(path=self.figurePath+"/posterior_ldcmmnu_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnu = par.extractParams(range(len(self.cosmoPar.ILCDMMnu)), marg=True)
#         sCosmoLCDMMnu[:, iPhotoz] = parCosmoLCDMMnu.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnu = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozLCDMMnu[:, iPhotoz] = parPhotozLCDMMnu.paramUncertainties(marg=True)
#         
#         # LCDM + Mnu + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnuW0Wa + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:            
#            par.printParams(path=self.figurePath+"/posterior_lcdmmnuw0wa_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnuW0Wa = par.extractParams(range(len(self.cosmoPar.ILCDMMnuW0Wa)), marg=True)
#         sCosmoLCDMMnuW0Wa[:, iPhotoz] = parCosmoLCDMMnuW0Wa.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnuW0Wa = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozLCDMMnuW0Wa[:, iPhotoz] = parPhotozLCDMMnuW0Wa.paramUncertainties(marg=True)
#         
#         # LCDM + Mnu + curv
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMMnuCurv + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:            
#            par.printParams(path=self.figurePath+"/posterior_lcdmmnucurv_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMMnuCurv = par.extractParams(range(len(self.cosmoPar.ILCDMMnuCurv)), marg=True)
#         sCosmoLCDMMnuCurv[:, iPhotoz] = parCosmoLCDMMnuCurv.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMMnuCurv = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozLCDMMnuCurv[:, iPhotoz] = parPhotozLCDMMnuCurv.paramUncertainties(marg=True)
#         
#         # LCDM + w0,wa
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMW0Wa + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmw0wa_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMW0Wa = par.extractParams(range(len(self.cosmoPar.ILCDMW0Wa)), marg=True)
#         sCosmoLCDMW0Wa[:, iPhotoz] = parCosmoLCDMW0Wa.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMW0Wa = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozLCDMW0Wa[:, iPhotoz] = parPhotozLCDMW0Wa.paramUncertainties(marg=True)
#         
#         # LCDM + w0,wa + curvature
#         # reject unwanted cosmo params
#         I = self.cosmoPar.ILCDMW0WaCurv + range(self.cosmoPar.nPar, self.fullPar.nPar)
#         par = newPar.extractParams(I, marg=False)
#         if iPhotoz==0 or iPhotoz==nPhotoz-1:
#            par.printParams(path=self.figurePath+"/posterior_lcdmw0wacurv_outlierprior_"+floatExpForm(photoz)+strGphotoz+"_"+name+".txt")
#         # cosmology
#         parCosmoLCDMW0WaCurv = par.extractParams(range(len(self.cosmoPar.ILCDMW0WaCurv)), marg=True)
#         sCosmoLCDMW0WaCurv[:, iPhotoz] = parCosmoLCDMW0WaCurv.paramUncertainties(marg=True)
#         # photo-z
#         parPhotozLCDMW0WaCurv = par.extractParams(range(-nCij, 0), marg=True)
#         sPhotozLCDMW0WaCurv[:, iPhotoz] = parPhotozLCDMW0WaCurv.paramUncertainties(marg=True)
#
#
#      ##################################################################################
#      # Degradation of cosmo. par., depending on photo-z prior
#
#      def plotDegradation(sCosmo, parCosmo, path):
#         fig=plt.figure(0)
#         ax=fig.add_subplot(111)
#         #
#         ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
#         #
#         for iPar in range(parCosmo.nPar):
##         for iPar in range(len(sCosmo)):
#            ax.plot(Photoz/(self.nBins-1.), sCosmo[iPar, :] / sCosmo[iPar, 0], label=parCosmo.namesLatex[iPar])
#         #
#         ax.set_xscale('log', nonposx='clip')
#         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
#         ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Perfect photo-z}$')
#         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
#         #
#         fig.savefig(self.figurePath+path, bbox_inches='tight')
#         fig.clf()
#
#      # Full: LCDM + Mnu + curv + w0,wa
#      plotDegradation(sCosmoFull, parCosmoFull, "/outlierreq_cosmo_deg_full"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu
#      plotDegradation(sCosmoLCDMMnu, parCosmoLCDMMnu, "/outlierreq_cosmo_deg_lcdmmnu"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu + w0,wa
#      plotDegradation(sCosmoLCDMMnuW0Wa, parCosmoLCDMMnuW0Wa, "/outlierreq_cosmo_deg_lcdmmnuw0wa"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu + curv
#      plotDegradation(sCosmoLCDMMnuCurv, parCosmoLCDMMnuCurv, "/outlierreq_cosmo_deg_lcdmmnucurv"+strGphotoz+"_"+name+".pdf")
#      # LCDM + w0,wa
#      plotDegradation(sCosmoLCDMW0Wa, parCosmoLCDMW0Wa, "/outlierreq_cosmo_deg_lcdmw0wa"+strGphotoz+"_"+name+".pdf")
#      # LCDM + w0,wa + curvature
#      plotDegradation(sCosmoLCDMW0WaCurv, parCosmoLCDMW0WaCurv, "/outlierreq_cosmo_deg_lcdmw0wacurv"+strGphotoz+"_"+name+".pdf")
#
#
#      ##################################################################################
#      # Relative cosmo. par. uncertainty, as a function of photo-z prior
#      # !!! these plots don't end up very useful: the difference between uncertainties of 
#      # different parameters are orders of magnitude, while the evolution with photo-z prior
#      # is only tens of percent, so all the curves look flat
#
##      def relatError(sigma, par):
##         """Computes relative uncertainty, except if the fiducial parameter is zero.
##         In that case, return 1/absolute_uncertainty.
##         """
##         result = np.zeros_like(sigma)
##         for iPar in range(par.nPar):
##            if par.fiducial[iPar]==0.:
##               result[iPar] = sigma[iPar]
##            else:
##               result[iPar] = sigma[iPar] / par.fiducial[iPar]
##         return result
##
##      def plotRelative(sP, par, path):
##         # compute relative uncertainty
##         sPOverP = relatError(sP, par)
##         
##         fig=plt.figure(0)
##         ax=fig.add_subplot(111)
##         #
##         for iPar in range(par.nPar):
##            ax.plot(Photoz, sPOverP[iPar, :], label=par.namesLatex[iPar])
##         #
##         ax.set_xscale('log', nonposx='clip')
##         ax.set_yscale('log', nonposy='clip')
###         ax.legend(loc=2)
##         ax.legend(loc=2, labelspacing=0.1, frameon=True, handlelength=1.)
##         ax.set_ylabel(r'$\sigma_\text{Param} / \text{Param}$')
##         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
##         #
##         fig.savefig(self.figurePath+path)
##         fig.clf()
##      
##      # Full: LCDM + Mnu + curv + w0,wa
##      plotRelative(sCosmoFull, parCosmoFull, "/outlierreq_cosmo_rel_full"+strGphotoz+"_"+name+".pdf")
##      # LCDM + Mnu
##      plotRelative(sCosmoLCDMMnu, parCosmoLCDMMnu, "/outlierreq_cosmo_rel_lcdmmnu"+strGphotoz+"_"+name+".pdf")
##      # LCDM + Mnu + w0,wa
##      plotRelative(sCosmoLCDMMnuW0Wa, parCosmoLCDMMnuW0Wa, "/outlierreq_cosmo_rel_lcdmmnuw0wa"+strGphotoz+"_"+name+".pdf")
##      # LCDM + Mnu + curv
##      plotRelative(sCosmoLCDMMnuCurv, parCosmoLCDMMnuCurv, "/outlierreq_cosmo_rel_lcdmmnucurv"+strGphotoz+"_"+name+".pdf")
##      # LCDM + w0,wa
##      plotRelative(sCosmoLCDMW0Wa, parCosmoLCDMW0Wa, "/outlierreq_cosmo_rel_lcdmw0wa"+strGphotoz+"_"+name+".pdf")
##      # LCDM + w0,wa + curvature
##      plotRelative(sCosmoLCDMW0WaCurv, parCosmoLCDMW0WaCurv, "/outlierreq_cosmo_rel_lcdmw0wacurv"+strGphotoz+"_"+name+".pdf")
#
#
#      ##################################################################################
#      # Comparing various param combinations
#
#      fig=plt.figure(10)
#      ax=fig.add_subplot(111)
#      #
#      for iPar in range(parCosmoLCDMMnuW0Wa.nPar):
#         ax.plot(Photoz, sCosmoFull[iPar, :] / sCosmoLCDMMnuW0Wa[iPar, :], label=parCosmoFull.namesLatex[iPar])
#      #
#      ax.set_xscale('log', nonposx='clip')
##      ax.legend(loc=2)
#      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
#      ax.set_ylabel(r'$\sigma_\text{Param}^\text{Full} / \sigma_\text{Param}^\text{no curv.}$')
#      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
#      #
#      fig.savefig(self.figurePath+"/outlierreq_cosmo_full_over_lcdmmnuw0wa"+strGphotoz+"_"+name+".pdf")
#      fig.clf()
#
#
#      ##################################################################################
#      # Photo-z posterior, as a function of photo-z prior
#
#      def plotPhotoZPosterior(sPhotoz, parPhotoz, path):
#         fig=plt.figure(1)
#         ax=fig.add_subplot(111)
#         #
#         ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
#         ax.axhline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray')
#         ax.plot(Photoz/(self.nBins-1), Photoz/(self.nBins-1), c='gray', ls=':', lw=1, label=r'Posterior=Prior')
#         #
#         color = 'r'
#         darkLight = 0.
#         #print "!!!!!!! number of photo-z parameters varied:", parPhotoz.nPar
#         for iPar in range(parPhotoz.nPar):
#         #for iPar in range(self.nBins):
#            color = 'r'
#            color = darkerLighter(color, amount=darkLight)
#            darkLight += -0.75 * 1./parPhotoz.nPar
#            ax.plot(Photoz/(self.nBins-1), sPhotoz[iPar, :], color=color, alpha=0.3)
#         #
#         ax.set_xscale('log', nonposx='clip')
#         ax.set_yscale('log', nonposy='clip')
#         ax.legend(loc=2)
#         ax.set_ylabel(r'Posterior on outlier fraction $c_{ij}$')
#         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
#         #
#         fig.savefig(self.figurePath+path, bbox_inches='tight')
#         fig.clf()
#
#      # Full: LCDM + Mnu + curv + w0,wa
#      plotPhotoZPosterior(sPhotozFull, parPhotozFull, "/outlierreq_outliers_full"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu
#      plotPhotoZPosterior(sPhotozLCDMMnu, parPhotozLCDMMnu, "/outlierreq_outliers_lcdmmnu"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu + w0,wa
#      plotPhotoZPosterior(sPhotozLCDMMnuW0Wa, parPhotozLCDMMnuW0Wa, "/outlierreq_outliers_lcdmmnuw0wa"+strGphotoz+"_"+name+".pdf")
#      # LCDM + Mnu + curv
#      plotPhotoZPosterior(sPhotozLCDMMnuCurv, parPhotozLCDMMnuCurv, "/outlierreq_outliers_lcdmmnucurv"+strGphotoz+"_"+name+".pdf")
#      # LCDM + w0,wa
#      plotPhotoZPosterior(sPhotozLCDMW0Wa, parPhotozLCDMW0Wa, "/outlierreq_outliers_lcdmw0wa"+strGphotoz+"_"+name+".pdf")
#      # LCDM + w0,wa + curvature
#      plotPhotoZPosterior(sPhotozLCDMW0WaCurv, parPhotozLCDMW0WaCurv, "/outlierreq_outliers_lcdmw0wacurv"+strGphotoz+"_"+name+".pdf")
#




   ##################################################################################
   ##################################################################################


#   def shearBiasRequirements(self, mask=None):
#      if mask is None:
#         mask=self.lMaxMask
#
#      # get the Fisher matrix
#      fisherData, fisherPosterior = self.generateFisher(mask=mask)
#
#      # values of shear priors to try
#      nM = 101
#      M = np.logspace(np.log10(1.e-5), np.log10(1.), nM, 10.)
#      
#      # parameters to plot
#      sigmasFull = np.zeros((self.fullPar.nPar, nM))
#      
#      for iM in range(nM):
#         m = M[iM]
#         # update the shear bias priors
#         shearMultBiasPar = ShearMultBiasParams(nBins=self.nBins, mStd=m)
#         
#         # update the full parameter object
#         newPar = self.cosmoPar.copy()
#         newPar.addParams(self.galaxyBiasPar)
#         newPar.addParams(shearMultBiasPar)
#         newPar.addParams(self.photoZPar)
#         # get the new posterior Fisher matrix
#         newPar.fisher += fisherData
#         
#         # LCDM + Mnu + curvature + w0, wa
#         # compute uncertainties with prior
#         invFisher = np.linalg.inv(newPar.fisher)
#         # get marginalized uncertainties
#         std = np.sqrt(np.diag(invFisher))
#         sigmasFull[:, iM] = std
#
#
#
#      # cosmological parameters
#      IPar = range(self.cosmoPar.nPar)
#      fig=plt.figure(0)
#      ax=fig.add_subplot(111)
#      #
#      # fiducial value
#      ax.axvline(0.005, color='gray', alpha=0.5)
#      #
#      for iPar in IPar:
#         ax.plot(M, sigmasFull[iPar, :] / sigmasFull[iPar, 0], label=self.fullPar.namesLatex[iPar])
#      #
#      ax.set_xscale('log', nonposx='clip')
#      ax.legend(loc=2)
#      ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Perfect shear bias}$')
#      ax.set_xlabel(r'Shear bias prior')
#
#      # shear bias parameters
#      fig=plt.figure(1)
#      ax=fig.add_subplot(111)
#      #
#      # fiducial prior
#      ax.axvline(0.005, color='gray')
#      #
#      IPar = self.cosmoPar.nPar+self.galaxyBiasPar.nPar
#      IPar += np.arange(self.nBins)
#      for iPar in IPar:
#         color = plt.cm.autumn((iPar-IPar[0])/(len(IPar)-1.))
#         ax.plot(M, sigmasFull[iPar, :], color=color, label=self.fullPar.namesLatex[iPar])
#      #
#      ax.set_xscale('log', nonposx='clip')
#      ax.set_yscale('log', nonposy='clip')
#      ax.legend(loc=2, labelspacing=0.05, handlelength=0.4, borderaxespad=0.01)
#      ax.set_ylabel(r'$\sigma_\text{Param}$')
#      ax.set_xlabel(r'Shear bias prior')
#
#      plt.show()
#



   ###############################################################
   ###############################################################
   ###############################################################

   def computePosterior(self, fisherData=None, ICosmoPar=None, dzStd=0.002, szStd=0.003, outliersStd=0.05):
      #print("Compute posterior uncertainties")
      if fisherData is None:
         fisherData = self.fisherDataGks
      if ICosmoPar is None:
         ICosmoPar = self.cosmoPar.IFull

      # update the photo-z priors
      newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=dzStd, szStd=szStd, outliers=0.1, outliersStd=outliersStd)
      # if g and s bins are different
      if self.photoZSPar is not None:
         newPhotoZPar.addParams(PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=dzStd, szStd=szStd, outliers=0.1, outliersStd=outliersStd))

      # update the full parameter object
      newPar = self.cosmoPar.copy()
      newPar.addParams(self.galaxyBiasPar)
      newPar.addParams(self.shearMultBiasPar)
      newPar.addParams(newPhotoZPar)

      #print "check shapes"
      #print newPar.nPar
      #print fisherData.shape, self.fisherDataGs.shape 

      # get the new posterior Fisher matrix, including the prior
      newPar.fisher += fisherData

      # Fix the unwanted cosmological parameters 
      I = ICosmoPar + range(self.cosmoPar.nPar, self.fullPar.nPar)
      parFull = newPar.extractParams(I, marg=False)
      # get the marginalized uncertainties
      #tStart = time()
      sFull = parFull.paramUncertainties(marg=True)
      #tStop = time()
      #print("Marginalizing unwanted parameters took "+str(tStop-tStart)+" sec")

      return parFull, sFull





   def computePosteriorImposeSamegs(self, fisherData=None, ICosmoPar=None, dzStd=0.002, szStd=0.003, outliersStd=0.05):
      '''If the Fisher object has different g and s bins,
      this will return the uncertainties one would get if the g and s bins were the same.
      '''
      #print("Compute posterior uncertainties")
      if fisherData is None:
         fisherData = self.fisherDataGks
      if ICosmoPar is None:
         ICosmoPar = self.cosmoPar.IFull

      # update the photo-z priors
      newPhotoZPar = PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=dzStd, szStd=szStd, outliers=0.1, outliersStd=outliersStd)
      # if g and s bins are different
      if self.photoZSPar is not None:
         newPhotoZPar.addParams(PhotoZParams(nBins=self.nBins, dzFid=0., szFid=0.05, dzStd=dzStd, szStd=szStd, outliers=0.1, outliersStd=outliersStd))

      # update the full parameter object
      # to get full prior
      newParPrior = self.cosmoPar.copy()
      newParPrior.addParams(self.galaxyBiasPar)
      newParPrior.addParams(self.shearMultBiasPar)
      newParPrior.addParams(newPhotoZPar)

      # Copy and get Fisher from the data
      newParData = newParPrior.copy()
      newParData.fisher = fisherData.copy()

      # force the g photo-z params to be equal to the s photo-z params
      iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar + self.photoZPar.nPar
      iMax = iMin + self.photoZSPar.nPar
      I = range(iMin, iMax)   # the parameters to keep: shear photo-z
      iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar
      iMax = iMin + self.photoZPar.nPar
      J = range(iMin, iMax)   # the parameters to discard: galaxy photo-z
      # do it on the data Fisher
      newParData = newParData.imposeEqualParams(I, J)

      # Do not double count the prior
#      # This would double count the prior on photo-z
#      newParPrior = newParPrior.imposeEqualParams(I, J)
      I = list( set(range(newParPrior.nPar)) - set(J) )
      newParPrior = newParPrior.extractParams(I)

      # get the new posterior Fisher matrix
      # add the prior to the data Fisher
      newParData.fisher += newParPrior.fisher

      # Fix the unwanted cosmological parameters 
      I = ICosmoPar + range(self.cosmoPar.nPar, newParData.nPar)
      parFull = newParData.extractParams(I, marg=False)
      # get the marginalized uncertainties
      #tStart = time()
      sFull = parFull.paramUncertainties(marg=True)
      #tStop = time()
      #print("Marginalizing unwanted parameters took "+str(tStop-tStart)+" sec")

      return parFull, sFull








   def varyGPhotozPrior(self, fisherData, ICosmoPar, outliersStd=0.05):
      #print("Starting sequence: varying Gaussian photo-z prior")
      # values of photo-z priors to try
      nPhotoz = 21
      Photoz = np.logspace(np.log10(1.e-5), np.log10(0.2), nPhotoz, 10.)

      # results to output
      sCosmo = np.zeros((len(ICosmoPar), nPhotoz))
      nGPhotoz = 2 * self.nBins
      nCij = self.nBins * (self.nBins-1)
      sGPhotoz = np.zeros((nGPhotoz, nPhotoz))
      sOutlierPhotoz = np.zeros((nCij, nPhotoz))

      for iPhotoz in range(nPhotoz):
         #print(str(iPhotoz))
         #tStart = time()
         photoz = Photoz[iPhotoz]

         # compute the forecast with this prior
         parFull, sFull = self.computePosterior(fisherData, ICosmoPar, dzStd=photoz, szStd=1.5*photoz, outliersStd=0.05)
         # extract cosmology
         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
         # extract Gaussian photoz
         sGPhotoz[:,iPhotoz] = sFull[-nGPhotoz-nCij:-nCij]
         # extract outlier photoz
         sOutlierPhotoz[:,iPhotoz] = sFull[-nCij:]

         #tStop = time()
         #print("Took "+str(tStop-tStart)+" sec")

      return Photoz, sCosmo, sGPhotoz, sOutlierPhotoz

   def varyGPhotozPriorImposeSamegs(self, fisherData, ICosmoPar, outliersStd=0.05):
      #print("Starting sequence: varying Gaussian photo-z prior")
      # values of photo-z priors to try
      nPhotoz = 21
      Photoz = np.logspace(np.log10(1.e-5), np.log10(0.2), nPhotoz, 10.)

      # results to output
      sCosmo = np.zeros((len(ICosmoPar), nPhotoz))
      nGPhotoz = 2 * self.nBins
      nCij = self.nBins * (self.nBins-1)
      sGPhotoz = np.zeros((nGPhotoz, nPhotoz))
      sOutlierPhotoz = np.zeros((nCij, nPhotoz))

      for iPhotoz in range(nPhotoz):
         #print(str(iPhotoz))
         #tStart = time()
         photoz = Photoz[iPhotoz]

         # compute the forecast with this prior
         parFull, sFull = self.computePosteriorImposeSamegs(fisherData, ICosmoPar, dzStd=photoz, szStd=1.5*photoz, outliersStd=0.05)
         # extract cosmology
         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
         # extract Gaussian photoz
         sGPhotoz[:,iPhotoz] = sFull[-nGPhotoz-nCij:-nCij]
         # extract outlier photoz
         sOutlierPhotoz[:,iPhotoz] = sFull[-nCij:]

         #tStop = time()
         #print("Took "+str(tStop-tStart)+" sec")

      return Photoz, sCosmo, sGPhotoz, sOutlierPhotoz



   def varyOutlierPhotozPrior(self, fisherData, ICosmoPar, dzStd=0.002, szStd=0.003):
      '''Works both if the g and s bins are the same or different.
      If they are different, the photo-z posterior uncertainties returned
      are those of the shear bins.
      '''
      #print("Starting sequence: varying outlier photo-z prior")
      # values of photo-z priors to try
      nPhotoz = 21
      Photoz = np.logspace(np.log10(1.e-4), np.log10(5.), nPhotoz, 10.)

      # results to output
      sCosmo = np.zeros((len(ICosmoPar), nPhotoz))
      nGPhotoz = 2 * self.nBins
      nCij = self.nBins * (self.nBins-1)
      sGPhotoz = np.zeros((nGPhotoz, nPhotoz))
      sOutlierPhotoz = np.zeros((nCij, nPhotoz))

      for iPhotoz in range(nPhotoz):
         #print(str(iPhotoz))
         #tStart = time()
         photoz = Photoz[iPhotoz]

         # compute the forecast with this prior
         parFull, sFull = self.computePosterior(fisherData, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=photoz)

         # extract cosmology
         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
         # extract Gaussian photoz
         sGPhotoz[:,iPhotoz] = sFull[-nGPhotoz-nCij:-nCij]
         # extract outlier photoz
         sOutlierPhotoz[:,iPhotoz] = sFull[-nCij:]

#         # case with different g and s bins
#         # extract cosmology
#         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
#         # extract Gaussian photoz for the source bins
#         iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar + self.photoZPar.nPar
#         iMax = iMin + nGPhotoZ
#         sGPhotoz[:,iPhotoz] = sFull[iMin:iMax]
#         # extract outlier photoz
#         iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar + self.photoZPar.nPar + nGPhotoZ
#         iMax = iMin + nCij
#         sOutlierPhotoz[:,iPhotoz] = sFull[iMin:iMax]

         #tStop = time()
         #print("Took "+str(tStop-tStart)+" sec")

      return Photoz, sCosmo, sGPhotoz, sOutlierPhotoz

   def varyOutlierPhotozPriorImposeSamegs(self, fisherData, ICosmoPar, dzStd=0.002, szStd=0.003):
      '''Works both if the g and s bins are the same or different.
      If they are different, the photo-z posterior uncertainties returned
      are those of the shear bins.
      '''
      #print("Starting sequence: varying outlier photo-z prior")
      # values of photo-z priors to try
      nPhotoz = 21
      Photoz = np.logspace(np.log10(1.e-4), np.log10(5.), nPhotoz, 10.)

      # results to output
      sCosmo = np.zeros((len(ICosmoPar), nPhotoz))
      nGPhotoz = 2 * self.nBins
      nCij = self.nBins * (self.nBins-1)
      sGPhotoz = np.zeros((nGPhotoz, nPhotoz))
      sOutlierPhotoz = np.zeros((nCij, nPhotoz))

      for iPhotoz in range(nPhotoz):
         #print(str(iPhotoz))
         #tStart = time()
         photoz = Photoz[iPhotoz]

         # compute the forecast with this prior
         parFull, sFull = self.computePosteriorImposeSamegs(fisherData, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=photoz)

         # extract cosmology
         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
         # extract Gaussian photoz
         sGPhotoz[:,iPhotoz] = sFull[-nGPhotoz-nCij:-nCij]
         # extract outlier photoz
         sOutlierPhotoz[:,iPhotoz] = sFull[-nCij:]

#         # case with different g and s bins
#         # extract cosmology
#         sCosmo[:,iPhotoz] = sFull[:len(ICosmoPar)]
#         # extract Gaussian photoz for the source bins
#         iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar + self.photoZPar.nPar
#         iMax = iMin + nGPhotoZ
#         sGPhotoz[:,iPhotoz] = sFull[iMin:iMax]
#         # extract outlier photoz
#         iMin = self.cosmoPar.nPar + self.galaxyBiasPar.nPar + self.shearMultBiasPar.nPar + self.photoZPar.nPar + nGPhotoZ
#         iMax = iMin + nCij
#         sOutlierPhotoz[:,iPhotoz] = sFull[iMin:iMax]

         #tStop = time()
         #print("Took "+str(tStop-tStart)+" sec")

      return Photoz, sCosmo, sGPhotoz, sOutlierPhotoz




   def plotGPhotozRequirements(self, ICosmoPar, name, fish2=None):
      # Self-calibration of Gaussian photo-z: compare data sets
      Photoz, sCosmoGks, sGPhotozGks, sOutlierPhotozGks = self.varyGPhotozPrior(self.fisherDataGks, ICosmoPar, outliersStd=0.05)
      Photoz, sCosmoGs, sGPhotozGs, sOutlierPhotozGs = self.varyGPhotozPrior(self.fisherDataGs, ICosmoPar, outliersStd=0.05)
      Photoz, sCosmoGsnonull, sGPhotozGsnonull, sOutlierPhotozGsnonull = self.varyGPhotozPrior(self.fisherDataGsnonull, ICosmoPar, outliersStd=0.05)


      ###############################################################
      ###############################################################
      # Degradation in cosmological parameters
      
      # Put the cosmo parameter set in the title
      if ICosmoPar==self.cosmoPar.ILCDM:
         title = r'$\Lambda$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMW0:
         title = r'$w_0$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMW0Wa:
         title = r'$w_0w_a$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMMnu:
         title = r'$\Lambda$CDM + $M_\nu$'
      elif ICosmoPar==self.cosmoPar.ILCDMCurv:
         title = r'$\Lambda$CDM + curvature'
      else:
         title = r''


      # get the names of the cosmo params
      parCosmo = self.cosmoPar.extractParams(ICosmoPar, marg=False)

      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='--', lw=1, c='gray', label=r'Requirement')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz, sCosmoGs[iPar, :] / sCosmoGs[iPar, 0], c=colors[iPar], label=parCosmo.namesLatex[iPar])
         #ax.plot(Photoz, sCosmoGks[iPar, :] / sCosmoGks[iPar, 0], c=colors[iPar], ls='--')
         #ax.plot(Photoz, sCosmoGsnonull[iPar, :] / sCosmoGsnonull[iPar, 0], c=colors[iPar], ls=':')
      #
      ax.set_xscale('log', nonposx='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Perfect photo-z}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      ax.set_title(title)
      #
      path = "/gphotozreq_deg_cosmo_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='--', lw=1, c='gray', label=r'Requirement')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz, sCosmoGks[iPar,:] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{Param, gks} / \sigma_\text{Param}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      ax.set_title(title)
      #
      path = "/gphotozreq_cosmo_vs_gks_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='--', lw=1, c='gray', label=r'Requirement')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz, sCosmoGsnonull[iPar,:] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{Param, no null} / \sigma_\text{Param}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      ax.set_title(title)
      #
      path = "/gphotozreq_cosmo_vs_gsnonull_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()




      ###############################################################
      ###############################################################
      # Gaussian photo-z posteriors
      
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='--', lw=1, color='gray', label=r'Fiducial')
      #ax.axvline(0.002, ls='--', lw=1, color='r', label=r'Requirement')
      #ax.axvline(0.003, ls='--', lw=1, color='b', label=r'Requirement')
      ax.axhline(0.002, ls='--', lw=1, color='b')
      ax.axhline(0.003, ls='--', lw=1, color='r')
      #ax.plot(Photoz, Photoz, c='gray', ls='--')
      #ax.plot(Photoz, 1.5*Photoz, c='r', ls='--')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      for iPar in range(self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz, sGPhotozGs[iPar, :], color=color)
         #ax.plot(Photoz, sGPhotozGks[iPar, :], color=color, ls='--')
         #ax.plot(Photoz, sGPhotozGsnonull[iPar, :], color=color, ls=':')
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      darkLight = 0.
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz, sGPhotozGs[iPar, :], color=color)
         #ax.plot(Photoz, sGPhotozGks[iPar, :], color=color, ls='--')
         #ax.plot(Photoz, sGPhotozGsnonull[iPar, :], color=color, ls=':')
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=True, handlelength=1.)
      ax.set_ylabel(r'Gaussian photo-z posterior')
      ax.set_xlabel(r'Gaussian photo-z prior')
      #
      path = "/gphotozreq_gphotoz_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='-', lw=1, color='gray', label=r'Requirement')
      #ax.axhline(0.002, ls='-', lw=1, color='gray')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      darkLight = 0.
      for iPar in range(self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz, sGPhotozGks[iPar, :] / sGPhotozGs[iPar, :], color=color)
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      darkLight = 0.
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz, sGPhotozGks[iPar, :] / sGPhotozGs[iPar, :], color=color)
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{G photo-z, gks} / \sigma_\text{G photo-z}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      #
      path = "/gphotozreq_gphotoz_vs_gks_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(0.002, ls='-', lw=1, color='gray', label=r'Requirement')
      #ax.axhline(0.002, ls='-', lw=1, color='gray')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      darkLight = 0.
      for iPar in range(self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz, sGPhotozGsnonull[iPar, :] / sGPhotozGs[iPar, :], color=color)
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      darkLight = 0.
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = - 0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz, sGPhotozGsnonull[iPar, :] / sGPhotozGs[iPar, :], color=color)
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{G photo-z, no null} / \sigma_\text{G photo-z}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      #
      path = "/gphotozreq_gphotoz_vs_gsnonull_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()



      ###############################################################
      ###############################################################
      # Degradation in outlier photo-z
      
      nCij = self.nBins * (self.nBins - 1)

      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
      #ax.axhline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray')
      #ax.plot(Photoz/(self.nBins-1), Photoz/(self.nBins-1), c='gray', ls=':', lw=1, label=r'Posterior=Prior')
      #
      for iPar in range(nCij):
         darkLight = -0.5 * iPar/nCij
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGs[iPar, :], color=color, alpha=0.3)
         #ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGks[iPar, :], color=color, alpha=0.3, ls='--')
         #ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGsnonull[iPar, :], color=color, alpha=0.3, ls=':')
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2)
      ax.set_ylabel(r'Posterior on outlier fraction $c_{ij}$')
      ax.set_xlabel(r'Gaussian photo-z prior')
      #
      path = "/gphotozreq_outlierphotoz_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()
      


      ###############################################################
      ###############################################################
      ###############################################################
      # Compare the two fisher Forecast objects


      if fish2 is not None:
         # Do the same calculation for the second Fisher forecast
         Photoz, sCosmoGks2, sGPhotozGks2, sOutlierPhotozGks2 = fish2.varyGPhotozPrior(fish2.fisherDataGks, ICosmoPar, outliersStd=0.05)
         Photoz, sCosmoGs2, sGPhotozGs2, sOutlierPhotozGs2 = fish2.varyGPhotozPrior(fish2.fisherDataGs, ICosmoPar, outliersStd=0.05)
         Photoz, sCosmoGsnonull2, sGPhotozGsnonull2, sOutlierPhotozGsnonull2 = fish2.varyGPhotozPrior(fish2.fisherDataGsnonull, ICosmoPar, outliersStd=0.05)


         # Check that I recover the case of same g s bins 
         # from the different g s bins calculation
         Photoz, sCosmoGks3, sGPhotozGks3, sOutlierPhotozGks3 = fish2.varyGPhotozPriorImposeSamegs(fish2.fisherDataGks, ICosmoPar, outliersStd=0.05)
         Photoz, sCosmoGs3, sGPhotozGs3, sOutlierPhotozGs3 = fish2.varyGPhotozPriorImposeSamegs(fish2.fisherDataGs, ICosmoPar, outliersStd=0.05)
         Photoz, sCosmoGsnonull3, sGPhotozGsnonull3, sOutlierPhotozGsnonull3 = fish2.varyGPhotozPriorImposeSamegs(fish2.fisherDataGsnonull, ICosmoPar, outliersStd=0.05)



         
         ###############################################################
         ###############################################################
         # Compare cosmological parameters

         # get the names of the cosmo params
         parCosmo = self.cosmoPar.extractParams(ICosmoPar, marg=False)

         
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         # fiducial prior
         ax.axvline(0.002, ls='--', lw=1, c='gray', label=r'Requirement')
         prop_cycle = plt.rcParams['axes.prop_cycle']
         colors = prop_cycle.by_key()['color']
         for iPar in range(parCosmo.nPar):
            ax.plot(Photoz, sCosmoGs2[iPar, :] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
            ax.plot(Photoz, sCosmoGs3[iPar, :] / sCosmoGs[iPar,:], c=colors[iPar], ls='--')
         #
         ax.set_xscale('log', nonposx='clip')
         #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
         ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
         ax.set_ylabel(r'$\sigma_\text{Param, diff bins} / \sigma_\text{Param, same bins}$')
         ax.set_xlabel(r'Gaussian photo-z prior')
         ax.set_title(title)
         #
         path = "/gphotozreq_cosmo_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         #plt.show()
         fig.clf()


         ###############################################################
         ###############################################################
         # Gaussian photo-z posteriors

         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         # fiducial prior
         ax.axvline(0.002, ls='-', lw=1, color='gray', label=r'Requirement')
         #
         # photo-z shifts
         # add legend entry
         ax.plot([], [], color='r', label=r'$\delta z$')
         for iPar in range(self.nBins):
            darkLight = - 0.5 * iPar/self.nBins
            color = darkerLighter('r', amount=darkLight)
            ax.plot(Photoz, sGPhotozGs2[iPar, :]/sGPhotozGs[iPar, :], color=color)
         #
         # photo-z scatter
         # add legend entry
         ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
         darkLight = 0.
         for iPar in range(self.nBins, 2*self.nBins):
            darkLight = - 0.5 * iPar/self.nBins
            color = darkerLighter('b', amount=darkLight)
            ax.plot(Photoz, sGPhotozGs2[iPar, :]/sGPhotozGs[iPar, :], color=color)
         #
         ax.set_xscale('log', nonposx='clip')
         #ax.set_yscale('log', nonposy='clip')
         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
         ax.set_ylabel(r'$\sigma_\text{Param, diff bins} / \sigma_\text{Param, same bins}$')
         ax.set_xlabel(r'Gaussian photo-z prior')
         #
         path = "/gphotozreq_gphotoz_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         #plt.show()
         fig.clf()


         ###############################################################
         ###############################################################
         # Degradation in outlier photo-z
         
         nCij = self.nBins * (self.nBins - 1)

         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
         #
         for iPar in range(nCij):
            darkLight = -0.5 * iPar/nCij
            color = darkerLighter('r', amount=darkLight)
            ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGs2[iPar, :]/sOutlierPhotozGs[iPar, :], color=color, alpha=0.3)
         #
         ax.set_xscale('log', nonposx='clip')
         #ax.set_yscale('log', nonposy='clip')
         ax.legend(loc=2)
         ax.set_ylabel(r'$\sigma_{c_{ij} \text{ diff bins}} / \sigma_{c_{ij} \text{ same bins}}$')
         ax.set_xlabel(r'Gaussian photo-z prior')
         #
         path = "/gphotozreq_outlierphotoz_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         fig.clf()












   def plotOutlierPhotozRequirements(self, ICosmoPar, name, fish2=None):
      # Self-calibration of Gaussian photo-z: compare data sets
      Photoz, sCosmoGks, sGPhotozGks, sOutlierPhotozGks = self.varyOutlierPhotozPrior(self.fisherDataGks, ICosmoPar, dzStd=0.002, szStd=0.003)
      Photoz, sCosmoGs, sGPhotozGs, sOutlierPhotozGs = self.varyOutlierPhotozPrior(self.fisherDataGs, ICosmoPar, dzStd=0.002, szStd=0.003)
      Photoz, sCosmoGsnonull, sGPhotozGsnonull, sOutlierPhotozGsnonull = self.varyOutlierPhotozPrior(self.fisherDataGsnonull, ICosmoPar, dzStd=0.002, szStd=0.003)


      ###############################################################
      ###############################################################
      # Degradation in cosmological parameters
      
      # get the names of the cosmo params
      parCosmo = self.cosmoPar.extractParams(ICosmoPar, marg=False)

      # Put the cosmo parameter set in the title
      if ICosmoPar==self.cosmoPar.ILCDM:
         title = r'$\Lambda$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMW0:
         title = r'$w_0$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMW0Wa:
         title = r'$w_0w_a$CDM'
      elif ICosmoPar==self.cosmoPar.ILCDMMnu:
         title = r'$\Lambda$CDM + $M_\nu$'
      elif ICosmoPar==self.cosmoPar.ILCDMCurv:
         title = r'$\Lambda$CDM + curvature'
      else:
         title = r''

      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz/(self.nBins-1), sCosmoGs[iPar, :] / sCosmoGs[iPar, 0], c=colors[iPar], label=parCosmo.namesLatex[iPar])
      #
      ax.set_xscale('log', nonposx='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      #ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Perfect photo-z}$')
      ax.set_ylabel(r'Degradation in posterior uncertainty')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      ax.set_title(title)
      #
      path = "/outlierphotozreq_deg_cosmo_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz/(self.nBins-1), sCosmoGks[iPar,:] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{Param, gks} / \sigma_\text{Param}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      ax.set_title(title)
      #
      path = "/outlierphotozreq_cosmo_vs_gks_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(Photoz/(self.nBins-1), sCosmoGsnonull[iPar,:] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{Param, no null} / \sigma_\text{Param}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      ax.set_title(title)
      #
      path = "/outlierphotozreq_cosmo_vs_gsnonull_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()




      ###############################################################
      ###############################################################
      # Degradation in Gaussian photo-z
      
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      for iPar in range(self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGs[iPar,:]/sGPhotozGs[iPar,0], color=color)
         #ax.plot(Photoz, sGPhotozGks[iPar, :], color=color, ls='--')
         #ax.plot(Photoz, sGPhotozGsnonull[iPar, :], color=color, ls=':')
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGs[iPar,:]/sGPhotozGs[iPar,0], color=color)
         #ax.plot(Photoz, sGPhotozGks[iPar, :], color=color, ls='--')
         #ax.plot(Photoz, sGPhotozGsnonull[iPar, :], color=color, ls=':')
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_ylabel(r'Degradation in Gaussian photo-z posterior')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_deg_gphotoz_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      for iPar in range(self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGks[iPar,:]/sGPhotozGs[iPar,:], color=color)
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGks[iPar,:]/sGPhotozGs[iPar,:], color=color)
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{G photo-z, gks} / \sigma_\text{G photo-z}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_gphotoz_vs_gks_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      # fiducial prior
      ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
      #
      # photo-z shifts
      # add legend entry
      ax.plot([], [], color='r', label=r'$\delta z$')
      for iPar in range(self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('r', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGsnonull[iPar,:]/sGPhotozGs[iPar,:], color=color)
      #
      # photo-z scatter
      # add legend entry
      ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
      for iPar in range(self.nBins, 2*self.nBins):
         darkLight = -0.5 * iPar/self.nBins
         color = darkerLighter('b', amount=darkLight)
         ax.plot(Photoz/(self.nBins-1), sGPhotozGsnonull[iPar,:]/sGPhotozGs[iPar, :], color=color)
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_ylabel(r'$\sigma_\text{G photo-z, no null} / \sigma_\text{G photo-z}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_gphotoz_vs_gsnonull_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()



      ###############################################################
      ###############################################################
      # Outlier photo-z posteriors
      
      nCij = self.nBins * (self.nBins - 1)

      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
      ax.axhline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray')
      #ax.plot(Photoz/(self.nBins-1), Photoz/(self.nBins-1), c='gray', ls=':', lw=1, label=r'Posterior=Prior')
      #
      color = 'r'
      darkLight = 0.
      for iPar in range(nCij):
         color = 'r'
         color = darkerLighter(color, amount=darkLight)
         darkLight += -0.5 * 1./nCij
         ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGs[iPar, :], color=color, alpha=0.3)
         #ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGks[iPar, :], color=color, alpha=0.3, ls='--')
         #ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGsnonull[iPar, :], color=color, alpha=0.3, ls=':')
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2)
      ax.set_ylabel(r'Posterior on outlier fraction $c_{ij}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_outlierphotoz_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()



      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
      #
      color = 'r'
      darkLight = 0.
      for iPar in range(nCij):
         color = 'r'
         color = darkerLighter(color, amount=darkLight)
         darkLight += -0.5 * 1./nCij
         ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGks[iPar,:]/ sOutlierPhotozGs[iPar,:], color=color, alpha=0.3)
      #
      ax.set_xscale('log', nonposx='clip')
      #ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2)
      ax.set_ylabel(r'Posterior on outlier fraction $c_{ij}$')
      ax.set_ylabel(r'$\sigma_{c_{ij}, \text{gks}} / \sigma_{c_{ij}}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_outlierphotoz_vs_gks_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()


      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
      #
      color = 'r'
      darkLight = 0.
      for iPar in range(nCij):
         color = 'r'
         color = darkerLighter(color, amount=darkLight)
         darkLight += -0.5 * 1./nCij
         ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGsnonull[iPar,:]/ sOutlierPhotozGs[iPar,:], color=color, alpha=0.3)
      #
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=2)
      ax.set_ylabel(r'Posterior on outlier fraction $c_{ij}$')
      ax.set_ylabel(r'$\sigma_{c_{ij}, \text{no null}} / \sigma_{c_{ij}}$')
      ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
      #
      path = "/outlierphotozreq_outlierphotoz_vs_gsnonull_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      fig.clf()
      

      ###############################################################
      ###############################################################
      ###############################################################
      # Compare the two fisher Forecast objects


      if fish2 is not None:
         # Do the same calculation for the second Fisher forecast
         Photoz, sCosmoGks2, sGPhotozGks2, sOutlierPhotozGks2 = fish2.varyOutlierPhotozPrior(fish2.fisherDataGks, ICosmoPar, dzStd=0.002, szStd=0.003)
         Photoz, sCosmoGs2, sGPhotozGs2, sOutlierPhotozGs2 = fish2.varyOutlierPhotozPrior(fish2.fisherDataGs, ICosmoPar, dzStd=0.002, szStd=0.003)
         Photoz, sCosmoGsnonull2, sGPhotozGsnonull2, sOutlierPhotozGsnonull2 = fish2.varyOutlierPhotozPrior(fish2.fisherDataGsnonull, ICosmoPar, dzStd=0.002, szStd=0.003)

         # Check that I recover the case of same g s bins
         # from the different g s bins calculation
         Photoz, sCosmoGks3, sGPhotozGks3, sOutlierPhotozGks3 = fish2.varyOutlierPhotozPriorImposeSamegs(fish2.fisherDataGks, ICosmoPar, dzStd=0.002, szStd=0.003)
         Photoz, sCosmoGs3, sGPhotozGs3, sOutlierPhotozGs3 = fish2.varyOutlierPhotozPriorImposeSamegs(fish2.fisherDataGs, ICosmoPar, dzStd=0.002, szStd=0.003)
         Photoz, sCosmoGsnonull3, sGPhotozGsnonull3, sOutlierPhotozGsnonull3 = fish2.varyOutlierPhotozPriorImposeSamegs(fish2.fisherDataGsnonull, ICosmoPar, dzStd=0.002, szStd=0.003)



         
         ###############################################################
         ###############################################################
         # Compare cosmological parameters

         # get the names of the cosmo params
         parCosmo = self.cosmoPar.extractParams(ICosmoPar, marg=False)

         
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         # fiducial prior
         ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
         prop_cycle = plt.rcParams['axes.prop_cycle']
         colors = prop_cycle.by_key()['color']
         for iPar in range(parCosmo.nPar):
            ax.plot(Photoz, sCosmoGs2[iPar, :] / sCosmoGs[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
            ax.plot(Photoz, sCosmoGs3[iPar, :] / sCosmoGs[iPar,:], c=colors[iPar], ls='--')
         #
         ax.set_xscale('log', nonposx='clip')
         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
         ax.set_ylabel(r'$\sigma_\text{Param, diff bins} / \sigma_\text{Param, same bins}$')
         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
         ax.set_title(title)
         #
         path = "/outlierphotozreq_cosmo_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         #plt.show()
         fig.clf()


         ###############################################################
         ###############################################################
         # Gaussian photo-z posteriors

         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         # fiducial prior
         ax.axvline(self.photoZPar.priorStd[-1], ls='--', lw=1, color='gray', label=r'Fiducial')
         #
         # photo-z shifts
         # add legend entry
         ax.plot([], [], color='r', label=r'$\delta z$')
         for iPar in range(self.nBins):
            darkLight = - 0.5 * iPar/self.nBins
            color = darkerLighter('r', amount=darkLight)
            ax.plot(Photoz, sGPhotozGs2[iPar, :]/sGPhotozGs[iPar, :], color=color)
         #
         # photo-z scatter
         # add legend entry
         ax.plot([], [], color='b', label=r'$\sigma_z / (1+z)$')
         darkLight = 0.
         for iPar in range(self.nBins, 2*self.nBins):
            darkLight = - 0.5 * iPar/self.nBins
            color = darkerLighter('b', amount=darkLight)
            ax.plot(Photoz, sGPhotozGs2[iPar, :]/sGPhotozGs[iPar, :], color=color)
         #
         ax.set_xscale('log', nonposx='clip')
         #ax.set_yscale('log', nonposy='clip')
         ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
         ax.set_ylabel(r'$\sigma_\text{Param, diff bins} / \sigma_\text{Param, same bins}$')
         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
         #
         path = "/outlierphotozreq_gphotoz_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         #plt.show()
         fig.clf()


         ###############################################################
         ###############################################################
         # Outlier photo-z posteriors
         
         nCij = self.nBins * (self.nBins - 1)

         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         ax.axvline(self.photoZPar.priorStd[-1], ls='-', lw=1, color='gray', label=r'Fiducial')
         #
         color = 'r'
         darkLight = 0.
         for iPar in range(nCij):
            color = 'r'
            color = darkerLighter(color, amount=darkLight)
            darkLight += -0.5 * 1./nCij
            ax.plot(Photoz/(self.nBins-1), sOutlierPhotozGs2[iPar, :]/sOutlierPhotozGs[iPar, :], color=color, alpha=0.3)
         #
         ax.set_xscale('log', nonposx='clip')
         ax.set_yscale('log', nonposy='clip')
         ax.legend(loc=2)
         ax.set_ylabel(r'$\sigma_{c_{ij} \text{ diff bins}} / \sigma_{c_{ij} \text{ same bins}}$')
         ax.set_xlabel(r'Prior on outlier fraction $c_{ij}$')
         #
         path = "/outlierphotozreq_outlierphotoz_"+name+"_vs_2.pdf"
         fig.savefig(self.figurePath+path, bbox_inches='tight')
         fig.clf()





   def plotSummaryComparison(self, ICosmoPar=None, name=""):
      if ICosmoPar is None:
         ICosmoPar = self.cosmoPar.IFull

      # compute the various posterior constraints we want to compare
      parPlanck = self.cosmoPar.extractParams(ICosmoPar, marg=False)
      sPlanck = parPlanck.paramUncertainties()
      #
      parGs, sGs = self.computePosterior(self.fisherDataGs, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGsNoprior, sGsNoprior = self.computePosterior(self.fisherDataGs, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)
      #
      parGks, sGks = self.computePosterior(self.fisherDataGks, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGksNoprior, sGksNoprior = self.computePosterior(self.fisherDataGks, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)
      #
      parGsonnull, sGsnonull = self.computePosterior(self.fisherDataGsnonull, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGsnonullNoprior, sGsnonullNoprior = self.computePosterior(self.fisherDataGsnonull, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)

      # extract the cosmological parameters only
      ICosmo = range(len(ICosmoPar))
      sGsCosmo = sGs[ICosmo]
      sGsNopriorCosmo = sGsNoprior[ICosmo]
      sGksCosmo = sGks[ICosmo]
      sGksNopriorCosmo = sGksNoprior[ICosmo]
      sGsnonullCosmo = sGsnonull[ICosmo]
      sGsnonullNopriorCosmo = sGsnonullNoprior[ICosmo]

      # plot cosmo constraints VS fiducial case
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      #ax.plot(sPlanck/sGsCosmo, c='gray', ls='--', marker='o', alpha=0.2, label=r'Planck prior')
      #
      ax.plot(sGsnonullCosmo/sGsCosmo, c='r', ls='-', marker='o', label=r'gs no null')
      ax.plot(sGsnonullNopriorCosmo/sGsCosmo, c='r', ls='--', marker='o', alpha=0.5, label=r'gs no null no prior')
      #
      ax.plot(np.ones_like(ICosmo), c='k', ls='-', marker='o', label=r'gs')
      ax.plot(sGsNopriorCosmo/sGsCosmo, c='k', ls='--', marker='o', alpha=0.5, label=r'gs no prior')
      #
      ax.plot(sGksCosmo/sGsCosmo, c='b', ls='-', marker='o', label=r'gks')
      ax.plot(sGksNopriorCosmo/sGsCosmo, c='b', ls='--', marker='o', alpha=0.5, label=r'gks no prior')
      #
      #ax.legend(loc=2, labelspacing=0.1, frameon=False)#, handlelength=1.)
      ax.legend(loc=2, labelspacing=0.1, frameon=True, framealpha=0.6)#, handlelength=1.)
      ax.set_xticks(ICosmo)
      ax.set_xticklabels(parGs.namesLatex[ICosmo], fontsize=24)
      [l.set_rotation(45) for l in ax.get_xticklabels()]
      ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Param fiducial}$')
      #
      fig.savefig(self.figurePath+"/comparison_cosmo_"+name+"_vs_gs.pdf", bbox_inches='tight')
      fig.clf()

      # plot cosmo constraints VS Planck only
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot(np.ones_like(ICosmo), c='gray', ls='--', marker='o', label=r'Planck prior')
      #
      ax.plot(sGsnonullCosmo/sPlanck, c='r', ls='-', marker='o', label=r'gs no null')
      ax.plot(sGsnonullNopriorCosmo/sPlanck, c='r', ls='--', marker='o', label=r'gs no null no prior')
      #
      ax.plot(sGsCosmo/sPlanck, c='k', ls='-', marker='o', label=r'gs')
      ax.plot(sGsNopriorCosmo/sPlanck, c='k', ls='--', marker='o', label=r'gs no prior')
      #
      ax.plot(sGksCosmo/sPlanck, c='b', ls='-', marker='o', label=r'gks')
      ax.plot(sGksNopriorCosmo/sPlanck, c='b', ls='--', marker='o', label=r'gks no prior')
      #
      ax.legend(loc=2, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.set_xticks(ICosmo)
      ax.set_xticklabels(parGs.namesLatex[ICosmo], fontsize=24)
      [l.set_rotation(45) for l in ax.get_xticklabels()]
      ax.set_ylabel(r'$\sigma_\text{Param} / \sigma_\text{Param fiducial}$')
      #
      fig.savefig(self.figurePath+"/comparison_cosmo_"+name+"_vs_planck.pdf", bbox_inches='tight')
      fig.clf()







   def plotFomComparison(self, ICosmoPar=None, name=""):
      '''Assumes that the last two cosmological parameters are w0,wa
      FOM = 1/sqrt(det(Cov(w0,wa))).
      '''
      if ICosmoPar is None:
         ICosmoPar = self.cosmoPar.IFull

      # extract the cosmological parameters only
      ICosmo = range(len(ICosmoPar))

      # Planck prior
      parPlanck = self.cosmoPar.extractParams(ICosmoPar, marg=False)
      parPlanck = parPlanck.extractParams(ICosmo[-2:], marg=True)
      fomPlanck = np.sqrt(np.linalg.det(parPlanck.fisher))

      # compute the various posterior constraints we want to compare

      # gs
      parGs, sGs = self.computePosterior(self.fisherDataGs, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGs = parGs.extractParams(ICosmo[-2:], marg=True)
      fomGs = np.sqrt(np.linalg.det(parGs.fisher))
      #
      parGsNoprior, sGsNoprior = self.computePosterior(self.fisherDataGs, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)
      parGsNoprior = parGsNoprior.extractParams(ICosmo[-2:], marg=True)
      fomGsNoprior = np.sqrt(np.linalg.det(parGsNoprior.fisher))

      # gks
      parGks, sGks = self.computePosterior(self.fisherDataGks, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGks = parGks.extractParams(ICosmo[-2:], marg=True)
      fomGks = np.sqrt(np.linalg.det(parGks.fisher))
      #
      parGksNoprior, sGksNoprior = self.computePosterior(self.fisherDataGks, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)
      parGksNoprior = parGksNoprior.extractParams(ICosmo[-2:], marg=True)
      fomGksNoprior = np.sqrt(np.linalg.det(parGksNoprior.fisher))

      # gs no null
      parGsnonull, sGsnonull = self.computePosterior(self.fisherDataGsnonull, ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      parGsnonull = parGsnonull.extractParams(ICosmo[-2:], marg=True)
      fomGsnonull = np.sqrt(np.linalg.det(parGsnonull.fisher))
      #
      parGsnonullNoprior, sGsnonullNoprior = self.computePosterior(self.fisherDataGsnonull, ICosmoPar, dzStd=10., szStd=10., outliersStd=10.)
      parGsnonullNoprior = parGsnonullNoprior.extractParams(ICosmo[-2:], marg=True)
      fomGsnonullNoprior = np.sqrt(np.linalg.det(parGsnonullNoprior.fisher))


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot([fomPlanck], c='k', marker='s')
      ax.plot([np.nan, fomGsnonull, fomGs, fomGks], 'ko-', label=r'fiducial')
      ax.plot([np.nan, fomGsnonullNoprior, fomGsNoprior, fomGksNoprior], 'ko--', label=r'no photoz prior')
      #
      ax.legend(loc=4, fontsize='x-small', labelspacing=0.1)
      ax.set_yscale('log', nonposy='clip')
      ax.set_xticks(range(4))
      ax.set_xticklabels(['Planck prior', 'gs no null', 'gs', 'gks'], rotation=45, ha='right')
      ax.set_ylabel(r'Dark energy FOM')
      #
      fig.savefig(self.figurePath+"/compare_fom_"+name+".pdf", bbox_inches='tight')
      #plt.show()
      fig.clf()








   ###############################################################
   ###############################################################


   def computeBiasFromOutliers(self, fisherData=None, ICosmoPar=None, cijBias=0.1):
      '''Compute the bias in desired parameters when one incorrectly
      assumes no photo-z outliers.
      cijBias is the fractional bias on the cij parameters
      dTheta_i = - (F^-1)_{ij} F_{j alpha} dPsi_alpha
      where:
      dPsi_alpha are the error in the fixed parameters
      dTheta_i are the resulting biases in the inferred parameters.
      '''
      if fisherData is None:
         fisherData = self.fisherDataGks
      if ICosmoPar is None:
         ICosmoPar = self.cosmoPar.IFull

      # get the posterior Fisher matrix
      par, s = self.computePosterior(fisherData=fisherData, ICosmoPar=ICosmoPar, dzStd=0.002, szStd=0.003, outliersStd=0.05)
      fisher = par.fisher
      # extract the sub-Fisher matrix for the inferred parameters only
      nCij = self.nBins * (self.nBins-1)
      fisherPartial = fisher[:-nCij, :-nCij]
      invFisherPartial = np.linalg.inv(fisherPartial)

      # dPsi: errors in the assumed fixed params
      wrongParFiducial = np.zeros(par.nPar)
      # assume (wrongly) no outliers
      wrongParFiducial[-nCij:] = self.photoZPar.fiducial[-nCij:] * cijBias
      
      # dTheta: bias in the inferred parameters
      dTheta = - np.dot(invFisherPartial, np.dot(fisher[:-nCij,:], wrongParFiducial))
      # uncertainties on the inferred parameters,
      # when the cij are fixed, not marginalized,
      # to assess the significance of the bias
      sTheta = np.sqrt(np.diag(invFisherPartial))  #s[:-nCij]
      return dTheta, sTheta


   def varyBiasFromOutliers(self, fisherData=None, ICosmoPar=None):
      # values of Cij fractional bias to try
      nCijBias = 21
      CijBias = np.logspace(np.log10(1.e-2), np.log10(1.), nCijBias, 10.)
      # results to output
      nCij = self.nBins * (self.nBins-1)
      nPar = self.fullPar.nPar - self.cosmoPar.nPar + len(ICosmoPar) - nCij
      dTheta = np.zeros((nPar, nCijBias))
      sTheta = np.zeros((nPar, nCijBias))
      for iCijBias in range(nCijBias):
         cijBias = CijBias[iCijBias]
         # Compute the bias on inferred parameters
         dTheta[:,iCijBias], sTheta[:,iCijBias] = self.computeBiasFromOutliers(fisherData=fisherData, ICosmoPar=ICosmoPar, cijBias=cijBias)
      return CijBias, dTheta, sTheta




   def plotBiasFromOutliers(self, ICosmoPar, name):

      # compute the biases for the different data sets
      CijBias, dThetaGks, sThetaGks = self.varyBiasFromOutliers(fisherData=self.fisherDataGks, ICosmoPar=ICosmoPar)


      ###############################################################
      ###############################################################
      # Bias on cosmological parameters

      # get the names and fiducial values of the cosmo params
      parCosmo = self.cosmoPar.extractParams(ICosmoPar, marg=False)

#      print parCosmo.names[:len(ICosmoPar)]
#      print dThetaGks[:len(ICosmoPar),-1]
#      print dThetaGks[:len(ICosmoPar),-1] / parCosmo.fiducial[:len(ICosmoPar)]
#      print parCosmo.fiducial[:len(ICosmoPar)]

      # get the absolute c_ij bias from the relative c_ij bias
      CijAbsBias = self.photoZPar.fiducial[-1] * CijBias

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.axhline(0., c='k', ls='--')
      #
      prop_cycle = plt.rcParams['axes.prop_cycle']
      colors = prop_cycle.by_key()['color']
      for iPar in range(parCosmo.nPar):
         ax.plot(CijAbsBias, dThetaGks[iPar,:], c=colors[iPar], label=parCosmo.namesLatex[iPar])
         ax.fill_between(CijAbsBias, dThetaGks[iPar,:]-sThetaGks[iPar,:], dThetaGks[iPar,:]+sThetaGks[iPar,:], facecolor=colors[iPar], edgecolor='', alpha=0.4)

      #
      ax.set_xlim((CijAbsBias.min(), CijAbsBias.max()))
#      ax.set_ylim((-0.1, 0.1))
      #ax.set_yscale('symlog', linthreshy=0.05)
      #ax.axhline(-0.05, c='gray', lw=0.5)
      #ax.axhline(0.05, c='gray', lw=0.5)
      ax.set_xscale('log', nonposx='clip')
      #ax.legend(loc=3, labelspacing=0.1, frameon=False, handlelength=1.)
      ax.legend(loc=3, labelspacing=0.1, frameon=True, framealpha=0.6, handlelength=1.)
      ax.set_ylabel(r'Parameter bias')
      ax.set_xlabel(r'Additive bias to outlier $c_{ij}$')
      #
      path = "/bias_photozoutliers_cosmo_"+name+".pdf"
      fig.savefig(self.figurePath+path, bbox_inches='tight')
      #plt.show()
      fig.clf()



   ###############################################################
   ###############################################################



   def plotCosmoContours(self, ICosmoPar, fishers, fisherNames=None, colors=None, path=None):
      
      # Create the list of inverse Fisher matrices,
      # by fixing unwanted cosmo params
      # and marginalizing over all nuisance marams
      nFisher = np.shape(fishers)[0]
      invFishers = np.zeros((nFisher, len(ICosmoPar), len(ICosmoPar)))
      for iFisher in range(nFisher):

         par = self.fullPar
         par.fisher = fishers[iFisher,:,:]

         # Fix the unwanted cosmological parameters 
         I = ICosmoPar + range(self.cosmoPar.nPar, self.fullPar.nPar)
         par = par.extractParams(I, marg=False)
         
         # Marginalize over the nuisance parameters
         par = par.extractParams(range(len(ICosmoPar)), marg=True)
         
         invFishers[iFisher,:,:] = np.linalg.inv(par.fisher)

      # generate the contour plot
      par = self.fullPar
      par = par.extractParams(ICosmoPar, marg=False)  # only used for fiducial values and names

      par.plotContours(par=par, invFishers=invFishers, lim=4., colors=colors, fisherNames=fisherNames, path=path)
         


   def plotSummaryCosmoContours(self):
      # Compare Planck prior, gs, gks
      fishers=np.array([self.fullPar.fisher, self.fullPar.fisher+self.fisherDataGs, self.fullPar.fisher+self.fisherDataGks])

      # LCDMW0Wa
      self.plotCosmoContours(self.cosmoPar.ILCDMW0Wa, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=self.figurePath+"/contours_lcdmw0wa.pdf")
      # LCDMMnu
      self.plotCosmoContours(self.cosmoPar.ILCDMMnu, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=self.figurePath+"/contours_lcdmmnu.pdf")
      # LCDMCurv
      self.plotCosmoContours(self.cosmoPar.ILCDMCurv, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=self.figurePath+"/contours_lcdmcurv.pdf")


