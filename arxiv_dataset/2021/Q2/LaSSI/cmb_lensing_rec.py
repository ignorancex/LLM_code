from headers import *
###############################################################################
# get noise power spectrum for lensing displacement d
# from CMB lens reconstruction. Follows Hu & Okamoto 2002.
# These are the N_l^dd. The noise on convergence kappa is:
# N_l^kappakappa = l^2/4. * N_l^dd

class CMBLensRec(object):

   def __init__(self, CMB, save=False, nProc=1):
      self.CMB = CMB
      self.nProc = nProc

      # bounds for ell integrals
      self.lMin = self.CMB.lMin
      self.lMax = max(self.CMB.lMaxT, self.CMB.lMaxP)
      
      # values of ell to compute the reconstruction noise
      self.L = np.genfromtxt("./input/Lc.txt") # center of the bins for l
      self.Nl = len(self.L)
      
      # output file path
      self.directory = "./output/cmblensrec/"+str(self.CMB)
      self.path = self.directory+"/cmblensrecnoise.txt"
      # create folder if needed
      if not os.path.exists(self.directory):
         os.makedirs(self.directory)

      if save or (not os.path.exists(self.path)):
         self.SaveAll()
      self.LoadAll()
      
   
   def SaveAll(self):
      tStart0 = time()
      
      data = np.zeros((self.Nl, 17))
      data[:,0] = np.copy(self.L)
      
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      
      # diagonal covariances
      print "Computing A_TT"
      tStart = time()
      data[:,1] = np.array(pool.map(self.A_TT, self.L))
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      
      print "Computing A_TE"
      tStart = time()
      data[:,2] = np.array(pool.map(self.A_TE, self.L))
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing A_TB"
      tStart = time()
      data[:,3] = np.array(pool.map(self.A_TB, self.L))
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing A_EE"
      tStart = time()
      data[:,4] = np.array(pool.map(self.A_EE, self.L))
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing A_EB"
      tStart = time()
      data[:,5] = np.array(pool.map(self.A_EB, self.L))
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      
      # load diagonal covariances
      self.N_TT = data[:,1]
      self.N_TE = data[:,2]
      self.N_TB = data[:,3]
      self.N_EE = data[:,4]
      self.N_EB = data[:,5]
      
      # non-diagonal covariances
      print "Computing N_TT_TE"
      tStart = time()
      data[:,6] = np.array(pool.map(self.N_TT_TE, self.L))
      data[:,6] *= self.N_TT * self.N_TE / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TT_TB"
      tStart = time()
      data[:,7] = np.array(pool.map(self.N_TT_TB, self.L))
      data[:,7] *= self.N_TT * self.N_TB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TT_EE"
      tStart = time()
      data[:,8] = np.array(pool.map(self.N_TT_EE, self.L))
      data[:,8] *= self.N_TT * self.N_EE / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TT_EB"
      tStart = time()
      data[:,9] = np.array(pool.map(self.N_TT_EB, self.L))
      data[:,9] *= self.N_TT * self.N_EB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TE_TB"
      tStart = time()
      data[:,10] = np.array(pool.map(self.N_TE_TB, self.L))
      data[:,10] *= self.N_TE * self.N_TB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TE_EE"
      tStart = time()
      data[:,11] = np.array(pool.map(self.N_TE_EE, self.L))
      data[:,11] *= self.N_TE * self.N_EE / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TE_EB"
      tStart = time()
      data[:,12] = np.array(pool.map(self.N_TE_EB, self.L))
      data[:,12] *= self.N_TE * self.N_EB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TB_EE"
      tStart = time()
      data[:,13] = np.array(pool.map(self.N_TB_EE, self.L))
      data[:,13] *= self.N_TB * self.N_EE / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_TB_EB"
      tStart = time()
      data[:,14] = np.array(pool.map(self.N_TB_EB, self.L))
      data[:,14] *= self.N_TB * self.N_EB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      print "Computing N_EE_EB"
      tStart = time()
      data[:,15] = np.array(pool.map(self.N_EE_EB, self.L))
      data[:,15] *= self.N_EE * self.N_EB / self.L**2
      np.savetxt(self.path, data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      
      # load non-diagonal covariances
      self.N_TT_TE = data[:,6]
      self.N_TT_TB = data[:,7]
      self.N_TT_EE = data[:,8]
      self.N_TT_EB = data[:,9]
      self.N_TE_TB = data[:,10]
      self.N_TE_EE = data[:,11]
      self.N_TE_EB = data[:,12]
      self.N_TB_EE = data[:,13]
      self.N_TB_EB = data[:,14]
      self.N_EE_EB = data[:,15]
      
      # variance of mv estimator
      print "Computing N_mv"
      tStart = time()
      self.SaveNmv(data)
      tStop = time()
      print "took", (tStop-tStart)/60., "min"
      
      print "Total time:", (tStop-tStart0)/3600., "hours"
   


   # Noise for minimum variance displacement estimator,
   # from Hu & Okamoto 2002
   def SaveNmv(self, data):
      Nmv = np.zeros(self.Nl)
      N = np.zeros((5, 5))
      for il in range(self.Nl):
         # fill in only the non-zero components
         N[0,0] = self.N_TT[il]
         N[0,1] = N[1,0] = self.N_TT_TE[il]
         N[0,3] = N[3,0] = self.N_TT_EE[il]
         #
         N[1,1] = self.N_TE[il]
         N[1,3] = N[3,1] = self.N_TE_EE[il]
         #
         N[2,2] = self.N_TB[il]
         N[2,4] = N[4,2] = self.N_TB_EB[il]
         #
         N[3,3] = self.N_EE[il]
         #
         N[4,4] = self.N_EB[il]
         #
         try:
            Inv = np.linalg.inv(N)
            Nmv[il] = 1./np.sum(Inv)
         except:
            pass
      data[:,16] = Nmv
      np.savetxt(self.path, data)
      return



   def LoadAll(self):
      data = np.genfromtxt(self.path)
      # ell values
      self.L = data[:,0]
      self.Nl = len(self.L)
      # noises for displacement d
      # diagonal covariances
      self.N_TT = data[:,1]
      self.N_TE = data[:,2]
      self.N_TB = data[:,3]
      self.N_EE = data[:,4]
      self.N_EB = data[:,5]
      # non-diagonal covariances
      self.N_TT_TE = data[:,6]
      self.N_TT_TB = data[:,7]
      self.N_TT_EE = data[:,8]
      self.N_TT_EB = data[:,9]
      self.N_TE_TB = data[:,10]
      self.N_TE_EE = data[:,11]
      self.N_TE_EB = data[:,12]
      self.N_TB_EE = data[:,13]
      self.N_TB_EB = data[:,14]
      self.N_EE_EB = data[:,15]
      # variance of mv estimator
      self.N_mv = data[:,16]
   
      # interpolate the reconstruction noises
      # TT
      self.fN_d_TT = interp1d(self.L, self.N_TT, kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT = lambda l: l**2/4. * self.fN_d_TT(l)
      self.fN_phi_TT = lambda l: self.fN_d_TT(l) / l**2
      # mv
      self.fN_d_mv = interp1d(self.L, self.N_mv, kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_mv = lambda l: l**2/4. * self.fN_d_mv(l)
      self.fN_phi_mv = lambda l: self.fN_d_mv(l) / l**2
      # EB
      self.fN_d_EB = interp1d(self.L, self.N_EB, kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_EB = lambda l: l**2/4. * self.fN_d_EB(l)
      self.fN_phi_EB = lambda l: self.fN_d_EB(l) / l**2
   
   
   ###############################################################################
   # functions f_alpha from Hu & Okamoto 2002:
   # < X Y > = f_XY * phi
   # phi is the angle between l1 and l2
   
   def f_TT(self, l1, l2, phi):
      result = self.CMB.funlensedTT(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedTT(l2) * l2*(l2 + l1*np.cos(phi))
      return result

   def f_TE(self, l1, l2, phi):
      # typo in Hu & Okamoto 2002: cos(2phi) and not cos(phi)!
      result = self.CMB.funlensedTE(l1) * l1*(l1 + l2*np.cos(phi)) * np.cos(2.*phi)
      result += self.CMB.funlensedTE(l2) * l2*(l2 + l1*np.cos(phi))
      return result

   def f_TB(self, l1, l2, phi):
      result = self.CMB.funlensedTE(l1) * l1*(l1 + l2*np.cos(phi)) * np.sin(2.*phi)
      return result

   def f_EE(self, l1, l2, phi):
      result = self.CMB.funlensedEE(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedEE(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.cos(2.*phi)
      return result

   def f_EB(self, l1, l2, phi):
      result = self.CMB.funlensedEE(l1) * l1*(l1 + l2*np.cos(phi))
      result -= self.CMB.funlensedBB(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.sin(2.*phi)
      return result

   def f_BB(self, l1, l2, phi):
      result = self.CMB.funlensedBB(l1) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.funlensedBB(l2) * l2*(l2 + l1*np.cos(phi))
      result *= np.cos(2.*phi)
      return result
   

   ###############################################################################
   ###############################################################################
   # functions F_alpha from Hu & Okamoto 2002
   # F_XY = f_XY / 2*C*C


   def F_TT(self, l1, l2, phi):
      result = self.f_TT(l1, l2, phi)
      result /= self.CMB.ftotalTT(l1)
      result /= self.CMB.ftotalTT(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result

   def F_EE(self, l1, l2, phi):
      result = self.f_EE(l1, l2, phi)
      result /= self.CMB.ftotalEE(l1)
      result /= self.CMB.ftotalEE(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result

   def F_BB(self, l1, l2, phi):
      result = self.f_BB(l1, l2, phi)
      result /= self.CMB.ftotalBB(l1)
      result /= self.CMB.ftotalBB(l2)
      result /= 2.
      if not np.isfinite(result):
         result = 0.
      return result


   def F_TB(self, l1, l2, phi):
      result = self.f_TB(l1, l2, phi)
      result /= self.CMB.ftotalTT(l1)
      result /= self.CMB.ftotalBB(l2)
      if not np.isfinite(result):
         result = 0.
      return result

   def F_EB(self, l1, l2, phi):
      result = self.f_EB(l1, l2, phi)
      result /= self.CMB.ftotalEE(l1)
      result /= self.CMB.ftotalBB(l2)
      if not np.isfinite(result):
         result = 0.
      return result

   def F_TE(self, l1, l2, phi):
      numerator = self.CMB.ftotalEE(l1) * self.CMB.ftotalTT(l2) * self.f_TE(l1, l2, phi)
      numerator -= self.CMB.ftotalTE(l1) * self.CMB.ftotalTE(l2) * self.f_TE(l2, l1, -phi)
      denom = self.CMB.ftotalTT(l1)*self.CMB.ftotalTT(l2) * self.CMB.ftotalEE(l1)*self.CMB.ftotalEE(l2)
      denom -= ( self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2) )**2
      result = numerator / denom
      if not np.isfinite(result):
         result = 0.
      return result

   ###############################################################################
   # A_alpha from Hu & Okamoto 2002,
   # i.e. noise power spectrum of d_XY,
   # the lensing deflection estimator from XY.
   # Very important to enforce that lMin < l1,l2 < lMax
   # compute the integrals in ln(l) and not l, for speed
   # reduce the theta integral to [0,pi], by symmetry

   # returns the angle phi between l1 and l2,
   # given the angle theta between l1 and L=l1+l2
   def phi(self, L, l1, theta):
      x = L*np.cos(theta) - l1
      y = -L*np.sin(theta)  # y = L*np.sin(theta)
      result = np.arctan2(y,x)   # = 2.*np.arctan(y/(x+sqrt(x**2+y**2)))
      return result

   # returns the modulus of l2=L-l1,
   # given the moduli of L and l1 and the angle theta
   def l2(self, L, l1, theta):
      result = L**2 + l1**2 - 2.*L*l1*np.cos(theta)
      result = np.sqrt(result)
      return result
   
   # theta_min for the l1 integral
   # so that l2 > lMin
   def thetaMin(self, L, lnl1):
      l1 = np.log(lnl1)
      if (abs(L-l1)<self.lMin):
         theta_min = np.arccos((L**2+l1**2-self.lMin**2) / (2.*L*l1))
      else:
         theta_min = 0.
      return theta_min
   
   # theta_max for the l1 integral
   # so that l2 < lMax
   def thetaMax(self, L, lnl1):
      l1 = np.log(lnl1)
      if (l1>self.lMax-L):
         theta_max = np.arccos((L**2+l1**2-self.lMax**2) / (2.*L*l1))
      else:
         theta_max = np.pi
      return theta_max
   
   
   
#   def A_XY(self, ell):
#      # integrand
#      def integrand(x):
#         theta = x[1]
#         l1 = np.exp(x[0])
#         l2 = self.l2(ell, l1, theta)
#         if l2<self.lMin or l2>self.lMax:
#            return 0.
#         phi = self.phi(ell, l1, theta)
#         
#         # unlensed spectra
#         uTT1 = self.CMB.funlensedTT(l1)
#         uTT2 = self.CMB.funlensedTT(l2)
#         uEE1 = self.CMB.funlensedEE(l1)
#         uEE2 = self.CMB.funlensedEE(l2)
#         uBB1 = self.CMB.funlensedBB(l1)
#         uBB2 = self.CMB.funlensedBB(l2)
#         uTE1 = self.CMB.funlensedTE(l1)
#         uTE2 = self.CMB.funlensedTE(l2)
#         #
#         # lensed spectra
#         TT1 = self.CMB.ftotalTT(l1)
#         TT2 = self.CMB.ftotalTT(l2)
#         EE1 = self.CMB.ftotalEE(l1)
#         EE2 = self.CMB.ftotalEE(l2)
#         BB1 = self.CMB.ftotalBB(l1)
#         BB2 = self.CMB.ftotalBB(l2)
#         TE1 = self.CMB.ftotalTE(l1)
#         TE2 = self.CMB.ftotalTE(l2)
#         
#         # f
#         fTT = TT1 * l1*(l1 + l2*np.cos(phi))
#         fTT += TT2 * l2*(l2 + l1*np.cos(phi))
#         #
#         # typo in Hu & Okamoto 2002: cos(2phi) and not cos(phi)!!!
#         fTE = TE1 * l1*(l1 + l2*np.cos(phi)) * np.cos(2.*phi)
#         fTE += TE2 * l2*(l2 + l1*np.cos(phi))
#         #
#         fTB = TE1 * l1*(l1 + l2*np.cos(phi)) * np.sin(2.*phi)
#         #
#         fEE = EE1 * l1*(l1 + l2*np.cos(phi))
#         fEE += EE2 * l2*(l2 + l1*np.cos(phi))
#         fEE *= np.cos(2.*phi)
#         #
#         fEB = EE1 * l1*(l1 + l2*np.cos(phi))
#         fEB -= BB2 * l2*(l2 + l1*np.cos(phi))
#         fEB *= np.sin(2.*phi)
#         #
#         fBB = BB1 * l1*(l1 + l2*np.cos(phi))
#         fBB += BB2 * l2*(l2 + l1*np.cos(phi))
#         fBB *= np.cos(2.*phi)
#         
#         # F
#         FTT = fTT /TT1 /TT2 /2.
#         #
#         FEE = fEE /EE1 /EE2 /2.
#         #
#         FBB = fBB /BB1 /BB2 /2.
#         #
#         FTB = fTB /TT1 /BB2
#         #
#         FEB = fEB /EE1 /BB2
#         #
#         FTE = EE1*TT2*fTE - TE1*TE2*fTE
#         FTE /= TT1*TT2*EE1*EE2 - (TE1*TE2)**2
#
#         result = np.array([fTT*FTT, fTE*FTE, fTB*FTB, fEE*FEE, fEB*FEB])
#         result *= l1**2
#         result /= (2.*np.pi)**2
#         result *= 2.
##         print result
#         return {'a':result[0], 'b':result[1], 'c':result[2], 'd':result[3], 'e':result[4]}
#         return result.tolist()
#         
#      
#      integ = vegas.Integrator([[np.log(self.lMin), np.log(self.lMax)], [0., np.pi]])
#      result = integ(integrand, nitn=8, rtol=1.e-2)
##      answer = np.zeros(5, dtype=gv.GVar)
##      answer[:] = integ(integrand, nitn=8, rtol=1.e-2)
##      print type(answer)
#      #result = np.array([result[i].mean for i in range(len(result))], dtype='float')
#      
#      integ = None
#      integrand = None
#      
#      print "ok"
##      print (L**2 / result)
#      #return (ell**2 / result).tolist()



#   def test(self):
#      
#      data = np.zeros((self.Nl, 15))
#      
#      print "Computing all A_XY"
#      tStart = time()
#      #print np.shape(data[::,1:6])
#      #print np.shape(np.array(map(self.A_XY, self.L)))
#      
#
#      for i in range(7,self.Nl):
#         l = self.L[i]
#         a = self.A_XY(l)
#
##      for il in range(7):
###      print self.L[0]
##         d = self.A_XY(1.e1)
##         data[i,1:6] = d[:]
#
##      print data[::,1:6]
#      #      map(self.A_XY, self.L)
#
#      #      try:
#      #         self.A_XY(self.L[0])
#      #      except:
#      #         pass
##      for il in range(self.Nl):
##         ##         data[il, 1:6] = self.A_XY(self.L[il])
##         self.A_XY(self.L[il])
#
#      tStop = time()



   '''
   def A_TT(self, L):
      """Noise power spectrum of d_TT,
      the lensing deflection estimator from TT.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(theta, lnl1):
         l1 = np.exp(lnl1)
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TT(l1, l2, phi) * self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      result = integrate.dblquad(integrand, np.log(self.CMB.lMin), np.log(self.CMB.lMaxT), lambda lnl1: 0., lambda lnl1: np.pi, epsabs=0., epsrel=1.e-2)[0]
      result = L**2 / result
      print L, result

      if not np.isfinite(result):
         result = 0.
      return result
   '''


   def A_TT(self, L):
      """Noise power spectrum of d_TT,
      the lensing deflection estimator from TT.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TT(l1, l2, phi) * self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_TT.__func__, "integ"):
         self.A_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.A_TT.integ(integrand, nitn=8, neval=1000)

      result = self.A_TT.integ(integrand, nitn=1, neval=5000)
      # L^2: convert from N^{0 phi} to N^{0 d}
      result = L**2 / result.mean

      if not np.isfinite(result):
         result = 0.
      return result
   


   def A_TE(self, L):
      """Noise power spectrum of d_TE,
      the lensing deflection estimator from TE.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TE(l1, l2, phi) * self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
   
      # if first time, initialize integrator
      if not hasattr(self.A_TE.__func__, "integ"):
         self.A_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.A_TE.integ(integrand, nitn=8, neval=1000)
      result = self.A_TE.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result
   
   

   def A_TB(self, L):
      """Noise power spectrum of d_TB,
      the lensing deflection estimator from TB.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TB(l1, l2, phi) * self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first time, initialize integrator
      if not hasattr(self.A_TB.__func__, "integ"):
         self.A_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.A_TB.integ(integrand, nitn=8, neval=1000)
      result = self.A_TB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   def A_EE(self, L):
      """Noise power spectrum of d_EE,
      the lensing deflection estimator from EE.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_EE(l1, l2, phi) * self.F_EE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_EE.__func__, "integ"):
         self.A_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_EE.integ(integrand, nitn=8, neval=1000)
      result = self.A_EE.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   def A_EB(self, L):
      """Noise power spectrum of d_EB,
      the lensing deflection estimator from EB.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_EB(l1, l2, phi) * self.F_EB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_EB.__func__, "integ"):
         self.A_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_EB.integ(integrand, nitn=8, neval=1000)
      result = self.A_EB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result

   def A_BB(self, L):
      """Noise power spectrum of d_BB,
      the lensing deflection estimator from BB.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_BB(l1, l2, phi) * self.F_BB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_BB.__func__, "integ"):
         self.A_BB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.A_BB.integ(integrand, nitn=8, neval=1000)
      result = self.A_BB.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      return result


   ###############################################################################
   # N_alphabeta from Hu & Okamoto,
   # covariance of d_alpha and d_beta,
   # lensing deflection estimators from alpha and beta
   # without the factor A_alpha*A_beta/L^2,
   # for speed reasons.

   def N_TT_TE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_TE(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalTE(l2)
         result += self.F_TE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTT(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first time, initialize integrator
      if not hasattr(self.N_TT_TE.__func__, "integ"):
         self.N_TT_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TE.integ(integrand, nitn=1, neval=5000)
      return result.mean

   def N_TT_TB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_TB(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalTB(l2)
         result += 0.   #self.F_TB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTT(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_TB.__func__, "integ"):
         self.N_TT_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TB.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TT_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
         result += self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_TE.__func__, "integ"):
         self.N_TT_TE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_TE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_TE.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TT_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TT_EB.__func__, "integ"):
         self.N_TT_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TT_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TT_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TE_TB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_TB(l1, l2, phi)*self.CMB.ftotalTT(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_TB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalTE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_TB.__func__, "integ"):
         self.N_TE_TB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_TB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_TB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TE_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
         result += self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_EE.__func__, "integ"):
         self.N_TE_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_EE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_EE.integ(integrand, nitn=1, neval=5000)
      return result.mean
      

   def N_TE_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_TE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TE_EB.__func__, "integ"):
         self.N_TE_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TE_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TE_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean

   def N_TB_EE(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EE(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EE(l2, l1, -phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEB(l2)
         result *= self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TB_EE.__func__, "integ"):
         self.N_TB_EE.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TB_EE.integ(integrand, nitn=8, neval=1000)
      result = self.N_TB_EE.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_TB_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*min(self.CMB.lMaxT, self.CMB.lMaxP):
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>min(self.CMB.lMaxT, self.CMB.lMaxP):
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EB(l1, l2, phi)*self.CMB.ftotalTE(l1)*self.CMB.ftotalBB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalTB(l1)*self.CMB.ftotalEB(l2)
         result *= self.F_TB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_TB_EB.__func__, "integ"):
         self.N_TB_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(min(self.CMB.lMaxT, self.CMB.lMaxP))], [0., np.pi]])
         self.N_TB_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_TB_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   def N_EE_EB(self, L):
      """N_alpha_beta from Hu & Okamoto,
      covariance of d_alpha and d_beta,
      lensing deflection estimators from alpha and beta,
      without the factor A_alpha*A_beta/L^2,
      for speed reasons.
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = 0. #self.F_EB(l1, l2, phi)*self.CMB.ftotalEE(l1)*self.CMB.ftotalEB(l2)
         result += 0.   #self.F_EB(l2, l1, -phi)*self.CMB.ftotalEB(l1)*self.CMB.ftotalEE(l2)
         result *= self.F_EE(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_EE_EB.__func__, "integ"):
         self.N_EE_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.N_EE_EB.integ(integrand, nitn=8, neval=1000)
      result = self.N_EE_EB.integ(integrand, nitn=1, neval=5000)
      return result.mean


   ###############################################################################
   # tests and timing


   def plotNoise(self, fPkappa=None):
      # diagonal covariances: all
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      if fPkappa is None:
         # read C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         ax.plot(L, L**4/(2.*np.pi)*Pphi, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa * 4./(2.*np.pi), 'k-', lw=3, label=r'signal')
      #
#      # read N_mv for the true Planck map 2015 XV
#      data = np.genfromtxt("./output/cmblensrec/plancksmica/sl_nl_kk_planck15_xv.txt")
#      L = data[:,0]
#      Nkappa_mv = data[:,1]  # noise
#      Skappa = data[:,2] - data[:,1]  # signal
#      Sphi = Skappa * 4./L**4
#      Nphi_mv = Nkappa_mv * 4./L**4
#      ax.plot(L, L**4/(2.*np.pi) * Nphi_mv, 'k--')
#      ax.plot(L, L**4/(2.*np.pi) * Sphi, 'b--')
      #
      ax.plot(self.L, self.L**2 * self.N_TT/(2.*np.pi), c=plt.cm.rainbow(0.), lw=1.5, label=r'TT')
      ax.plot(self.L, self.L**2 * self.N_TE/(2.*np.pi), c=plt.cm.rainbow(1./5.), lw=1.5, label=r'TE')
      ax.plot(self.L, self.L**2 * self.N_TB/(2.*np.pi), c=plt.cm.rainbow(2./5.), lw=1.5, label=r'TB')
      ax.plot(self.L, self.L**2 * self.N_EE/(2.*np.pi), c=plt.cm.rainbow(3./5.), lw=1.5, label=r'EE')
      ax.plot(self.L, self.L**2 * self.N_EB/(2.*np.pi), c=plt.cm.rainbow(4./5.), lw=1.5, label=r'EB')
      ax.plot(self.L, self.L**2 * self.N_mv/(2.*np.pi), 'r', lw=3, label=r'min. var.')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$L^2C^{dd}_L / (2\pi) = 4C_L^\kappa  / (2\pi)$', fontsize=24)
      ax.set_ylim((3.e-11, 1.e-1))
      ax.set_xlim((10., 4.e4))
      #ax.set_title(r'Noise in lensing deflection reconstruction ('+self.CMB.name+')')
      #
      #path = "./figures/cmblensrec/"+str(self.CMB)+"/full_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #path = "/Users/Emmanuel/Desktop/cmblensrec_atmnoise.pdf"
      #path = "./figures/cmblensrec/summaries_s4/"+str(self.CMB)+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      #fig.clf()

      '''
      # diagonal covariances: mv versus signal
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.loglog(L, L**4/(2.*np.pi)*Pphi, 'k', lw=3, label=r'signal')
      #
      ax.loglog(self.L, self.L**2 * self.N_mv/(2.*np.pi), 'r', lw=3, label=r'mv')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell^2 C^{dd}_\ell / (2\pi)$')
      #ax.set_ylim((1.e-9, 1.e-5))
      ax.set_xlim((self.lMin, self.lMax))
      ax.set_title(r'Noise in lensing deflection reconstruction ('+self.CMB.name+')')
      #
      path = "./figures/cmblensrec/"+str(self.CMB)+"/short_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      

      # non-diagonal covariances
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TT_TE)/(2.*np.pi), 'k', lw=2, label=r'TT-TE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TT_EE)/(2.*np.pi), 'r', lw=2, label=r'TT-EE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TE_EE)/(2.*np.pi), 'g', lw=2, label=r'TE-EE')
      ax.loglog(self.L, self.L**2 * np.abs(self.N_TB_EB)/(2.*np.pi), 'c', lw=2, label=r'TB-EB')
      #
      ax.legend(loc=2)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell^2 | C_\ell | / (2\pi)$')
      ax.set_title(r'Noise in lensing deflection reconstruction ('+self.CMB.name+')')
      #
      path = "./figures/cmblensrec/"+str(self.CMB)+"/cross_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      '''
      plt.show()
   

   # SNR on Cld, including reconstruction noise
   # and cosmic var if wanted
   def snrTT(self, fClkappa, cosmicVar=False):
      Clkappa = np.array(map(fClkappa, self.L))
      # convert Clkappa to Cld
      Cld = 4. * Clkappa / self.L**2
      # get rid of artificial zeros in noise
      Noise = self.N_TT.copy()
      Noise[Noise==0.] = np.inf
      # compute total SNR
      if cosmicVar:
         Y = self.L * Cld**2 / (Cld + Noise)**2  # w cosmic variance
      else:
         Y = self.L * Cld**2 / Noise**2  # w/o cosmic variance
      snr2 = np.trapz(Y, self.L)
      return np.sqrt(snr2)


   def timeIntegrandA_TT(self, L):
      
      NL1 = 101
      Ntheta = 101
      L1 = np.logspace(np.log10(1.), np.log10(1.e4), NL1, 10.)
      Theta = np.linspace(0., 2.*np.pi, Ntheta)
      
      def integrand(l1, theta):
         l2 = self.l2(L, l1, theta)
         phi = self.phi(L, l1, theta)
         result = self.f_TT(l1, l2, phi) * self.F_TT(l1, l2, phi)
         result *= 2.*np.pi*l1
         result /= 2.*np.pi**2
         return result
      
      # time the integrand
      tStart = time()
      for l1 in L1:
         for theta in Theta:
            result = integrand(l1, theta)
      tStop = time()
      print "took", (tStop-tStart)/(NL1*Ntheta), "sec per evaluation,"
      
      # time the integral
      tStart = time()
      integral = integrate.dblquad(integrand, 0., 2.*np.pi, lambda theta: self.lMin, lambda theta: self.lMax, epsabs=0., epsrel=1.e-1)[0]
      result = L**2 / integral
      tStop = time()
      print "took", tStop-tStart, "sec for the integral"
      
      # took 400sec=7min with epsrel = 1.e-2, ie 2.e6 evaluations
      # took 200sec=3min with epsrel = 5.e-2, ie 1.e6 evaluations
      # took 100sec=2min with epsrel = 1.e-1, ie 6.e5 evaluations
      
      return

   ###############################################################################
   ###############################################################################
   # show which T_l contribute to kappa_L


   def snr2Density_TT(self, L, l1):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl1
      showing which l1 contribute to kappa_L.
      Normalized such that the integral wrt dlnl yields 1/N_L^kappa.
      This breaks the symmetry between l1 and l2,
      because l1 is fixed while l2 varies within l1-L and L1+L
      """

      # make sure L is within reconstruction range
      if L>2.*self.CMB.lMaxT:
         return 0.
      # make sure l1 is within the map range
      if l1<self.CMB.lMin or l1>self.CMB.lMaxT:
         return 0.

      # integrand
      def integrand(theta):
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TT(l1, l2, phi) * self.F_TT(l1, l2, phi)
         result *= l1**2   # integrand wrt dlnl1 * dphi
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      # convert it in terms of kappa, i.e d(1/N_L^kappa)/dlnl1
      result /= L**4 / 4.
      return result


   def snr2DensitySymLowL_TT(self, L, l):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl
      showing which l contribute to kappa_L,
      ONLY IN THE REGIME L << l.
      Normalized such that the integral wrt dlnl yields 1/N_L^kappa.
      Here l = (l1+l2)/2, and L = l1-l2,
      which makes l1 and l2 symmetric:
      they both vary between l-L/2 and l+L/2.
      This decomposition is inspired by the shear/dilation estimators
      """
      # make sure L is within reconstruction range
      if L>2.*self.CMB.lMaxT:
         return 0.

      # integrand: theta is the angle between L = (l1-l2)/2 and l = (l1+l2)/2.
      # We integrate over theta, at fixed l and L, ie varying l1 and l2 accordingly
      def integrand(theta):
         # derivatives of the unlensed power spectrum
         a = 1.+1.e-3
         dlnCldlnl = np.log(self.CMB.funlensedTT(l*a) / self.CMB.funlensedTT(l*a))
         dlnCldlnl /= 2.*np.log(a)
         dlnl2Cldlnl = dlnCldlnl + 2.
         # get l1 and l2 in a symmetric form
         l1 = np.sqrt((L/2.)**2 + l**2 - 2.*(L/2.)*l*np.cos(theta))
         l2 = np.sqrt((L/2.)**2 + l**2 + 2.*(L/2.)*l*np.cos(theta))
         if l1<self.CMB.lMin or l1>self.CMB.lMaxT:
            return 0.
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         # compute the integrand
         result = dlnl2Cldlnl + np.cos(2.*theta) * dlnCldlnl
         result *= self.CMB.funlensedTT(l)
         result *= 0.5*L**2   # conversion from kappa to phi, irrelevant
         result = result**2
         result /= self.CMB.ftotalTT(l1)
         result /= self.CMB.ftotalTT(l2)
         result /= 2.
         result *= l**2   # integrand wrt dlnl1 * dphi
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      # convert it in terms of kappa, i.e d(1/N_L^kappa)/dlnl1
      result /= L**4 / 4.
      return result



   def plotSnr2Density_TT_lines(self):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl
      showing which l contribute to kappa_L,
      ONLY IN THE REGIME L << l.
      Normalized such that the integral wrt dlnl yields 1/N_L^kappa.
      Here l = (l1+l2)/2, and L = l1-l2,
      which makes l1 and l2 symmetric:
      they both vary between l-L/2 and l+L/2.
      This decomposition is inspired by the shear/dilation estimators
      """
      # ell values for phi
      #L = np.logspace(np.log10(10.), np.log10(1.e4), 5, 10.)
      #L = np.array([10., 1.e2, 5.e2])
      L = [100.]
      
      # ell values for T
      L1 = np.logspace(np.log10(10.), np.log10(6.e4), 501, 10.)

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for iL in range(len(L)):
         f = lambda l1: self.snr2DensitySymLowL_TT(L[iL], l1)
         snr2Density = np.array(map(f, L1))
         # expected value if only counting number of modes
         dNLkappadlnl_theory = L1**2 / (np.pi)# * L[iL]**4

         # save the values for future use
         data = np.zeros((len(L1), 3))
         data[:,0] = L1.copy()
         data[:,1] = snr2Density
         data[:,2] = dNLkappadlnl_theory
         path = "./output/cmblensrec/"+self.CMB.name+"/dsnr2_dlnl_L"+intExpForm(L[iL])+".txt"
         print "saving to "+path
         np.savetxt(path, data)
         
         # normalize both, for plot
         dNLkappadlnl_theory /= np.max(snr2Density)
         snr2Density /= np.max(snr2Density)#[np.where(L1 <= 5.e3)]) # just to rescale in plot
         #ax.plot(L1, snr2Density, lw=2, c=plt.cm.jet(1.*iL/len(L)), label=r'$L_\phi=$'+str(int(L[iL])))
         ax.plot(L1, snr2Density, lw=2, c='r', label=r'$L_\phi=$'+str(int(L[iL])))
         ax.plot(L1, dNLkappadlnl_theory, 'r', ls='--', lw=1)
      #
      ax.legend(loc=1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_xlabel(r'$\ell_T$')
      ax.set_ylabel(r'$d\text{SNR}(\phi_L)^2/d\text{ln}\ell_T$')
      #ax.set_xlim((1., 5.e3))
      ax.set_ylim((0., 1.1))
      #
      path = "./figures/cmblensrec/dsnr2_dl_test.pdf"
      fig.savefig(path, bbox_inches='tight')
      
      plt.show()

      return L1, snr2Density
   
   
   
   def plotSnr2Density_TT_color(self):
   
      # multipoles of phi
      nL = 101  #51
      lnLMin = np.log10(1.)
      lnLMax = np.log10(self.lMax*(1.-1.e-1))   #np.log10(2.*self.lMax*(1.-1.e-3))
      dlnL = (lnLMax-lnLMin)/nL
      lnL = np.linspace(lnLMin, lnLMax, nL)
      lnLEdges = np.linspace(lnLMin-0.5*dlnL, lnLMax+0.5*dlnL, nL+1)
      L = 10.**lnL
      LEdges = 10.**lnLEdges

      # multipoles of T
      nL1 = 101  #51
      lnL1Min = np.log10(self.lMin*(1.+1.e-1))
      lnL1Max = np.log10(self.lMax*(1.-1.e-1))
      dlnL1 = (lnL1Max-lnL1Min)/nL1
      lnL1 = np.linspace(lnL1Min, lnL1Max, nL1)
      lnL1Edges = np.linspace(lnL1Min-0.5*dlnL1, lnL1Max+0.5*dlnL1, nL1+1)
      L1 = 10.**lnL1
      L1Edges = 10.**lnL1Edges
   
      # compute
      dSNR2dl = np.zeros((nL1, nL))
      for iL in range(nL):
         l = L[iL]
         for iL1 in range(nL1):
            l1 = L1[iL1]
            dSNR2dl[iL1, iL] = self.snr2Density_TT(l, l1)
         # normalize so that int dl1 dSNR2/dl1 = 1
         #dSNR2dl[:,iL] /= 1./self.A_TT(l)
   
      # make the color plot
      LL1,LL = np.meshgrid(L1Edges, LEdges, indexing='ij')
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      cp=ax.pcolormesh(LL1, LL, np.log10(dSNR2dl + 1.e-10), linewidth=0, rasterized=True, cmap=plt.cm.YlOrRd)
      #cp.set_clim(0., 1.)
      fig.colorbar(cp)
      #
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$L$')
      
      plt.show()
   
   
   ###############################################################################
   ###############################################################################
   # for a source field defined by dCl/dz = f(l, z),
   # compute the effective source redshift distribution,
   # appropriate for the quadratic estimator,
   # analogous to the dn/dz of galaxy lensing.
   
   def fz_TT(self, l1, l2, phi, z):
      result = self.CMB.fdCldz(l1, z) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.fdCldz(l2, z) * l2*(l2 + l1*np.cos(phi))
      return result
   
   
   # here l is the multipole of kappa considered
   # gives the effective source distribution at z
   def sourceDist_TT(self, L, z):
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      def integrand(x):
         l1 = np.exp(x[0])
         theta = x[1]
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.fz_TT(l1, l2, phi, z) * self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.   # by symmetry, we integrate over half the area
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.sourceDist_TT.__func__, "integ"):
         self.sourceDist_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.sourceDist_TT.integ(integrand, nitn=8, neval=1000)
      result = self.sourceDist_TT.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      print L, z, result
      return result
   
   
   def plotSourceDist_TT(self):
   
      # multipoles of phi
      nL = 3  #51
      lnLMin = np.log10(1.)
      lnLMax = np.log10(2.*self.lMax*(1.-1.e-3))
      dlnL = (lnLMax-lnLMin)/nL
      lnL = np.linspace(lnLMin, lnLMax, nL)
      lnLEdges = np.linspace(lnLMin-0.5*dlnL, lnLMax+0.5*dlnL, nL+1)
      L = 10.**lnL
      LEdges = 10.**lnLEdges
      
      # redshifts
      nZ = 3  #51
      zMin = 1.e-3
      zMax = 5.
      dZ = (zMax-zMin)/nZ
      Z = np.linspace(zMin, zMax, nZ)
      zEdges = np.linspace(zMin-0.5*dZ, zMax+0.5*dZ, nZ+1)
      
      # compute!!!
      SourceDistTT = np.zeros((nZ, nL))
      for iZ in range(nZ):
         z = Z[iZ]
         for iL in range(nL):
            l = L[iL]
            SourceDistTT[iZ, iL] = self.sourceDist_TT(l, z)
      
      print SourceDistTT
      
      # make the color plot
      ZZ,LL = np.meshgrid(zEdges, LEdges, indexing='ij')
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      cp=ax.pcolormesh(ZZ, LL, SourceDistTT, linewidth=0, rasterized=True, cmap=plt.cm.YlOrRd_r)
      #cp.set_clim(0., 1.)
      fig.colorbar(cp)
      #
      #ax.set_xscale('log')
      #ax.set_yscale('log')
      ax.set_xlabel(r'$z$')
      ax.set_ylabel(r'$L$')
      
      plt.show()
   
   
   
   
   ###############################################################################
   ###############################################################################
   # Order of magnitude for the effect of the unlensed trispectrum on the lensing noise
   # e.g., effect of CIB trispectrum on CIB lensing noise
   # This is really T / CC.
   # !!! not really used
   
   # L is the ell for phi
   def trispectrumCorrection(self, fTnondiag, L=20.):
      """multiplicative correction due to the trispectrum,
      i.e. T_{l,L-l,l,L-l} / C_l C_{L-l},
      for L given as input,
      and l in [1, lMaxT].
      """
      def f(arg):
         # here the arguments expected are |l|, |L-l|
         l1 = arg[0]
         l2 = arg[1]
         cc = self.CMB.ftotalTT(l1)*self.CMB.ftotalTT(l2)
         t = fTnondiag(l1, l2)
         return t/cc
      
      # initialize the array of l
      lx = np.linspace(1., self.CMB.lMaxT, 11)
      ly = np.linspace(1., self.CMB.lMaxT, 11)
      lx, ly = np.meshgrid(lx, ly, indexing='ij')
      #
      l = np.sqrt(lx**2 + ly**2)
      Lminusl = np.sqrt( (L-lx)**2 + ly**2 )
      Arg = np.array([(l.flatten()[i], Lminusl.flatten()[i]) for i in range(len(l.flatten()))])
      #
      Result = np.array(map(f, Arg))
      Result.reshape(np.shape(l))
      return lx, ly, Result
   
   
   def plotTrispectrumCorrection(self, fTnondiag, L=20.):
      lx, ly, Result = self.trispectrumCorrection(fTnondiag, L=20.)
      plt.pcolormesh(lx, ly, np.log(Result))
      plt.colorbar()
      plt.show()
      return lx, ly, result


   ###############################################################################
   ###############################################################################
   # Effect of white trispectrum on lensing noise
   # e.g., effect of CIB trispectrum on CIB lensing
   # here I assume that the trispectra of T, Q, U are white,
   # i.e. the trispectra of E and B are non-white.


   def relativeNoiseWhiteTrispec_TT(self, L):
      """Factor such that:
      N_L^\kappa = N_L^{0 \kappa} * (1 + factor * Trispectrum),
      where the trispectrum is assumed white.
      In other words, factor * Trispectrum is the relative increase in lensing noise
      due to the unlensed white trispectrum.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.relativeNoiseWhiteTrispec_TT.__func__, "integ"):
         self.relativeNoiseWhiteTrispec_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.relativeNoiseWhiteTrispec_TT.integ(integrand, nitn=8, neval=1000)
      result = self.relativeNoiseWhiteTrispec_TT.integ(integrand, nitn=1, neval=5000)

      result = result.mean**2
      result *= self.fN_phi_TT(L)
      if not np.isfinite(result):
         result = 0.
      return result



   def relativeNoiseWhiteTrispec_EB(self, L):
      """Factor such that:
      N_L^\kappa = N_L^{0 \kappa} * (1 + factor * Trispectrum),
      where the trispectrum is assumed white.
      In other words, factor * Trispectrum is the relative increase in lensing noise
      due to the unlensed white trispectrum.
      Here, we assume the trispectrum of Q and U are white and uncorrelated,
      so that the E and B trispectra are non-white.
      !!! Gives weird result in polarization: very large and noisy. Cancellations?
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.F_EB(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         # extra factors because the trispectrum of Q and U are white,
         # so the trispectrum EBEB isn't white:
         result *= l1**2 * np.cos(2.*theta)
#         result *= l1**2 * (np.cos(theta)**2 - np.sin(theta)**2)
         result *= 2. * (L - l1 * np.cos(theta)) * (- l1 * np.sin(theta))
#         result *= 2. * l2 * np.cos(theta+phi) * l2 * np.sin(theta+phi)
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.relativeNoiseWhiteTrispec_EB.__func__, "integ"):
         self.relativeNoiseWhiteTrispec_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=10, neval=10000)
#      result = self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=50, neval=5000)
      result = self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=10, neval=1000)

      result = result.mean**2
      result *= self.fN_phi_EB(L)
      # extra factor 2, due to conversion from Q/U trispectra to EBEB trispectrum
      result *= 2.
      if not np.isfinite(result):
         result = 0.
      return result


   def saveRelativeNoiseWhiteTrispec(self):
      path = self.directory+"/relative_noise_white_trispectrum.txt"
      data = np.zeros((self.Nl, 3))
      data[:,0] = self.L.copy()
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      print "relative noise from trispectrum: TT"
      data[:,1] = np.array(pool.map(self.relativeNoiseWhiteTrispec_TT, self.L))
      np.savetxt(path, data)
      print "relative noise from trispectrum: EB"
      data[:,2] = np.array(pool.map(self.relativeNoiseWhiteTrispec_EB, self.L))
      np.savetxt(path, data)


   def loadRelativeNoiseWhiteTrispec(self):
      path = self.directory+"/relative_noise_white_trispectrum.txt"
      data = np.genfromtxt(path)
      self.fRelativeNoiseWhiteTrispec_TT = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.fRelativeNoiseWhiteTrispec_EB = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)


   def plotRelativeNoiseWhiteTrispec(self):
      TT = self.fRelativeNoiseWhiteTrispec_TT(self.L)
      EB = self.fRelativeNoiseWhiteTrispec_EB(self.L)
      N_TT_G = self.fN_k_TT(self.L)
      N_EB_G = self.fN_k_EB(self.L)
      
      # Q and U trispectra from polarized point sources, in (muK)^4 sr^-3,
      # according to CMBPol white paper, Eq 20-22,
      # for a flux cut of 5mJy instead of 200mJy
      Trispec = 3.5e-20
      
      print "TT"
      print TT
      print "EB"
      print EB

      # conversion factor
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.loglog(self.L, TT, c='k', label=r'TT')
      ax.loglog(self.L, EB, c='b', label=r'EB')
      #
      ax.legend(loc=1)
      #ax.set_ylim((1.e-4, 1.))
      ax.set_xlabel(r'$L$')
      ax.set_ylabel(r'$N^{0, \text{NG}} / N^{0, \text{G}} / \mathcal{T}$')
      
      # compare noises
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
#      ax.loglog(self.L, N_TT_G, 'k--', label=r'TT G')
#      ax.loglog(self.L, TT * N_TT_G, 'k--', label=r'TT PS')
      #
      ax.loglog(self.L, N_EB_G, 'b', label=r'EB')
      ax.loglog(self.L, EB * N_EB_G * Trispec, 'b--', label=r'EB PS')
      #
      ax.legend(loc=1)
      #ax.set_ylim((1.e-4, 1.))
      ax.set_xlabel(r'$L$')
      ax.set_ylabel(r'N^{0 \; \kappa}$')

      plt.show()



















   ###############################################################################
   ###############################################################################
   # Multipole lensing estimators
   
   ###############################################################################
   # Azimuthal multipole moments of the lensing response f_TT,
   # with theta = (L, l), l1 = L/2 + l and l2 = L/2 - l.
   # Needed for shear/dilation estimators, in the squeezed limit L<<l,
   # and in the general (not squeezed) limit.
   
#   def config_multipole(self, L, l, t):
#      '''Returns l1, l2 and phi=(l1, l2),
#      given L, l and theta=(L,l),
#      where:
#      l1 = L/2 + l,
#      l2 = L/2 - l.
#      This configuration is relevant for azimuthal multipoles, especially in the squeezed limit L<<l.
#      '''
#      l1 = np.sqrt(l**2 + L**2/4. + l*L*np.cos(t)) # np.sqrt((L/2.+l*np.cos(t))**2 + (l*np.sin(t))**2)
#      l2 = np.sqrt(l**2 + L**2/4. - l*L*np.cos(t)) # np.sqrt((L/2.-l*np.cos(t))**2 + (l*np.sin(t))**2)
#
#      x1 = l + L/2.*np.cos(t)
#      y1 = L/2.*np.sin(t)
#      phi1 = np.arctan2(y1,x1)
#
#      x2 = l - L/2.*np.cos(t)
#      y2 = L/2.*np.sin(t)
#      phi2 = np.arctan2(y2,x2)
#
#      phi = phi1 + phi2 + np.pi
#      return l1, l2, phi, theta1, theta2

   def config_multipole(self, L, l, t):
      '''Returns l1, l2 and phi=(l1, l2),
      given L, l and theta=(L,l),
      where:
      l1 = L/2 + l,
      l2 = L/2 - l.
      This configuration is relevant for azimuthal multipoles, especially in the squeezed limit L<<l.
      '''
      l1 = np.sqrt(l**2 + L**2/4. + l*L*np.cos(t)) # np.sqrt((L/2.+l*np.cos(t))**2 + (l*np.sin(t))**2)
      l2 = np.sqrt(l**2 + L**2/4. - l*L*np.cos(t)) # np.sqrt((L/2.-l*np.cos(t))**2 + (l*np.sin(t))**2)

      x1 = L/2. + l*np.cos(t)
      y1 = l*np.sin(t)
      theta1 = np.arctan2(y1,x1)

      x2 = L/2. - l*np.cos(t)
      y2 = -l*np.sin(t)
      theta2 = np.arctan2(y2,x2)

      phi = theta2 - theta1
      return l1, l2, phi, theta1, theta2


   def f_TT_multipole_squeezed(self, L, l, m=0):
      '''Expected limit of the m-th multipole moment of f_TT when L<<l,
      to first order in L/l.
      Should match f_TT_multipole_interp in this limit.
      '''
      # derivative of the unlensed power spectrum
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = self.CMB.funlensedTT(lup) / self.CMB.funlensedTT(ldown)
         result = np.log(result) / (2.*e)
         return result
      
      if m==0:
         result = fdLnC0dLnl(l) + 2.  # = dln(l^2C0)/dlnl
      elif m==2:
         result = fdLnC0dLnl(l)
      else:
         result = 0. # I haven't computed the other multipoles in the squeezed limit
      result *= - self.CMB.funlensedTT(l)
      result *= L**2/2.
      return result


   def f_TT_multipole(self, L, l, m=0):
      '''m-th multipole moment of f_TT(l1, l2):
      \int dtheta/(2pi) f_TT(l1, l2) * cos(m theta),
      where:
      l1 = L/2 + l,
      l2 = L/2 - l.
      No consideration of lmin and lmax here, so that all the theta integrals
      cover the full circle.
      '''
      def integrand(t):
         l1, l2, phi, theta1, theta2 = self.config_multipole(L, l, t)
#         if l1<self.CMB.lMin or l1>self.CMB.lMaxT or l2<self.CMB.lMin or l2>self.CMB.lMaxT:
#            print "nope"
#            result = 0.
#         else:
         result = self.f_TT(l1, l2, phi)
         if m>0:
            result *= 2. * np.cos(m*t)
         return result / (np.pi) # because half the angular integration domain
      
      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      return result

   ###############################################################################
   # Interpolate the azimuthal multipole moments of the lensing response f_TT.

   def save_f_TT_multipole(self, m=0):
      '''Precompute f_TT_multipole(L, l, m),
      for speed.
      '''
      NL = 501
      Nl = 501
      Ell = np.logspace(np.log10(10.), np.log10(2.*self.CMB.lMaxT), NL)
      ell = np.logspace(np.log10(10.), np.log10(self.CMB.lMaxT), Nl)
      table = np.zeros((NL, Nl))
      print "precompute the "+str(m)+"-th multipole of f_TT"
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      for iL in range(NL):
         f = lambda il: self.f_TT_multipole(Ell[iL], ell[il], m)
         table[iL, :] = np.array(pool.map(f, range(Nl)))
         print "- done "+str(iL+1)+" of "+str(NL)
      # save the table
      np.savetxt(self.directory+"/Llong_fTTmultipole_m"+str(m)+".txt", Ell)
      np.savetxt(self.directory+"/lshort_fTTmultipole_m"+str(m)+".txt", ell)
      np.savetxt(self.directory+"/fTTmultipole_m"+str(m)+".txt", table)
      return


   def save_f_TT_multipoles(self):
      self.save_f_TT_multipole(m=0)
      self.save_f_TT_multipole(m=2)
      self.save_f_TT_multipole(m=4)
      self.save_f_TT_multipole(m=6)
      self.save_f_TT_multipole(m=8)


   def load_f_TT_multipole(self, m=0, test=False):
      '''Interpolate f_TT_multipole(L, l, m),
      for speed.
      '''
      print "interpolate the "+str(m)+"-th multipole of f_TT"
      Ell = np.genfromtxt(self.directory+"/Llong_fTTmultipole_m"+str(m)+".txt")
      ell = np.genfromtxt(self.directory+"/lshort_fTTmultipole_m"+str(m)+".txt")
      table = np.genfromtxt(self.directory+"/fTTmultipole_m"+str(m)+".txt")
      interp = RectBivariateSpline(np.log10(Ell), np.log10(ell), table, kx=1, ky=1, s=0)
      
      if m==0:
         self.f_TT_monopole_interp = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]
      elif m==2:
         self.f_TT_quadrupole_interp = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]
      elif m==4:
         self.f_TT_hexadecapole_interp = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]
      elif m==6:
         self.f_TT_multipole6_interp = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]
      elif m==8:
         self.f_TT_multipole8_interp = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]

      if test:
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         for iL in range(0, len(Ell), 50):
            L = Ell[iL]
            # interpolated multipole
            if m==0:
               f = lambda l: self.f_TT_monopole_interp(L,l)
            elif m==2:
               f = lambda l: self.f_TT_quadrupole_interp(L,l)
            elif m==4:
               f = lambda l: self.f_TT_hexadecapole_interp(L,l)
            elif m==6:
               f = lambda l: self.f_TT_multipole6_interp(L,l)
            elif m==8:
               f = lambda l: self.f_TT_multipole8_interp(L,l)
            Finterp = np.array(map(f, ell))
            ax.loglog(ell, np.abs(Finterp), c=plt.cm.winter(iL*1./len(Ell)), label=r'$L=$'+str(np.int(L)))
            #
            # theory multipole, to first order in L/l
            f = lambda l: self.f_TT_multipole_squeezed(L,l,m)
            Ftheory = np.array(map(f, ell))
            ax.loglog(ell, np.abs(Ftheory), c=plt.cm.winter(iL*1./len(Ell)), ls='--')
         #
         ax.loglog([], [], c=plt.cm.winter(0.), label=r'non-perturbative')
         ax.loglog([], [], c=plt.cm.winter(0.), ls='--', label=r'squeezed')
         ax.legend(loc=3)

         plt.show()


   def load_f_TT_multipoles(self, test=False):
      self.load_f_TT_multipole(m=0, test=test)
      self.load_f_TT_multipole(m=2, test=test)
      self.load_f_TT_multipole(m=4, test=test)
      self.load_f_TT_multipole(m=6, test=test)
      self.load_f_TT_multipole(m=8, test=test)


   def f_TT_multipole_interp(self, L, l, m=0):
      if m==0:
         result = self.f_TT_monopole_interp(L,l)
      elif m==2:
         result = self.f_TT_quadrupole_interp(L,l)
      elif m==4:
         result = self.f_TT_hexadecapole_interp(L,l)
      elif m==6:
         result = self.f_TT_multipole6_interp(L,l)
      elif m==8:
         result = self.f_TT_multipole8_interp(L,l)
      return result



   ###############################################################################
   # Noise of the lensing estimator from the m-th multipole moment of f_TT


   def N_k_TT_m(self, L, m=0, optimal=True):
      """Noise power spectrum of kappa from TT,
      using only the m-th multipole moment of the lensing response f_TT.
      Here theta=(L,l), with:
      l1 = L/2 + l,
      l2 = L/2 - l.
      This is not the same convention as in my standard QE integrals.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      # integrand
      def integrand(lnl):
         l = np.exp(lnl)
         
         # choose l bounds so that the theta integral can cover the full circle
         # otherwise, the multipole estimators will be biased
         if (np.abs(l-L/2)<self.CMB.lMin) or (l+L/2>self.CMB.lMaxT):
            result = 0.
         
         else:
            result = self.f_TT_multipole_interp(L, l, m)**2
            result *= l**2 / (2.*np.pi)
            if m>0:
               result /= 4.

            # use the optimal noise weighting: angular average of the Cl^total
            if optimal:
               # angular integrand
               def f(t):
                  l1, l2, phi, theta1, theta2 = self.config_multipole(L, l, t)
                  result = self.CMB.ftotalTT(l1) * self.CMB.ftotalTT(l2)
                  result *= np.cos(m*t)**2
                  result /= 2.*np.pi
                  result *= 2.
                  result *= 2.   # because integrating over half the domain
                  return result
               # compute angular integral
               integral = integrate.quad(f, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
            # else use the suboptimal intuitive noise
            else:
               integral = 2. * self.CMB.ftotalTT(l)**2
            result /= integral
   
         if not np.isfinite(result):
            result = 0.
         return result
      
      result = integrate.quad(integrand, np.log(1.), np.log(self.CMB.lMaxT), epsabs=0, epsrel=1.e-3)[0]
      result = (L**2/2.)**2 / result
      if not np.isfinite(result):
         result = 0.
      print "- done L="+str(L), result
      return result


   def N_k_TT_test(self, L):
      """Noise power spectrum of kappa from TT:
      should recover the standard Hu & Okamoto expression.
      This is a good test for my angular conversions.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      # integrand
      def integrand(x):
         l = np.exp(x[0])
         theta = x[1]
         l1, l2, phi, theta1, theta2 = self.config_multipole(L, l, theta)

         if (l1<self.CMB.lMin or l1>self.CMB.lMaxT) or (l2<self.CMB.lMin or l2>self.CMB.lMaxT):
            result = 0.
         else:
            result = self.f_TT(l1, l2, phi)
            result = result**2
            result /= 2. * self.CMB.ftotalTT(l1) * self.CMB.ftotalTT(l2)
            result *= l**2 / (2.*np.pi)**2
            result *= 2.   # because half of the angular integration domain
         if not np.isfinite(result):
            print "problem:", L, l, theta, l1, l2, phi
            result = 0.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_k_TT_test.__func__, "integ"):
         self.N_k_TT_test.__func__.integ = vegas.Integrator([[np.log(1.), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.N_k_TT_test.integ(integrand, nitn=8, neval=1000)
      result = self.N_k_TT_test.integ(integrand, nitn=1, neval=5000)
      result = (L**2/2.)**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      print "- done L="+str(int(L))
      return result


   def save_N_k_TT_m(self):
      path = self.directory+"/Nk_TT_m.txt"
      data = np.zeros((self.Nl, 9))
      data[:,0] = self.L.copy()
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      print "test: recover Hu & Okamoto 2002"
      data[:,1] = np.array(pool.map(self.N_k_TT_test, self.L))
      np.savetxt(path, data)
      print "Noise: m=0"
      f = lambda L: self.N_k_TT_m(L, m=0)
      data[:,2] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=0, suboptimal"
      f = lambda L: self.N_k_TT_m(L, m=0, optimal=False)
      data[:,3] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=2"
      f = lambda L: self.N_k_TT_m(L, m=2)
      data[:,4] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=2, suboptimal"
      f = lambda L: self.N_k_TT_m(L, m=2, optimal=False)
      data[:,5] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=4"
      f = lambda L: self.N_k_TT_m(L, m=4)
      data[:,6] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=6"
      f = lambda L: self.N_k_TT_m(L, m=6)
      data[:,7] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "Noise: m=8"
      f = lambda L: self.N_k_TT_m(L, m=8)
      data[:,8] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      '''
      print "Save noise on no-monopole lensing estimator"
      data[:,5] = np.array(pool.map(self.N_k_TT_nomono, self.L))
      np.savetxt(path, data)
      '''


   def load_N_k_TT_m(self):
      path = self.directory+"/Nk_TT_m.txt"
      data = np.genfromtxt(path)
      self.fN_k_TT_test = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_monopole = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_monopole_suboptimal = interp1d(data[:,3], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_quadrupole = interp1d(data[:,0], data[:,4], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_quadrupole_suboptimal = interp1d(data[:,0], data[:,5], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_m4 = interp1d(data[:,0], data[:,6], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_m6 = interp1d(data[:,0], data[:,7], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k_TT_m8 = interp1d(data[:,0], data[:,8], kind='linear', bounds_error=False, fill_value=0.)
      #self.fN_k_TT_nomonopole = interp1d(data[:,0], data[:,5], kind='linear', bounds_error=False, fill_value=0.)
   

   def plotNoiseMultipoles(self, fPkappa=None):
      N_qe = self.fN_k_TT(self.L)
      N_m0 = self.fN_k_TT_monopole(self.L)
      N_m0_suboptimal = self.fN_k_TT_monopole_suboptimal(self.L)
      N_m2 = self.fN_k_TT_quadrupole(self.L)
      N_m2_suboptimal = self.fN_k_TT_quadrupole_suboptimal(self.L)
      N_m4 = self.fN_k_TT_m4(self.L)
      N_m6 = self.fN_k_TT_m6(self.L)
      N_m8 = self.fN_k_TT_m8(self.L)
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot(self.L, N_m8, c=plt.cm.winter(1.), label=r'm=8')
      ax.plot(self.L, N_m6, c=plt.cm.winter(0.75), label=r'm=6')
      ax.plot(self.L, N_m4, c=plt.cm.winter(0.5), label=r'm=4')
      #
      #ax.plot(self.L, N_m2_suboptimal, ls='--', c=plt.cm.winter(0.25), label=r'm=2 sub')
      ax.plot(self.L, N_m2, c=plt.cm.winter(0.25), label=r'm=2')
      #
      #ax.plot(self.L, N_m0_suboptimal, ls='--', c=plt.cm.winter(0.), label=r'm=0 sub')
      ax.plot(self.L, N_m0, c=plt.cm.winter(0.), label=r'm=0')
      #
      ax.plot(self.L, 1./(1./N_m0+1./N_m2+1./N_m4+1./N_m6+1./N_m8), c='k', ls='--', label=r'm=0-8')
      #
      ax.plot(self.L, N_qe, 'k', label=r'QE')
      #ax.plot(self.L, self.fN_k_TT_nomonopole(self.L), c='g', label=r'no monopole')
      #ax.plot(self.L, self.fN_k_TT_test(self.L), 'b--', label=r'QE test')
      #
      if fPkappa is None:
         # read C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         ax.plot(L, L**4 / 4. * Pphi, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa * 4./(2.*np.pi), 'k-', lw=3, label=r'signal')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$C_L^\kappa$', fontsize=24)

      plt.show()



   ###############################################################################
   ###############################################################################
   # Primary multiplicative bias from lensed foregrounds
   
   
   def f_TT_fg(self, l1, l2, phi, fCfg):
      '''fCfg: unlensed foreground power spectrum
      '''
      result = fCfg(l1) * l1*(l1 + l2*np.cos(phi))
      result += fCfg(l2) * l2*(l2 + l1*np.cos(phi))
      return result


   def A_TT_fg(self, L, fCfg):
      """Multiplicative bias m from lensed foreground,
      such that:
      <phi_QE> = m * phi_foreground.
      m = N^{0, phi} * (integral similar to N^0, but with one Cfg instead of C0)
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(x):
         theta = x[1]
         l1 = np.exp(x[0])
         l2 = self.l2(L, l1, theta)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         phi = self.phi(L, l1, theta)
         result = self.f_TT_fg(l1, l2, phi, fCfg) * self.F_TT(l1, l2, phi)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.A_TT.__func__, "integ"):
         self.A_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.A_TT.integ(integrand, nitn=8, neval=1000)

      result = self.A_TT.integ(integrand, nitn=1, neval=5000)
#      result = self.A_TT.integ(integrand, nitn=8, neval=5000)
#      result = self.A_TT.integ(integrand, nitn=4, neval=1000)

      result = result.mean

      # multiply by N^{0 phi}, to get dimensionless multiplicative bias
      result *= self.fN_phi_TT(L)
      
      if not np.isfinite(result):
         result = 0.
      return result


   def save_foreground_multiplicative_bias(self):
      path = self.directory+"/foreground_multiplicative_bias.txt"
      data = np.zeros((self.Nl, 6))
      data[:,0] = self.L.copy()
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      print "lensed CMB: test"
      f = lambda l: self.A_TT_fg(l, self.CMB.funlensedTT)
      data[:,1] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "lensed CIB: multiplicative bias"
      f = lambda l: self.A_TT_fg(l, self.CMB.fCIB)
      data[:,2] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "lensed kSZ: multiplicative bias"
      f = lambda l: self.A_TT_fg(l, self.CMB.fkSZ)
      data[:,3] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "lensed tSZ: multiplicative bias"
      f = lambda l: self.A_TT_fg(l, self.CMB.ftSZ)
      data[:,4] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)
      print "lensed radio PS: multiplicative bias"
      f = lambda l: self.A_TT_fg(l, self.CMB.fradioPoisson)
      data[:,5] = np.array(pool.map(f, self.L))
      np.savetxt(path, data)


   def load_foreground_multiplicative_bias(self):
      path = self.directory+"/foreground_multiplicative_bias.txt"
      data = np.genfromtxt(path)
      self.fm_CMB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.fm_CIB = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.fm_kSZ = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=False, fill_value=0.)
      self.fm_tSZ = interp1d(data[:,0], data[:,4], kind='linear', bounds_error=False, fill_value=0.)
      self.fm_radioPS = interp1d(data[:,0], data[:,5], kind='linear', bounds_error=False, fill_value=0.)


   def plotForegroundMultiplicativeBias(self):
      mCMB = self.fm_CMB(self.L)
      mCIB = self.fm_CIB(self.L)
      mkSZ = self.fm_kSZ(self.L)
      mtSZ = self.fm_tSZ(self.L)
      mradioPS = self.fm_radioPS(self.L)
      
      cCmb = 'k'
      cKsz = 'r'
      cTsz = 'b'
      cCib = 'g'
      cCibTsz = 'm'
      cRadiops = 'y'
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.loglog(self.L, mCMB, c=cCmb, label=r'CMB (test)')
      ax.loglog(self.L, mCIB, c=cCib, label=r'CIB')
      ax.loglog(self.L, -mCIB, c=cCib, ls='--')
      ax.loglog(self.L, mkSZ, c=cKsz, label=r'kSZ')
      ax.loglog(self.L, -mkSZ, c=cKsz, ls='--')
      ax.loglog(self.L, mtSZ, c=cTsz, label=r'tSZ')
      ax.loglog(self.L, -mtSZ, c=cTsz, ls='--')
      ax.loglog(self.L, mradioPS, c=cRadiops, label=r'radioPS')
      ax.loglog(self.L, -mradioPS, c=cRadiops, ls='--')
      #
      ax.legend(loc=1)
      #ax.set_ylim((1.e-4, 1.))
      ax.set_xlabel(r'$L$')
      ax.set_ylabel(r'Lensing multiplicative bias')

      plt.show()


   ###############################################################################
   ###############################################################################
   # Secondary multiplicative bias from lensed foregrounds


   def F_TT_cart(self, l1, l2):
      """Weight function for the numerator of the quadratic estimator for phi.
      l1, l2 are 2d vectors.
      Quantity is symmetric in l1,l2, as expected.
      Enforces the lmin, lmax boundaries.
      """
      l1norm = np.sqrt(np.sum(l1**2))
      l2norm = np.sqrt(np.sum(l2**2))
      if (l1norm<self.CMB.lMin) or (l1norm>self.CMB.lMaxT) or (l2norm<self.CMB.lMin) or (l2norm>self.CMB.lMaxT):
         result = 0.
      else:
         L = l1 + l2
         result = self.CMB.funlensedTT(l1norm) * np.dot(L, l1)
         result += self.CMB.funlensedTT(l2norm) * np.dot(L, l2)
         result /= self.CMB.ftotalTT(l1norm)
         result /= self.CMB.ftotalTT(l2norm)
         result /= 2.
      return result


   def alpha_cart(self, l1, l2):
      """Coupling between unlensed map and convergence map,
      producing the first order lensed map.
      l1, l2 are 2d vectors.
      Quantity is not symmetric in l1,l2, as expected.
      No lmin, lmax boundaries to enforce.
      """
      l1norm = np.sqrt(np.sum(l1**2))
      if l1norm==0.:
         result = 0.
      else:
         result = -2. * np.dot(l1, l2) / l1norm**2
      return result


   def secondaryLensedForegroundBias(self, fCf, fCkkf, L0):
      """Computes the secondary lensed foreground bias to CMB lensing, at multipole L0.
      Gives the bias on C_ell^kappa_CMB.
      Assumes that the QE uses the unlensed CMB power in the lensing response (numerator).
      fCf: unlensed foreground power spectrum
      fCkkf: cross-power spectrum of kappa_CMB and kappa_foreground
      """
      # Make L0 a 2d vector, along x-axis
      L0 = np.array([L0, 0.])

      def integrand(pars):
         lnLnorm = pars[0]
         thetaL = pars[1]
         lnlnorm = pars[2]
         thetal = pars[3]
         L = np.array([np.exp(lnLnorm) * np.cos(thetaL), np.exp(lnLnorm) * np.sin(thetaL)])
         l = np.array([np.exp(lnlnorm) * np.cos(thetal), np.exp(lnlnorm) * np.sin(thetal)])

         # factors in common between term 1 and term 2
         result = self.F_TT_cart(l, L0-l)
         result *= self.F_TT_cart(l-L-L0, L-l)
         result *= self.alpha_cart(L, l-L)

         # Term 1: L0
         term1 = self.alpha_cart(-L, l-L0)
         term1 *= self.CMB.funlensedTT(np.sqrt(np.sum((L0-l)**2)))
         term1 *= fCf(np.sqrt(np.sum((l-L)**2)))
         term1 *= fCkkf(np.sqrt(np.sum(L**2)))

         # Term 2: L0
         term2 = self.alpha_cart(-L, L0+L-l)
         term2 *= self.CMB.funlensedTT(np.sqrt(np.sum((l-L)**2)))
         term2 *= fCf(np.sqrt(np.sum((l-L-L0)**2)))
         term2 *= fCkkf(np.sqrt(np.sum(L**2)))

         # factor from symmetries
         result *= 8. * (term1 + term2)
         # factors of pi from Fourier convention
         result /= (2.*np.pi)**4
         # normalize properly for the phi quadratic estimator squared
         result *= self.fN_phi_TT(L0[0])**2
         # convert from phi squared to kappa squared
         result *= (-L0[0]**2/2.)**2
         # jacobian for polar coordinates, with log norm
         result *= np.exp(lnLnorm)**2
         result *= np.exp(lnlnorm)**2
         # compensate for halving the integration domain
         result *= 2.
         return result

      # if first time, initialize integrator
      if not hasattr(self.secondaryLensedForegroundBias.__func__, "integ"):
         print "wah"
         self.secondaryLensedForegroundBias.__func__.integ = vegas.Integrator([[np.log(1.), np.log(2.*self.CMB.lMaxT)], # L-l is limited: goes into QE
                                                                              [0., np.pi],   # keep only half the domain (symmetry)
                                                                              [np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)],   # l is limited: goes into QE
                                                                              [0., 2.*np.pi]])

         self.secondaryLensedForegroundBias.integ(integrand, nitn=10, neval=1e6) # 7h33m run: 20, 1e5
      print "hoohoo"
      result = self.secondaryLensedForegroundBias.integ(integrand, nitn=30, neval=6e6) # 7h33m run: 60, 5e5
      print "weehee"

#         self.secondaryLensedForegroundBias.integ(integrand, nitn=1, neval=1e4) # 7h33m run: 20, 1e5
#      print "hoohoo"
#      result = self.secondaryLensedForegroundBias.integ(integrand, nitn=5, neval=6e4) # 7h33m run: 60, 5e5
#      print "weehee"


      print L0[0], result.sdev/result.mean
      print result.summary()
      return result.mean, result.sdev








