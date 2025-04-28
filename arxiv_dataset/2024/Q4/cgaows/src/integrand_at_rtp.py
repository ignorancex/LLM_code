import numpy as np
import scipy as sp

def evaluate_switching_func(r, rc, cusped_eval=True):
       """ Evaluate the switching function and its derivatives.

       The switching function is defined as:
          b(r) = 1 - 10 (r/rc)^3 + 15 (r/rc)^4 - 6 (r/rc)^5
       This function is one at r = 0, zero at r = rc, and
       has zero 1st and 2nd derivatives at r = 0 and r = rc.

       params:
                  r - A numpy array of the radii values to evaluate at.
                 rc - The cutoff radius for the switching function.
        cusped_eval - If True, the switching function is evaluated, else set to 0
       
       return:
           Three numpy arrays:
               - The array of switching function values at the given radii.
               - The array of 1st derivatives with respect to r.
               - The array of 2nd derivatives with respect to r.
       """
       if cusped_eval == False:
         return np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
       else:
         s = r / rc
         dsdr = 1.0 / rc
         s2 = s * s
         s3 = s2 * s
         s4 = s2 * s2
         s5 = s2 * s3
         b = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
         dbds = -30.0 * s2 + 60.0 * s3 - 30.0 * s4
         d2bds2 = -60.0 * s + 180.0 * s2 - 120.0 * s3
         dbdr = dbds * dsdr
         d2bdr2 = d2bds2 * dsdr * dsdr # note that d2sdr2 is zero
         return b, dbdr, d2bdr2

def evaluate_slater_func(r, a0, alpha):
       """ Evaluate the slater-type function and its derivatives.
       params:
                  r - A numpy array of the radii values to evaluate at.
                 a0 - The out-front coefficient of the slater function.
              alpha - The exponent of the slater function.
       """
       f = a0 * np.exp(-alpha * r)
       dfdr = -alpha * f
       d2fdr2 = -alpha * dfdr
       return f, dfdr, d2fdr2

def evaluate_gaussian(x, y, z, a):
       """ Evaluate a gaussian function.
       params:
           x - A numpy array of the x values to evaluate at.
           y - A numpy array of the y values to evaluate at.
           z - A numpy array of the z values to evaluate at.
           a - The exponent of the gaussian function.
       return:
           A numpy array of the values of the gaussian function at the given x, y, and z values.
       """
       r2 = x * x + y * y + z * z
       return np.exp(-a * r2)

def evaluate_s_gaussian_and_derivs(x, y, z, a):
       """ Evaluate a gaussian function and its xyz 1st and 2nd derivatives.
       params:
           x - A numpy array of the x values to evaluate at.
           y - A numpy array of the y values to evaluate at.
           z - A numpy array of the z values to evaluate at.
           a - The exponent of the gaussian function.
       return:
           Three numpy arrays:
               - The array of gaussian function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
       """
       x2 = x * x
       y2 = y * y
       z2 = z * z
       r2 = x2 + y2 + z2
       r = np.sqrt(r2)
       r3 = r2 * r
       N = (2.0 * a / np.pi)**0.75
       f = np.exp(-a * r2)
       two_ar = 2.0 * a * r
       dfdr = -two_ar * f
       d2fdr2 = -2.0 * a * f - two_ar * dfdr
       drdx = x / r
       drdy = y / r
       drdz = z / r
       d2rdx2 = ( y2 + z2 ) / r3
       d2rdy2 = ( x2 + z2 ) / r3
       d2rdz2 = ( x2 + y2 ) / r3
       d2rdxdy = -x * y / r3
       d2rdxdz = -x * z / r3
       d2rdydz = -y * z / r3
       dfdxyz = np.zeros([3, x.size])
       dfdxyz[0,:] = dfdr * drdx
       dfdxyz[1,:] = dfdr * drdy
       dfdxyz[2,:] = dfdr * drdz
       d2fdxyz2 = np.zeros([3, 3, x.size])
       d2fdxyz2[0,0,:] = dfdr * d2rdx2  + d2fdr2 * drdx * drdx
       d2fdxyz2[1,1,:] = dfdr * d2rdy2  + d2fdr2 * drdy * drdy
       d2fdxyz2[2,2,:] = dfdr * d2rdz2  + d2fdr2 * drdz * drdz
       d2fdxyz2[0,1,:] = dfdr * d2rdxdy + d2fdr2 * drdx * drdy
       d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]
       d2fdxyz2[0,2,:] = dfdr * d2rdxdz + d2fdr2 * drdx * drdz
       d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]
       d2fdxyz2[1,2,:] = dfdr * d2rdydz + d2fdr2 * drdy * drdz
       d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]
       f = f * N
       dfdxyz = dfdxyz * N
       d2fdxyz2 = d2fdxyz2 * N
       return f, dfdxyz, d2fdxyz2

def evaluate_p_gaussian_and_derivs(x, y, z, a, l):
       """ Evaluate a p-type gaussian function and its xyz 1st and 2nd derivatives.
       params:
           x - A numpy array of the x values to evaluate at.
           y - A numpy array of the y values to evaluate at.
           z - A numpy array of the z values to evaluate at.
           a - The exponent of the gaussian function.
           l - 2px, 2py, or 2pz
       return:
           Three numpy arrays:
               - The array of gaussian function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
       """
       x2 = x * x
       y2 = y * y
       z2 = z * z
       x3 = x * x * x
       y3 = y * y * y
       z3 = z * z * z
       r2 = x2 + y2 + z2
       r = np.sqrt(r2)
       r3 = r2 * r
       a2 = a * a
       N = 2.0 * a**0.5 * (2.0 * a / np.pi)**0.75
       g = np.exp(-a * r2)

       dfdxyz = np.zeros([3, x.size])
       d2fdxyz2 = np.zeros([3, 3, x.size])
       if l == 0 or l == 1:
               raise RuntimeError('l must be equal to 2, 3, or 4 for the p-type Gaussian function')
       if l == 2:
               f = x * g       
               dfdxyz[0,:] = g * (1.0 - 2.0 * a * x2)                    
               dfdxyz[1,:] = -2.0 * a * x * y * g                        
               dfdxyz[2,:] = -2.0 * a * x * z * g                        
               d2fdxyz2[0,0,:] = g * (-6.0 * a * x + 4.0 * a2 * x3)      
               d2fdxyz2[1,1,:] = 2.0 * a * x * g * (2.0 * a * y2 - 1.0)
               d2fdxyz2[2,2,:] = 2.0 * a * x * g * (2.0 * a * z2 - 1.0)
               d2fdxyz2[0,1,:] = -2.0 * a * y * g * (1.0 - 2.0 * a * x2) 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]                         
               d2fdxyz2[0,2,:] = -2.0 * a * z * g * (1.0 - 2.0 * a * x2) 
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]                         
               d2fdxyz2[1,2,:] = 4.0 * a2 * x * y * z * g
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]
       if l == 3:
               f = y * g       
               dfdxyz[0,:] = -2.0 * a * y * x * g                       
               dfdxyz[1,:] = g * (1.0 - 2.0 * a * y2)                   
               dfdxyz[2,:] = -2.0 * a * y * z * g                       
               d2fdxyz2[0,0,:] = 2.0 * a * y * g * (2.0 * a * x2 - 1.0)
               d2fdxyz2[1,1,:] = g * (-6.0 * a * y + 4.0 * a2 * y3)     
               d2fdxyz2[2,2,:] = 2.0 * a * y * g * (2.0 * a * z2 - 1.0)
               d2fdxyz2[0,1,:] = -2.0 * a * x * g * (1.0 - 2.0 * a * y2) 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]                         
               d2fdxyz2[0,2,:] = 4.0 * a2 * x * y * z * g 
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]
               d2fdxyz2[1,2,:] = -2.0 * a * z * g * (1.0 - 2.0 * a * y2) 
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]                         
       if l == 4:
               f = z * g       
               dfdxyz[0,:] = -2.0 * a * z * x * g
               dfdxyz[1,:] = -2.0 * a * z * y * g
               dfdxyz[2,:] = g * (1.0 - 2.0 * a * z2) 
               d2fdxyz2[0,0,:] = 2.0 * a * z * g * (2.0 * a * x2 - 1.0)
               d2fdxyz2[1,1,:] = 2.0 * a * z * g * (2.0 * a * y2 - 1.0)
               d2fdxyz2[2,2,:] = g * (-6.0 * a * z + 4.0 * a2 * z3)
               d2fdxyz2[0,1,:] = 4.0 * a2 * x * y * z * g 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:] 
               d2fdxyz2[0,2,:] = -2.0 * a * x * g * (1.0 - 2.0 * a * z2) 
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]                         
               d2fdxyz2[1,2,:] = -2.0 * a * y * g * (1.0 - 2.0 * a * z2) 
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]                         
       f = f * N
       dfdxyz = dfdxyz * N
       d2fdxyz2 = d2fdxyz2 * N
       return f, dfdxyz, d2fdxyz2

def evaluate_d_gaussian_and_derivs(x, y, z, a, l):
       """ Evaluate a p-type gaussian function and its xyz 1st and 2nd derivatives.
       params:
           x - A numpy array of the x values to evaluate at.
           y - A numpy array of the y values to evaluate at.
           z - A numpy array of the z values to evaluate at.
           a - The exponent of the gaussian function.
           l - 5=3xy, 6=yz, 7=z^2, 8=xz, 9=x^2-y^2
       return:
           Three numpy arrays:
               - The array of gaussian function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
       """
       x2 = x * x
       y2 = y * y
       z2 = z * z
       x3 = x * x * x
       y3 = y * y * y
       z3 = z * z * z
       x4 = x * x * x * x
       y4 = y * y * y * y
       z4 = z * z * z * z
       r2 = x2 + y2 + z2
       r = np.sqrt(r2)
       r3 = r2 * r
       a2 = a * a
       N = (2048.0 * a**7.0 / np.pi**3.0)**0.25 
       #N = (32.0 * a**7.0 / np.pi**3.0)**0.25 
       g = np.exp(-a * r2)

       dfdxyz = np.zeros([3, x.size])
       d2fdxyz2 = np.zeros([3, 3, x.size])
       if l < 5:
               raise RuntimeError('l must be equal to 5, 6, 7, 8, or 9 for the d-type Gaussian function')
       if l == 5:
               f = x * y * g       
               dfdxyz[0,:] = y * (1.0 - 2.0 * a * x2) * g # good 
               dfdxyz[1,:] = x * (1.0 - 2.0 * a * y2) * g # good
               dfdxyz[2,:] = -2.0 * a * x * y * z * g     # good
               d2fdxyz2[0,0,:] = 2.0 * a * x * y * (2.0 * a * x2 - 3.0) * g #good 
               d2fdxyz2[1,1,:] = 2.0 * a * x * y * (2.0 * a * y2 - 3.0) * g #good
               d2fdxyz2[2,2,:] = 2.0 * a * x * y * (2.0 * a * z2 - 1.0) * g #good 
               d2fdxyz2[0,1,:] = (2.0 * a * x2 - 1.0) * (2.0 * a * y2 - 1.0) * g #good 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:] 				 #good
               d2fdxyz2[0,2,:] = 2.0 * a * y * z * (2.0 * a * x2 - 1.0) * g      #good
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]			         #good
               d2fdxyz2[1,2,:] = 2.0 * a * x * z * (2.0 * a * y2 - 1.0) * g      #good
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]			         #good
       if l == 6:
               f = y * z * g       
               dfdxyz[0,:] = -2.0 * a * x * y * z * g	  # good
               dfdxyz[1,:] = z * (1.0 - 2.0 * a * y2) * g # good
               dfdxyz[2,:] = y * (1.0 - 2.0 * a * z2) * g # good
               d2fdxyz2[0,0,:] = 2.0 * a * y * z * (2.0 * a * x2 - 1.0) * g #good
               d2fdxyz2[1,1,:] = 2.0 * a * y * z * (2.0 * a * y2 - 3.0) * g #good
               d2fdxyz2[2,2,:] = 2.0 * a * y * z * (2.0 * a * z2 - 3.0) * g #good
               d2fdxyz2[0,1,:] = 2.0 * a * x * z * (2.0 * a * y2 - 1.0) * g #good 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]			    #good
               d2fdxyz2[0,2,:] = 2.0 * a * x * y * (2.0 * a * z2 - 1.0) * g #good
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]			    #good
               d2fdxyz2[1,2,:] = (2.0 * a * y2 - 1.0) * (2.0 * a * z2 - 1.0) * g #good
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]			         #good
       if l == 7:
               M = 1.0 / 144.0**0.25
               f = M * (3.0 * z2 - r2) * g       		#good
               dfdxyz[0,:] = M * 2.0 * x * (a * x2 + a * y2 - 2.0 * a * z2 - 1.0) * g #good
               dfdxyz[1,:] = M * 2.0 * y * (a * x2 + a * y2 - 2.0 * a * z2 - 1.0) * g #good
               dfdxyz[2,:] = M * -2.0 * z * (-a * x2 - a * y2 + 2.0 * a * z2 - 2.0) * g   #good
               d2fdxyz2[0,0,:] = M * -2.0 * (2.0 * a2 * x4 + a * x2 * (2.0 * a * y2 - 4.0 * a * z2 - 5.0) - a * y2 + 2.0 * a * z2 + 1.0) * g #good
               d2fdxyz2[1,1,:] = M * -2.0 * (2.0 * a2 * y4 + a * y2 * (2.0 * a * x2 - 4.0 * a * z2 - 5.0) - a * x2 + 2.0 * a * z2 + 1.0) * g #good
               d2fdxyz2[2,2,:] = M * -2.0 * ((2.0 * a2 * z2 - a) * (x2 + y2) - 4.0 * a2 * z4 + 10.0 * a * z2 - 2.0) * g #good
               d2fdxyz2[0,1,:] = M * -4.0 * a * x * y * (a * x2 + a * y2 - 2.0 * a * z2 - 2.0) * g #good 
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]					           #good	
               d2fdxyz2[0,2,:] = M * -4.0 * a * x * z * (a * x2 + a * y2 - 2.0 * a * z2 + 1.0) * g #good
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]                                                   #good
               d2fdxyz2[1,2,:] = M * -4.0 * a * y * z * (a * x2 + a * y2 - 2.0 * a * z2 + 1.0) * g #good
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]                                                   #good
       if l == 8:
               f = x * z * g       
               dfdxyz[0,:] = z * (1.0 - 2.0 * a * x2) * g #good
               dfdxyz[1,:] = -2.0 * a * x * y * z * g	  #good
               dfdxyz[2,:] = x * (1.0 - 2.0 * a * z2) * g #good
               d2fdxyz2[0,0,:] = 2.0 * a * x * z * (2.0 * a * x2 - 3.0) * g #good 
               d2fdxyz2[1,1,:] = 2.0 * a * x * z * (2.0 * a * y2 - 1.0) * g #good
               d2fdxyz2[2,2,:] = 2.0 * a * x * z * (2.0 * a * z2 - 3.0) * g #good
               d2fdxyz2[0,1,:] = 2.0 * a * y * z * (2.0 * a * x2 - 1.0) * g #good
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]                            #good
               d2fdxyz2[0,2,:] = (2.0 * a * x2 - 1.0) * (2.0 * a * z2 - 1.0) * g #good
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]                                 #good
               d2fdxyz2[1,2,:] = 2.0 * a * x * y * (2.0 * a * z2 - 1.0) * g #good
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]                            #good
       if l == 9:
               M = 0.5
               f = M * (x2 - y2) * g       
               dfdxyz[0,:] = M * -2.0 * x * (a * x2 - a * y2 - 1.0) * g #good
               dfdxyz[1,:] = M * 2.0 * y * (a * y2 - a * x2 - 1.0) * g  #good
               dfdxyz[2,:] = M * -2.0 * a * z * (x2 - y2) * g           #good
               d2fdxyz2[0,0,:] = M * 2.0 * (2.0 * a2 * x4 - a * x2 * (2.0 * a * y2 + 5.0) + a * y2 + 1.0) * g #good
               d2fdxyz2[1,1,:] = M * 2.0 * (x2 * (2.0 * a2 * y2 - a) - 2.0 * a2 * y4 + 5.0 * a * y2 - 1.0) * g #good
               d2fdxyz2[2,2,:] = M * 2.0 * a * (2.0 * a * z * z - 1.0) * (x * x - y * y) * g                   #good
               d2fdxyz2[0,1,:] = M * 4.0 * a2 * x * y * (x2 - y2) * g #good
               d2fdxyz2[1,0,:] = d2fdxyz2[0,1,:]                      #good
               d2fdxyz2[0,2,:] = M * 4.0 * a * x * z * (a * x2 - a * y2 - 1.0) * g #good
               d2fdxyz2[2,0,:] = d2fdxyz2[0,2,:]                                   #good
               d2fdxyz2[1,2,:] = M * -4.0 * a * y * z * (a * y2 - a * x2 - 1.0) * g #good
               d2fdxyz2[2,1,:] = d2fdxyz2[1,2,:]                                   #good
       f = f * N
       dfdxyz = dfdxyz * N
       d2fdxyz2 = d2fdxyz2 * N
       return f, dfdxyz, d2fdxyz2

def evaluate_s_type_gaussian_sum_and_derivs(x, y, z, coeffs, exponents, coeffs_core, exponents_core, proj, orthogonalized):
       """ Evaluate a linear combination of gaussian functions and its xyz 1st and 2nd derivatives.

       params:
                x - A numpy array of the x values to evaluate at.
                y - A numpy array of the y values to evaluate at.
                z - A numpy array of the z values to evaluate at.
           coeffs - The coefficients of the gaussian functions.
        exponents - The exponents of the gaussian functions.
      coeffs_core - The coefficients of the gaussian functions for the core orbital
   exponents_core - The exponents of the gaussian functions for the corbital
             proj - The projection for the orthogonalzied orbital

       return:
           Three numpy arrays:
               - The array of function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
 
       TO-DO: ADD LOGIC FOR ORTHOGONALIZED ORBS
       
       """
       if len(coeffs) != len(exponents):
               raise ValueError("The number of coefficients and exponents must match.")
       #print("orthogonalized: ", orthogonalized, flush=True) 
       f = 0.0
       dfdxyz = np.zeros([3, x.size])
       d2fdxyz2 = np.zeros([3, 3, x.size])
       for i in range(len(coeffs)):
               g, dgdxyz, d2gdxyz2 = evaluate_s_gaussian_and_derivs(x, y, z, exponents[i])
               f = f + coeffs[i] * g
               dfdxyz = dfdxyz + coeffs[i] * dgdxyz
               d2fdxyz2 = d2fdxyz2 + coeffs[i] * d2gdxyz2
       if orthogonalized ==True:
               #print("made it into ortho for loop", flush=True)
               f_core = 0.0
               dfdxyz_core = np.zeros([3, x.size])
               d2fdxyz2_core = np.zeros([3, 3, x.size])
               for i in range(len(coeffs_core)):
                       g_core, dgdxyz_core, d2gdxyz2_core = evaluate_s_gaussian_and_derivs(x, y, z, exponents_core[i])
                       f_core = f_core + coeffs_core[i] * g_core
                       dfdxyz_core = dfdxyz_core + coeffs_core[i] * dgdxyz_core
                       d2fdxyz2_core = d2fdxyz2_core + coeffs_core[i] * d2gdxyz2_core
               f = f - proj * f_core
               dfdxyz = dfdxyz - proj * dfdxyz_core
               d2fdxyz2 = d2fdxyz2 - proj * d2fdxyz2_core
       return f, dfdxyz, d2fdxyz2

def evaluate_p_type_gaussian_sum_and_derivs(x, y, z, coeffs, exponents, l):
       """ Evaluate a linear combination of gaussian functions and its xyz 1st and 2nd derivatives.

       params:
                x - A numpy array of the x values to evaluate at.
                y - A numpy array of the y values to evaluate at.
                z - A numpy array of the z values to evaluate at.
           coeffs - The coefficients of the gaussian functions.
        exponents - The exponents of the gaussian functions.
                l - 2px, 2py, or 2pz

       return:
           Three numpy arrays:
               - The array of function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
 
       TO-DO: ADD LOGIC FOR ORTHOGONALIZED ORBS
       
       """
       if len(coeffs) != len(exponents):
               raise ValueError("The number of coefficients and exponents must match.")
       f = 0.0
       dfdxyz = np.zeros([3, x.size])
       d2fdxyz2 = np.zeros([3, 3, x.size])
       for i in range(len(coeffs)):
               g, dgdxyz, d2gdxyz2 = evaluate_p_gaussian_and_derivs(x, y, z, exponents[i], l)
               f = f + coeffs[i] * g
               dfdxyz = dfdxyz + coeffs[i] * dgdxyz
               d2fdxyz2 = d2fdxyz2 + coeffs[i] * d2gdxyz2
       return f, dfdxyz, d2fdxyz2

def evaluate_d_type_gaussian_sum_and_derivs(x, y, z, coeffs, exponents, l):
       """ Evaluate a linear combination of gaussian functions and its xyz 1st and 2nd derivatives.

       params:
                x - A numpy array of the x values to evaluate at.
                y - A numpy array of the y values to evaluate at.
                z - A numpy array of the z values to evaluate at.
           coeffs - The coefficients of the gaussian functions.
        exponents - The exponents of the gaussian functions.
                l - 5=3xy, 6=yz, 7=z^2, 8=xz, 9=x^2-y^2

       return:
           Three numpy arrays:
               - The array of function values at the given x, y, and z values.
               - A 3 x n array holding the 1st derivatives.
               - A 3 x 3 x n array holding the 2nd derivatives.
 
       TO-DO: ADD LOGIC FOR ORTHOGONALIZED ORBS
       
       """
       if len(coeffs) != len(exponents):
               raise ValueError("The number of coefficients and exponents must match.")
       f = 0.0
       dfdxyz = np.zeros([3, x.size])
       d2fdxyz2 = np.zeros([3, 3, x.size])
       for i in range(len(coeffs)):
               g, dgdxyz, d2gdxyz2 = evaluate_d_gaussian_and_derivs(x, y, z, exponents[i], l)
               f = f + coeffs[i] * g
               dfdxyz = dfdxyz + coeffs[i] * dgdxyz
               d2fdxyz2 = d2fdxyz2 + coeffs[i] * d2gdxyz2
       return f, dfdxyz, d2fdxyz2

def integrand_at_rtp_general(r, theta, phi, Z, vfoc, evlb, evls, evlg, n, m, rc, full_eN_pot, int_type):
       """ Evaluate r^2 sin(theta) ( b s + (1-b) f ) ( -Z/r - (1/2) nabla ) ( b s + (1-b) g )
       params:
           r - A numpy array of the radii values to evaluate at.
       theta - The angle away from the z axis to evaluate at.
         phi - The angle away from the x axis within the xy plane.
           Z - The charge of the nucleus that sits at our origin
               of integration and is zeta, the exponent of our 
               slater functions for s cusps.
       vfoc - The "vector from other center", a length-3 numpy array
              holding the x, y, and z components of the vector that
              points from the other center to our integration origin.
       evlb - An object that "evaluates b".  Given r, it should
              return three arrays that hold the values and first and
              second derivatives of the switching function b(r)
              at the various values within r.  Specifically, it
              returns b(r), db/dr, and d^2b/dr^2, returns 0 arrays 
              for vanilla gaussian.
       evls - An object that "evaluates s" at each value within r,
              where s(r) is the slater-type function centered on
              our origin of integration.  evls(r) should return
              three arrays: s(r), ds/dr, and d^2s/dr^2.
       evlg - An object that "evaluates g" and its derivatives at
              each value within r, where g(xoc, yoc, zoc) is the
              gaussian-based basis function centered on the
              other center.  xoc, yoc, and zoc are the x, y, and z
              parts of the vector pointing from the other center
              to the position given by r, theta, and phi.
              evlg(xoc, yoc, zoc) should return three arrays:
                - An    [nr] sized array of the values of g.
                - A   [3,nr] sized array of the values of the 1st derivatives of g.
                - A [3,3,nr] sized array of the values of the 2nd derivatives of g.
              nr here is the number of values within r.
          n - The power of (r/rc) used in the bra.
          m - The power of (r/rc) used in the ket.
         rc - The radius of the sphere we are integrating inside.
full_eN_pot - None if elec-nuclear potential for only the cusped nuclei, 
              length 2 list for total elec-nuclear energy containing:
                [charge of all nuclei array of length num_nuc, 
                 num_nuc x 3 numpy array holding the x, y, and z components of the 
                 vector that points from each nuclear center to our integration origin].
   int_type - Which type of integral we are evaluating (0 = < bQr^n | O | bQr^m > , 1 = < (1-b)X | O | bQr^m >, 2 = < (1-b)X | O | (1-b)X > ) 
       """

       # First, get sin and cos values for theta and phi
       sint = np.sin(theta)
       cost = np.cos(theta)
       sinp = np.sin(phi)
       cosp = np.cos(phi)

       # evaluate some useful intermediates
       r2 = r * r
       r2sint = r2 * sint
       rsint = r * sint

       # First, get the x, y, and z parts of the vectors that point
       # from the other center to our positions.
       #print("Length of vfoc: ", len(vfoc), flush=True)
       xoc = vfoc[0] + r * sint * cosp
       yoc = vfoc[1] + r * sint * sinp
       zoc = vfoc[2] + r * cost

       # Next, get the derivatives of these that we will need.
       # Note that the 2nd derivatives w.r.t. r are zero.
       # Note that all derivatives of zoc w.r.t. phi are zero.
       dxocdr = sint * cosp
       dyocdr = sint * sinp
       dzocdr = cost
       dxocdt =  r * cost * cosp
       dyocdt =  r * cost * sinp
       dzocdt = -r * sint
       d2xocdt2 = -r * sint * cosp
       d2yocdt2 = -r * sint * sinp
       d2zocdt2 = -r * cost
       
       # Derivatives w.r.t. phi require special handling to avoid division by zero.
       # Below, we will need (dxocdp)^2 / sint and d2xocdp2 / sint  and the analogous terms for yoc.
       dxyoc_dp_products_over_sint = np.zeros([2, 2, r.size])
       dxyoc_dp_products_over_sint[0,0,:] =  r2 * sint * sinp * sinp  # dxocdp = -r * sint * sinp
       dxyoc_dp_products_over_sint[1,1,:] =  r2 * sint * cosp * cosp  # dyocdp =  r * sint * cosp
       dxyoc_dp_products_over_sint[0,1,:] = -r2 * sint * cosp * sinp
       dxyoc_dp_products_over_sint[1,0,:] = dxyoc_dp_products_over_sint[0,1,:]
       d2xyoc_dp2_over_sint = np.zeros([2, r.size])
       d2xyoc_dp2_over_sint[0,:] = -r * cosp  # d2xocdp2 = -r * sint * cosp
       d2xyoc_dp2_over_sint[1,:] = -r * sinp  # d2yocdp2 = -r * sint * sinp

       # evaluate the values and derivatives of the switching function
       b, dbdr, d2bdr2 = evlb(r)
       one_minus_b = 1.0 - b

       # evaluate the values and derivatives of the slater function
       s, dsdr, d2sdr2 = evls(r)

       # Evaluate the values and xzy derivatives of the gaussian-based function.
       g, dg, d2g = evlg(xoc, yoc, zoc)

       # evaluate the powers of r/rc we need and their derivatives
       #if type(m) != int or type(n) != int:
       #       raise TypeError("We are assuming that n and m are integers.")
       if m == 1 or n == 1:
              raise ValueError("We are assuming that n and m are not 1.")
       r_over_rc = r / rc
       bra_rrc = 1.0
       if n != 0:
              bra_rrc = np.power(r_over_rc, 1.0 * n )
       ket_rrc = 1.0
       if m != 0:
              ket_rrc = np.power(r_over_rc, 1.0 * m )
       d_ket_rrc_dr = 0.0
       if m != 0:
              d_ket_rrc_dr = ( 1.0 * m ) * np.power(r_over_rc, m - 1.0) / rc
       d2_ket_rrc_dr2 = 0.0
       if m == 2:
              d2_ket_rrc_dr2 = 2.0 / ( rc * rc )
       elif m > 2:
              d2_ket_rrc_dr2 = ( m * ( m - 1.0 ) ) * np.power(r_over_rc, m - 2.0) / ( rc * rc )

       # put together the value of the overall bra function
       if int_type == 0:
              bra = b * s * bra_rrc
       elif int_type == 1 or int_type == 2:
              bra = one_minus_b * g
       else: 
              bra = b * s * bra_rrc + one_minus_b * g
             
       # put together the value of the overall ket function
       if int_type == 0 or int_type == 1:
              ket = b * s * ket_rrc
       elif int_type == 2:
              ket = one_minus_b * g
       else:
              ket = b * s * ket_rrc + one_minus_b * g

       # put together the value of the overall bra function
       #bra = b * s * bra_rrc + one_minus_b * g

       # put together the value of the overall ket function
       #ket = b * s * ket_rrc + one_minus_b * g

       # For total elec-nuc potential
       if full_eN_pot != None: 

         z_array = full_eN_pot[0].reshape(-1,1)

         # Get the x, y, and z parts of the vectors that point
         # from all nuclei to our positions w/ dim [num_nuc, n].
         x_all_eN = full_eN_pot[1][:,0].reshape(-1,1) + r * sint * cosp # [num_nuc, 1] + [n, ] 
         y_all_eN = full_eN_pot[1][:,1].reshape(-1,1) + r * sint * sinp
         z_all_eN = full_eN_pot[1][:,2].reshape(-1,1) + r * cost
         #print("x_all_eN dim", x_all_eN.shape)

         r_btw_all_nuc = np.sqrt(x_all_eN*x_all_eN + y_all_eN*y_all_eN + z_all_eN*z_all_eN)  # 
        
         r_cont = (r * r) / r_btw_all_nuc # [n,] / [num_nuc, n] 
         #print("CHECK SHAPES", full_eN_pot[0].shape, full_eN_pot[1].shape, sinp.shape, cost.shape, cosp.shape, x_all_eN.shape, y_all_eN.shape, z_all_eN.shape, bra.shape, z_array.shape, r.shape, r_cont.shape, r_btw_all_nuc.shape, sint.shape, ket.shape, flush=True)
         retval = bra * np.sum( -z_array * r_cont , axis=0) * sint * ket # sum_num_nuc{ [num_nuc, 1] / [num_nuc, n] } = [n, ]
         #print("r_cont shape for all nuc should be [num_nuc by n]: ", r_cont.shape, flush=True)
         #print("all eNpot vs 1nuc eNpot: ", np.sum( -z_array * r_cont , axis=0), "\n", ( -Z * r ), flush=True) #, " r_cont for all nuc and r (one should match)", r_cont, r, flush=True)
       else:
         # Initialize the return value (the integrand) with the nuclear attraction term,
         # remembering to include the volume element so that we don't divide by zero.
         retval = bra * ( -Z * rsint ) * ket

       # Get a 3 x 1 array holding dxocdr, dyocdr, and dzocdr, and also
       # a 3 x 3 x 1 array of the products of dxocdr, dyocdr, and dzocdr
       dxyzoc_dr_list = [dxocdr, dyocdr, dzocdr]
       dxyzoc_dr = np.zeros([3, 1])
       dxyzoc_dr_products = np.zeros([3, 3, 1])
       for i, di in enumerate(dxyzoc_dr_list):
              dxyzoc_dr[i,0] = di
       for i, di in enumerate(dxyzoc_dr_list):
              for j, dj in enumerate(dxyzoc_dr_list):
                     dxyzoc_dr_products[i,j,0] = di * dj

       # Get a 3 x nr array holding dxocdt, dyocdt, and dzocdt, and also
       #     a 3 x nr array holding d2xocdt2, d2yocdt2, and d2zocdt2, and also
       # a 3 x 3 x nr array of the products of dxocdt, dyocdt, and dzocdt
       dxyzoc_dt          = np.zeros([   3, r.size])
       d2xyzoc_dt2        = np.zeros([   3, r.size])
       dxyzoc_dt_products = np.zeros([3, 3, r.size])
       dxyzoc_dt_list = [dxocdt, dyocdt, dzocdt]
       d2xyzoc_dt2_list = [d2xocdt2, d2yocdt2, d2zocdt2]
       for i in range(len(dxyzoc_dt_list)):
              dxyzoc_dt[i,:] = dxyzoc_dt_list[i]
              d2xyzoc_dt2[i,:] = d2xyzoc_dt2_list[i]
              for j in range(len(dxyzoc_dt_list)):
                     dxyzoc_dt_products[i,j,:] = dxyzoc_dt_list[i] * dxyzoc_dt_list[j]

       # Get a 2 x nr array holding d2xocdp2_over_sint and d2yocdp2_over_sint, and also
       # a 2 x 2 x nr array of the products of dxocdt, dyocdt, and dzocdt

       # Do chain rule to get the r derivatives of the ket gaussian-based function.
       # Remember that second derivatives of xoc, yoc, and zoc w.r.t. r are zero.
       dgdr = np.sum(dg * dxyzoc_dr, axis=0)
       d2gdr2 = np.sum(d2g * dxyzoc_dr_products, axis=(0,1))

       # Do chain rule to get the theta derivatives of the ket gaussian-based function.
       dgdt = np.sum(dg * dxyzoc_dt, axis=0)
       d2gdt2 = np.sum(d2g * dxyzoc_dt_products, axis=(0,1)) + np.sum(dg * d2xyzoc_dt2, axis=0)

       # Do chain rule to get the phi derivatives of the ket gaussian-based function.
       # Remember that all phi derivatives of zoc are zero, and we only need the 2nd derivative.
       # Again, we need to think ahead to avoid division by zero, so we will actually
       # compute the the quotient of the 2nd derivative and sint.
       d2gdp2_over_sint = np.sum(d2g[:2,:2,:] * dxyoc_dp_products_over_sint, axis=(0,1)) + np.sum(dg[:2,:] * d2xyoc_dp2_over_sint, axis=0)

       #################################################################################################################
       # Now, the messy business of the laplacian, which when acted on a function F gives:
       #
       #    d2F/dr2 + (2/r) dFdr + (1/r2) d2F/dt2 + (cost / r2 sint) dF/dt + (1 / r2 sin2t ) d2F/dp2
       #
       # Let's first build up r^2 sin(theta) ( - (1/2) nabla ) ( b s (r/rc)^m + (1-b) g ) in the variable nabterm.
       #################################################################################################################

       if int_type == 0 or int_type == 1:
              # Get contribution from d2F/dr2 term in which both r derivatives act on b.
              nabterm = -0.5 * r2sint * d2bdr2 * ( s * ket_rrc )

              # Get contribution from d2F/dr2 term in which only one r derivative acts on b.
              nabterm = nabterm - r2sint * dbdr * ( dsdr * ket_rrc + s * d_ket_rrc_dr )

              # Get contribution from d2F/dr2 term in which no r derivatives act on b.
              nabterm = nabterm - 0.5 * r2sint * ( b * ( d2sdr2 * ket_rrc + 2.0 * dsdr * d_ket_rrc_dr + s * d2_ket_rrc_dr2 ) )

              # Get contribution from (2/r) dF/dr term in which one r derivative acts on b.
              nabterm = nabterm - rsint * dbdr * ( s * ket_rrc )

              # Get contribution from (2/r) dF/dr term in which no r derivatives act on b.
              nabterm = nabterm - rsint * ( b * ( dsdr * ket_rrc + s * d_ket_rrc_dr ) )

              # For the remaining contributions, remember that b, s, and ket_rrc are functions of r only.

       elif int_type == 2:
              # Get contribution from d2F/dr2 term in which both r derivatives act on b.
              nabterm = -0.5 * r2sint * d2bdr2 * ( -1.0 * g )

              # Get contribution from d2F/dr2 term in which only one r derivative acts on b.
              nabterm = nabterm - r2sint * dbdr * ( -1.0 * dgdr )

              # Get contribution from d2F/dr2 term in which no r derivatives act on b.
              nabterm = nabterm - 0.5 * r2sint * ( one_minus_b * d2gdr2 )

              # Get contribution from (2/r) dF/dr term in which one r derivative acts on b.
              nabterm = nabterm - rsint * dbdr * ( -1.0 * g )

              # Get contribution from (2/r) dF/dr term in which no r derivatives act on b.
              nabterm = nabterm - rsint * ( one_minus_b * dgdr )

              # For the remaining contributions, remember that b, s, and ket_rrc are functions of r only.

              # Get contribution from (1/r2) d2F/dt2 term.
              nabterm = nabterm - 0.5 * sint * one_minus_b * d2gdt2

              # Get contribution from (cost / r2 sint) dF/dt term.
              nabterm = nabterm - 0.5 * cost * one_minus_b * dgdt

              # Get contribution from (1 / r2 sin2t ) d2F/dp2 term.
              # This is where thinking ahead to avoid division by zero pays off.
              nabterm = nabterm - 0.5 * one_minus_b * d2gdp2_over_sint
       else:
              # Get contribution from d2F/dr2 term in which both r derivatives act on b.
              nabterm = -0.5 * r2sint * d2bdr2 * ( s * ket_rrc - g )

              # Get contribution from d2F/dr2 term in which only one r derivative acts on b.
              nabterm = nabterm - r2sint * dbdr * ( dsdr * ket_rrc + s * d_ket_rrc_dr - dgdr )

              # Get contribution from d2F/dr2 term in which no r derivatives act on b.
              nabterm = nabterm - 0.5 * r2sint * ( b * ( d2sdr2 * ket_rrc + 2.0 * dsdr * d_ket_rrc_dr + s * d2_ket_rrc_dr2 ) + one_minus_b * d2gdr2 )

              # Get contribution from (2/r) dF/dr term in which one r derivative acts on b.
              nabterm = nabterm - rsint * dbdr * ( s * ket_rrc - g )

              # Get contribution from (2/r) dF/dr term in which no r derivatives act on b.
              nabterm = nabterm - rsint * ( b * ( dsdr * ket_rrc + s * d_ket_rrc_dr ) + one_minus_b * dgdr )

              # For the remaining contributions, remember that b, s, and ket_rrc are functions of r only.

              # Get contribution from (1/r2) d2F/dt2 term.
              nabterm = nabterm - 0.5 * sint * one_minus_b * d2gdt2

              # Get contribution from (cost / r2 sint) dF/dt term.
              nabterm = nabterm - 0.5 * cost * one_minus_b * dgdt

              # Get contribution from (1 / r2 sin2t ) d2F/dp2 term.
              # This is where thinking ahead to avoid division by zero pays off.
              nabterm = nabterm - 0.5 * one_minus_b * d2gdp2_over_sint

       # Multiply nabterm by the bra function and add it to the integrand.
       retval = retval + bra * nabterm

       # Return the integrand, which is an array of values, one for each value within r.
       return retval

def integrand_at_rtp_s(r, theta, phi, Z, vfoc, evlb, evls, evlg, n, m, rc, int_type):
       """ Evaluate r^2 sin(theta) ( b s + (1-b) f ) ( -Z/r - (1/2) nabla ) ( b s + (1-b) g )
       params:
           r - A numpy array of the radii values to evaluate at.
       theta - The angle away from the z axis to evaluate at.
         phi - The angle away from the x axis within the xy plane.
           Z - The charge of the nucleus that sits at our origin
               of integration and is zeta, the exponent of our 
               slater functions for s cusps.
       vfoc - The "vector from other center", a length-3 numpy array
              holding the x, y, and z components of the vector that
              points from the other center to our integration origin.
       evlb - An object that "evaluates b".  Given r, it should
              return three arrays that hold the values and first and
              second derivatives of the switching function b(r)
              at the various values within r.  Specifically, it
              returns b(r), db/dr, and d^2b/dr^2, returns 0 arrays 
              for vanilla gaussian.
       evls - An object that "evaluates s" at each value within r,
              where s(r) is the slater-type function centered on
              our origin of integration.  evls(r) should return
              three arrays: s(r), ds/dr, and d^2s/dr^2.
       evlg - An object that "evaluates g" and its derivatives at
              each value within r, where g(xoc, yoc, zoc) is the
              gaussian-based basis function centered on the
              other center.  xoc, yoc, and zoc are the x, y, and z
              parts of the vector pointing from the other center
              to the position given by r, theta, and phi.
              evlg(xoc, yoc, zoc) should return three arrays:
                - An    [nr] sized array of the values of g.
                - A   [3,nr] sized array of the values of the 1st derivatives of g.
                - A [3,3,nr] sized array of the values of the 2nd derivatives of g.
              nr here is the number of values within r.
          n - The power of (r/rc) used in the bra.
          m - The power of (r/rc) used in the ket.
         rc - The radius of the sphere we are integrating inside.
   int_type - Which type of integral we are evaluating (0 = < bQr^n | O | bQr^m > , 1 = < (1-b)X | O | bQr^m >, 2 = < (1-b)X | O | (1-b)X > ) 
       """

       # First, get sin and cos values for theta and phi
       sint = np.sin(theta)
       cost = np.cos(theta)
       sinp = np.sin(phi)
       cosp = np.cos(phi)

       # evaluate some useful intermediates
       r2 = r * r
       r2sint = r2 * sint
       rsint = r * sint

       # First, get the x, y, and z parts of the vectors that point
       # from the other center to our positions.
       #print("Length of vfoc: ", len(vfoc), flush=True)
       xoc = vfoc[0] + r * sint * cosp
       yoc = vfoc[1] + r * sint * sinp
       zoc = vfoc[2] + r * cost

       # evaluate the values and derivatives of the switching function
       b, dbdr, d2bdr2 = evlb(r)
       one_minus_b = 1.0 - b

       # evaluate the values and derivatives of the slater function
       s, dsdr, d2sdr2 = evls(r)

       # Evaluate the values and xzy derivatives of the gaussian-based function.
       g, dg, d2g = evlg(xoc, yoc, zoc)

       # evaluate the powers of r/rc we need and their derivatives
       #if type(m) != int or type(n) != int:
       #       raise TypeError("We are assuming that n and m are integers.")
       if m == 1 or n == 1:
              raise ValueError("We are assuming that n and m are not 1.")
       r_over_rc = r / rc
       bra_rrc = 1.0
       if n != 0:
              bra_rrc = np.power(r_over_rc, 1.0 * n )
       ket_rrc = 1.0
       if m != 0:
              ket_rrc = np.power(r_over_rc, 1.0 * m )

       # put together the value of the overall bra function
       if int_type == 0:
              bra = b * s * bra_rrc
       elif int_type == 3:
              bra = b * s * bra_rrc + one_minus_b * g
       else:
              bra = one_minus_b * g

       # put together the value of the overall ket function
       if int_type == 0 or int_type == 1:
              ket = b * s * ket_rrc
       elif int_type == 3:
              ket = b * s * ket_rrc + one_minus_b * g
       else:
              ket = one_minus_b * g

       # Initialize the return value (the integrand) with the nuclear attraction term,
       # remembering to include the volume element so that we don't divide by zero.
       retval = bra * ket * r2sint

       # Return the integrand, which is an array of values, one for each value within r.
       return retval


def get_big_grid(rc):
       rgrid = rc * 0.5 * ( 1.0 + np.array( [
                                   -9.993050417357721704e-01,
                                   -9.963401167719552198e-01,
                                   -9.910133714767442870e-01,
                                   -9.833362538846259771e-01,
                                   -9.733268277899109755e-01,
                                   -9.610087996520537690e-01,
                                   -9.464113748584027652e-01,
                                   -9.295691721319395695e-01,
                                   -9.105221370785028245e-01,
                                   -8.893154459951141400e-01,
                                   -8.659993981540927699e-01,
                                   -8.406292962525803159e-01,
                                   -8.132653151227975385e-01,
                                   -7.839723589433413853e-01,
                                   -7.528199072605319397e-01,
                                   -7.198818501716107709e-01,
                                   -6.852363130542332703e-01,
                                   -6.489654712546573112e-01,
                                   -6.111553551723932776e-01,
                                   -5.718956462026340004e-01,
                                   -5.312794640198945650e-01,
                                   -4.894031457070529556e-01,
                                   -4.463660172534640869e-01,
                                   -4.022701579639916258e-01,
                                   -3.572201583376681255e-01,
                                   -3.113228719902109698e-01,
                                   -2.646871622087674236e-01,
                                   -2.174236437400070832e-01,
                                   -1.696444204239928033e-01,
                                   -1.214628192961205583e-01,
                                   -7.299312178779904237e-02,
                                   -2.435029266342442905e-02,
                                   2.435029266342442905e-02,
                                   7.299312178779904237e-02,
                                   1.214628192961205583e-01,
                                   1.696444204239928033e-01,
                                   2.174236437400070832e-01,
                                   2.646871622087674236e-01,
                                   3.113228719902109698e-01,
                                   3.572201583376681255e-01,
                                   4.022701579639916258e-01,
                                   4.463660172534640869e-01,
                                   4.894031457070529556e-01,
                                   5.312794640198945650e-01,
                                   5.718956462026340004e-01,
                                   6.111553551723932776e-01,
                                   6.489654712546573112e-01,
                                   6.852363130542332703e-01,
                                   7.198818501716107709e-01,
                                   7.528199072605319397e-01,
                                   7.839723589433413853e-01,
                                   8.132653151227975385e-01,
                                   8.406292962525803159e-01,
                                   8.659993981540927699e-01,
                                   8.893154459951141400e-01,
                                   9.105221370785028245e-01,
                                   9.295691721319395695e-01,
                                   9.464113748584027652e-01,
                                   9.610087996520537690e-01,
                                   9.733268277899109755e-01,
                                   9.833362538846259771e-01,
                                   9.910133714767442870e-01,
                                   9.963401167719552198e-01,
                                   9.993050417357721704e-01,
                            ] ) )

       w = rc * 0.5 * np.array( [
                     1.783280721694215174e-03,
                     4.147033260562923290e-03,
                     6.504457968979654274e-03,
                     8.846759826364391024e-03,
                     1.116813946013146645e-02,
                     1.346304789671823147e-02,
                     1.572603047602508242e-02,
                     1.795171577569730156e-02,
                     2.013482315353009450e-02,
                     2.227017380838300711e-02,
                     2.435270256871085309e-02,
                     2.637746971505462723e-02,
                     2.833967261425970191e-02,
                     3.023465707240249531e-02,
                     3.205792835485145320e-02,
                     3.380516183714178668e-02,
                     3.547221325688232341e-02,
                     3.705512854024015090e-02,
                     3.855015317861559127e-02,
                     3.995374113272034955e-02,
                     4.126256324262348590e-02,
                     4.247351512365359766e-02,
                     4.358372452932346430e-02,
                     4.459055816375654541e-02,
                     4.549162792741811429e-02,
                     4.628479658131437469e-02,
                     4.696818281620999957e-02,
                     4.754016571483030140e-02,
                     4.799938859645831724e-02,
                     4.834476223480295431e-02,
                     4.857546744150345597e-02,
                     4.869095700913975144e-02,
                     4.869095700913975144e-02,
                     4.857546744150345597e-02,
                     4.834476223480295431e-02,
                     4.799938859645831724e-02,
                     4.754016571483030140e-02,
                     4.696818281620999957e-02,
                     4.628479658131437469e-02,
                     4.549162792741811429e-02,
                     4.459055816375654541e-02,
                     4.358372452932346430e-02,
                     4.247351512365359766e-02,
                     4.126256324262348590e-02,
                     3.995374113272034955e-02,
                     3.855015317861559127e-02,
                     3.705512854024015090e-02,
                     3.547221325688232341e-02,
                     3.380516183714178668e-02,
                     3.205792835485145320e-02,
                     3.023465707240249531e-02,
                     2.833967261425970191e-02,
                     2.637746971505462723e-02,
                     2.435270256871085309e-02,
                     2.227017380838300711e-02,
                     2.013482315353009450e-02,
                     1.795171577569730156e-02,
                     1.572603047602508242e-02,
                     1.346304789671823147e-02,
                     1.116813946013146645e-02,
                     8.846759826364391024e-03,
                     6.504457968979654274e-03,
                     4.147033260562923290e-03,
                     1.783280721694215174e-03,
                     ] )
       return rgrid, w

# GENERALIZE TO ALL NUC ELEC-NUC POTENTIAL AND VANILLLA GAUSSIAN FOR ONE_ELEC_ENERGY
def test_integrand_at_rtp_general(l, rc, n, m, Z, a0, gc, ge, vfoc, gc_core, ge_core, proj, int_type, use_big_grid=False, orthogonalized=False, H=True, full_eN_pot=None, cusped_eval=True):
       """ Test the integrand_at_rtp function by evaluating the integrand.
       params:
                      l - The angular momentum of the AO (1s = 0, 2s/3s = 1, 2px = 2, 2py = 3, 2pz = 4)
                     rc - The cutoff distance for the switching function.
                      n - The power of (r/rc) used in the bra.
                      m - The power of (r/rc) used in the ket.
                      Z - The charge of the nucleus that sits at our origin
                          of integration and is zeta, the exponent of our 
                          slater functions for s cusps.
                     a0 - The a0 value for the slater function.
                     gc - The guassian linear combination coefficients.
                     ge - The guassian exponents.
                   vfoc - The length-3 xyz vector from the other center to our origin of integration.
                gc_core - The guassian linear combination coefficients of the corbital
                ge_core - The guassian exponents of the corbital
                   proj - Projection needed for the orthogonalized orbitals
            full_eN_pot - Default None for elec-nuclear potential for only the cusped nuclei, 
                          for total elec-nuclear energy, input list containing:
                          [charge of all nuclei array, 
                          num_nuc x 3 xyz array from all nuclei to our origin of integration] 
            cusped_eval - Default to True (cusped orbital evaluation), 
                          if False, switching function set to 0 to evaluate vanilla gaussian.
               int_type - Which type of integral we are evaluating (0 = < bQr^n | O | bQr^m > , 1 = < (1-b)X | O | bQr^m >, 2 = < (1-b)X | O | (1-b)X > ) 
       """
       
       # Get the grid of points and weights for a 16-point Gauss-Legendre quadrature
       # along r, which we will use to evaluate the integrand of a double integral
       # over theta and phi that will be handled by scipy's dblquad.
       g16_for_dbl = np.array(
                       [ -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
                         -0.2816035507792589, -0.0950125098376375,  0.0950125098376375,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
                          0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499, ] )
       rgrid = rc * 0.5 * ( g16_for_dbl + 1.0 )
       w = rc * 0.5 * np.array(
                           [ 0.0271524594117565, 0.0622535239386476, 0.0951585116824923, 0.1246289712555336, 0.1495959888165764, 0.1691565193950021,
                             0.1826034150449233, 0.1894506104550681, 0.1894506104550681, 0.1826034150449233, 0.1691565193950021, 0.1495959888165764,
                             0.1246289712555336, 0.0951585116824923, 0.0622535239386476, 0.0271524594117565, ] )

       # if requested, use a bigger 64-point grid
       if use_big_grid:
              rgrid, w = get_big_grid(rc)

       # set the slater function exponent
       csi = Z

       # get the function to evaluate b and its derivatives, return 0 arrays for vanilla gaussian evaluation
       evlb = lambda r: evaluate_switching_func(r, rc, cusped_eval)

       # get the function to evaluate s and its derivatives
       evls = lambda r: evaluate_slater_func(r, a0, csi)

       # get a function to evaluate the gaussian-based function and its derivatives
       if l < 2:
         evlg = lambda x, y, z: evaluate_s_type_gaussian_sum_and_derivs(x, y, z, gc, ge, gc_core, ge_core, proj, orthogonalized) 
       elif l == 2 or l == 3 or l == 4:       
         evlg = lambda x, y, z: evaluate_p_type_gaussian_sum_and_derivs(x, y, z, gc, ge, l) 
       elif l == 5 or l == 6 or l == 7 or l == 8 or l == 9:       
         evlg = lambda x, y, z: evaluate_d_type_gaussian_sum_and_derivs(x, y, z, gc, ge, l) 
       else:
         raise RuntimeError('orb_type does not match l = {0,9}')

       # prepare the integrand function for dblquad
       if (H == True):
         integrand_for_dbl = lambda t, p: np.sum( w * integrand_at_rtp_general(rgrid, t, p, Z, vfoc, evlb, evls, evlg, n, m, rc, full_eN_pot, int_type) )
       else:  
         integrand_for_dbl = lambda t, p: np.sum( w * integrand_at_rtp_s(rgrid, t, p, Z, vfoc, evlb, evls, evlg, n, m, rc, int_type) )

       # do the integration
       dbl_num, dbl_err = sp.integrate.dblquad(integrand_for_dbl, 0.0, 2.0*np.pi, 0.0, np.pi)

       #print(f"dbl_num = {dbl_num:20.12e}, dbl_err = {dbl_err:.2e}")
  
       return dbl_num, dbl_err
