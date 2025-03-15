'''
Routines for use in conducting Hamiltonian Monte Carlo with variables constrained to
the regular simplex.
'''

''' TODO
* Exploit diagonal mass when given
* Use sparse matrices (list-of-lists?) for big probability matrix
* allow additional parameters (like galaxy bias)
* introduce bias into clustering
* Have descent cut step size
* Remake Fisher matrix at selected f?
'''

import numpy as np
import scipy.stats


class SimplexHMC:
    '''Class that will execute the HMC constrained to simplex for a subset of the parameters. Maintains
    an internal state with current position and momentum. The Hamiltonian's
    state variables q are the (negative) log of the subset of quantities f that
    sum to unity to form the simplex plus the other parameters of the model that
    are not subject to this constraint.  This precludes the possibility of
    boundary crossing, but there is still a boundary set at `qMax` where
    a bounce step is implemented, to keep probabilities from running away.

    The iterative "RATTLE" algorithm is used to solve the constrained dynamics
    of each time step.

    External classes (those defining the probability on the simplex) use
    the f's as variables.  But internally the dynamics (and momenta) use q.
    '''
    
    # Return codes for samples
    accept = 0   # Accepted new state
    reject = 1   # Rejected step by Metropolis
    tooManyBounces = 2 # Step rejected from too many bounces
    didNotConverge = 3 # Newton iteration failure in RATTLE

    def __init__(self, fStart, negLogP, dNegLogPdF,
                 mass=None,
                 dt = 0.2, 
                 tRange=(0.1*np.pi,2.1*np.pi),
                 qMax = -np.log(1e-8),
                 sumConstraint=1e-3,
                 maxBounces = 30):
        '''
        Set up class to perform HMC operations for the case where 
        constraint of sum(f)=1 is in place.  
        fStart:   vector of f values at start of chain
        thetaStart: vector of theta values at start of chain
        negLogP:  function of f and theta that yields -log P(f) where P is the dist. to sample
        dNegLogPdF: function of f yielding gradient vector of -log(P) w.r.t. f and theta
        mass:     mass matrix (=local Hessian, ideally) for the f's and theta's. Can be changed
                  later with `setMass`, defaults to identity.
        dt:       time step used for Hamiltonian leapfrog integrator
        tRange:   range of integration times used for each sample.  Each time
                  step chooses a duration at random uniformly from this interval.
        qMax:     maximum allowed value of -log(f).  HMC trajectory will
                  bounce off of this boundary, incrementing bounceCount
        sumConstraint:  For mass matrix, a soft Gaussian constraint is
                  set on sum(f)=1+-sumConstraint.  This is needed rather
                  than infinite mass since finite trajectories have
                  finite component perp to boundary.  1/sqrt(N_tot) is a
                  good value for Dirichlet distributions with N_tot counts.
        maxBounces: One sample will abort if its trajectory bounces more than
                  this.

        '''
        self.tolerance = 1e-8   # Tolerance for simplex constraint
        self.negLogP = negLogP
        self.dNegLogPdF = dNegLogPdF
        self.dt = dt
        self.tRange = tRange
        self.qMax = qMax
        self.maxBounces = maxBounces

        self.p = np.zeros_like(fStart)  # Initialize momentum to zero
        # Check for valid starting conditions
        if np.any(fStart<=0.):
            raise ValueError("Starting f is on or outside simplex border")

        self.q = -np.log(fStart)
        if np.abs(np.sum(fStart)-1) > self.tolerance:
            print("Warning: HMC is given unnormalized fStart, sum(f)=",np.sum(fStart))
            # raise exception ???
            # Renormalize:
            self.renormalizeQ(self.q)
        self.f = np.exp(-self.q)

        self.setMass(mass,sumConstraint=sumConstraint)

        # Set initial coordinate and first force calculation
        self.setQ(self.q)
        
        # Initialize some potentially interesting operation counters
        self.resetCounters()
        return
    
    def setMass(self,hessianF=None,sumConstraint=1e-3):
        ''' Initialize mass matrix for q's, given a guess at Hessian for f's.
        Use formula that assume we are near the minimum potential for q's,
        and include the factors that accrue from Jacobian is transformation
        of probability from f to q space.

        If None is given, then identity matrix is assumed.

        The current value of self.f is used to execute tranformations.

        sumConstraint is the 1-sigma width of soft prior of sum(f). '''

        # Build up the Hessian
        fsq = np.dot(self.f,self.f)    # Norm-squared of f vector
        v = fsq - np.square(self.f)    # v_i = f^2 - f_i^2
        if hessianF is None:
            hF = np.eye(len(self.f))
        else:
            hF = hessianF.copy()

        # Needed intermediate
        hFk = np.dot(hF,np.square(self.f)) # \sum_k f_k^2 hF_ik, vector
        hFkk = np.dot(hFk,np.square(self.f)) #\sum_kl f_k^2 F_l^2 hF_kl, scalar
        hFk *= fsq
        # Build the bracketed quantity; fsq term is from Jacobian
        hQ = (fsq*fsq)*hF - hFk[:,np.newaxis] - hFk[np.newaxis,:] + (hFkk-fsq)
        # Factors 1/(v_i v_j)
        hQ /= v[:,np.newaxis]
        hQ /= v[np.newaxis,:]
        # Add the term for softened delta function on sum(f):
        hQ += sumConstraint**(-2)
        # Factors f_i f_j
        hQ *= self.f[:,np.newaxis]
        hQ *= self.f[np.newaxis,:]
        # Now add last part of Jacobian term to diagonal
        nf = hQ.shape[0]
        hQ.ravel()[::nf+1] += (fsq*fsq)/np.square(v)
        
        # This is now our mass matrix
        self.mass = hQ

        # These are not valid and will be calculated when needed:
        self._mCholesky = None
        self._mInverse = None
        self._mInverseF = None
        self._pGenerator = None
    
    @property
    def mInverse(self):
        ''' Calculate the Cholesky decomposition of the mass matrix if needed
        and make an inverse and a momentum generator from it.'''
        if self._mInverse is None:
            # Do the Cholesky decomposition
            print("Inverting:") #???
            self._mCholesky = scipy.linalg.cho_factor(self.mass, lower=True)
            self._mInverse = scipy.linalg.cho_solve(self._mCholesky,np.eye(self.mass.shape[0]))
            print("...done:") #???
            # Make RNG for momentum
            self._pGenerator = MultivariateGaussian(cholesky=self._mCholesky)
        return self._mInverse

    @property
    def pGenerator(self):
        '''The random generator for conjugate momenta'''
        if self._pGenerator is None:
            # Force creation of a pGenerator
            self.mInverse
        return self._pGenerator
    
    @property
    def mInverseF(self):
        if self._mInverseF is None:
            self._mInverseF = np.matmul(self.mInverseF,self.f)
        return self._mInverseF
       
    def resetCounters(self):
        # Reset `stepCount` and `bounceCount`
        self.stepCount = 0
        self.bounceCount = 0
        return
    
    def setQ(self,q):
        # Reset internal q value, calculate matching f, clear relevant caches
        np.copyto(self.q,q)
        self.f = np.exp(-self.q)
        # Invalidates these saved quantities:
        self._mInverseF = None
        self._force = None
        return
    
    @property
    def force(self):
        # Access force at currently cached q
        if self._force is None:
            self._force = self.getForce(self.f)
            # Get rid of force component normal to current constraint
            self.projectP(self._force)
        return self._force
    
    def renormalizeQ(self,q):
        # Rescale q vector so \sum f_i = 1  (additive on q's since they're log)
        q += np.log(np.sum(np.exp(-q)))
        return
    def projectP(self,p):
        '''Take a momentum-space vector and project away motion normal
        to constraint at current q, f.
        At end, f * mInverse * p = 0 '''
        p -= (np.dot(self.mInverseF,p)/np.dot(self.mInverseF,self.f)) * self.f
        return

    def newMomentum(self):
        ''' Draw new momentum from multivariate Gaussian, project out
        component normal to constraint'''
        self.p = self.pGenerator.sample()
        # Project away the motion normal to the constraint, so
        # Ones * mass^-1 * p = 0
        self.projectP(self.p)
        return

    # The routines below operate at specified f/q/p, not self values

    def getForce(self,f):
        # Calculate (q-space) force at given f values
        # Convert from f to q by multiplying by -f and subtracting 1
        return self.dNegLogPdF(f) * f - 1
    def pe(self,q):
        # Return (q-space) potential energy = -log P(q)
        f = np.exp(-q)
        return self.negLogP(f) + np.sum(q)
    def hamilton(self,p,q):
        # Return total energy at given p,q
        return self.pGenerator.negLogP(p)+self.pe(q)
    def sample(self, debug=0, pStart=None,wzpz=None):
        ''' Take one HMC integration, execute Metropolis test at end.
        Returns integer signaling result - accepted (0) or rejected (1).
        The state q after the sample is available as member self.q, or
        the corresponding f is available as self.f.
        
        If pStart is given, it is used as initial momentum.  Otherwise
        a momentum is chosen from a multivariate Gaussian as the HMC
        algorithm requires.

        The WZplus3sDir instance being used as potential can be
        given in wzpz for debugging purposes.

        Returns: flag,dH  where flag is the result of sampling and dH is
        the change in Hamiltonian during the step, if it concludes.
        '''
        
        if pStart is None:
            # Get new momentum
            self.newMomentum()
        else:
            # Use the supplied momentum
            self.p = np.array(pStart)
            self.projectP(self.p)
            
        # Save energy at start of integration (???Save PE from last time??)
        h0 = self.hamilton(self.p,self.q)
        hStartStep = h0
        
        if debug>1:
            print("Starting H:",h0)
        
         # Choose number of leapfrog steps (integration time)
        tTotal = np.random.uniform(low=self.tRange[0],high=self.tRange[1])
        nSteps = int(np.floor(tTotal / self.dt))
        if nSteps == 0:
            nSteps = 1   # Take at least one step
        if debug > 0:
            print("...integrating for {:d} steps".format(nSteps))
        # Save starting q in case step is rejected
        self.qStartSample = np.array(self.q)

        '''
        We're going to enter a loop where each trip is an integration
        step.  This will use the RATTLE algorithm as 2 half-steps
        to meet constraints rather than use straight-up leapfrog.
        '''
        fullStep = True    # Begin with full steps
        while nSteps > 0:
            if fullStep:
                # Set up for a fresh full step
                tStep = self.dt  # Begin with a full step.
                tStepLeft = tStep # This is the total time remaining in a parent step
                bounceIndex = -1 # Coordinate index where this integration should bounce at end
                nBounce = 0 # Number of bounces this step
                
            self.stepCount = self.stepCount+1
            pHalf = self.p + (0.5*tStep)*self.force  # First kick
            q1 = self.q + tStep*np.matmul(self.mInverse,pHalf)
            f1 = np.exp(-q1)
            c = np.sum(f1)-1.
            # Now execute Newton iterations 
            newtonCount = 0
            while np.abs(c) > self.tolerance:
                if debug>2:
                    print('iter',newtonCount,'c',c) 
                newtonCount = newtonCount+1
                if newtonCount > 10: #### Set this value???
                    # Too many iterations.  Reject step
                    self.setQ(self.qStartSample)
                    return self.didNotConverge,0
                tweak = c / np.dot(f1, self.mInverseF)
                # Update position and momentum
                pHalf += (tweak/tStep) * self.f
                q1 += tweak * self.mInverseF
                f1 = np.exp(-q1)
                c = np.sum(f1)-1.

            # The Newton iteration is successful.  Have we crossed a bound?
            xx = q1>self.qMax
            if bounceIndex>=0:
                # Do not repeat bounce on the same coordinate.
                # It will just be a slight recalculation of the crossing.
                xx[bounceIndex] = False 
            crossings = np.where(xx)[0]
            if len(crossings) > 0: 
                self.bounceCount = self.bounceCount+1  # Total bounces
                nBounce = nBounce + 1 # Bounces on this step
                if nBounce > self.maxBounces:
                    # Too many bounces in one step.  Reject sample
                    self.setQ(self.qStartSample)
                    return self.tooManyBounces,0
                    
                tCross = (self.qMax-self.q[crossings]) / (q1[crossings]-self.q[crossings])
                # Find the earliest crossing if more than one
                if len(crossings)>1:
                    # See which boundary is crossed first
                    ii = np.argmin(tCross)
                else:
                    # Use first and only crossing
                    ii = 0
                bounceIndex = crossings[ii]
                tStep = tStep * tCross[ii]
                if debug>1:
                    print("Bounce",nBounce,"on axis",bounceIndex,"at tStep",tStep)
                    print("Crossings",crossings,"tCross",tCross)
                # Now go back and repeat the first part of RATTLE
                # using the shorter time interval.
                fullStep = False
                continue
                   
            # Successul Newton iteration with no boundary crossing,
            # save the q1 and calculate force
            q0 = self.q.copy() ### For potential debugging of the step
            p0 = self.p.copy() ###
            
            self.setQ(q1)  # ??? This is repeating the np.exp(-q1) an extra time

            p1 = pHalf + (0.5*tStep)*self.force  # 2nd kick from force at q1
            # Project into tangent bundle at self.q=q1:
            self.projectP(p1)
            self.p = p1

            if bounceIndex>=0:
                # We just finished a pre-bounce step.
                # Reverse the momentum appropriately
                self.bounce(bounceIndex)
                # Set time for the remainder of the step
                tStepLeft = tStepLeft - tStep
                tStep = tStepLeft
                # Reset bounce indicator
                bounceIndex = -1
            else:
                # We just finished a post-bounce step, meaning
                # we have completed a full time step now
                nSteps = nSteps-1
                fullStep = True

            if debug>1:
                # Report total energy after step
                hh = self.hamilton(self.p,self.q)
                print(".{:d}...end H".format(nSteps), hh)
                if debug>3 and np.abs(hh-hStartStep)>2.:
                    # We have encountered a very non-energy-conserving
                    # integration step.  Poke into this some more,
                    # what part of energy change is it??
                    print("---max Q, dQ", np.max(q1),np.max(q1-q0))
                    print("---H change:", hh-hStartStep)
                    print("---KE change:", self.pGenerator.negLogP(p1) - self.pGenerator.negLogP(p0))
                    print("---PE change:", self.pe(q1) - self.pe(q0))
                    f0 = np.exp(-q0)
                    f1 = np.exp(-q1)
                    print("---Force0*dq prediction:", np.dot(self.getForce(f0),q1-q0))
                    print("---Force1*dq prediction:", np.dot(self.getForce(f1),q1-q0))
                    if wzpz is not None:
                        # More detailed breakdown of PE change in WZPZ if we have it
                        print("---PE components:")
                        print("-----p3sDir change:",wzpz.pz.negLogP(f1)-wzpz.pz.negLogP(f0))
                        for i,wz in enumerate(wzpz.wz):
                            b,wzl = wz
                            dE = wzl.negLogP(wzpz._f2nz(f1,b)) \
                                - wzl.negLogP(wzpz._f2nz(f0,b))
                            print("-----wz {:d} change:".format(i),dE)

                        print("---Force components:")
                        df = f1-f0
                        print("-----p3sDir change:",
                                  np.dot((wzpz.pz.dNegLogPdF(f1)-wzpz.pz.dNegLogPdF(f0)),df))
                        for i,wz in enumerate(wzpz.wz):
                            b,wzl = wz
                            n1 = wzpz._f2nz(f1,b)
                            n0 = wzpz._f2nz(f0,b)
                            dF = wzl.dNegLogPdF(n1) - wzl.dNegLogPdF(n0)
                            print("-----wz {:d} change:".format(i),np.dot(dF,n1-n0))
                            if i==3:
                                print("-------vs z:\n",(dF*(n1-n0))[wzl.z_select],
                                          "\n",dF[wzl.z_select],
                                          "\n",(n1-n0)[wzl.z_select])
                            
                                
                hStartStep = hh  ## Update for next step's debugging
            # End of step loop
        # Now completed the integration.
        # Accept or reject the new value, using prob(new) / prob(old)
        # Rescale q first to clear up any roundoff accumulations
        self.renormalizeQ(self.q)
        self.projectP(self.p)
        h1 = self.hamilton(self.p,self.q)
        dLogP = h0-h1
        if debug>0:
            print("dLogP:",dLogP,"H0,1:",h0,h1)

        # apply Metropolis criterion
        if np.random.uniform() < np.exp(dLogP):
            if (debug>0):
                print("Acceptance at p2/p1=",np.exp(dLogP))
            return self.accept,dLogP
        else:
            self.setQ(self.qStartSample)
            if debug>0:
                print("Rejection at p2/p1=",np.exp(dLogP))
            return self.reject,dLogP

    def bounce(self,bounceIndex):
        ''' Alter the momentum for a bounce off of wall at
        the given coordinate index. '''
        normal = np.zeros_like(self.q)
        normal[bounceIndex] = 1.  # Now have normal to the boundary
        # Project away component producing impulse outside tangent bundle
        self.projectP(normal)
        mInverseN = np.matmul(self.mInverse,normal)
        impulse = np.dot(self.p,mInverseN) / np.dot(normal,mInverseN)
        self.p -= (2*impulse)*normal
        return

    def descend(self, firstPEStep=10., maxSteps=100, minimumShift=0.01,
                     stepFactor=2., debug=0):
        ''' Use the force calculation to do a gradient descent
        on the potential in q space.  Recalculates the gradient at
        every step and adjusts next step size upwards if a gradient is
        successful.  Tries a reduced step size if PE does not
        decrease by minimumShift.
        Convergence is when the gradient step cannot give more 
        than minimumShift.

        firstPEStep: initial step size is chosen to yield this PE
                     decrease if gradient is constant.
        stepFactor:  increase or decrease step size by this factor
                     on decrease or increase of PE.
        maxSteps:    stop descent after this many steps.
        minimumShift: stop descent when gradient step becomes
                     smaller than will yield this PE drop

        Returns: True if descent fails to converge in maxSteps.
        '''
        
        stepCount = 0
        q0 = np.array(self.q)
        f0 = np.array(self.f)
        pe0 = self.pe(q0)
        if debug>0:
            print("Starting PE:",pe0)

        chisqStep = firstPEStep
        while stepCount < maxSteps:
            if debug>0:
                print("..Iteration",stepCount)
            force0 = self.getForce(f0)
            # Project away force normal to constraint.  No mass in use, just put p*f0 = 0.
            force0 -= (np.dot(force0,f0)/np.dot(f0,f0))*f0 
            # Rescale force to be a step of unit (or less) linearized chisq change
            fScale = np.dot(force0,force0)
            if fScale > 1:
                force0 /= np.dot(force0,force0)

            # Make a first attempt using last chisqStep:
            q1 = q0 + chisqStep*force0
            self.renormalizeQ(q1) # Put q onto manifold
            pe1 = self.pe(q1)
            if debug>0:
                print("...chisqStep",chisqStep,"yields PE",pe1)
            
            if pe1 < pe0-minimumShift:
                # increase step size for next attempt
                chisqStep *= stepFactor
            else:
                # Try smaller steps of the same force til it's significantly downhill
                while pe1 > pe0-minimumShift:
                    chisqStep /= stepFactor
                    if chisqStep < minimumShift:
                        # quit - not getting anywhere.
                        if debug>0:
                            print("Done - chisqStep too small now")
                        self.setQ(q0)
                        return False
                    q1 = q0 + chisqStep*force0
                    self.renormalizeQ(q1) # Put q onto manifold
                    pe1 = self.pe(q1)
                    if debug>0:
                        print("...chisqStep",chisqStep,"yields PE",pe1)

            # We have achieved a reduction.  Move along
            np.copyto(q0,q1)
            f0 = np.exp(-q1)
            pe0 = pe1
            stepCount = stepCount+1

        # Get here from too many steps.  Save wherever we are.
        self.setQ(q0)
        return True



class SimplexGeneralHMC:
    '''Class that will execute the HMC constrained to simplex. Maintains
    an internal state with current position and momentum. The Hamiltonian's
    state variables q are the (negative) log of the quantities f that
    sum to unity to form the simplex.  This precludes the possibility of
    boundary crossing, but there is still a boundary set at `qMax` where
    a bounce step is implemented, to keep probabilities from running away.

    The iterative "RATTLE" algorithm is used to solve the constrained dynamics
    of each time step.

    External classes (those defining the probability on the simplex) use
    the f's as variables.  But internally the dynamics (and momenta) use q.
    '''
    
    # Return codes for samples
    accept = 0   # Accepted new state
    reject = 1   # Rejected step by Metropolis
    tooManyBounces = 2 # Step rejected from too many bounces
    didNotConverge = 3 # Newton iteration failure in RATTLE

    def __init__(self, fStart, thetaStart, negLogP, dNegLogPdF,
                dNegLogPdTheta,
                 massF=None, massTheta = None,
                 dt = 0.2, 
                 tRange=(0.1*np.pi,2.1*np.pi),
                 qMax = -np.log(1e-8),
                 sumConstraint=1e-3,
                 maxBounces = 30):
        '''
        Set up class to perform HMC operations for the case where 
        constraint of sum(f)=1 is in place.  
        fStart:   vector of f values at start of chain
        negLogP:  function of f that yields -log P(f) where P is the dist. to sample
        dNegLogPdF: function of f yielding gradient vector of -log(P) w.r.t. f
        dNegLogPdF: function of f yielding gradient vector of -log(P) w.r.t. theta
        mass:     mass matrix (=local Hessian, ideally) for the f's. Can be changed
                  later with `setMass`, defaults to identity.
        dt:       time step used for Hamiltonian leapfrog integrator
        tRange:   range of integration times used for each sample.  Each time
                  step chooses a duration at random uniformly from this interval.
        qMax:     maximum allowed value of -log(f).  HMC trajectory will
                  bounce off of this boundary, incrementing bounceCount
        sumConstraint:  For mass matrix, a soft Gaussian constraint is
                  set on sum(f)=1+-sumConstraint.  This is needed rather
                  than infinite mass since finite trajectories have
                  finite component perp to boundary.  1/sqrt(N_tot) is a
                  good value for Dirichlet distributions with N_tot counts.
        maxBounces: One sample will abort if its trajectory bounces more than
                  this.

        '''
        self.tolerance = 1e-8   # Tolerance for simplex constraint
        self.negLogP = negLogP
        self.dNegLogPdF = dNegLogPdF
        self.dNegLogPdTheta = dNegLogPdTheta
        self.dt = dt
        self.tRange = tRange
        self.qMax = qMax
        self.maxBounces = maxBounces

        self.pF = np.zeros_like(fStart)  # Initialize momentum to zero
        self.pTheta = np.zeros_like(thetaStart)
        # Check for valid starting conditions
        if np.any(fStart<=0.):
            raise ValueError("Starting f is on or outside simplex border")

        self.q = -np.log(fStart)
        if np.abs(np.sum(fStart)-1) > self.tolerance:
            print("Warning: HMC is given unnormalized fStart, sum(f)=",np.sum(fStart))
            # raise exception ???
            # Renormalize:
            self.renormalizeQ(self.q)
        self.f = np.exp(-self.q)
        self.theta = thetaStart


        self.setMassF(massF,sumConstraint=sumConstraint)
        self.setMassTheta(massTheta)

        # Set initial coordinate and first force calculation
        self.setQ(self.q, self.theta)
        
        # Initialize some potentially interesting operation counters
        self.resetCounters()
        return
    
    def setMassF(self,hessianF=None,sumConstraint=1e-3):
        ''' Initialize mass matrix for q's, given a guess at Hessian for f's.
        Use formula that assume we are near the minimum potential for q's,
        and include the factors that accrue from Jacobian is transformation
        of probability from f to q space.

        If None is given, then identity matrix is assumed.

        The current value of self.f is used to execute tranformations.

        sumConstraint is the 1-sigma width of soft prior of sum(f). '''

        # Build up the Hessian
        fsq = np.dot(self.f,self.f)    # Norm-squared of f vector
        v = fsq - np.square(self.f)    # v_i = f^2 - f_i^2
        if hessianF is None:
            hF = np.eye(len(self.f))
        else:
            hF = hessianF.copy()

        # Needed intermediate
        hFk = np.dot(hF,np.square(self.f)) # \sum_k f_k^2 hF_ik, vector
        hFkk = np.dot(hFk,np.square(self.f)) #\sum_kl f_k^2 F_l^2 hF_kl, scalar
        hFk *= fsq
        # Build the bracketed quantity; fsq term is from Jacobian
        hQ = (fsq*fsq)*hF - hFk[:,np.newaxis] - hFk[np.newaxis,:] + (hFkk-fsq)
        # Factors 1/(v_i v_j)
        hQ /= v[:,np.newaxis]
        hQ /= v[np.newaxis,:]
        # Add the term for softened delta function on sum(f):
        hQ += sumConstraint**(-2)
        # Factors f_i f_j
        hQ *= self.f[:,np.newaxis]
        hQ *= self.f[np.newaxis,:]
        # Now add last part of Jacobian term to diagonal
        nf = hQ.shape[0]
        hQ.ravel()[::nf+1] += (fsq*fsq)/np.square(v)
        
        # This is now our mass matrix
        self.massF = hQ

        # These are not valid and will be calculated when needed:
        self._mCholeskyF = None
        self._mInverseF = None
        self._pGeneratorF = None
        self._mInverseQ = None
    
    @property
    def mInverseQ(self):
        ''' Calculate the Cholesky decomposition of the mass matrix if needed
        and make an inverse and a momentum generator from it.'''
        if self._mInverseQ is None:
            # Do the Cholesky decomposition
            print("Inverting:") #???
            self._mCholeskyQ = scipy.linalg.cho_factor(self.massF, lower=True)
            self._mInverseQ = scipy.linalg.cho_solve(self._mCholeskyQ,np.eye(self.massF.shape[0]))
            print("...done:") #???
            # Make RNG for momentum
            self._pGeneratorF = MultivariateGaussian(cholesky=self._mCholeskyQ)
        return self._mInverseQ

    
    @property
    def mInverseF(self):
        if self._mInverseF is None:
            self._mInverseF = np.matmul(self.mInverseQ,self.f)
        return self._mInverseF

    @property
    def pGeneratorF(self):
        '''The random generator for conjugate momenta'''
        if self._pGeneratorF is None:
            # Force creation of a pGenerator
            self.mInverseQ
        return self._pGeneratorF


    def setMassTheta(self, hessianTheta = None):
        if hessianTheta is None:
            self.massTheta = np.eye(len(self.theta))
        else:
            self.massTheta = hessianTheta

        self._mCholeskyTheta = None 
        self._mInverseTheta = None 
        self._pGeneratorTheta = None

    @property
    def mInverseTheta(self):
        ''' Calculate the Cholesky decomposition of the mass matrix if needed
        and make an inverse and a momentum generator from it.'''
        if self._mInverseTheta is None:
            # Do the Cholesky decomposition
            #print("Inverting:") #???
            self._mCholeskyTheta = scipy.linalg.cho_factor(self.massTheta, lower=True)
            self._mInverseTheta = scipy.linalg.cho_solve(self._mCholeskyTheta,np.eye(self.massTheta.shape[0]))
            #print("...done:") #???
            # Make RNG for momentum
            self._pGeneratorTheta = MultivariateGaussian(cholesky=self._mCholeskyTheta)
        return self._mInverseTheta

    
    @property
    def pGeneratorTheta(self):
        '''The random generator for conjugate momenta'''
        if self._pGeneratorTheta is None:
            # Force creation of a pGenerator
            self.mInverseTheta
        return self._pGeneratorTheta
    

    def resetCounters(self):
        # Reset `stepCount` and `bounceCount`
        self.stepCount = 0
        self.bounceCount = 0
        return
    
    def setParams(self,q, theta):
        # Reset internal q value, calculate matching f, clear relevant caches
        np.copyto(self.q,q)
        self.f = np.exp(-self.q)
        # Invalidates these saved quantities:
        self._mInverseF = None
        self._forceF = None

        np.copyto(self.theta,theta)
        #self._mInverseTheta = None
        self._forceTheta = None

        return
    
    @property
    def forceF(self):
        # Access force at currently cached q
        if self._forceF is None:
            self._forceF = self.getForceF(self.f,self.theta)
            # Get rid of force component normal to current constraint
            self.projectP(self._forceF)
        return self._forceF

    @property
    def forceTheta(self):
        # Access force at currently cached q
        if self._forceTheta is None:
            self._forceTheta = self.getForceTheta(self.f, self.theta)
        return self._forceTheta

    def renormalizeQ(self,q):
        # Rescale q vector so \sum f_i = 1  (additive on q's since they're log)
        q += np.log(np.sum(np.exp(-q)))
        return 

    def projectP(self,p):
        '''Take a momentum-space vector and project away motion normal
        to constraint at current q, f.
        At end, f * mInverse * p = 0 '''
        p -= (np.dot(self.mInverseF,p)/np.dot(self.mInverseF,self.f)) * self.f
        return
    def newMomentum(self):
        ''' Draw new momentum from multivariate Gaussian, project out
        component normal to constraint'''
        self.pF = self.pGeneratorF.sample()
        self.pTheta = self.pGeneratorTheta.sample()
        # Project away the motion normal to the constraint, so
        # Ones * mass^-1 * p = 0
        self.projectP(self.pF)
        return

    # The routines below operate at specified f/q/p, not self values

    def getForceF(self,f,theta):
        # Calculate (q-space) force at given f values
        # Convert from f to q by multiplying by -f and subtracting 1
        return self.dNegLogPdF(f,theta) * f - 1

    def getForceTheta(self,f,theta):
        # Calculate (theta-space) force at given f values
        # Convert from f to q by multiplying by -f and subtracting 1
        return self.dNegLogPdTheta(f,theta)

    def pe(self,q,theta):
        # Return (q-space) potential energy = -log P(q)
        f = np.exp(-q)
        return self.negLogP(f, theta) + np.sum(q)

    def hamilton(self,pF,ptheta,q,theta):
        # Return total energy at given p,q
        return self.pGeneratorF.negLogP(pF)+self.pGeneratorTheta.negLogP(ptheta)+self.pe(q,theta)
    def sample(self, debug=0, pStartF=None,pStartTheta=None,wzpz=None):
        ''' Take one HMC integration, execute Metropolis test at end.
        Returns integer signaling result - accepted (0) or rejected (1).
        The state q after the sample is available as member self.q, or
        the corresponding f is available as self.f.
        
        If pStart is given, it is used as initial momentum.  Otherwise
        a momentum is chosen from a multivariate Gaussian as the HMC
        algorithm requires.

        The WZplus3sDir instance being used as potential can be
        given in wzpz for debugging purposes.

        Returns: flag,dH  where flag is the result of sampling and dH is
        the change in Hamiltonian during the step, if it concludes.
        '''
        
        if pStartF is None:
            # Get new momentum
            self.newMomentum()
        else:
            # Use the supplied momentum
            self.pF = np.array(pStartF)
            self.projectP(self.pF)
            self.pTheta = np.array(pStartTheta)

        # Save energy at start of integration (???Save PE from last time??)
        h0 = self.hamilton(self.pF,self.pTheta,self.q,self.theta)
        hStartStep = h0
        
        if debug>1:
            print("Starting H:",h0)
        
         # Choose number of leapfrog steps (integration time)
        tTotal = np.random.uniform(low=self.tRange[0],high=self.tRange[1])
        nSteps = int(np.floor(tTotal / self.dt))
        if nSteps == 0:
            nSteps = 1   # Take at least one step
        if debug > 0:
            print("...integrating for {:d} steps".format(nSteps))
        # Save starting q in case step is rejected
        self.qStartSample = np.array(self.q)
        self.thetaStartSample = np.array(self.theta)

        '''
        We're going to enter a loop where each trip is an integration
        step.  This will use the RATTLE algorithm as 2 half-steps
        to meet constraints rather than use straight-up leapfrog.
        '''
        fullStep = True    # Begin with full steps
        while nSteps > 0:
            if fullStep:
                # Set up for a fresh full step
                tStep = self.dt  # Begin with a full step.
                tStepLeft = tStep # This is the total time remaining in a parent step
                bounceIndex = -1 # Coordinate index where this integration should bounce at end
                nBounce = 0 # Number of bounces this step
                
            self.stepCount = self.stepCount+1
            pHalfF = self.pF + (0.5*tStep)*self.forceF  # First kick
            q1 = self.q + tStep*np.matmul(self.mInverseQ,pHalfF)
            f1 = np.exp(-q1)
            c = np.sum(f1)-1.
            
            pHalfTheta = self.pTheta - (0.5 * tStep) * self.forceTheta
            theta1 = self.theta + tStep * np.matmul(self.mInverseTheta, pHalfTheta)

            # Now execute Newton iterations 
            newtonCount = 0
            while np.abs(c) > self.tolerance:
                if debug>2:
                    print('iter',newtonCount,'c',c) 
                newtonCount = newtonCount+1
                if newtonCount > 10: #### Set this value???
                    # Too many iterations.  Reject step
                    self.setParams(self.qStartSample, self.thetaStartSample)
                    return self.didNotConverge,0
                tweak = c / np.dot(f1, self.mInverseF)
                #print(tweak, self.mInverseF)
                # Update position and momentum
                pHalfF += (tweak/tStep) * self.f
                q1 += np.dot(self.mInverseF, tweak)
                f1 = np.exp(-q1)
                c = np.sum(f1)-1.

            # The Newton iteration is successful.  Have we crossed a bound?
            xx = q1>self.qMax
            if bounceIndex>=0:
                # Do not repeat bounce on the same coordinate.
                # It will just be a slight recalculation of the crossing.
                xx[bounceIndex] = False 
            crossings = np.where(xx)[0]
            if len(crossings) > 0: 
                self.bounceCount = self.bounceCount+1  # Total bounces
                nBounce = nBounce + 1 # Bounces on this step
                if nBounce > self.maxBounces:
                    # Too many bounces in one step.  Reject sample
                    self.setParams(self.qStartSample, self.thetaStartSample)
                    return self.tooManyBounces,0
                    
                tCross = (self.qMax-self.q[crossings]) / (q1[crossings]-self.q[crossings])
                # Find the earliest crossing if more than one
                if len(crossings)>1:
                    # See which boundary is crossed first
                    ii = np.argmin(tCross)
                else:
                    # Use first and only crossing
                    ii = 0
                bounceIndex = crossings[ii]
                tStep = tStep * tCross[ii]
                if debug>1:
                    print("Bounce",nBounce,"on axis",bounceIndex,"at tStep",tStep)
                    print("Crossings",crossings,"tCross",tCross)
                # Now go back and repeat the first part of RATTLE
                # using the shorter time interval.
                fullStep = False
                continue
                   
            # Successul Newton iteration with no boundary crossing,
            # save the q1 and calculate force
            self.setQ(q1,theta1)  # ??? This is repeating the np.exp(-q1) an extra time

            p1F = pHalfF + (0.5*tStep)*self.forceF  # 2nd kick at q from force at q1, theta1
            p1Theta = pHalfTheta - (0.5*tStep)*self.forceTheta  # 2nd kick at theta from force at q1, theta1
            

            # Project into tangent bundle at self.q=q1:
            self.projectP(p1F)
            self.pF = p1F
            self.pTheta = p1Theta

            q0 = self.q.copy() ### For potential debugging of the step
            p0F = self.pF.copy() ###
            
            theta0 = self.theta.copy()
            p0Theta = self.pTheta.copy()

            if bounceIndex>=0:
                # We just finished a pre-bounce step.
                # Reverse the momentum appropriately
                self.bounce(bounceIndex)
                # Set time for the remainder of the step
                tStepLeft = tStepLeft - tStep
                tStep = tStepLeft
                # Reset bounce indicator
                bounceIndex = -1
            else:
                # We just finished a post-bounce step, meaning
                # we have completed a full time step now
                nSteps = nSteps-1
                fullStep = True
                                
            # End of step loop
        # Now completed the integration.
        # Accept or reject the new value, using prob(new) / prob(old)
        # Rescale q first to clear up any roundoff accumulations
        self.renormalizeQ(self.q)
        self.projectP(self.pF)
        h1 = self.hamilton(self.pF,self.pTheta,self.q,self.theta)
        dLogP = h0-h1
        if debug>0:
            print("dLogP:",dLogP,"H0,1:",h0,h1)

        # apply Metropolis criterion
        if np.random.uniform() < np.exp(dLogP):
            if (debug>0):
                print("Acceptance at p2/p1=",np.exp(dLogP))
            return self.accept,dLogP
        else:
            self.setParams(self.qStartSample, self.thetaStartSample)
            if debug>0:
                print("Rejection at p2/p1=",np.exp(dLogP))
            return self.reject,dLogP

    def setQ(self,q, theta):
        # Reset internal q value, calculate matching f, clear relevant caches
        np.copyto(self.q,q)
        self.f = np.exp(-self.q)
        np.copyto(self.theta, theta)
        # Invalidates these saved quantities:
        self._mInverseF = None
        self._forceF = None
        self._mInverseTheta = None
        self._forceTheta = None
        return

    def bounce(self,bounceIndex):
        ''' Alter the momentum for a bounce off of wall at
        the given coordinate index. '''
        normal = np.zeros_like(self.q)
        normal[bounceIndex] = 1.  # Now have normal to the boundary
        # Project away component producing impulse outside tangent bundle
        self.projectP(normal)
        mInverseN = np.matmul(self.mInverseF,normal)
        impulse = np.dot(self.pF,mInverseN) / np.dot(normal,mInverseN)
        self.pF -= (2*impulse)*normal
        return

    def descend(self, firstPEStep=10., maxSteps=100, minimumShift=0.01,
                     stepFactor=2., debug=0, learning_rate = 0.01):
        ''' Use the force calculation to do a gradient descent
        on the potential in q space.  Recalculates the gradient at
        every step and adjusts next step size upwards if a gradient is
        successful.  Tries a reduced step size if PE does not
        decrease by minimumShift.
        Convergence is when the gradient step cannot give more 
        than minimumShift.

        firstPEStep: initial step size is chosen to yield this PE
                     decrease if gradient is constant.
        stepFactor:  increase or decrease step size by this factor
                     on decrease or increase of PE.
        maxSteps:    stop descent after this many steps.
        minimumShift: stop descent when gradient step becomes
                     smaller than will yield this PE drop

        Returns: True if descent fails to converge in maxSteps.
        '''
        
        stepCount = 0
        q0 = np.array(self.q)
        f0 = np.array(self.f)
        theta0 = np.array(self.theta)
        pe0 = self.pe(q0, theta0)

        if debug>0:
            print("Starting PE:",pe0)

        if debug > 0 :
            print("First descend theta at fixed f")

        while stepCount < maxSteps:
            if debug>0:
                print("..Iteration",stepCount, 'pe', pe0, theta0)
            force0Theta = self.getForceTheta(f0, theta0)
            #same for force in theta
 
            theta1 = theta0 - learning_rate*force0Theta
            pe1 = self.pe(q0, theta1)

            if pe1 > pe0-minimumShift:
                break
            theta0 = theta1
            pe0 = pe1
            stepCount += 1

            self.setParams(q0,theta1)


        stepCount = 0

        chisqStep = firstPEStep
        if debug > 0:
            print("Now descend f at fixed theta")
        while stepCount < maxSteps:
            if debug>0:
                print("..Iteration",stepCount, 'pe', pe0)
            force0F = self.getForceF(f0, theta1)
            # Project away force normal to constraint.  No mass in use, just put p*f0 = 0.
            force0F -= (np.dot(force0F,f0)/np.dot(f0,f0))*f0 
            # Rescale force to be a step of unit (or less) linearized chisq change
            fScale = np.dot(force0F,force0F)
            if fScale > 1:
                force0F /= np.dot(force0F,force0F)
            

            # Make a first attempt using last chisqStep:
            q1 = q0 + chisqStep*force0F

            self.renormalizeQ(q1) # Put q onto manifold
            pe1 = self.pe(q1, theta0)
            if debug>0:
                print("...chisqStep",chisqStep,"yields PE",pe0)
            if pe1 < pe0-minimumShift:
                # increase step size for next attempt
                chisqStep *= stepFactor
            else:
                # Try smaller steps of the same force til it's significantly downhill
                while pe1 > pe0-minimumShift:
                    chisqStep /= stepFactor
                    if chisqStep < minimumShift:
                        # quit - not getting anywhere.
                        if debug>0:
                            print("Done - chisqStep too small now")
                        self.setQ(q1,theta1)
                        stepCount = maxSteps
                        break
                    q1 = q0 + chisqStep*force0F
                    self.renormalizeQ(q1) # Put q onto manifold
                    pe1 = self.pe(q1, theta1)


                    if debug>0:
                        print("...chisqStep",chisqStep,"yields PE",pe1)

            # We have achieved a reduction.  Move along
            np.copyto(q0,q1)
            np.copyto(theta0,theta1)
            f0 = np.exp(-q1)
            pe0 = pe1
            stepCount = stepCount+1

        # Get here from too many steps.  Save wherever we are.
        self.setParams(q0,theta0)

        if debug > 0:
            print("Now walk both together for a few steps")

        stepCount = 0
        chisqStep = firstPEStep
        while stepCount < maxSteps:
            force0F = self.getForceF(f0, theta0)
            force0F -= (np.dot(force0F,f0)/np.dot(f0,f0))*f0 
            fScale = np.dot(force0F,force0F)
            if fScale > 1:
                force0F /= np.dot(force0F,force0F)

            force0Theta = self.getForceTheta(f0, theta0)

            # Make a first attempt using last chisqStep:
            q1 = q0 + chisqStep*force0F
            theta1 = theta0 - learning_rate*force0Theta

            self.renormalizeQ(q1) # Put q onto manifold
            pe1 = self.pe(q1, theta1)
            if debug>0:
                print("...chisqStep",chisqStep,"yields PE",pe0)
            if pe1 < pe0-minimumShift:
                # increase step size for next attempt
                chisqStep *= stepFactor
            else:
                # Try smaller steps of the same force til it's significantly downhill
                while pe1 > pe0-minimumShift:
                    chisqStep /= stepFactor
                    if chisqStep < minimumShift:
                        # quit - not getting anywhere.
                        if debug>0:
                            print("Done - chisqStep too small now")
                        self.setQ(q1,theta0)
                        stepCount = maxSteps
                        break
                    q1 = q0 + chisqStep*force0F
                    theta1 = theta0 - learning_rate*force0Theta

                    self.renormalizeQ(q1) # Put q onto manifold
                    pe1 = self.pe(q1, theta1)


                    if debug>0:
                        print("...chisqStep",chisqStep,"yields PE",pe1)

            # We have achieved a reduction.  Move along
            np.copyto(q0,q1)
            np.copyto(theta0, theta1)
            self.setParams(q0,theta0)

            f0 = np.exp(-q1)
            pe0 = pe1
            stepCount = stepCount+1

        return True
    

class GeneralHMC:
    '''Class that will execute the HMC constrained to simplex. Maintains
    an internal state with current position and momentum. The Hamiltonian's
    state variables q are the (negative) log of the quantities f that
    sum to unity to form the simplex.  This precludes the possibility of
    boundary crossing, but there is still a boundary set at `qMax` where
    a bounce step is implemented, to keep probabilities from running away.

    The iterative "RATTLE" algorithm is used to solve the constrained dynamics
    of each time step.

    External classes (those defining the probability on the simplex) use
    the f's as variables.  But internally the dynamics (and momenta) use q.
    '''
    
    # Return codes for samples
    accept = 0   # Accepted new state
    reject = 1   # Rejected step by Metropolis
    tooManyBounces = 2 # Step rejected from too many bounces
    didNotConverge = 3 # Newton iteration failure in RATTLE

    def __init__(self, thetaStart, negLogP,dnegLogPdTheta,
                 massTheta = None,
                 dt = 0.2, 
                 tRange=(0.1*np.pi,2.1*np.pi),):
        '''
        Set up class to perform HMC operations for the case where 
        constraint of sum(f)=1 is in place.  
        fStart:   vector of f values at start of chain
        negLogP:  function of f that yields -log P(f) where P is the dist. to sample
        dNegLogPdF: function of f yielding gradient vector of -log(P) w.r.t. f
        dNegLogPdF: function of f yielding gradient vector of -log(P) w.r.t. theta
        mass:     mass matrix (=local Hessian, ideally) for the f's. Can be changed
                  later with `setMass`, defaults to identity.
        dt:       time step used for Hamiltonian leapfrog integrator
        tRange:   range of integration times used for each sample.  Each time
                  step chooses a duration at random uniformly from this interval.
        qMax:     maximum allowed value of -log(f).  HMC trajectory will
                  bounce off of this boundary, incrementing bounceCount
        sumConstraint:  For mass matrix, a soft Gaussian constraint is
                  set on sum(f)=1+-sumConstraint.  This is needed rather
                  than infinite mass since finite trajectories have
                  finite component perp to boundary.  1/sqrt(N_tot) is a
                  good value for Dirichlet distributions with N_tot counts.
        maxBounces: One sample will abort if its trajectory bounces more than
                  this.

        '''
        self.negLogP = negLogP
        self.dNegLogPdTheta = dnegLogPdTheta
        self.dt = dt
        self.tRange = tRange
        self.pTheta = np.zeros_like(thetaStart)
        self.theta = thetaStart


        self.setMassTheta(massTheta)

        # Set initial coordinate and first force calculation
        self.setParams(self.theta)
        
        # Initialize some potentially interesting operation counters
        self.resetCounters()
        return
    

    def setMassTheta(self, hessianTheta = None):
        if hessianTheta is None:
            self.massTheta = np.eye(len(self.theta))
        else:
            self.massTheta = hessianTheta

        self._mCholeskyTheta = None 
        self._mInverseTheta = None 
        self._pGeneratorTheta = None

    @property
    def mInverseTheta(self):
        ''' Calculate the Cholesky decomposition of the mass matrix if needed
        and make an inverse and a momentum generator from it.'''
        if self._mInverseTheta is None:
            # Do the Cholesky decomposition
            #print("Inverting:") #???
            self._mCholeskyTheta = scipy.linalg.cho_factor(self.massTheta, lower=True)
            self._mInverseTheta = scipy.linalg.cho_solve(self._mCholeskyTheta,np.eye(self.massTheta.shape[0]))
            #print("...done:") #???
            # Make RNG for momentum
            self._pGeneratorTheta = MultivariateGaussian(cholesky=self._mCholeskyTheta)
        return self._mInverseTheta

    
    @property
    def pGeneratorTheta(self):
        '''The random generator for conjugate momenta'''
        if self._pGeneratorTheta is None:
            # Force creation of a pGenerator
            self.mInverseTheta
        return self._pGeneratorTheta
    

    def resetCounters(self):
        # Reset `stepCount` and `bounceCount`
        self.stepCount = 0
        self.bounceCount = 0
        return
    
    def setParams(self,theta):
        # Reset internal q value, calculate matching f, clear relevant caches
        np.copyto(self.theta,theta)
        #self._mInverseTheta = None
        self._forceTheta = None

        return
    

    @property
    def forceTheta(self):
        # Access force at currently cached q
        if self._forceTheta is None:
            self._forceTheta = self.getForceTheta(self.theta)
        return self._forceTheta

    def newMomentum(self):
        ''' Draw new momentum from multivariate Gaussian, project out
        component normal to constraint'''
        self.pTheta = self.pGeneratorTheta.sample()
        # Project away the motion normal to the constraint, so
        return

    # The routines below operate at specified f/q/p, not self values

    def getForceTheta(self,theta):
        # Calculate (theta-space) force at given f values
        # Convert from f to q by multiplying by -f and subtracting 1
        return self.dNegLogPdTheta(theta)

    def pe(self,theta):
        # Return (q-space) potential energy = -log P(q)
        return self.negLogP(theta) 

    def hamilton(self,ptheta,theta):
        # Return total energy at given p,q
        return self.pGeneratorTheta.negLogP(ptheta)+self.pe(theta)
    def sample(self, debug=0,pStartTheta=None,wzpz=None):
        ''' Take one HMC integration, execute Metropolis test at end.
        Returns integer signaling result - accepted (0) or rejected (1).
        The state q after the sample is available as member self.q, or
        the corresponding f is available as self.f.
        
        If pStart is given, it is used as initial momentum.  Otherwise
        a momentum is chosen from a multivariate Gaussian as the HMC
        algorithm requires.

        The WZplus3sDir instance being used as potential can be
        given in wzpz for debugging purposes.

        Returns: flag,dH  where flag is the result of sampling and dH is
        the change in Hamiltonian during the step, if it concludes.
        '''
        
        if pStartTheta is None:
            # Get new momentum
            self.newMomentum()
        else:
            # Use the supplied momentum
            self.pTheta = np.array(pStartTheta)

        # Save energy at start of integration (???Save PE from last time??)
        h0 = self.hamilton(self.pTheta,self.theta)
        hStartStep = h0
        
        if debug>1:
            print("Starting H:",h0)
        
         # Choose number of leapfrog steps (integration time)
        tTotal = np.random.uniform(low=self.tRange[0],high=self.tRange[1])
        nSteps = int(np.floor(tTotal / self.dt))
        if nSteps == 0:
            nSteps = 1   # Take at least one step
        if debug > 0:
            print("...integrating for {:d} steps".format(nSteps))
        # Save starting q in case step is rejected
        self.thetaStartSample = np.array(self.theta)

        '''
        We're going to enter a loop where each trip is an integration
        step.  This will use the RATTLE algorithm as 2 half-steps
        to meet constraints rather than use straight-up leapfrog.
        '''
        fullStep = True    # Begin with full steps
        while nSteps > 0:
            if fullStep:
                # Set up for a fresh full step
                tStep = self.dt  # Begin with a full step.
                tStepLeft = tStep # This is the total time remaining in a parent step
                bounceIndex = -1 # Coordinate index where this integration should bounce at end
                nBounce = 0 # Number of bounces this step
                
            self.stepCount = self.stepCount+1
            
            pHalfTheta = self.pTheta - (0.5 * tStep) * self.forceTheta
            theta1 = self.theta + tStep * np.matmul(self.mInverseTheta, pHalfTheta)


            # save the theta1 and calculate force
            self.setParams(theta1)  # ??? This is repeating the np.exp(-q1) an extra time

            p1Theta = pHalfTheta - (0.5*tStep)*self.forceTheta  # 2nd kick at theta from force at q1, theta1
            

            self.pTheta = p1Theta
            
            theta0 = self.theta.copy()
            p0Theta = self.pTheta.copy()
            nSteps -= 1

                                
        # Now completed the integration.
        # Accept or reject the new value, using prob(new) / prob(old)
        # Rescale q first to clear up any roundoff accumulations
        h1 = self.hamilton(self.pTheta,self.theta)
        dLogP = h0-h1
        if debug>0:
            print("dLogP:",dLogP,"H0,1:",h0,h1)

        # apply Metropolis criterion
        if np.random.uniform() < np.exp(dLogP):
            if (debug>0):
                print("Acceptance at p2/p1=",np.exp(dLogP))
            return self.accept,dLogP
        else:
            self.setParams(self.thetaStartSample)
            if debug>0:
                print("Rejection at p2/p1=",np.exp(dLogP))
            return self.reject,dLogP


    def descend(self, firstPEStep=10., maxSteps=100, minimumShift=0.01,
                     stepFactor=2., debug=0, learning_rate = 0.01):
        ''' Use the force calculation to do a gradient descent
        on the potential in q space.  Recalculates the gradient at
        every step and adjusts next step size upwards if a gradient is
        successful.  Tries a reduced step size if PE does not
        decrease by minimumShift.
        Convergence is when the gradient step cannot give more 
        than minimumShift.

        firstPEStep: initial step size is chosen to yield this PE
                     decrease if gradient is constant.
        stepFactor:  increase or decrease step size by this factor
                     on decrease or increase of PE.
        maxSteps:    stop descent after this many steps.
        minimumShift: stop descent when gradient step becomes
                     smaller than will yield this PE drop

        Returns: True if descent fails to converge in maxSteps.
        '''
        
        stepCount = 0
        theta0 = np.array(self.theta)
        pe0 = self.pe(theta0)

        if debug>0:
            print("Starting PE:",pe0)

        while stepCount < maxSteps:
            if debug>0:
                print("..Iteration",stepCount, 'pe', pe0, theta0)
            force0Theta = self.getForceTheta(theta0)
 
            theta1 = theta0 - learning_rate*force0Theta
            pe1 = self.pe(theta1)

            if pe1 > pe0-minimumShift:
                break
            theta0 = theta1
            pe0 = pe1
            stepCount += 1

            self.setParams(theta1)


        return True
    
##################################################################

'''Following are classes that satisfy the interface for
probabilities to be used by SimplexHMC.  This interface
must include negLogP(), dNegLogPdF(), and hessian() methods.
All methods take f's as arguments and take derivatives w.r.t.
f components.  The fisher() method if implemented should yield
some approximation to the expectation of Hessian.
Simplex constraint should NOT be included
in the potential or derivatives.
'''

class MultivariateGaussian:
    # Multivariate random deviate with fixed cov and zero mean.  Has sample()
    # as well as basic SimplexHMC interface, and is called by SimplexHMC for
    # momentum sampling.
    
    def __init__(self, cov=None, cholesky=None):
        # Create a multivariate Gaussian by providing either its covariance
        # matrix or its cholesky decomposition from scipy.linalg.cho_factor.
        # The class need save only the cholesky
        if cov is not None:
            if cholesky is not None:
                raise ValueError("Cannot provide both covariance and its Cholesky")
            if len(cov.shape)!=2 or not np.all(np.isclose(cov.T,cov,rtol=1e-5,atol=1e-8)):
                raise RuntimeError("Covariance matrix is not symmetric")
            # Symmetrize the covariance
            tmp = 0.5*(cov.T + cov)
            # Do the Cholesky and zero above the diagonal
            self.L = np.tril(scipy.linalg.cho_factor(tmp,lower=True)[0])
        else:
            if cholesky is None:
                raise ValueError("Must provide one of covariance or cholesky")
            if not cholesky[1]:
                raise ValueError("Cholesky must be lower triangular")
            self.L = np.tril(cholesky[0])  # Make a local copy, zeroed above diagonal
        # Precompute log(sqrt | 2 pi cov | )
        self.neglogNorm = self.L.shape[0]*0.5*np.log(2*np.pi) + np.sum(np.log(np.diag(self.L)))
        # Initialize a unit Gaussian deviate
        self.gauss = scipy.stats.norm()
        return
    def sample(self, n=None):
        '''Generate n samples from the distribution.
        Returns a single vector if n is None, else an m x n array where m
        is the dimension of the random variable.
        '''
        m = self.L.shape[0]
        if n is None:
            v = self.gauss.rvs(size=m)
        elif n>0:
            v = self.gauss.rvs(size=m*n).reshape((m,n))
        else:
            raise ValueError("Number of samples must be positive")
        return np.matmul(self.L, v)
    def negLogP(self, x):
        # Return negative log of probability at vector x
        if x.shape!=(self.L.shape[0],):
            raise ValueError("Wrong dimension for input vector")
        v = scipy.linalg.solve_triangular(self.L, x, lower=True, check_finite=False)
        return 0.5 * np.sum(v*v) + self.neglogNorm
    def dNegLogPdF(self, f):
        return scipy.linalg.cho_solve((self.L,True),f) 
    def hessian(self,f=None):
        # Fisher matrix is inverse of covariance
        return scipy.linalg.cho_solve((self.L,True),np.eye(self.L.shape[0]))
    def fisher(self):
        return self.hessian()


class PDirichlet:
    ''' Dirichlet distribution for N draws per category.  The N values
        are equal to $\alpha-1$ in the version on Wikipedia.  The
        Fisher matrix here does not include the delta function on sum(f).
    '''
    def __init__(self, N=None, alpha=None):
        '''Initialize with vector of counts of events per bin'''
        if N is not None and alpha is None:
            self.N = N
        elif N is None and alpha is not None:
            self.N = alpha-1
        else:
            raise ValueError("Must specify one of N or alpha for PDirichlet")

        # Normalization factor for means and such
        self.alpha0 = np.sum(self.N+1)        
        return
    def negLogP(self, f):
        return -np.sum( self.N * np.log(f))
    def dNegLogPdF(self, f):
        if np.any(f<=0.):
            raise ValueError("Non-positive f for Dirichlet")
        return -self.N/f
    def hessian(self,f):
        n = len(self.N)
        hess = np.zeros((n,n),dtype=float)
        # Diagonal matrix in the absence of simplex constraint:
        hess.ravel()[::n+1] = self.N/np.square(f)
    def fisher(self):
        # We will set up Fisher to allow for variance normal to the simplex
        # otherwise the Fisher elements are infinite.
        fish = np.diag( self.alpha0 / (self.N+1))
        fish *= self.alpha0+1
        return fish
    def mean(self):
        # Return the expectation value of the distribution
        return (self.N+1) / self.alpha0


class ConvolutionProbability:
    '''Class that represents the probability distribution of nCat category fractions
    when given nEvent events that have known likelihood of generation from each category.
    Satisfies the interface needed for HMC.
    '''
    def __init__(self, likelihood, nominalF=None):
        '''
        likelihood = array L of shape (nEvent, nCat) where L_ij is the likelihood
          of generating event i's data given an object in category j.
        nominalF = array of length nCat with a guess at the fractions f_j, in each bin,
        used to estimate Fisher matrix.  If None, will bootstrap an observed 
        distribution from L.
        '''
        s = likelihood.shape
        if len(s)!=2:
            raise ValueError("likelihood array is not 2d")
        self.likelihood = np.array(likelihood)
        self.nCat = s[1]
        self.nEvent = s[0]
        if nominalF is None:
            # Bootstrap an initial F estimate from L's by calculating
            # mean of galaxy distribution with uniform f.
            n = likelihood / np.sum(likelihood,axis=1)[:,np.newaxis]
            f = np.sum(n,axis=0)  
            # Then apply these f's and refine probabilities again
            n = likelihood * f  # broadcasting will do f[np.newaxis,:]
            n /= np.sum(n,axis=1)[:,np.newaxis]  # Normalize prob to 1 for each event
            f = np.sum(n,axis=0)   # Sum probs in each category
            f /= np.sum(f)         # Normalize category probabilities to unity
            self.nominalF = f
        elif nominalF.shape == (nCat,):
            # Renormalize the input
            norm = np.sum(nominalF)
            self.nominalF = nominalF / norm
        else:
            raise ValueError("Wrong dimensions for nominalF")
        return
    def negLogP(self,f):
        # prob = \prod_i \sum_c(L_ic*f_c) 
        return -np.sum(np.log(np.matmul(self.likelihood,f)))
    def dNegLogPdF(self, f):
        # dU/df_j =  \sum_i (L_ij  / \sum_k (L_ik f_k) 
        norms = np.matmul(self.likelihood,f)
        return -np.sum( self.likelihood/norms[:,np.newaxis], axis=0)
    def hessian(self,f):
        # ???? Write this
        return
    def fisher(self):
        # We will approximate the Fisher matrix by the second derivative
        # of -log P evaluated at the nominalF values.
        norms = np.matmul(self.likelihood, self.nominalF)
        tmp = self.likelihood / norms[:,np.newaxis]  # Now L_ik / \sum_k(L_ik f_k)
        result = np.einsum('ij,ik',tmp,tmp,optimize=True) #object-wise outer product
        # Constrain radial variance - again using a finite width.
        ## result += 10.*self.nEvent ###Could tweak this number, higher = less variance
        return result

class ProbabilityProduct:
    def __init__(self,elements):
        '''Create a probability that is the product of a list of
        probability elements.'''
        self.elements = elements
    # Now simply define outputs as sums of both distributions    
    def negLogP(self,f):
        return sum([e.negLogP(f) for e in self.elements])
    def dNegLogPdF(self,f):
        return sum([e.dNegLogPdF(f) for e in self.elements])
    def hessian(self,f):
        return sum([e.hessian(f) for e in self.elements])
    def fisher(self):
        return sum([e.fisher() for e in self.elements])


class PZCZProbability:
    '''Class that has the SimplexHMC interface implemented for our
    BHM using clustering and photometry jointly for redshifts.
    We will start with a case of known densities (no bias factors)'''
    def __init__(self, photoLikelihood, clusterLikelihood, useTZ,
                       chunkSize=1024):
        '''Create a probability defined by
        $P(f) \propto \prod_i \sum_{tz} L_{it} rho_{iz} f_{tz}$
        where i indexes target galaxies, t indexes galaxy types,
        and z indexes redshifts.  L_{it} is the likelihood of obtaining
        galaxy observables given the type t.  rho_{iz} is the
        estimated (relative) galaxy density at position of galaxy i
        at redshift z.  And f_{tz} is the relative abundance of 
        galaxies of type t at redshift z, such that \sum_{t,z} f_{tz}=1.
        
        Inputs:
        * photoLikelihood is an N_g x N_t matrix giving L_{it}
        * clusterLikelihood is N_g x N_z matrix giving rho_{iz}
        * tzUse is a (2 x N_f) integer array, or a tuple of
          2 arrays of length N_f, that gives the 
          t and z indices of all of the f_{tz} values that will
          be allowed to be nonzero.
        * chunkSize gives the number of galaxies in each iteration
          of the outer computation loop ??? and later for multitasking.
          If None then we do everything in one set of numpy
          operations, which uses more intermediate memory.
        '''
        sp = photoLikelihood.shape
        sc = clusterLikelihood.shape
        
        if len(sp)!=2 or len(sc)!=2 or len(useTZ)!=2:
            raise ValueError("Input arrays are not 2d")
        self.nG = sp[0]  # Number of galaxies
        self.nT = sp[1]  # Number of types
        self.nZ = sc[1]  # Number of redshifts
        self.nF = len(useTZ[0]) # Number of type/z combos
        if sc[0]!=self.nG:
            raise ValueError("clusterLikelihood and photoLikelihood have"\
                            " different numbers of galaxies")
        if len(useTZ[1])!=self.nF:
            raise ValueError("useTZ is not of shape 2xNf")
        
        # Copy the inputs
        self.fT = np.array(useTZ[0])  # The type indices of the f's
        self.fZ = np.array(useTZ[1])  # and the redshift indices

        self.photoL = np.array(photoLikelihood)
        self.clusterL = np.array(clusterLikelihood)

            
        # Set up a set of slices of the galaxy dimension that we
        # will use to break up the biggest loop
        self.gSlices = []
        if chunkSize is None:
            chunkSize = self.nG
        for gStart in range(0,self.nG,chunkSize):
            self.gSlices.append(slice(gStart, min(gStart+chunkSize,self.nG)))
        return
    
    def negLogP(self,f):
        # Return -log(probability) at specified f
        out = 0.
        for s in self.gSlices:
            tmp1 = np.array(self.photoL[s,self.fT])
            tmp1 *= np.array(self.clusterL[s,self.fZ])
            n = np.matmul(tmp1,f)
            '''
            # Equivalent but slower??
            n = np.einsum('ik,ik,k->i',self.photoL[s,self.fT],
                            self.clusterL[s,self.fZ],f, optimize=True)
            '''
            out -= np.sum(np.log(n))
        return out
    def dNegLogPdF(self,f):
        # Return deriv of negLogP w.r.t. f at specified f
        out = np.zeros_like(f)
        for s in self.gSlices:
            #'''
            tmp1 = np.array(self.photoL[s,self.fT])
            tmp1 *= np.array(self.clusterL[s,self.fZ])
            n = np.matmul(tmp1,f)
            out -= np.sum(tmp1 / n[:,np.newaxis], axis=0)
            '''
            # An equivalent way (but slower?)
            n = np.einsum('ik,ik,k->i',self.photoL[s,self.fT],
                            self.clusterL[s,self.fZ],f, optimize=True)
            out -= np.sum(self.photoL[s,self.fT] * self.clusterL[s,self.fZ] / n[:,np.newaxis],
                            axis=0)
            '''
        return out
    def hessian(self,f):
        # ???? Write this ???
        raise Exception
    def fisher(self, nominalF):
        ''' We will approximate the Fisher matrix by the second derivative
            of -log P evaluated at the nominalF. It should be a 1-d array of 
            dimension N_f.  It will be renormalized to sum to unity.
        '''
        ff = nominalF / np.sum(nominalF)
        
        out = np.zeros( (self.nF,self.nF), dtype=float)
        for s in self.gSlices:
            # Combine photo and cluster L's at all (tz)'s
            tmp = self.photoL[s,self.fT] * self.clusterL[s,self.fZ]
            n = np.matmul(tmp, nominalF)  # Sum probabilities for each galaxy
            tmp /=  n[:,np.newaxis]   # Normalize all probs per galaxy
            out += np.einsum('ij,ik',tmp,tmp,optimize=True) #object-wise outer product
        # Put finite constraint on sum of f's, at some size large but not
        # enough to generate roundoff problems ???
        out += 10.*self.nG
        return out  
    def redshiftDistribution(self, f, zValues=None):
        '''Given fixed values of the tz bin fractions f, return the expectation
        value of the parent redshift distribution.
        Assumes that the input array f sums to unity.
        zValues = an array of dimension nZ giving z values for each bin.

        Return: if zValues=None, then just an array of dimension nZ giving
                fractions per z bin.
                If zValues is given, then also returns the mean z value.
        '''
        out = np.zeros(self.nZ, dtype=float)
        for i in range(self.nF):
            out[self.fZ[i]] += f[i]
        if zValues is None:
            return out
        else:
            return out, np.sum(out*zValues)/np.sum(out)
    
class BHMPropability:
    '''Class to interface with SimplexHMC that embodies the Bayesian
    posterior on the f's - includes clustering, photometry likelihoods
    *and* a Dirichlet prior on the f's. 
    '''
    def __init__(self, photoLikelihood, clusterLikelihood, priorTZ,
                       chunkSize=1024):
        '''Create a probability defined by MICE-style files:
        $P(f) \propto Pr(f) \prod_i \sum_{tz} L_{it} rho_{iz} f_{tz} \times $
        where i indexes target galaxies, t indexes galaxy types,
        and z indexes redshifts.  L_{it} is the likelihood of obtaining
        galaxy observables given the type t.  rho_{iz} is the
        estimated (relative) galaxy density at position of galaxy i
        at redshift z.  And f_{tz} is the relative abundance of 
        galaxies of type t at redshift z, such that \sum_{t,z} f_{tz}=1.
        $Pr(f)$ is prior on them, which is taken as Dirichlet
        with alpha's given by the priorTZ array. ???
        
        Inputs:
        * photoLikelihood is an N_g x N_t matrix giving L_{it}
        * clusterLikelihood is N_g x N_z matrix giving rho_{iz}
        * priorTZ is an N_t x N_z matrix giving the priors on f_{tz}
          to be a Dirichlet distribution with the \alpha values in
          this file.  tz combinations with zero in this file will be
          assumed to be fixed at zero (i.e. this type is assumed to
          have 0 chance of being at this redshift).  
        * chunkSize gives the number of galaxies in each iteration
          of the outer computation loop ??? and later for multitasking.
          If None then we do everything in one set of numpy
          operations, which uses more intermediate memory.
        '''

        # Check some dimensions
        if len(photoLikelihood.shape)!=2 or \
           len(clusterLikelihood.shape)!=2 or \
           len(priorTZ.shape)!=2:
            raise ValueError('Input arrays are not 2d')
        self.nG = photoLikelihood.shape[0]
        self.nT = photoLikelihood.shape[1]
        self.nZ = clusterLikelihood.shape[1]
        if clusterLikelihood.shape[0]!=self.nG or \
           priorTZ.shape!=(self.nT,self.nZ):
            raise ValueError("Input array dimensions do not match")
            
        # Now set up the photo/cluster likelihood, using the prior
        # as an initial guess
        ftz = np.where(priorTZ>0.)
        fPrior = priorTZ[ftz]
        self.nF = len(fPrior)
        self.pzwz = PZCZProbability(photoLikelihood,clusterLikelihood,
                                   ftz, chunkSize=chunkSize)
        # Make Dirichlet for the f prior
        self.prior = PDirichlet(alpha=fPrior)
        return
    
    # Now simply define outputs as sums of both distributions    
    def negLogP(self,f):
        return self.prior.negLogP(f)+self.pzwz.negLogP(f)
    def dNegLogPdF(self,f):
        return self.prior.dNegLogPdF(f)+self.pzwz.dNegLogPdF(f)
    def hessian(self,f):
        return self.prior.hessian(f) + self.pzwz.hessian(f)
    def fisher(self, nominalF=None):
        # Evaluate Fisher matrix, using 2nd derivs at nominalF if given,
        # else use the mean of he prior.
        if nominalF is None:
            nominalF = self.prior.mean()
        return self.prior.fisher() + self.pzwz.fisher(nominalF=nominalF)
    def redshiftDistribution(self, f):
        # Get mean redshift distribution given f's, as per PZCZProbability
        return self.pzwz.redshiftDistribution(f)



