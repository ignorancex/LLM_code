import numpy as np
import types
import operator
import time
import torch

realmin = np.finfo(np.float64).tiny
ccc = 20000

def maxc(x,v):
  return x*(x>=v) + v*(x < v)
  
def isscalar(v):
  return not hasattr(v,"shape") and type(v) in ["int", "float"]

def isa(f):
  return isinstance(f, types.FunctionType)

def GPSR_Basic(y,A,tau,device,dtype,**varargin):   
  # flag for initial x (can take any values except 0,1,2)
  Initial_X_supplied = 3333

  # Set the defaults for the optional parameters
  stopCriterion = 3
  tolA = 0.01
  tolD = 0.0001
  debias = 0
  maxiter = 10000
  maxiter_debias = 500
  miniter = 5
  miniter_debias = 0
  init = 0
  compute_mse = 0
  AT = 0
  verbose = 1
  continuation = 0
  cont_steps = 5
  firstTauFactorGiven = 0

  # sufficient decrease parameter for GP line search
  mu = 0.1
  # backtracking parameter for line search
  lambda_backtrack = 0.5

  # Set the defaults for outputs that may not be computed
  debias_start = 0
  x_debias = None
  mses = np.empty((ccc,1))
  times = np.empty((ccc,1))
  objective = np.empty((ccc,1))
  lambdas=np.empty((ccc,1))

  # Read the optional parameters
  for k in varargin.keys():
    uk = k.upper()
    value = varargin[k]

    if uk=='STOPCRITERION':
      stopCriterion = value
    elif uk=='TOLERANCEA':       
      tolA = value
    elif uk=='TOLERANCED':
      tolD = value
    elif uk=='DEBIAS':
      debias = value
    elif uk=='MAXITERA':
      maxiter = value
    elif uk=='MAXITERD':
      maxiter_debias = value
    elif uk=='MINITERA':
      miniter = value
    elif uk=='MINITERD':
      miniter_debias = value
    elif uk=='INITIALIZATION':
      if hasattr(value,"shape") and np.prod(value.shape) > 1:   # we have an initial x
        init = Initial_X_supplied    # some flag to be used below
        x = value
      else:
        init = value
    elif uk=='CONTINUATION':
      continuation = value  
    elif uk=='CONTINUATIONSTEPS': 
      cont_steps = value
    elif uk=='FIRSTTAUFACTOR':
      firstTauFactor = value
      firstTauFactorGiven = 1
    elif uk=='TRUE_X':
      compute_mse = 1
      true = value
    elif uk=='AT':
      AT = value
    elif uk=='VERBOSE':
      verbose = value
    elif uk =="ITERS":
        iters = value
    else:
      # Hmmm, something wrong with the parameter string
      print('Unrecognized option:', value)


  if stopCriterion not in [0,1,2,3,4]:
    print(['Unknown stopping criterion'])

  # if A is a function handle, we have to check presence of AT,
  if isa(A) and not isa(AT):
    print(['The function handle for transpose of A is missing'])

  # if A is a matrix, we find out dimensions of y and x,
  # and create function handles for multiplication by A and A',
  # so that the code below doesn't have to distinguish between
  # the handle/not-handle cases
  if not isa(A):
    AT = lambda x: A.T @ x
    A = lambda x:  A @ x

  # from this point down, A and AT are always function handles.

  # Precompute A'*y since it'll be used a lot
  Aty = AT(y)

  # Initialization
  if init==0:   # initialize at zero, using AT to find the size of x
      x = AT(torch.zeros(y.shape),dtype=dtype,device=device)
  elif init==1:   # initialize randomly, using AT to find the size of x
      x = torch.randn((AT(torch.zeros(y.shape,dtype=dtype,device=device))).shape,dtype=dtype,device=device)
  elif init==2:   # initialize x0 = A'*y
      x = Aty 
  elif init==Initial_X_supplied:
      # initial x was given as a function argument just check size
      if not operator.eq(A(x).shape, y.shape):
        print(['Size of initial x is not compatible with A']) 
  else:
      print(['Unknown ''Initialization'' option'])

  # now check if tau is an array if it is, it has to 
  # have the same size as x
  if hasattr(tau,"shape") and np.prod(tau.shape) > 1:
    try:
        dummy = x*tau
    except:
        print(['Parameter tau has wrong dimensions it should be scalar or size(x)']),
        
  # if the true x was given, check its size
  if compute_mse and not operator.eq(true.shape, x.shape):  
    print(['Initial x has incompatible size']) 

  # if tau is scalar, we check its value if it's large enough,
  # the optimal solution is the zero vector
  if isscalar(tau):
    aux = AT(y)
    max_tau = abs(aux).max()
    if tau >= max_tau:
        x = torch.zeros(aux.shape,dtype=dtype,device=device)
        if debias:
          x_debias = x
        objective[1] = 0.5 * (y.T @ y).cpu().numpy()
        times[1] = 0
        if compute_mse:
            mses[1] = np.power(true.cpu().numpy(), 2).sum()
        return

  # initialize u and v
  u =  x*(x >= 0)
  v = -x*(x <  0)

  # define the indicator vector or matrix of nonzeros in x
  nz_x = (x != 0.0)
  num_nz_x = nz_x.sum()

  # Compute and store initial value of the objective function
  resid =  y - A(x)
  f = 0.5*(resid.T @ resid) + (tau*u).sum() + (tau*v).sum()

  # auxiliary vector on ones, same size as x
  onev = torch.ones(x.shape,dtype=dtype,device=device)

  # start the clock
  t0 = time.time()

  # store given tau, because we're going to change it in the
  # continuation procedure
  final_tau = tau

  # store given stopping criterion and threshold, because we're going 
  # to change them in the continuation procedure
  final_stopCriterion = stopCriterion
  final_tolA = tolA

  # set continuation factors
  if continuation and cont_steps > 1:
    # If tau is scalar, first check top see if the first factor is 
    # too large (i.e., large enough to make the first 
    # solution all zeros). If so, make it a little smaller than that.
    # Also set to that value as default
    if isscalar(tau):
        if (firstTauFactorGiven == 0) or (firstTauFactor*tau >= max_tau):
          firstTauFactor = 0.8*max_tau / tau
          print('parameter FirstTauFactor too large changing')
    cont_factors = 10 ** torch.arange(np.log10(firstTauFactor),0,np.log10(1/firstTauFactor)/(cont_steps-1),device=device,dtype=dtype)
  else:
    cont_factors = [1]
    cont_steps = 1

  iter = 1
  if compute_mse:
    mses[iter] = (np.power((x-true).cpu().numpy(),2)).sum()

  # loop for continuation
  for cont_loop in range(1, cont_steps+1):

      tau = final_tau * cont_factors[cont_loop-1]
      
      if verbose:
          print('\nSetting tau = %8.4f\n' % (tau))
      
      if cont_loop == cont_steps:
        stopCriterion = final_stopCriterion
        tolA = final_tolA
      else:
        stopCriterion = 3
        tolA = 1e-3
      
      # Compute and store initial value of the objective function
      resid =  y - A(x)
      f = 0.5*(resid.T @ resid) + \
              (tau*u).sum() + (tau*v).sum()

      objective[iter] = f.cpu().numpy()
      times[iter] = time.time() - t0
      
      # Compute the useful quantity resid_base
      resid_base = y - resid

      # control variable for the outer loop and iteration counter
      # cont_outer = (norm(projected_gradient) > 1.e-5) 

      keep_going = 1

      if verbose:
        print('\nInitial obj=%10.6e, nonzeros=%7d\n' % (f,num_nz_x))

      while keep_going:

        x_previous = x

        # compute gradient
        temp = AT(resid_base)
        term  =  temp - Aty
        gradu =  term + tau
        gradv = -term + tau

        # set search direction
        #du = -gradu dv = -gradv dx = du-dv 
        dx = gradv-gradu
        old_u = u 
        old_v = v

        # calculate useful matrix-vector product involving dx
        auv = A(dx)
        dGd = auv.T @ auv

        # calculate unconstrained minimizer along this direction, use this
        # as the first guess of steplength parameter lambda
        #  lambda0 = - (gradu(:)'*du(:) + gradv(:)'*dv(:)) / dGd

        # use instead a first guess based on the "conditional" direction
        condgradu = ((old_u>0) | (gradu<0)) * gradu
        condgradv = ((old_v>0) | (gradv<0)) * gradv
        auv_cond = A(condgradu-condgradv)
        dGd_cond = auv_cond.T @ auv_cond
        lambda0 = (gradu.T @ condgradu + gradv.T @ condgradv) / (dGd_cond + realmin)

        # loop to determine steplength, starting wit the initial guess above.
        lambdaA = lambda0 
        while 1:
          # calculate step for this lambda and candidate point
          du = torch.maximum(u-lambdaA*gradu,torch.zeros_like(u)) - u 
          u_new = u + du
          dv = torch.maximum(v-lambdaA*gradv,torch.zeros_like(u)) - v 
          v_new = v + dv
          dx = du-dv 
          x_new = x + dx

          # evaluate function at the candidate point
          resid_base = A(x_new)
          resid = y - resid_base
          f_new = 0.5*(resid.T @ resid) + (tau*u_new).sum() + (tau*v_new).sum()   
          # test sufficient decrease condition
          if f_new <= f + mu * (gradu.T @ du + gradv.T @ dv):
            #disp('OK')  
            break
          lambdaA = lambdaA * lambda_backtrack
          if verbose:
            print('    reducing lambda to %6.2e\n' % (lambdaA))
        u = u_new 
        v = v_new 
        prev_f = f 
        f = f_new
        uvmin = torch.minimum(u,v) 
        u = u - uvmin 
        v = v - uvmin 
        x = u-v

        # calculate nonzero pattern and number of nonzeros (do this *always*)
        nz_x_prev = nz_x
        nz_x = (x!=0.0)
        num_nz_x = (nz_x).sum()
        
        iter = iter + 1
        objective[iter] = f.cpu().numpy()
        times[iter] = time.time()-t0
        lambdas[iter] = lambdaA.cpu().numpy()

        if compute_mse:
          err = true - x
          mses[iter] = (err.T @ err).cpu().numpy()
        
        # print out stuff
        if verbose:
          if iters is not None and iter>=iters:
            keep_going=False
            break
          print('It =%4d, obj=%9.5e, lambda=%6.2e, nz=%8d   ' % (iter, f, lambdaA, num_nz_x))
        
    
        if stopCriterion==0:
          # compute the stopping criterion based on the change 
          # of the number of non-zero components of the estimate
          num_changes_active = (nz_x!=nz_x_prev).sum()
          if num_nz_x >= 1:
            criterionActiveSet = num_changes_active
          else:
            criterionActiveSet = tolA / 2
          keep_going = (criterionActiveSet > tolA)
          if verbose:
            print('Delta n-zeros = %d (target = %e)\n' %(criterionActiveSet , tolA))
        elif stopCriterion==1:
          # compute the stopping criterion based on the relative
          # variation of the objective function.
          criterionObjective = abs(f-prev_f)/(prev_f)
          keep_going =  (criterionObjective > tolA)
          if verbose:
            print('Delta obj. = %e (target = %e)\n' %(criterionObjective , tolA))
        elif stopCriterion==2:
          # stopping criterion based on relative norm of step taken
          delta_x_criterion = dx.norm()/x.norm()
          keep_going = (delta_x_criterion > tolA)
          if verbose:
            print('Norm(delta x)/norm(x) = %e (target = %e)\n' %(delta_x_criterion,tolA))
        elif stopCriterion==3:
          # compute the "LCP" stopping criterion - again based on the previous
          # iterate. Make it "relative" to the norm of x.
          w = [ torch.minimum(gradu, old_u), torch.minimum(gradv, old_v) ]
          criterionLCP = torch.linalg.norm(w, ord=np.inf)
          criterionLCP = criterionLCP / max([1.0e-6, torch.linalg.norm(old_u, ord=np.inf), torch.linalg.norm(old_v, ord=np.inf)])
          keep_going = (criterionLCP > tolA)
          if verbose:
            print('LCP = %e (target = %e)\n'%(criterionLCP,tolA)) 
        elif stopCriterion==4:
          # continue if not yeat reached target value tolA
          keep_going = (f > tolA)
          if verbose:
            print('Objective = %e (target = %e)\n' %(f,tolA)) 
        else:
          print(['Unknwon stopping criterion'])
          
        # take no less than miniter
        if iter<=miniter:
          keep_going = 1
        else: #and no more than maxiter iterations  
            if iter > maxiter:
                keep_going = 0

  # Print results
  if verbose:
    print('\nFinished the main algorithm!\nResults:\n')
    print('||A x - y ||_2^2 = %10.3e\n' %(resid.T @ resid))
    print('||x||_1 = %10.3e\n',(abs(x)).sum())
    print('Objective function = %10.3e\n' %(f))
    nz_x = (x!=0.0) 
    num_nz_x = (nz_x).sum()
    print('Number of non-zero components = %d\n' %(num_nz_x))
    print('CPU time so far = %10.3e\n' %(times[iter]))
    print('\n')

  # If the 'Debias' option is set to 1, we try to remove the bias from the l1
  # penalty, by applying CG to the least-squares problem obtained by omitting
  # the l1 term and fixing the zero coefficients at zero.

  # do this only if the reduced linear least-squares problem is
  # overdetermined, otherwise we are certainly applying CG to a problem with a
  # singular Hessian

  if (debias and ((x!=0).sum()!=0)):
      
      if (num_nz_x > len(y)):
        if verbose:
          print('\n')
          print('Debiasing requested, but not performed\n')
          print('There are too many nonzeros in x\n\n')
          print('nonzeros in x: %8d, length of y: %8d\n' %(num_nz_x, len(y)))

      elif (num_nz_x==0):
        if verbose:
          print('\n')
          print('Debiasing requested, but not performed\n')
          print('x has no nonzeros\n\n')
      else:
        if verbose:
          print('\n')
          print('Starting the debiasing phase...\n\n')
        
        x_debias = x
        zeroind = (x_debias!=0) 
        cont_debias_cg = 1
        debias_start = iter
        
        # calculate initial residual
        resid = A(x_debias)
        resid = resid-y
        resid_prev = eps * torch.ones(resid.shape,dtype=dtype,device=device)
        
        rvec = AT(resid)
        
        # mask out the zeros
        rvec = rvec * zeroind
        rTr_cg = rvec.T @ rvec
        
        # set convergence threshold for the residual || RW x_debias - y ||_2
        tol_debias = tolD * (rvec.T @ rvec)
        
        # initialize pvec
        pvec = -rvec
        
        # main loop
        while cont_debias_cg:
          
          # calculate A*p = Wt * Rt * R * W * pvec
          RWpvec = A(pvec)      
          Apvec = AT(RWpvec)
          
          # mask out the zero terms
          Apvec = Apvec * zeroind
          
          # calculate alpha for CG
          alpha_cg = rTr_cg / (pvec.T @ Apvec)
          
          # take the step
          x_debias = x_debias + alpha_cg * pvec
          resid = resid + alpha_cg * RWpvec
          rvec  = rvec  + alpha_cg * Apvec
          
          rTr_cg_plus = rvec.T @ rvec
          beta_cg = rTr_cg_plus / rTr_cg
          pvec = -rvec + beta_cg * pvec
          
          rTr_cg = rTr_cg_plus
          
          iter = iter+1
          
          objective[iter] = (0.5*(resid.T @ resid) + (tau * abs(x_debias)).sum()).cpu().numpy()
          times[iter] = cputime - t0
          
          if compute_mse:
            err = true - x_debias
            mses[iter] = (err.T @ err).cpu().numpy()
          
          if verbose:
            # in the debiasing CG phase, always use convergence criterion
            # based on the residual (this is standard for CG)
            print(' Iter = %5d, debias resid = %13.8e, convergence = %8.3e\n' %(iter, resid.T @ resid, rTr_cg / tol_debias))
          
          cont_debias_cg = (iter-debias_start <= miniter_debias )| \
                          ((rTr_cg > tol_debias) & \
                          (iter-debias_start <= maxiter_debias))
        
        if verbose:
          print('\nFinished the debiasing phase!\nResults:\n')
          print('||A x - y ||^2_2 = %10.3e\n' %(resid.T @ resid))
          print('||x||_1 = %10.3e\n' %((abs(x)).sum()))
          print('Objective function = %10.3e\n' %(f))
          nz = (x_debias!=0.0)
          print('Number of non-zero components = %d\n' %((nz).sum()))
          print('CPU time so far = %10.3e\n' %(times[iter]))
          print('\n')
      
      if compute_mse:
        mses = mses/len(true)

  return [x,x_debias,objective,times,debias_start,mses]
      


