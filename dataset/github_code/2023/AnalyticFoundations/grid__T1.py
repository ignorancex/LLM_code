import torch as th
import gridutil

# an (n+1)-point equally spaced grid of 1d torus [-pi,pi] with spacing dx
# lt endpt: x[0] :~: -pi
# rt endpt: x[n] :~:  pi-dx

class grid__T1():
	
	def __init__(self,n_input,type=th.float64):
		
		self.pi = 3.141592653589793238462643383279502884197169399375105820974
		self.n  = n_input
		self.x  = th.linspace(-self.pi,self.pi,self.n+2,dtype=type)[0:self.n+1]		# grid points x_0,...,x_n
		self.dx = (self.x[1] - self.x[0]).item()									# grid spacing
		self.k  = th.zeros(self.n+1,dtype=type)										# Fourier wavenumbers

		if (self.n+1)%2 == 0:														# Compute Fourier wave numbers
    
			m = int( (self.n+1)/2 )
			v = th.arange(0,m)
			self.k[0:m]        = v
			self.k[m:self.n+1] = -1.0-v.flip(0)
    
		else:
    
			m = int(self.n/2)
			v = th.arange(1,m+1)
			self.k[1:m+1]        = v
			self.k[m+1:self.n+1] = -v.flip(0)

	def to(self,device): 

		self.x  = self.x.to(device)
		self.k  = self.k.to(device)

	def derivative(self,funcs,p): 

		# compute p-th order derivative of periodic functions in funcs via fft
		# computation expands along first dimension

		F        = th.rfft(funcs,1,onesided=False)
		Ip       = (1j)**p
		xi_r     = th.pow(self.k,p)*Ip.real
		xi_i     = th.pow(self.k,p)*Ip.imag
		tmp      = F[:,:,0]*xi_r - F[:,:,1]*xi_i
		smp      = F[:,:,1]*xi_r + F[:,:,0]*xi_i
		F[:,:,0] = tmp
		F[:,:,1] = smp

		return th.ifft(F,1)[:,:,0]
	
	def primitive(self,funcs): 

		# compute mean-zero primitive of periodic functions in funcs via fft
		# computation expands along first dimension

		c        = th.mean(funcs,1).view(funcs.size(0),1)
		F        = th.rfft(funcs-c,1,onesided=False)
		xi       = th.pow(self.k,-1)
		xi[0]    = 1.0
		tmp      = -F[:,:,0]*xi
		F[:,:,0] = F[:,:,1]*xi
		F[:,:,1] = tmp

		return th.ifft(F,1)[:,:,0] + c*self.x

	def P0(self,funcs): 

		# compute zero-Dirichlet primitive of (f - mean(f)) for periodic 
		#	functions *funcs
		# computation expands along first dimension

		out = funcs.clone()
		out.sub_( th.mean(funcs,1).unsqueeze(1) )
		out = self.primitive(out)

		return out - out[:,0].unsqueeze(1)

	def diffuse(self,funcs,tau): 

		# compute inv(I - tau /\)*funcs subject to periodic boundary conditions via fft
		# computation expands along first dimension

		assert(tau>=0)
		F        = th.rfft(funcs,1,onesided=False)
		xi       = self.dx/( self.dx + 2.0*(tau/self.dx)*(1.0- th.cos(2.0*self.pi*self.k/(self.n+1))) )
		F[:,:,0] = F[:,:,0]*xi
		F[:,:,1] = F[:,:,1]*xi
    
		return th.ifft(F,1)[:,:,0]

	def integrate(self,funcs): 

		# compute the integral of periodic funcs over T1 via trapezoid quadrature
		#     (i.e. Gaussian quadrature for trigonometric polynomials)
		# computation expands along first dimension

		return( th.sum(funcs,1)*self.dx )

	def interpolate(self,pts,funcs): 

		# Use fourth-order WENO interpolation to obtain function values at *pts
		#	from known values *funcs at grid points
		
		assert( funcs.size(0) == pts.size(0)  )
		
		val = th.zeros_like(pts)
		gridutil.interpolate(val,pts,funcs)
		
		return val

	@staticmethod
	def expint(xinp): 

		# Compute the exponential integral functions 
		#	E_0(-x) = exp(-x) with E_{j+1}(-x) = (1 - E_j(-x))/x

		E0 = th.exp(-xinp)
		E1 = (1.0 - E0)/xinp
		E2 = (1.0 - E1)/xinp
		E3 = (0.5 - E2)/xinp		

		x   = xinp[th.abs(xinp)<.03]
		x2  = x*x
		x3  = x*x2
		x4  = x*x3
		x5  = x*x4
		x6  = x*x5

		E1[th.abs(xinp)<.03] = (1   - x/2  + x2/6   - x3/24  + x4/120  - x5/720   + x6/5040  )
		E2[th.abs(xinp)<.03] = (1/2 - x/6  + x2/24  - x3/120 + x4/720  - x5/5040  + x6/40320 )
		E3[th.abs(xinp)<.03] = (1/6 - x/24 + x2/120 - x3/720 + x4/5040 - x5/40320 + x6/362880)

		return (E0,E1,E2,E3)

	def propagate4(self,state,force0,forceh,force1,t,dt): 

		# Use 4th-order pseudo-exponential integrator to approximate one time-step
		#	u(t) --> u(t+dt) 
		# of the evolution u_t = u_xx + force starting from u(t) = *state*
		# The input dt is an m x 1 Tensor of timesteps
		# The input *force0/h/1* should be a tensor of same size as input *state
		# 		force0 is forcing function at time t
		# 		forceh is forcing function at time t+dt/2
		# 		force1 is forcing function at time t+dt

		U  = th.rfft(state,1,onesided=False)
		F0 = th.rfft(force0,1,onesided=False)
		Fh = th.rfft(forceh,1,onesided=False)
		F1 = th.rfft(force1,1,onesided=False)

		dF = 4.0*Fh - 3.0*F0 - F1
		sF = 4.0*( F0 + F1 - 2.0*Fh )
		E  = grid__T1.expint(dt*self.k*self.k)

		U[:,:,0] = E[0]*U[:,:,0] + dt*( F0[:,:,0]*E[1] + dF[:,:,0]*E[2] + sF[:,:,0]*E[3] )
		U[:,:,1] = E[0]*U[:,:,1] + dt*( F0[:,:,1]*E[1] + dF[:,:,1]*E[2] + sF[:,:,1]*E[3] )


		return th.ifft(U,1)[:,:,0]

	def propagate2(self,state,force0,force1,dt): 

		# Use 2nd-order pseudo-exponential integrator to approximate one time-step
		#		u(t) --> u(t+dt) 
		# of the evolution u_t = u_xx + force starting from u(t) = *state*
		# The input dt is an m x 1 Tensor of timesteps
		# The input *force0/1* should be a tensor of same size as input *state
		# 		force0 is forcing function at time t
		# 		force1 is forcing function at time t+dt

		U  = th.rfft(state,1,onesided=False)
		F0 = th.rfft(force0,1,onesided=False)
		F1 = th.rfft(force1,1,onesided=False)
		
		dF = F1-F0
		E  = grid__T1.expint(dt*self.k*self.k)

		U[:,:,0] = E[0]*U[:,:,0] + dt*( F0[:,:,0]*E[1] + dF[:,:,0]*E[2] )
		U[:,:,1] = E[0]*U[:,:,1] + dt*( F0[:,:,1]*E[1] + dF[:,:,1]*E[2] )

		return th.ifft(U,1)[:,:,0]
