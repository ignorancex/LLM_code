import torch as th
import grid__T1
import ifceutil

# a collection of m closed, two-dimensional interfaces
# represented via the data
# 	cm 	:~:	 centers of mass
#	ln  :~:  lengths of interface
# 	th  :~:  tangent angles over T1

class ifce__T1():
	
	def __init__(self,m_input,n_input,type=th.float64):
		
		self.G   = grid__T1.grid__T1(n_input,type)
		self.n   = n_input
		self.m   = m_input
		self.eql = False									# False: not constant speed, True: constant speed
		self.cms = th.zeros(self.m,2,dtype=type)			# centers of mass
		self.sig = th.zeros(self.m,dtype=type)				# speed of interface ( i.e. length/2pi )
		self.tht = th.zeros(self.m,self.n+1,dtype=type)		# tangent angles
		self.gam = th.zeros(self.m,self.n+1,2,dtype=type)	# (x,y) positions of interfaces

	def __init__(self,gam_input):

		assert(gam_input.ndim == 3)

		self.m   = gam_input.size(0)
		self.n   = gam_input.size(1) - 1
		self.G   = grid__T1.grid__T1(self.n,gam_input.dtype)
		self.eql = False
		self.cms = th.zeros(self.m,2,dtype=gam_input.dtype)			
		self.sig = th.zeros(self.m,dtype=gam_input.dtype)				
		self.tht = th.zeros(self.m,self.n+1,dtype=gam_input.dtype)
		self.gam = gam_input.clone()

		self.resample()
		self.gamtotht()
		self.thttogam()

	def to(self,device): 

		self.G.to(device)
		self.cms = self.cms.to(device)
		self.sig = self.sig.to(device)
		self.tht = self.tht.to(device)
		self.gam = self.gam.to(device)

	def resample(self): 

		# compute the change of variable map eta( xi(x) ) = x at grid points
		#	and use interpolation of gam(x) to compute constant speed representation
		#		psi(x) = gam( xi(x) )
		#	of each interface

		dgx = self.G.derivative(self.gam[:,:,0],1) 
		dgy = self.G.derivative(self.gam[:,:,1],1)
		spd = th.sqrt( dgx*dgx + dgy*dgy )

		assert( th.min(spd) > 1e-8 )

		lng = self.G.integrate(spd)
		spd = 1.0/spd												# right hand side of ODE xi'(x) = f( xi(x) ) at grid points

		xi  = th.zeros_like(self.tht)
		ifceutil.chofvar(xi,spd,lng)
		psx = self.G.interpolate(xi,self.gam[:,:,0].clone())		# perform resampling in each coordinate
		psy = self.G.interpolate(xi,self.gam[:,:,1].clone())

		self.sig.copy_(lng/(2.0*self.G.pi))
		self.gam[:,:,0].copy_(psx)
		self.gam[:,:,1].copy_(psy)
		self.cms[:,0]   = self.G.integrate(psx)/(2.0*self.G.pi)		# update center of mass
		self.cms[:,1]   = self.G.integrate(psy)/(2.0*self.G.pi)
		self.eql        = True

	def gamtotht(self): 

		# Given a constant speed representation gam(x), construct a continuous 
		#	tangent angle function tht(x) so that [-sin tht(x), cos tht(x)]
		#	represents the unit tangent to gam(x) at each point

		assert( self.eql )
		dgx = self.G.derivative(self.gam[:,:,0],1)
		dgy = self.G.derivative(self.gam[:,:,1],1)
		spd = th.sqrt( dgx*dgx + dgy*dgy )
		ang = th.atan2(-dgx/spd,dgy/spd)
		ifceutil.cleanth(self.tht,ang)

	def thttogam(self): 

		# Reconstruct a constant speed representation gam(x) of an interface
		#	from its tangent angle function tht(x), its speed (self.sig)
		#	and its center of mass (self.cms)

		self.gam[:,:,0] = self.G.primitive(-th.sin(self.tht) )*self.sig.unsqueeze(1) + self.cms[:,0].unsqueeze(1)
		self.gam[:,:,1] = self.G.primitive( th.cos(self.tht) )*self.sig.unsqueeze(1) + self.cms[:,1].unsqueeze(1)

	@staticmethod
	def newtonstep(sx,sy): 

		# Perform one-step of a Newton update
		#	(sx,sy) --> (sx,sy) - dF\F
		# on the zero-mean problem
		#	0 = F(lx,ly) = mean( (sx+lx,sy+ly)/norm(sx+lx,sx+lx) )
		# whose solution corresponds to a unit-length tangent (sx+lx,sy+ly)
		# with zero mean

		mg = th.sqrt( sx*sx + sy*sy )
		Fx = th.mean( sx/mg,1 )
		Fy = th.mean( sy/mg,1 )
		da = th.mean( (mg*mg-sx*sx)/(mg**3) , 1 )
		db = th.mean(-(sx*sy)/(mg**3) , 1 )
		dc = th.mean( (mg*mg - sy*sy)/(mg**3) , 1 )
		px = (dc*Fx - db*Fy)/(da*dc-db*db)
		py = (da*Fy - db*Fx)/(da*dc-db*db)

		sx.sub_(px.unsqueeze(1))
		sy.sub_(py.unsqueeze(1))

	def normalize(self): 

		# Re-initialize tangent angle function tht(x) so that the induced tangent
		#	[-sin tht(x),cos tht(x)] has zero mean

		t0 = self.tht[:,0].clone()
		sx = -th.sin( self.tht )
		sy =  th.cos( self.tht )

		ifce__T1.newtonstep(sx,sy)
		ifce__T1.newtonstep(sx,sy)
		ifce__T1.newtonstep(sx,sy)
		ifce__T1.newtonstep(sx,sy)
		
		spd = th.sqrt( sx*sx + sy*sy )
		ang = th.atan2(-sx/spd,sy/spd)
		ifceutil.cleanth(self.tht,ang)
		self.tht = self.tht - self.tht[:,0].unsqueeze(1) + t0.unsqueeze(1)
	
