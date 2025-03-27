import torch as th
import ifce__T1
import ifceutil
import math

class NLFront(ifce__T1.ifce__T1):

	def __init__(self,gam_input,c_input,g_input):
		super(NLFront,self).__init__(gam_input)
		self.normalize()
		self.thttogam()
		
		self.nspd = c_input
		self.kern = g_input
		self.tstp = math.inf
		self.Fpre = th.zeros_like(self.tht)
		self.Fcur = th.zeros_like(self.tht)
		self.apre = th.zeros_like(self.tht)
		self.acur = th.zeros_like(self.tht)
		self.kpre = th.zeros_like(self.tht)
		self.kcur = th.zeros_like(self.tht)
		self.wind = th.round( (self.tht[:,-1] - self.tht[:,0])/(2.0*self.G.pi) ).unsqueeze(1)
		self.offs = (self.G.x+self.G.pi)*self.wind
		self.eta  = self.tht - self.offs

		self.forceupdate()

	def to(self,device):
		super(NLFront,self).to(device)

		self.Fpre = self.Fpre.to(device)
		self.Fcur = self.Fcur.to(device)
		self.apre = self.apre.to(device)
		self.acur = self.acur.to(device)
		self.kpre = self.kpre.to(device)
		self.kcur = self.kcur.to(device)
		self.wind = self.wind.to(device)
		self.offs = self.offs.to(device)
		self.eta  = self.eta.to(device)

	def nlforces(self):

		npt = self.tht.size(0)*self.tht.size(1)
		dst = th.zeros(npt,npt,dtype=self.tht.dtype)
		ifceutil.getdist(dst,self.gam)
		dst = dst.view(self.tht.size(0),self.tht.size(1),npt)
		wty = ( self.sig.unsqueeze(1) ).expand(self.tht.size(0),self.tht.size(1))
		wtz = self.sig.repeat_interleave(self.tht.size(1)).view(1,1,npt)
		dst = self.kern(dst)*self.G.dx
		dst = dst*( wty.unsqueeze(2) )
		dst = dst*wtz

		return th.sum( dst,2 )

	def forceupdate(self): 

		self.kpre.copy_(self.kcur)
		self.kcur = self.G.derivative(self.eta,1) + self.wind
		self.Fpre.copy_(self.Fcur)
		self.Fcur = self.nlforces() + self.nspd*self.sig.unsqueeze(1)
		self.apre.copy_(self.acur)
		self.acur = self.kcur*(self.kcur - self.Fcur)

	def advance(self,dt):

		# compute largest allowable timestep and extrapolate forces
		muc  = th.mean(self.acur,1)
		dmu  = th.mean(self.acur-self.apre,1)/self.tstp
		dtM  = th.min( .5*(self.sig**2)/( th.sqrt( th.abs(muc)+.5*th.abs(dmu)*(self.sig**2) ) ) ).item()
		dtc  = th.min( th.Tensor([dt,dtM]) ).item()
		rt   = dtc/self.tstp
		anxt = (1.0 + rt)*self.acur - rt*self.apre
		knxt = (1.0 + rt)*self.kcur - rt*self.kpre
		Fnxt = (1.0 + rt)*self.Fcur - rt*self.Fpre

		# compute updated lengths and time increment dW for each interface
		sig  = th.sqrt( self.sig**2 - 2.0*dtc*muc - dtc*dtc*dmu )			# updated speeds
		dW   = .5*dtc*( sig**(-2)+self.sig**(-2) )							# time increment for forced HE solve

		# compute transport coefficients P0(a) for each interface
		pcur = self.G.P0(self.acur)
		pnxt = self.G.P0(anxt)

		# compute and apply forces G0,G1 to update angular variables
		G0       = pcur*self.kcur - self.G.derivative(self.Fcur,1)
		G1       = pnxt*knxt      - self.G.derivative(Fnxt,1)
		self.eta = self.G.propagate2(self.eta,G0,G1,dW.unsqueeze(1))
		tnxt     = self.eta + self.offs										# updated angular variables

		# update center of mass
		ucur = th.mean( pcur*( -th.sin(self.tht) ) + self.Fcur*( th.cos(self.tht) ) , 1 )
		vcur = th.mean( pcur*(  th.cos(self.tht) ) + self.Fcur*( th.sin(self.tht) ) , 1 )
		unxt = th.mean( pnxt*( -th.sin(tnxt) )     + Fnxt*( th.cos(tnxt) ) , 1 )
		vnxt = th.mean( pnxt*(  th.cos(tnxt) )     + Fnxt*( th.sin(tnxt) ) , 1 )

		self.cms[:,0] = self.cms[:,0] + .5*dtc*( ucur*( self.sig**(-2) ) + unxt*( sig**(-2) ) )
		self.cms[:,1] = self.cms[:,1] + .5*dtc*( vcur*( self.sig**(-2) ) + vnxt*( sig**(-2) ) )

		# update variables, timestep and forces to current timestep
		self.tstp = dtc
		self.tht.copy_(tnxt)
		self.sig.copy_(sig)
		self.normalize()		
		self.thttogam()
		self.forceupdate()

		return self.tstp
