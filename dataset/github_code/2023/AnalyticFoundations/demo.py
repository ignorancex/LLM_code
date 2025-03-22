import torch as th
from matplotlib import pyplot as plt
from grid__T1 import grid__T1 as Grid
from NLFront import NLFront

c  = 1.0
R  = 10.0
r  = 5.0
S  = 44.2
C  = .5725
sg = 18.45
C  = C/(2.506628274631000502415765284811045253006986740609938316629*sg)

def g(s):

	return -C*th.exp(-s/(sg*sg))

G  = Grid(299)
cx = R*th.cos( G.x ).unsqueeze(0).unsqueeze(2) + 10.0
sx = R*th.sin( G.x ).unsqueeze(0).unsqueeze(2)
Ga = th.cat( (cx,sx),2 )

cx = r*th.cos( -G.x ).unsqueeze(0).unsqueeze(2) + 10.0
sx = r*th.sin( -G.x ).unsqueeze(0).unsqueeze(2)
Ps = th.cat( (cx,sx),2 )
Ga = th.cat( (Ga,Ps),0 )
Ps = NLFront(Ga,c,g)

tc = 0.
dt = 0.25*G.dx
tf = 5.0

ns = 0
while tc < tf:

	if ns%20 == 0:
		plt.axis('equal')
		plt.title('t = ' + str(tc) + '    dt = ' + str(dt) )
		plt.plot( Ps.gam[0,:,0],Ps.gam[0,:,1] )
		plt.plot( Ps.gam[1,:,0],Ps.gam[1,:,1] )
		plt.show(block=False)
		plt.pause(.001)
		plt.close()

	ns  = ns+1
	dt  = th.min( th.Tensor([dt]),th.Tensor([tf-tc]) ).item()
	dt  = Ps.advance(dt)
	tc  = tc + dt

nx = th.cos( Ps.tht )
ny = th.sin( Ps.tht )

print(Ps.sig[0].item())
print(Ps.sig[1].item())
