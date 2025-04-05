from scipy import *
from numpy import *
from pylab import *
import matplotlib.pyplot as pyplot

omde = genfromtxt('omegade.txt')
weff = genfromtxt('weff.txt')
weff2 = genfromtxt('eft_new_weff.txt')
a = genfromtxt('a_evol.txt')

figure(1)
plot(a,omde)
xlabel(r'$a$',fontsize=18)
ylabel(r'$\Omega_{\rm{DE}}$',fontsize=18)
pyplot.xscale('log')
savefig('omegede.pdf')

figure(2)
plot(a,weff)
xlabel(r'$a$',fontsize=18)
ylabel(r'$w_{\rm{eff}}$', fontsize=18)
pyplot.xscale('log')
xlim([1.0e-4,1.0])
ylim([-1.5,-0.2])
savefig('weff.pdf')

figure(3)
plot(a,weff2)
xlabel(r'$a$',fontsize=18)
ylabel(r'$w_{\rm{eff}}$', fontsize=18)
pyplot.xscale('log')
savefig('weff2.pdf')
show()
