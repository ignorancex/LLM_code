import numpy
import readfile


def Mstar(Mi, Mo=5.23e11/0.7, alpha=0.298, beta=1.99, logk=10.30,
          sigma=0.192):
    Mstar_fid = 2 / ((Mo/Mi)**alpha + (Mo/Mi)**beta) * 10**logk
    logMstar = numpy.random.normal(numpy.log10(Mstar_fid), sigma)
    return 10**logMstar


def Minfall(Mstarbins