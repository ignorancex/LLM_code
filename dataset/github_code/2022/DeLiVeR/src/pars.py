import math

# unit conversion
MeV = 1e-3
gev2nb = 389379.3656

######################
### SM parameters ####
######################

# lepton masses - PDG values #
me_ = 0.5109989461*MeV #
mmu_ = 0.1056583745
mtau_ = 1.77686
mlep_ = [me_,mmu_,mtau_,0.,0.,0.]
# quark masses #
md_ = 4.67*MeV
mu_ = 2.16*MeV
ms_ = 93.*MeV
mc_ =   1.27
mb_ = 4.18
mt_ = 172.76

############
# mesons ###
############
# masses and widths from PDG

## light mesons ##
#pion
mpi0_ = 134.9768*MeV
mpi_  = 139.57039*MeV
# rho mass
mRho  = .7755
gRho  = .1494
mRho1 =1.459
gRho1 =0.4
mRho2 =1.72
gRho2 =0.25
# omega mass
mOmega=.78265
gOmega=.00849
# a1 mass
ma1=1.23
ga1=.2
# f0 mass
mf0=1.35
gf0=0.2


#lightest charmed meson mass
mD0 = 1.864
#lightest bottom meson mass
mUps = 9.46

# fundamental constants #
hbar = 6.58211951e-25 # GeV.s
cLight = 299792458 # m/s
ge   = 3.02822e-1  # Electromagnetic coupling (unitless).
