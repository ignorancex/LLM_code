import numpy as np

##########################################################
# Finley & Matt (2017/2018) Torque Formulations ##########
# Version 1.0 : 12th April 2018 ##########################
##########################################################
# Function:                                              #
# FM18_surf-Takes polar strengths of the lowest harmonic #
#	    components and predicts an average Alfv\'en  #
#	    radius which produces a braking torque on 	 #
#	    the star.                                    #
##########################################################
# Input Parameters:                                      #
# Bdip- Polar Dipole Strength in Gauss                   #
# Bquad- Polar Quadrupole Strength in Gauss              #
# Boct- Polar Octupole Strength in Gauss                 #
# Mdot- Mass Loss Rate in Solar per Year                 #
# Period- Rotation Period in Days                        #
# Mass- Stellar Mass in Solar Units                      #
# Radius- Stellar Radius in Solar Units                  #
# Info- Print Topology and Rotation Information          #
##########################################################

def FM18_surf(Bdip, Bquad, Boct, Mdot, Period, Mass, Radius, Info):

	#########################################################
	#Stellar Parameters:                                    #
	#########################################################
	R_star = 6.96e10*(Radius)
	M_star = 1.99e33*(Mass)
	Omega_star = 2*np.pi/Period/24/3600
	v_esc = np.sqrt( 2*(6.67e-8)*M_star/R_star )
	f = Omega_star*R_star/v_esc*np.sqrt(2)
	if (f < 0.1 and Info==True):
		print('SLOW ROTATOR REGIME, f = '+str(f))
	elif (Info==True):
		print('FAST ROTATOR REGIME, f = '+str(f))
	Btot = abs(Bdip) + abs(Bquad) + abs(Boct)
	Mdot = Mdot*( (2e-14)*(1.99e33)/(365.25*24*3600) )
	Upsilon = Btot**2*R_star**2/Mdot/v_esc
	Rdip = Bdip/Btot
	Rquad = Bquad/Btot
	Roct = Boct/Btot


	#########################################################
	# Derived Simulation Parameters:                        #
	#########################################################
	K1 = 1.53
	m1 = 0.229
	K2 = 1.70
	m2 = 0.134
	K3 = 1.80
	m3 = 0.087
	Kvr = 0.2

	#########################################################
	# Calculate Average Alfv\'en Radius using equation (17) #
	#########################################################
	RA_dip = K1*(Rdip)**(2*m1)*(Upsilon/np.sqrt(1+f**2/Kvr**2))**(m1)
	RA_quad = K2*(Rquad+Rdip)**(2*m2)*(Upsilon/np.sqrt(1+f**2/Kvr**2))**(m2)
	RA_oct = K3*(Roct+Rquad+Rdip)**(2*m3)*(Upsilon/np.sqrt(1+f**2/Kvr**2))**(m3)
	##################### Max Function ######################
	if ( RA_dip>RA_quad and RA_dip>RA_oct and Rdip>0. ):
		RA = RA_dip
		if (Info==True):
			print('DIPOLE DOMINATED')
	elif (RA_quad>RA_dip and RA_quad>RA_oct and Rquad>0.):
		RA=(RA_quad)
		if (Info==True):
			print('MULTIPOLAR DOMINATED')
	else:
		RA=(RA_oct)
		if (Info==True):
			print('MULTIPOLAR DOMINATED')
	#########################################################

	#########################################################
	#Calculate Angular Momentum Loss Rate, equation (14)    #
	#########################################################
	Torque = Mdot*Omega_star*R_star**2*np.power(RA,2)

	return Upsilon, RA, Torque


##########################################################
# Metcalfe et al. (2022) Torque Analysis #################
# Version 1.0 :  8th April 2022 ##########################
##########################################################
# Function:                                              #
# torqueDiagnostic-Calculates the FM18 torque for StarA  #
#	    and StarB, and evaluates the relative        #
#	    influence of changing each stellar parameter #
#	    in the braking law.                          #
##########################################################
# Input Parameters:                                      #
# fromStarA- Dictionary* of StarA Properties             #
# toStarB- Dictionary* of StarB Properties               #
# RhkFieldChange- Change in <B> given difference in R'HK #
# Dictionary*- Name, Bdip, Bquad, Boct, Mdot, Period,    #
#	    Mass, and Radius.                            #
##########################################################


def torqueDiagnostic(fromStarA,toStarB,FieldChange,ndec=5):

	#########################################################
	#StarA Wind Torque:                                     #
	#########################################################
	print('['+fromStarA["Name"]+', Bd='+str(fromStarA["Bdip"])+', Bq='+str(fromStarA["Bquad"])+', Bo='+str(fromStarA["Boct"])+']')
	Upsilon, RA, Torque1 = FM18_surf(Bdip=fromStarA["Bdip"], Bquad=fromStarA["Bquad"], Boct=fromStarA["Boct"], Mdot=fromStarA["Mdot"], Period=fromStarA["Period"], Mass=fromStarA["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('Torque:',round(Torque1/1e30,ndec),'x10^30erg') #fiducial model StarA
	print(' ')
	#########################################################

	# COMPARE TO StarA MODEL #
	print('# COMPARE TO '+fromStarA["Name"]+' MODEL #')
	print(' ')

	#########################################################
	#StarB Wind Torque:                                     #
	#########################################################
	print('['+toStarB["Name"]+', Bd='+str(toStarB["Bdip"])+', Bq='+str(toStarB["Bquad"])+', Bo='+str(toStarB["Boct"])+']')
	Upsilon, RA, Torque2 = FM18_surf(Bdip=toStarB["Bdip"], Bquad=toStarB["Bquad"], Boct=toStarB["Boct"], Mdot=toStarB["Mdot"], Period=toStarB["Period"], Mass=toStarB["Mass"], Radius=toStarB["Radius"], Info=False)
	print('Torque:',round(Torque2/1e30,ndec),'x10^30erg') #fiducial model StarB
	print('['+fromStarA["Name"]+' -> '+toStarB["Name"]+']:',round(Torque2/Torque1-1.,ndec))
	#########################################################

	#########################################################
	#Evaluate the Influence of Changing Each Parameter:     #
	#########################################################
	print('#parameter study#')
	Upsilon, RA, Torque3 = FM18_surf(Bdip=FieldChange*fromStarA["Bdip"], Bquad=FieldChange*fromStarA["Bquad"], Boct=FieldChange*fromStarA["Boct"], Mdot=fromStarA["Mdot"], Period=fromStarA["Period"], Mass=fromStarA["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('<B>:',round(Torque3/Torque1-1.,ndec))
	Upsilon, RA, Torque = FM18_surf(Bdip=toStarB["Bdip"], Bquad=toStarB["Bquad"], Boct=toStarB["Boct"], Mdot=fromStarA["Mdot"], Period=fromStarA["Period"], Mass=fromStarA["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('Morph:',round(Torque/Torque3-1.,ndec)) # StarB morphology at same <B>
	print('both:',round(Torque/Torque1-1.,ndec))
	Upsilon, RA, Torque = FM18_surf(Bdip=fromStarA["Bdip"], Bquad=fromStarA["Bquad"], Boct=fromStarA["Boct"], Mdot=toStarB["Mdot"], Period=fromStarA["Period"], Mass=fromStarA["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('Mdot:',round(Torque/Torque1-1.,ndec))
	Upsilon, RA, Torque = FM18_surf(Bdip=fromStarA["Bdip"], Bquad=fromStarA["Bquad"], Boct=fromStarA["Boct"], Mdot=fromStarA["Mdot"], Period=toStarB["Period"], Mass=fromStarA["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('Prot:',round(Torque/Torque1-1.,ndec))
	Upsilon, RA, Torque = FM18_surf(Bdip=fromStarA["Bdip"], Bquad=fromStarA["Bquad"], Boct=fromStarA["Boct"], Mdot=fromStarA["Mdot"], Period=fromStarA["Period"], Mass=toStarB["Mass"], Radius=fromStarA["Radius"], Info=False)
	print('M:',round(Torque/Torque1-1.,ndec))
	Upsilon, RA, Torque = FM18_surf(Bdip=fromStarA["Bdip"], Bquad=fromStarA["Bquad"], Boct=fromStarA["Boct"], Mdot=fromStarA["Mdot"], Period=fromStarA["Period"], Mass=fromStarA["Mass"], Radius=toStarB["Radius"], Info=False)
	print('R:',round(Torque/Torque1-1.,ndec))
	print(' ')
	#########################################################

# FIDUCIAL MODELS #

# NOTE: the model parameters for HD 76151 were correctly listed in Table 1 of Metcalfe+2022, but were incorrect in the
#       original release of this code. The correct values yield a torque of 4.17 x10^30erg (down from 4.77), and modify
#       some of the percentages reported in Section 3 (-69% → -70%; -65% → -63%; -4% → +16%; +0.4% → +0.7%). The source
#       was an update to the SED radius and the scaled mass, which changed Fx and thus Mdot. Conclusions are unchanged.
#
#HD76151 = {"Name":'HD76151', "Bdip":5.13, "Bquad":2.88, "Boct":1.34, "Mdot":7.5, "Period":20.5, "Mass":1.04, "Radius":1.023,

_HD76151 = {"Name":'HD76151', "Bdip":5.13, "Bquad":2.88, "Boct":1.34, "Mdot":8.3, "Period":20.5, "Mass":1.05, "Radius":0.964,
            "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.7,0.7], "un_Period":[0.3,0.3], "un_Mass":[0.06,0.06], "un_Radius":[0.018,0.018]}

_18Sco = {"Name":'18 Sco', "Bdip":1.34, "Bquad":2.01, "Boct":0.864, "Mdot":0.87, "Period":22.7, "Mass":1.02, "Radius":1.01,
	  "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.32,0.32], "un_Period":[0.5,0.5], "un_Mass":[0.03,0.03], "un_Radius":[0.009,0.009]}

_16CygA_Bd = {"Name":'16 Cyg A', "Bdip":0.5, "Bquad":0, "Boct":0, "Mdot":0.92, "Period":20.5, "Mass":1.072, "Radius":1.223,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.16,0.16], "un_Period":[1.1,2.0], "un_Mass":[0.013,0.013], "un_Radius":[0.005,0.005]}
_16CygB_Bd = {"Name":'16 Cyg B', "Bdip":0.9, "Bquad":0, "Boct":0, "Mdot":0.57, "Period":21.2, "Mass":1.038, "Radius":1.113,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.13,0.13], "un_Period":[1.5,1.8], "un_Mass":[0.047,0.047], "un_Radius":[0.016,0.016]}

_16CygA_Bq = {"Name":'16 Cyg A', "Bdip":0, "Bquad":2.5, "Boct":0, "Mdot":0.92, "Period":20.5, "Mass":1.072, "Radius":1.223,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.16,0.16], "un_Period":[1.1,2.0], "un_Mass":[0.013,0.013], "un_Radius":[0.005,0.005]}
_16CygB_Bq = {"Name":'16 Cyg B', "Bdip":0, "Bquad":4.5, "Boct":0, "Mdot":0.57, "Period":21.2, "Mass":1.038, "Radius":1.113,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.13,0.13], "un_Period":[1.5,1.8], "un_Mass":[0.047,0.047], "un_Radius":[0.016,0.016]}

_16CygA_Bo = {"Name":'16 Cyg A', "Bdip":0, "Bquad":0, "Boct":5.8, "Mdot":0.92, "Period":20.5, "Mass":1.072, "Radius":1.223,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.16,0.16], "un_Period":[1.1,2.0], "un_Mass":[0.013,0.013], "un_Radius":[0.005,0.005]}
_16CygB_Bo = {"Name":'16 Cyg B', "Bdip":0, "Bquad":0, "Boct":31.3, "Mdot":0.57, "Period":21.2, "Mass":1.038, "Radius":1.113,
	      "un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":[0.13,0.13], "un_Period":[1.5,1.8], "un_Mass":[0.047,0.047], "un_Radius":[0.016,0.016]}


ndec=5 # Number of decimal places for the outputs

print('# EVOLUTIONARY SEQUENCE #')
print(' ')

torqueDiagnostic(_HD76151,_18Sco,0.395) # HD 76151 x (1.18/2.99) => 0.395
torqueDiagnostic(_HD76151,_16CygA_Bd,0.115) # HD 76151 x [(0.5/1.45)/2.99] => 0.115
torqueDiagnostic(_HD76151,_16CygB_Bd,0.208) # HD 76151 x [(0.9/1.45)/2.99] => 0.208
torqueDiagnostic(_HD76151,_16CygA_Bq,0.467) # HD 76151 x [(2.5/1.79)/2.99] => 0.467
torqueDiagnostic(_HD76151,_16CygB_Bq,0.841) # HD 76151 x [(4.5/1.79)/2.99] => 0.841
torqueDiagnostic(_HD76151,_16CygA_Bo,0.937) # HD 76151 x [(5.8/2.07)/2.99] => 0.937
torqueDiagnostic(_HD76151,_16CygB_Bo,5.057) # HD 76151 x [(31.3/2.07)/2.99] => 5.057
print(' ')

torqueDiagnostic(_18Sco,_16CygA_Bd,0.345) # 18 Sco x [(0.5/1.45)/1.18] => 0.345
torqueDiagnostic(_18Sco,_16CygB_Bd,0.526) # 18 Sco x [(0.9/1.45)/1.18] => 0.526
torqueDiagnostic(_18Sco,_16CygA_Bq,1.184) # 18 Sco x [(2.5/1.79)/1.18] => 1.184
torqueDiagnostic(_18Sco,_16CygB_Bq,2.131) # 18 Sco x [(4.5/1.79)/1.18] => 2.131
torqueDiagnostic(_18Sco,_16CygA_Bo,2.375) # 18 Sco x [(5.8/2.07)/1.18] => 2.375
torqueDiagnostic(_18Sco,_16CygB_Bo,12.81) # 18 Sco x [(31.3/2.07)/1.18] => 12.81
print(' ')

