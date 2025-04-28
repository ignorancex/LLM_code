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

_HD22049 = {"Name":'eps Eri', "Bdip":14.6, "Bquad":8.78, "Boct":5.90, "Mdot":30, "Period":12, "Mass":0.86, "Radius":0.694,
"un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['N/A','N/A'], "un_Period":['0.5','0.5'], "un_Mass":['0.05','0.05'], "un_Radius":['0.014','0.014']}

_HD185144 = {"Name":'sig Dra', "Bdip":5.68, "Bquad":4.82, "Boct":4.76, "Mdot":3.05, "Period":27, "Mass":0.84, "Radius":0.772,
"un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['0.79','1.17'], "un_Period":['0.5','0.5'], "un_Mass":['0.01','0.01'], "un_Radius":['0.005','0.005']}

_HD10476 = {"Name":'107 Psc', "Bdip":4.24, "Bquad":2.77, "Boct":1.37, "Mdot":0.93, "Period":35, "Mass":0.87, "Radius":0.813,
"un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['0.28','0.39'], "un_Period":['0.5','0.5'], "un_Mass":['0.05','0.05'], "un_Radius":['0.012','0.012']}

_HD103095 = {"Name":'HD103095', "Bdip":0.57, "Bquad":0.0, "Boct":0.0, "Mdot":0.30, "Period":31, "Mass":0.60, "Radius":0.641,
"un_Bdip":['0.03','0.03'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['0.08','0.10'], "un_Period":['0.5','0.5'], "un_Mass":['0.04','0.04'], "un_Radius":['0.017','0.017']}

_HD219134 = {"Name":'HD219134', "Bdip":2.39, "Bquad":4.05, "Boct":1.19, "Mdot":0.5, "Period":42, "Mass":0.80, "Radius":0.724,
"un_Bdip":['N/A','N/A'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['N/A','N/A'], "un_Period":['0.9','0.9'], "un_Mass":['0.05','0.05'], "un_Radius":['0.015','0.015']}

_HD166620 = {"Name":'HD166620', "Bdip":1.10, "Bquad":0.00, "Boct":0.00, "Mdot":0.92, "Period":43, "Mass":0.80, "Radius":0.782,
"un_Bdip":['0.40','0.42'], "un_Bquad":['N/A','N/A'], "un_Boct":['N/A','N/A'], "un_Mdot":['0.23','0.22'], "un_Period":['0.5','0.5'], "un_Mass":['0.05','0.05'], "un_Radius":['0.012','0.012']}

ndec=5 # Number of decimal places for the outputs

print('# EVOLUTIONARY SEQUENCE #')
print(' ')

torqueDiagnostic(_HD22049,_HD185144,0.419759) # Delta logR'HK=-0.377 => 0.419759 
print(' ')
torqueDiagnostic(_HD185144,_HD10476,0.831764) # Delta logR'HK=-0.080 => 0.831764
print(' ')
torqueDiagnostic(_HD185144,_HD103095,0.862979) # Delta logR'HK=-0.064 => 0.862979
print(' ')
torqueDiagnostic(_HD185144,_HD219134,0.874984) # Delta logR'HK=-0.058 => 0.874984
print(' ')
torqueDiagnostic(_HD185144,_HD166620,0.753356) # Delta logR'HK=-0.123 => 0.753356
print(' ')
