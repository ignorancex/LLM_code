import numpy as np
import math
import numpy.linalg as LI
from scipy import linalg as LA

#MaxLik fce

# Stokes to Rho
sigma1 = np.array([[1., 0.], [0., -1.]])
sigma2 = np.array([[0., 1.], [1., 0.]])
sigma3 = np.array([[0., -1j], [1j, 0.]])
sigma = np.array([sigma1, sigma2, sigma3])

stokesFromRho = lambda rho_ : np.real(np.array([
    np.trace(np.dot(rho_,sigma1)),
    np.trace(np.dot(rho_,sigma2)),
    np.trace(np.dot(rho_,sigma3))
]))

rhoFromStokes = lambda stokes_ : np.array([
    [1+stokes_[0], stokes_[1]-1j*stokes_[2]],
    [stokes_[1]+1j*stokes_[2], 1-stokes_[0]]
])/2.

def Matrix_distance (matrix1, matrix2):
	difference12 = matrix1 - matrix2
	difference12dag = np.transpose(difference12).conj() #dag = hermitovske sdruzeni
	distance12 = np.sqrt(np.trace(np.dot(difference12, difference12dag)))
	return distance12
	
def Fidelity (rho1,rho2):
	return math.pow((np.trace(np.dot(LA.sqrtm(rho1),np.dot(rho2,LA.sqrtm(rho1))))).real,2)

def Purity (rho):
	return ((np.trace(LI.matrix_power(rho,2))).real)

def Maxlik(in_data, projectors, epsilon = 1e-10):
	#normalizace countu
	in_data = np.array(list(in_data))
	total_data = float(sum(in_data))
	fi = in_data / total_data
	#print (in_data)
	#print (total_data)
	#print (fi)

	rho_old = np.eye(2) / 2
	#print (rho_old)
	while True:
		#pravdepodobnosti P
		probabilities = []
		for projector in projectors:
			prob = np.trace(np.dot(rho_old, projector))
			probabilities.append(prob)

		# R
		R_parts = []
		for i in range(0,6):
			part =  projectors[i] * fi[i] / probabilities[i]
			#print (fi[i])
			R_parts.append(part)
			#print projectors[i]
		R = sum(R_parts)

		#Novy odhad
		rho_new = np.dot(np.dot(R, rho_old),R)
		#print (rho_new)
		rho_new = rho_new / np.trace(rho_new)
		
		#kontrola vzdalenosti
		distance = Matrix_distance(rho_new, rho_old)
		if distance < epsilon:
			break

		# Pokud neni prilis dobre, zacneme novou iteraci
		rho_old = rho_new
		
	stokes = stokesFromRho(rho_new)
	return ([rho_new,stokes])
