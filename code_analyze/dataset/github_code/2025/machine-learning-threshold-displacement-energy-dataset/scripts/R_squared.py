#Plots raw Ed vs predicted Ed from analytical formula and calculates R squared value
import numpy as np
import math
import matplotlib.pyplot as plt


filename = '/directory/to/file/eds_all_method.csv' # your filepath to raw data file
Matrix = []
xyz = open(filename, "r")
for line in xyz:
    # Generate Matrix with the xyz data
    Matrix.append(line.split(','))

#Define the analytical expression obtained from SISSO
def fit1(density,E_ioniz,l_bond):
    ed = 1.296178732584652e+01 + 2.729416958744306e+02 * ((float(density) / float(E_ioniz)) / (float(l_bond)**3)) # change to the relevant analytical expression
    return ed
    
#generate matrix to work with data
NewMatrix = []
for j in range(len(Matrix)):
    NewMatrix.append([])

#Generate lists with ed values
ed_predict = np.zeros(len(Matrix)-1)
ed_exp = np.zeros(len(Matrix)-1)
for i in range(0,len(Matrix)-1):
    ed_predict[i] = fit1(Matrix[i+1][7], Matrix[i+1][9], Matrix[i+1][4])
    ed_exp[i] = Matrix[i+1][1]

#Calculate r^2
SSres = 0
for i in range(len(ed_exp)):
    SSres += (ed_exp[i]-ed_predict[i])**2
SStot = 0
for i in range(len(ed_exp)):
    SStot += (ed_exp[i]-np.mean(ed_exp))**2

R2 = round(1 - SSres/SStot,5)


#Plotting
plt.figure(figsize=(10,10))
plt.title('All methods data, Model dim:1', fontsize=25)   #method1
plt.scatter(ed_predict,ed_exp, label = f'$R^2 =$ %s'%R2, marker = 'x', color = 'k')
plt.xlabel(r'$c_0 + a_0 \frac{ \rho }{ E_{ioniz} \ell_{bond}^3 }$', fontsize=20)
plt.ylabel(r'$E_d$ (eV)', fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(axis='both', direction = 'in', length = 10, labelsize = 17, which = 'both', right = 'true', top = 'true')
plt.savefig('all_d1_R2.pdf',bbox_inches='tight')
plt.show()


