import numpy as np
import numba
from numba import njit

#below are the symmetrized F2 and G2 kernels
@njit
def K2(k1,k2,K12):
    kdot = (K12*K12-k1*k1-k2*k2)/2/k1/k2
    kernel1 = kdot*(k1/k2 + k2/k1)
    kernel2 = kdot*kdot

    FF = 5./7. +  1./2. * kernel1 + 2./7. * kernel2
    GG = 3./7. +  1./2. * kernel1 + 4./7. * kernel2
    return FF, GG

@njit
def K2new(k1,k2,K12):

    s1 = k1*k1
    s14 = k2*k2
    s4 = K12*K12

    
    alpha1 = (1 - s14/s1 + s4/s1)/2
    alpha2 = (1 - s1/s14 + s4/s14)/2
    beta = (-s4/s1 - s4/s14 + s4*s4/s1/s14)/4
    
    FF = 5./7. * (alpha1 + alpha2)/2. + 2./7. * beta
    GG = 3./7. * (alpha1 + alpha2)/2. + 4./7. * beta

    return FF, GG

# # @njit
# def alphakernel(k1,k2,K12): #gives back both permutations alpha(k1,k2) and alpha(k2,k1)
#     alpha1 = (1+K12**2-
#     return numerator/k1**2 , numerator/k2**2

# # @njit
# def betakernel(k1,k2,K12):
#     return K12**2 * (K12**2 - k1**2 - k2**2) / (4*k1**2 * k2**2)


#below are the F3 and G3 permutations
# @njit
def KPerm(k1,k2,k3,K14,k4):
    FF,GG = K2(k2,k3,K14)

    s1 = k1*k1
    s14 = K14*K14
    s4 = k4*k4

    alpha1 = (1 - s14/s1 + s4/s1)/2
    alpha2 = (1 - s1/s14 + s4/s14)/2
    beta = (-s4/s1 - s4/s14 + s4*s4/s1/s14)/4
    # print(alpha1,alpha2,beta)

    FFF = 1/18 * (7*alpha1 * FF + 2*beta*GG + GG*(7*alpha2 + 2*beta))
    GGG = 1/18 * (3*alpha1 * FF + 6*beta*GG + GG*(3*alpha2 + 6*beta))

    return FFF, GGG

#below are the symmetrized F3 and G3 kernels
# @njit
def K3(k1,k2,k3,k4,K12,K13,K14):

    F1,G1 = KPerm(k1,k2,k3,K14,k4) 
    F2,G2 = KPerm(k2,k1,k3,K13,k4)
    F3,G3 = KPerm(k3,k1,k2,K12,k4)
    return (F1+F2+F3)/3,(G1+G2+G3)/3

#Below is the total trispectrum as a single shape
def SecondaryTotal(k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,P1,P2,P3,P4,P12,P13,P14,T21,alpha,beta,gamma,delta,eta,nu):
    def Secondary2Perm(k1,k2,k3,k4,K13,K14,mu1,mu2,mu3,mu4,mu13,mu14,P1,P2,P13,P14):
        FF113, GG113 = K2(k1,K13,k3)
        FF213, GG213 = K2(k2,K13,k4)
        FF114, GG114 = K2(k1,K14,k4)
        FF214, GG214 = K2(k2,K14,k3)

        pre1 = alpha + T21 * mu1
        pre2 = alpha + T21 * mu2
 
        result = 4*pre1*pre2*(P14*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21)*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21) + P13*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21)*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21))
        
        return result*P1*P2

    def Secondary3Perm(k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14): #this is the permutation with k4 special
        FF12, GG12 = K2(k1,k2,K12)
        FF13, GG13 = K2(k1,k3,K13)
        FF23, GG23 = K2(k2,k3,K14)
        FFF, GGG = K3(k1,k2,k3,k4,K12,K13,K14)
                
        pre1 = alpha + T21 * mu1
        pre2 = alpha + T21 * mu2
        pre3 = alpha + T21 * mu3
        res = 2*eta*(FF12 + FF13 + FF23) + 6*delta*FFF + 2*beta*(mu1 + mu2 + mu3) + 2*gamma*(FF23*mu1 + FF13*mu2 + FF12*mu3) + 2*alpha*(GG12*mu12 + GG13*mu13 + GG23*mu14 + mu1*mu2 + mu1*mu3 + mu2*mu3) + 6*nu + 2*(2*GG23*mu1*mu14 + 2*GG13*mu13*mu2 + 2*GG12*mu12*mu3 + 3*mu1*mu2*mu3 + 3*GGG*mu4)*T21
        return res*pre1*pre2*pre3

    permk3k4 = Secondary2Perm(k1,k2,k3,k4,K13,K14,mu1,mu2,mu3,mu4,mu13,mu14,P1,P2,P13,P14)
    permk2k4 = Secondary2Perm(k1,k3,k2,k4,K12,K14,mu1,mu3,mu2,mu4,mu12,mu14,P1,P3,P12,P14)
    permk1k4 = Secondary2Perm(k3,k2,k1,k4,K13,K12,mu3,mu2,mu1,mu4,mu13,mu12,P3,P2,P13,P12)
    permk1k3 = Secondary2Perm(k4,k2,k3,k1,K12,K14,mu4,mu2,mu3,mu1,mu12,mu14,P4,P2,P12,P14)
    permk2k3 = Secondary2Perm(k1,k4,k3,k2,K13,K12,mu1,mu4,mu3,mu2,mu13,mu12,P1,P4,P13,P12)
    permk1k2 = Secondary2Perm(k3,k4,k1,k2,K13,K14,mu3,mu4,mu1,mu2,mu13,mu14,P3,P4,P13,P14)

    permk4 = Secondary3Perm(k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14)*P1*P2*P3 #permutation with k4 special
    permk3 = Secondary3Perm(k1,k2,k4,k3,K12,K14,K13,mu1,mu2,mu4,mu3,mu12,mu14,mu13)*P1*P2*P4 #permutation with k4<->k3
    permk2 = Secondary3Perm(k1,k4,k3,k2,K14,K13,K12,mu1,mu4,mu3,mu2,mu14,mu13,mu12)*P1*P4*P3 #permutation with k4<->k2
    permk1 = Secondary3Perm(k4,k2,k3,k1,K13,K12,K14,mu4,mu2,mu3,mu1,mu13,mu12,mu14)*P4*P2*P3  #permutation with k4<->k1

    # return permk1k2 + permk1k3 + permk1k4 + permk2k3 + permk2k4 + permk3k4
    return permk1 + permk2 + permk3 + permk4 + permk1k2 + permk1k3 + permk1k4 + permk2k3 + permk2k4 + permk3k4



#below is the secondary trispectrum in 7 shapes

def Secondary1122Perm7(S1,k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,P1,P2,P3,P4,P12,P13,P14,T21,alpha,beta,gamma,delta,eta,nu):
        FF113, GG113 = K2(k1,K13,k3)
        FF213, GG213 = K2(k2,K13,k4)
        FF114, GG114 = K2(k1,K14,k4)
        FF214, GG214 = K2(k2,K14,k3)

        pre1 = alpha + T21 * mu1
        pre2 = alpha + T21 * mu2
    
        if S1 == 0: #T21
            res = P14*(4*(mu1*mu14 + GG114*mu4)*pre1*pre2*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21) + 4*(mu14*mu2 + GG214*mu3)*pre1*pre2*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21) + 4*mu2*pre1*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21)*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21) + 
                        4*mu1*pre2*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21)*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21)) + P13*(4*(mu13*mu2 + GG213*mu4)*pre1*pre2*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21) + 4*(mu1*mu13 + GG113*mu3)*pre1*pre2*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21) + 
                        4*mu2*pre1*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21)*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21) + 4*mu1*pre2*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21)*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21))
        elif S1 == 1: #alpha
            res = P14*(4*mu14*pre1*pre2*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21) + 4*mu14*pre1*pre2*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21) + 4*pre1*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21)*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21) + 
                        4*pre2*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21)*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21)) + P13*(4*mu13*pre1*pre2*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21) + 4*mu13*pre1*pre2*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21) + 
                        4*pre1*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21)*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21) + 4*pre2*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21)*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21))
        elif S1 == 2: #beta
            res = 4*P14*pre1*pre2*(2*beta + FF114*gamma + FF214*gamma + mu14*pre1 + mu14*pre2 + GG214*mu3*T21 + GG114*mu4*T21) + 4*P13*pre1*pre2*(2*beta + FF113*gamma + FF213*gamma + mu13*pre1 + mu13*pre2 + GG113*mu3*T21 + GG213*mu4*T21)
        elif S1 == 3: #gamma
            res = 4*P14*pre1*pre2*(FF114*(beta + FF214*gamma + mu14*pre2 + GG214*mu3*T21) + FF214*(beta + FF114*gamma + mu14*pre1 + GG114*mu4*T21)) + 4*P13*pre1*pre2*(FF213*(beta + FF113*gamma + mu13*pre1 + GG113*mu3*T21) + FF113*(beta + FF213*gamma + mu13*pre2 + GG213*mu4*T21))
        else: res = np.zeros(k1.shape) #delta eta nu do not exist for 1122

        return res

def Secondary1113Perm7(S1,k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,T21,alpha,beta,gamma,delta,eta,nu): #this is the permutation with k4 special
    FF12, GG12 = K2(k1,k2,K12)
    FF13, GG13 = K2(k1,k3,K13)
    FF23, GG23 = K2(k2,k3,K14)
    FFF, GGG = K3(k1,k2,k3,k4,K12,K13,K14)

    pre1 = alpha + T21 * mu1
    pre2 = alpha + T21 * mu2
    pre3 = alpha + T21 * mu3
    if S1 == 0: #T21
        res = 2*(2*GG23*mu1*mu14 + 2*GG13*mu13*mu2 + 2*GG12*mu12*mu3 + 3*mu1*mu2*mu3 + 3*GGG*mu4)*pre1*pre2*pre3 + (mu3*pre1*pre2 + mu2*pre1*pre3 + mu1*pre2*pre3)*(2*eta*(FF12 + FF13 + FF23) + 6*delta*FFF + 2*beta*(mu1 + mu2 + mu3) + 2*gamma*(FF23*mu1 + FF13*mu2 + FF12*mu3) + 
2*alpha*(GG12*mu12 + GG13*mu13 + GG23*mu14 + mu1*mu2 + mu1*mu3 + mu2*mu3) + 6*nu + 2*(2*GG23*mu1*mu14 + 2*GG13*mu13*mu2 + 2*GG12*mu12*mu3 + 3*mu1*mu2*mu3 + 3*GGG*mu4)*T21)
    elif S1 == 1: #alpha
        res = 2*(GG12*mu12 + GG13*mu13 + GG23*mu14 + mu1*mu2 + mu1*mu3 + mu2*mu3)*pre1*pre2*pre3 + (pre1*pre2 + pre1*pre3 + pre2*pre3)*(2*eta*(FF12 + FF13 + FF23) + 6*delta*FFF + 2*beta*(mu1 + mu2 + mu3) + 2*gamma*(FF23*mu1 + FF13*mu2 + FF12*mu3) + 
   2*alpha*(GG12*mu12 + GG13*mu13 + GG23*mu14 + mu1*mu2 + mu1*mu3 + mu2*mu3) + 6*nu + 2*(2*GG23*mu1*mu14 + 2*GG13*mu13*mu2 + 2*GG12*mu12*mu3 + 3*mu1*mu2*mu3 + 3*GGG*mu4)*T21)
    elif S1 == 2: #beta
        res = 2*(mu1 + mu2 + mu3)*pre1*pre2*pre3
    
    elif S1 == 3: #gamma
        res = 2*(FF23*mu1 + FF13*mu2 + FF12*mu3)*pre1*pre2*pre3
    elif S1 == 4: #delta
        res = 6*FFF*pre1*pre2*pre3
    elif S1 == 5: #eta
        res = 2*(FF12 + FF13 + FF23)*pre1*pre2*pre3
    elif S1 == 6: #nu
        res = 6*pre1*pre2*pre3

    return res

def Secondary7(S1,k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,P1,P2,P3,P4,P12,P13,P14,T21,alpha,beta,gamma,delta,eta,nu):
     permk3k4 = Secondary1122Perm7(S1,k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,P1,P2,P3,P4,P12,P13,P14,T21,alpha,beta,gamma,delta,eta,nu)*P1*P2
     permk2k4 = Secondary1122Perm7(S1,k1,k3,k2,k4,K13,K12,K14,mu1,mu3,mu2,mu4,mu13,mu12,mu14,P1,P3,P2,P4,P13,P12,P14,T21,alpha,beta,gamma,delta,eta,nu)*P1*P3
     permk1k4 = Secondary1122Perm7(S1,k3,k2,k1,k4,K14,K13,K12,mu3,mu2,mu1,mu4,mu14,mu13,mu12,P3,P2,P1,P4,P14,P13,P12,T21,alpha,beta,gamma,delta,eta,nu)*P2*P3
     permk1k3 = Secondary1122Perm7(S1,k4,k2,k3,k1,K13,K12,K14,mu4,mu2,mu3,mu1,mu13,mu12,mu14,P4,P2,P3,P1,P13,P12,P14,T21,alpha,beta,gamma,delta,eta,nu)*P2*P4
     permk2k3 = Secondary1122Perm7(S1,k1,k4,k3,k2,K14,K13,K12,mu1,mu4,mu3,mu2,mu14,mu13,mu12,P1,P4,P3,P2,P14,P13,P12,T21,alpha,beta,gamma,delta,eta,nu)*P1*P4
     permk1k2 = Secondary1122Perm7(S1,k3,k4,k1,k2,K12,K13,K14,mu3,mu4,mu1,mu2,mu12,mu13,mu14,P3,P4,P1,P2,P12,P13,P14,T21,alpha,beta,gamma,delta,eta,nu)*P3*P4

     permk4 = Secondary1113Perm7(S1,k1,k2,k3,k4,K12,K13,K14,mu1,mu2,mu3,mu4,mu12,mu13,mu14,T21,alpha,beta,gamma,delta,eta,nu)*P1*P2*P3 #permutation with k4 spec
     permk3 = Secondary1113Perm7(S1,k1,k2,k4,k3,K12,K14,K13,mu1,mu2,mu4,mu3,mu12,mu14,mu13,T21,alpha,beta,gamma,delta,eta,nu)*P1*P2*P4 #permutation with k4<->k3
     permk2 = Secondary1113Perm7(S1,k1,k4,k3,k2,K14,K13,K12,mu1,mu4,mu3,mu2,mu14,mu13,mu12,T21,alpha,beta,gamma,delta,eta,nu)*P1*P4*P3 #permutation with k4<->k2
     permk1 = Secondary1113Perm7(S1,k4,k2,k3,k1,K13,K12,K14,mu4,mu2,mu3,mu1,mu13,mu12,mu14,T21,alpha,beta,gamma,delta,eta,nu)*P4*P2*P3 #permutation with k4<->k1

     return permk1 + permk2 + permk3 + permk4 + permk1k2 + permk1k3 + permk1k4 + permk2k3 + permk2k4 + permk3k4