def BSec(k1,k2,k3):
    kdot = (k3*k3-k1*k1-k2*k2)/2/k1/k2
    kernel1 = kdot*(k1/k2 + k2/k1)
    kernel2 = kdot*kdot

    F = 5./7. +  1./2. * kernel1 + 2./7. * kernel2
    # S = kernel2 - 1/3

    BgravF = 2*F
    # BgravS = 2*S

    return BgravF

#below are the symmetrized F2 and G2 kernels
def K2(k1,k2,K12):
    kdot = (K12*K12-k1*k1-k2*k2)/2/k1/k2
    kernel1 = kdot*(k1/k2 + k2/k1)
    kernel2 = kdot*kdot

    FF = 5./7. +  1./2. * kernel1 + 2./7. * kernel2
    GG = 3./7. +  1./2. * kernel1 + 4./7. * kernel2
    return FF, GG

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

def K3(k1,k2,k3,k4,K12,K13,K14):

    F1,G1 = KPerm(k1,k2,k3,K14,k4) 
    F2,G2 = KPerm(k2,k1,k3,K13,k4)
    F3,G3 = KPerm(k3,k1,k2,K12,k4)
    return (F1+F2+F3)/3,(G1+G2+G3)/3

#Below is the total trispectrum as a single shape
def TSec(k1,k2,k3,k4,K12,K13,K14,P1,P2,P3,P4,P12,P13,P14):
    def Secondary2Perm(k1,k2,k3,k4,K13,K14,P1,P2,P13,P14):
        FF113, GG113 = K2(k1,K13,k3)
        FF213, GG213 = K2(k2,K13,k4)
        FF114, GG114 = K2(k1,K14,k4)
        FF214, GG214 = K2(k2,K14,k3)
 
        result = 4*(P14*(FF214)*(FF114) + P13*(FF113)*(FF213))
        return result*P1*P2
        return 0

    def Secondary3Perm(k1,k2,k3,k4,K12,K13,K14): #this is the permutation with k4 special
        FFF, GGG = K3(k1,k2,k3,k4,K12,K13,K14)
        result = 6*FFF
        return result

    permk3k4 = Secondary2Perm(k1,k2,k3,k4,K13,K14,P1,P2,P13,P14)
    permk2k4 = Secondary2Perm(k1,k3,k2,k4,K12,K14,P1,P3,P12,P14)
    permk1k4 = Secondary2Perm(k3,k2,k1,k4,K13,K12,P3,P2,P13,P12)
    permk1k3 = Secondary2Perm(k4,k2,k3,k1,K12,K14,P4,P2,P12,P14)
    permk2k3 = Secondary2Perm(k1,k4,k3,k2,K13,K12,P1,P4,P13,P12)
    permk1k2 = Secondary2Perm(k3,k4,k1,k2,K13,K14,P3,P4,P13,P14)

    permk4 = Secondary3Perm(k1,k2,k3,k4,K12,K13,K14)*P1*P2*P3 #permutation with k4 special
    permk3 = Secondary3Perm(k1,k2,k4,k3,K12,K14,K13)*P1*P2*P4 #permutation with k4<->k3
    permk2 = Secondary3Perm(k1,k4,k3,k2,K14,K13,K12)*P1*P4*P3 #permutation with k4<->k2
    permk1 = Secondary3Perm(k4,k2,k3,k1,K13,K12,K14)*P4*P2*P3  #permutation with k4<->k1

    return permk1 + permk2 + permk3 + permk4 + permk1k2 + permk1k3 + permk1k4 + permk2k3 + permk2k4 + permk3k4