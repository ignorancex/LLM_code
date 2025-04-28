def BSec4Pressure(S1,k1,k2,k3,mu1,mu2,mu3,Pb1,Pb2,Pb3,Pbc1,Pbc2,Pbc3,alpha,beta,gamma,T21):
    kJ = 300
    kdot = (-k1*k1-k2*k2+k3*k3)/2/k1/k2
    kernel1 = (k1/k2 + k2/k1)*kdot
    kernel2 = kdot*kdot

    F2 = 5./7. + 1./2.*kernel1 + 2./7.*kernel2
    G2 = 3./7. + 1./2.*kernel1 + 4./7.*kernel2

    pre1 = (alpha+T21*mu1)
    pre2 = (alpha+T21*mu2)

    kappa1 = k1*k1/kJ/kJ
    kappa2 = k2*k2/kJ/kJ
    kappa3 = k3*k3/kJ/kJ
    sigma = 1./(10./3. + kappa3)

    Fcal2 = (F2 + 3./14. * kappa3)
    Fcal2tildePP = sigma*(F2*Pbc1*Pbc2 + 7./3.*Fcal2*Pb1*Pb2)

    Gcal2tildePP = 2.*Fcal2tildePP -  (1. + kdot*(k1**2 + k2**2)/2/k1/k2)*Pb1*Pb2

    if S1 == 0: #alpha
        return 2*(pre1 + pre2)*(Fcal2tildePP*gamma + Gcal2tildePP*mu3*T21) + Pb1*Pb2*(2*beta*(pre1 + pre2) + mu2*(pre1*pre2 + alpha*(pre1 + pre2)) + mu1*(pre1*pre2 + alpha*(pre1 + pre2) + 2*mu2*(pre1 + pre2)*T21))
    if S1 == 1: #beta
        return 2*Pb1*Pb2*pre1*pre2
    if S1 == 2: #gamma
        return 2*Fcal2tildePP*pre1*pre2
    if S1 == 3: #T21
        return Pb1*Pb2*(mu2*(2*beta + alpha*mu2)*pre1 + mu1**2*pre2*(alpha + 2*mu2*T21) + mu1*(2*beta*pre2 + mu2*(alpha*pre1 + alpha*pre2 + 2*pre1*pre2) + 2*mu2**2*pre1*T21)) + 2*(Fcal2tildePP*gamma*(mu2*pre1 + mu1*pre2) + Gcal2tildePP*mu3*(pre1*pre2 + mu2*pre1*T21 + mu1*pre2*T21))

    return 0