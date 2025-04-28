import numpy as np

def guillot(tau, p):
    gamma = 10**(p[2]*np.tanh(np.log10(tau)/p[3]))
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (gamma - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
    return newTau

def pierrehumbert(tau, p):
    gamma = 10**(p[2]*np.tanh(np.log10(tau)/p[3]))
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (1 - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
    return newTau

def nDeriv(f, x, args):
    delta = f(x+0.00001, args) - f(x, args)
    return delta/0.00001

def lyapunovExp(f, x0, args):
    lyExp = 0
    x = x0
    for i in range(50):
        x = f(x,args)
    for i in range(1000):
        lyExp += np.log(np.abs(nDeriv(f,x,args)))
        x = f(x,args)
    lyExp /= 1000
    return lyExp

chaoticP1 = []
chaoticP2 = []
chaoticP3 = []
chaoticP4 = []
lyapunovExponent = []

p1s = np.linspace(0,10,25)
p2s = np.linspace(20,40,25)
p3s = np.linspace(0,2,25)

for i in range(50000):
    p1 = np.random.random()*0.4
    p2 = np.random.random()*20 + 20
    p3 = np.random.random()*2
    p4 = 0.5
    d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))

    lyExp = lyapunovExp(guillot, 0 , [d,p2,p3,p4])
    if lyExp > 0:
        chaoticP1.append(p1)
        chaoticP2.append(p2)
        chaoticP3.append(p3)
        chaoticP4.append(p4)
        lyapunovExponent.append(lyExp)

file = open("./data/chaoticPointsGuillot2.txt", "w")
for i in range(len(chaoticP1)):
    text = "{0}, {1}, {2}, {3:.5f} \n"
    file.write(text.format(chaoticP1[i],chaoticP2[i],chaoticP3[i],lyapunovExponent[i]))
file.close()
