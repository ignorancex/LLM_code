import math
def cp_steel(T):
    #  https://www.setaram.com/application-notes/an372-heat-capacity-of-a-steel-betwwen-50c-and-1550c-liquid-state/
    cp = 821.0 - 0.434*T + 0.000232*T**2
    xmax = 1000.0
    xmin = 300.0
    cp = min(xmax,max(xmin,cp))
  #  cp = cp*0.6 # Tune DEBUG
    return cp


def Rho_steel(T):
    xc = 0.0  # Carbon content
    Rho = 8320 - 0.835*(T - 273.15) + (-83.2 + 8.35e-3*(T-273.15))*xc
    xmax = 8500.0
    xmin = 6500.0
    Rho = min(xmax,max(xmin,Rho))
    return Rho


def beta_steel(T):
    xc = 0.0
    return 1.2e-4
#     return 3.5e-5  # https://courses.lumenlearning.com/physics/chapter/13-2-thermal-expansion-of-solids-and-liquids/
#     return (-0.835 + 8.35*xc)/Rho_steel(T)


def my_steel(T):
    xc = 0.0 # Carbon weight fraction
    T_ = max(min(T,2000.0),1500.0)
    my = (8.767 + 34.62*math.exp(T_/1000) - 12.57*xc - 6.52e-5*T_**2 - 1.3e-5*xc**3+9.40e-3*T_*xc**2)/1000.0
#     xmax = 1.0e-2
#     xmin = 1.0e-4
#     my = min(xmax,max(xmin,my))
    return my

def Lam_steel(T):
    return 2.15e-2*(T-1818)+33.3
