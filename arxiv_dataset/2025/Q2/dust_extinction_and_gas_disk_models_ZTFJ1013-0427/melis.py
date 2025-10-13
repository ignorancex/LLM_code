## We solve Equations 2, 7, 8 in Melis+, 2010

import numpy as np
import scipy

## Defining Constants
kb = 1.3806e-16
mH = 1.67356e-24
h = 6.626e-27
c = 2.998e10
h = 6.62618e-27

def melis_pol_par(Ts, nuL = 3.46e14, hnuI = 1.2817e-11):
    mca = 40*mH
    
    ## Grid of distances (in units of stellar radius)
    r_grid = np.arange(20,300,1)
    
    ## Equation 3
    xI = hnuI/(kb*Ts)
    
    ## Solve integral in Eq. 2
    def func(x):
        return ((x**3)-(xI*(x**2)))*np.exp(-x)
    
    integral,_ = scipy.integrate.quad(func, xI, np.inf)
    
    ## LHS factor without Tg and exponent
    fact_lhs = 12*np.sqrt(2*kb/mca)*2*np.pi*h*(nuL**4)/(c**3)
    
    ## RHS factor without Tg and r
    fact_rhs = (2/3)*(2*h/(c**2))*((kb*Ts/h)**4)*integral
    
    ## Taking LHS factor to the right and devding by (r/R_star)**3
    fact = fact_rhs/fact_lhs
    rhss = fact/(r_grid**3)
    
    ## Tg grid to find the solution from minimum distance
    tg_grid = np.arange(100, 21000, 1)
    
    ## The exponent in LHS
    exp = np.exp(-h*nuL/(kb*tg_grid))
    lhss = np.sqrt(tg_grid)*exp
    
    ## Solve for the gas temperature using grids
    temps = []
    for rhs in rhss:
        diff = abs(lhss - rhs)
        min_index = np.where(diff==min(diff))[0][0]
        temps.append(tg_grid[min_index])
        
    ## Fit with a polynomial for future use
    melis_pol_par = np.polyfit(r_grid, temps, deg = 10)
    
    return melis_pol_par

if __name__ == "__main__":
  melis_pol_par = melis_pol_par(21243) ## At temperature of WDJ1013
  print(melis_pol_par)
