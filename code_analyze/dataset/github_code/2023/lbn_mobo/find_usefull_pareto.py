
import numpy as np
import oapackage

def refine_pareto(surrogate_pareto):
    pareto = oapackage.ParetoDoubleLong()
    ParetoFront = surrogate_pareto.T
    for ii in range(0, ParetoFront.shape[1]):
        w = oapackage.doubleVector((ParetoFront[0, ii], ParetoFront[1, ii]))
        pareto.addvalue(w, ii)
    lst = pareto.allindices()  # the indices of the Pareto optimal designs
    usefull_pareto = ParetoFront[:, lst].T
    return usefull_pareto, lst