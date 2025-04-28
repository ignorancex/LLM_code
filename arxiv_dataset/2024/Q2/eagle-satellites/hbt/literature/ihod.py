"""
iHOD (Zu & Mandelbaum 2015)


"""
from numpy import exp, log10, sin


def logMh(logMstar, h=0.7):
    """Equation 51"""
    # convert logMstar from the given h to h=1 as in the paper
    logMstar = logMstar + 2*log10(h)
    logMh = 4.41 / (1 + exp(-1.82*(logMstar-11.18))) \
            + 11.12 * sin(-0.12*(logMstar-23.37))
    # convert logMh from h=1 to the desired h
    return logMh - log10(h)