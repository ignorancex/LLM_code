#!/usr/bin/env python

import logging                                                                  
import shutil                                                                   
import argparse                                                                 
import numpy                                                                    
import pycbc                                                                    
from pycbc.inference import io                                                  
from scipy.special import logsumexp 

"""Reweights QNM samples from a prior uniform in amp330 to a prior uniform
in estimated mass ratio beteen 1/25 and 1.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', required=True)
parser.add_argument('--output-file', required=True)

opts = parser.parse_args()

if opts.input_file == opts.output_file:
    raise ValueError("input can't be same as output")

# the prior boundaries that were used in the original run
# the priors were uniform on this
qbounds = {'xphm': (1, 25), 'nrsur': (1, 6)}
qmax = qbounds['xphm'][1]

# the conversion between amp330 and q is:
# q = (alpha33 + amp330) / (alpha33 - amp330),
# where alpha33 is:
alpha33 = 0.4433

pycbc.init_logging(True)
logging.info("Copying input to output")
shutil.copy(opts.input_file, opts.output_file)

logging.info('Reweighting samples')
fp = io.loadfile(opts.output_file, 'a')
samples = fp.read_samples(['amp330', 'logwt'],
                          raw_samples=True)
# amp -> q estimate (where q < 1)
def q_from_amp330(amp330, alpha33):
    return (alpha33 - amp330) / (alpha33 + amp330)

# set anything with an amp > 0.4433 to 0
mask = samples['amp330 >= {}'.format(alpha33)]
qest = q_from_amp330(samples['amp330'], alpha33)

logjacobian = samples['log(2*{alpha}) - 2*log({alpha}+amp330)'
                      .format(alpha=alpha33)]
# set anything with q < 1/25 to 0 and renormalize to the new prior
mask = qest < 1./qmax
logpnew = logjacobian + numpy.log(qmax) - numpy.log(qmax - 1)
logpnew[mask] = -numpy.inf
ampmax = 0.5
logporig = -numpy.log(ampmax)

logwt = samples['logwt']
logwt += logpnew - logporig

# renormalize and save
logz = fp.log_evidence[0]
lognorm = logsumexp(logwt - logz)
logging.info("lognorm is: %f", lognorm)
logwt -= lognorm
fp['samples/logwt'][:] = logwt
fp.close()
