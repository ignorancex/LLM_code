# there is no specific likelihood code for this experiment, because it
# falls in the category of CMB experiments described in the "mock CMB"
# format. The class below inherits the properties of a general class
# "Likelihood_mock_cmb", which knows how to deal with all experiments in
# "mock CMB" format.

from montepython.likelihood_class import Likelihood_sptpol_500d_TEEE


class sptpol_500d_TEEE_lmax1000(Likelihood_sptpol_500d_TEEE):
    pass
