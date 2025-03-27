def derive_Smn(chain):
    omnuh2 = chain['cosmological_parameters--omnuh2']
    Smn = omnuh2 * 93.14
    name = "cosmological_parameters--smn"
    return Smn, name
