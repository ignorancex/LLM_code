"""
    This module contains the pDUNE geometry parameters.
"""

nb_tdc_ticks = 6000  # detector timeticks number
nb_ichannels = 800  # channel number in induction plane
nb_cchannels = 960  # channel number in collection plane
nb_apas = 6  # APAs number
nb_apa_channels = 2 * nb_ichannels + nb_cchannels  # number of channels per apa
nb_event_channels = nb_apas * nb_apa_channels  # total channel number

ElectronsToADC = 6.8906513e-3  # detector response: 1 e- = 6.8906513e-3 ADC
