import logging
from math import floor, log, sqrt


class Measurement:
    """Class to define the objects storing the information relative to a single measurement."""

    def __init__(self, label, arxiv, m, stat, sys=None, sys2=None):
        self.label = label
        self.arxiv = arxiv
        self.m = m
        self.stat = stat
        self.sys = sys
        self.sys2 = sys2

        if any(s in self.label for s in ['average', 'PDG']):
            self.color = 'k'
        elif 'BaBar' in self.label:
            self.color = 'g'
        elif 'Belle' in self.label:
            self.color = 'r'
        elif 'BES' in self.label:
            self.color = 'g'
        elif 'CDF' in self.label:
            self.color = 'm'
        elif 'CLEO' in self.label:
            self.color = 'm'
        elif 'CMS' in self.label:
            self.color = 'g'
        elif 'LHCb' in self.label:
            self.color = 'b'
        else:
            raise RuntimeError(f'The label {self.label} is not supported')

    def result_str(self, units=1):

        def _ndigits_to_print(stat, sys=None, sys2=None):
            """Given three uncertainties, return the number of digits after the comma to be printed according to PDG
            conventions. Assumes that the input measurements have the right number of digits according to PDG
            conventions (where the number of digits is set by the least precise measurement).
            """

            # Special case for publications not adhering to PDG conventions
            if (self.label == 'CLEO' and
                (self.arxiv == 'hep-ex/9705006' or
                 (self.arxiv == '0906.3198' and stat > 0.2))) or (self.label == 'Belle' and self.arxiv == '2103.09969'
                                                                  and stat > 0.1) or self.arxiv in [
                                                                      "1911.01114", "2105.01565", "2405.11606"
                                                                  ]:
                return 1

            min_unc = stat
            if sys:
                min_unc = min(min_unc, sys)
            if sys2:
                min_unc = min(min_unc, sys2)
            main_exp = floor(log(min_unc) / log(10))
            if main_exp > 0:
                logging.warning(f'There may be too many figures printed for the uncertainties ({stat}, {sys}, {sys2})')
                return 0
            min_unc = floor(min_unc * 10**(2 - main_exp))
            if min_unc < 354:
                ndig = 2
            else:
                ndig = 1
            logging.debug(
                f'The number of digits to be printed for ({stat}, {sys}, {sys2}) is {ndig} ({ndig - main_exp - 1} after the comma; main_exp = {main_exp})'
            )
            return ndig - main_exp - 1

        val = self.m / units
        stat = self.stat / units
        sys = self.sys / units if self.sys else None
        sys2 = self.sys2 / units if self.sys2 else None

        ndig = _ndigits_to_print(stat, sys, sys2)
        if ndig > 2:  # TODO (fix for measurements non conformant to PDG
            ndig = 2

        s = f'{{:+.{ndig}f}} $\\pm$ {{:.{ndig}f}}'.format(val, stat)
        if sys:
            s += f' $\\pm$ {{:.{ndig}f}}'.format(sys)
        if sys2:
            s += f' $\\pm$ {{:.{ndig}f}}'.format(sys2)
        s = s.replace('-', '\u2013')
        return s

    def err(self):
        if self.sys is None:
            return self.stat
        else:
            err2 = (self.stat)**2 + (self.sys)**2
            if self.sys2:
                err2 += (self.sys2)**2
            return sqrt(err2)
