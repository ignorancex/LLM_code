# -*- coding: utf-8 -*-
"""
Module for handling linelist data from the VALD3 database (http://vald.astro.uu.se/).


"""
import logging
import re
from io import StringIO
from os.path import dirname, join

import numpy as np
import pandas as pd
import pybtex.database
from astropy import units as u

from ..abund import Abund
from .linelist import LineList, LineListError

logger = logging.getLogger(__name__)


class ValdError(LineListError):
    """Vald Data File Error"""


class ValdFile(LineList):
    """Atomic data for a list of spectral lines."""

    citation_info = r"""
    @ARTICLE{2015PhyS...90e4005R,
        author = {{Ryabchikova}, T. and {Piskunov}, N. and {Kurucz}, R.~L. and
        {Stempels}, H.~C. and {Heiter}, U. and {Pakhomov}, Yu and
        {Barklem}, P.~S.},
        title = "{A major upgrade of the VALD database}",
        journal = {Physica Scripta},
        year = "2015",
        month = "May",
        volume = {90},
        number = {5},
        eid = {054005},
        pages = {054005},
        doi = {10.1088/0031-8949/90/5/054005},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    @ARTICLE{2000BaltA...9..590K,
        author = {{Kupka}, F.~G. and {Ryabchikova}, T.~A. and {Piskunov}, N.~E. and
        {Stempels}, H.~C. and {Weiss}, W.~W.},
        title = "{VALD-2 -- The New Vienna Atomic Line Database}",
        journal = {Baltic Astronomy},
        keywords = {ATOMIC DATA, METHODS: SPECTROSCOPIC, STARS: ABUNDANCES, STARS: CHEMICALLY PECULIAR},
        year = "2000",
        month = "Jan",
        volume = {9},
        pages = {590-594},
        doi = {10.1515/astro-2000-0420},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2000BaltA...9..590K},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    @ARTICLE{1999A&AS..138..119K,
        author = {{Kupka}, F. and {Piskunov}, N. and {Ryabchikova}, T.~A. and
        {Stempels}, H.~C. and {Weiss}, W.~W.},
        title = "{VALD-2: Progress of the Vienna Atomic Line Data Base}",
        journal = {\aaps},
        keywords = {ATOMIC DATA, TECHNIQUES: SPECTROSCOPIC, SUN: ABUNDANCES, STARS: ABUNDANCES, STARS: ATMOSPHERES, STARS: CHEMICALLY PECULIAR},
        year = "1999",
        month = "Jul",
        volume = {138},
        pages = {119-133},
        doi = {10.1051/aas:1999267},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1999A&AS..138..119K},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    @ARTICLE{1997BaltA...6..244R,
        author = {{Ryabchikova}, T.~A. and {Piskunov}, N.~E. and {Kupka}, F. and
        {Weiss}, W.~W.},
        title = "{The Vienna Atomic Line Database : Present State and Future Development}",
        journal = {Baltic Astronomy},
        keywords = {DATABASES:  ATOMIC LINE PARAMETERS, STELLAR SPECTROSCOPY},
        year = "1997",
        month = "Mar",
        volume = {6},
        pages = {244-247},
        doi = {10.1515/astro-1997-0216},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1997BaltA...6..244R},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    @ARTICLE{1995A&AS..112..525P,
        author = {{Piskunov}, N.~E. and {Kupka}, F. and {Ryabchikova}, T.~A. and
        {Weiss}, W.~W. and {Jeffery}, C.~S.},
        title = "{VALD: The Vienna Atomic Line Data Base.}",
        journal = {\aaps},
        keywords = {ATOMIC DATA, STARS: ABUNDANCES, ASTRONOMICAL DATA BASES: MISCELLANEOUS},
        year = "1995",
        month = "Sep",
        volume = {112},
        pages = {525},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1995A&AS..112..525P},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """

    acknowledgement = (
        r"This work has made use of the VALD database, operated at Uppsala University,"
        r"the Institute of Astronomy RAS in Moscow, and the University of Vienna."
    )

    def __init__(self, filename, medium=None):
        self.filename = filename
        self.atmo = None
        self.abund = None
        self.unit = None
        self.energy_unit = None
        linelist = self.loads(filename)

        super().__init__(
            linelist,
            lineformat=self.lineformat,
            medium=self.medium,
            citation_info=self.citation_info,
        )
        # Convert to desired medium
        if medium is not None:
            self.medium = medium

    @staticmethod
    def load(filename):
        """
        Read line data file from the VALD extract stellar service

        Parameters
        ----------
        filename : str
            Name of the VALD linelist file to read

        Returns
        -------
        vald : ValdFile
            Parsed vald file
        """
        return ValdFile(filename)

    def identify_valdtype(self, lines):
        """Determines whether the file was created with extract_all, extract_stellar, or extract_element
        and whether it is in long or short format

        Parameters
        ----------
        lines : list(str)
            file contents
        """
        header = lines[0].strip().split()

        if header[0] == "Damping":
            # short format extract all / extract element
            return "extract_all", "short"
        elif header[0] == "Lande":
            # long format extract all / extract element
            return "extract_all", "long"
        else:
            header = lines[1].strip().split()
            if header[0] == "Damping":
                # short format, extract stellar
                return "extract_stellar", "short"
            elif header[0] == "Lande":
                # long format, extract stellar
                return "extract_stellar", "long"

        raise ValueError("Could not identify ValdFile type")

    def loads(self, filename):
        logger.info("Loading VALD file %s", filename)

        with open(filename, "r") as file:
            lines = file.readlines()

        # Check for Warnings
        while lines[0].lstrip().startswith("WARNING"):
            logger.warning(f"VALD {lines[0].lstrip()}")
            lines = lines[1:]

        # Determine File type and format
        valdtype, fmt = self.identify_valdtype(lines)

        # Determine the number of lines in the file
        if valdtype == "extract_stellar":
            n = self.parse_header(lines[0])
            # Skip the info header if extract stellar
            lines = lines[1:]
        else:
            n = self.parse_nlines(lines, fmt, valdtype)

        # Determine the units and medium in the linelist
        self.parse_columns(lines[1])

        # Split the lines into the different parts
        try:
            if fmt == "long":
                linedata = lines[2 : 2 + n * 4]
                refdata = linedata[3::4]
                if valdtype == "extract_stellar":
                    atmodata = lines[2 + n * 4]
                    abunddata = lines[3 + n * 4 : 21 + n * 4]
            elif fmt == "short":
                linedata = lines[2 : 2 + n]
                refdata = linedata
                if valdtype == "extract_stellar":
                    atmodata = lines[2 + n]
                    abunddata = lines[3 + n : 21 + n]
        except IndexError:
            msg = "Linelist file is shorter than it should be according to the number of lines. Is it incomplete?"
            logger.error(msg)
            raise IOError(msg)

        # Process the individual parts
        linelist = self.parse_linedata(linedata, fmt=fmt, valdtype=valdtype)
        if valdtype == "extract_stellar":
            self.atmo = self.parse_valdatmo(atmodata)
            self.abund = self.parse_abund(abunddata)

        self.citation_info += self.parse_references(refdata, fmt)

        return linelist

    def parse_nlines(self, lines, fmt, valdtype):
        if valdtype == "extract_stellar":
            pattern = r"^ '\S+',$"
        else:
            pattern = r"\* oscillator|References"
        pattern = re.compile(pattern)
        for i in range(len(lines)):
            if re.match(pattern, lines[i]):
                # Offset by the two header lines
                n = i - 2
                break

        if fmt == "long":
            n //= 4
        self.nlines = n
        return self.nlines

    def parse_header(self, line):
        """
        Parse header line from a VALD line data file
        and sets the internal parameters

        Parameters
        ----------
        line : str
            header line of a vald file

        Raises
        ------
        ValdError
            If the header is not understood
        """
        words = [w.strip() for w in line.split(",")]
        # if len(words) < 5 or words[5] != "Wavelength region":
        #     raise ValdError(f"{self.filename} is not a VALD line data file")
        try:
            self.nlines = int(words[2])
            # self._wavelo = float(words[0])
            # self._wavehi = float(words[1])
            # self._nlines_proc = int(words[3])
            # self._vmicro = float(words[4])
            pass
        except:
            raise ValdError(f"{self.filename} is not a VALD line data file")
        return self.nlines

    def parse_columns(self, line):
        match = re.search(r"WL_(air|vac)\((.*?)\)", line)
        medium = match.group(1)
        unit = match.group(2)

        match = re.search(r"E_low\((.*?)\)", line)
        if match is None:
            match = re.search(r"Excit\((.*?)\)", line)
        energy_unit = match.group(1)

        if medium == "air":
            self._medium = "air"
        elif medium == "vac":
            self._medium = "vac"
        else:
            raise ValueError(
                "Could not determine the medium that the wavelength is based on (air or vacuum)"
            )

        if unit == "A":
            self.unit = u.AA
        elif unit == "nm":
            self.unit = u.nm
        elif unit == "cm^-1":
            self.unit = 1 / u.cm
        else:
            raise ValueError("Could not determine the unit of the wavelength")

        if energy_unit == "eV":
            self.energy_unit = u.eV
        elif energy_unit == "cm^-1":
            self.energy_unit = 1 / u.cm
        else:
            raise ValueError("could not determine the unit of the energy levels")

        return self.medium, self.unit, self.energy_unit

    def parse_linedata(self, lines, fmt="short", valdtype="extract_stellar"):
        """Parse line data from a VALD line data file

        Parameters
        ----------
        lines : list of str
            lines of the input data file
        fmt : {"short", "long"}, optional
            linelist format, short format has one
            line of data per spectral line, while the
            long format uses four lines per spectral line.
            The default is "short"

        Returns
        -------
        linelist : LineList
            the parsed linelist
        """

        if fmt == "short":
            names = [
                "species",
                "wlcent",
                "excit",
                "vmic",
                "gflog",
                "gamrad",
                "gamqst",
                "gamvw",
                "lande",
                "reference",
            ]
            if valdtype == "extract_stellar":
                names = names[:-1] + ["depth", names[-1]]

        elif fmt == "long":
            names = [
                "species",
                "wlcent",
                "gflog",
                "excit",
                "j_lo",
                "e_upp",
                "j_up",
                "lande_lower",
                "lande_upper",
                "lande",
                "gamrad",
                "gamqst",
                "gamvw",
            ]
            if valdtype == "extract_stellar":
                names += ["depth"]

            term_lower = lines[1::4]
            term_upper = lines[2::4]
            comment = lines[3::4]
            lines = lines[::4]

        data = StringIO("".join(lines))
        linelist = pd.read_csv(
            data,
            sep=",",
            names=names,
            header=None,
            quotechar="'",
            skipinitialspace=True,
            usecols=range(len(names)),
        )

        # Convert from cm^-1 to eV
        if self.energy_unit == 1 / u.cm:
            conversion_factor = 8065.544
            linelist["excit"] /= conversion_factor
            if fmt == "long":
                linelist["e_upp"] /= conversion_factor

        if fmt == "long":
            comment = [c.replace("'", "").strip() for c in comment]
            linelist["reference"] = comment

            # Parse energy level terms
            # Extract Stellar has quotation marks around the levels
            # extract element does not...
            if valdtype == "extract_stellar":
                term_lower = [t.strip()[8:-1].strip() for t in term_lower]
                term_upper = [t.strip()[8:-1].strip() for t in term_upper]
            else:
                term_lower = [t.strip()[8:].strip() for t in term_lower]
                term_upper = [t.strip()[8:].strip() for t in term_upper]

            term_lower = np.char.partition(term_lower, " ")[:, (0, 2)]
            term_lower = np.char.strip(term_lower)
            idx = term_lower[:, 1] == ""
            term_lower[idx, 1] = term_lower[idx, 0]
            term_lower = np.char.add(
                np.char.add(term_lower[:, 0], " "), term_lower[:, 1]
            )

            term_upper = np.char.partition(term_upper, " ")[:, (0, 2)]
            term_upper = np.char.strip(term_upper)
            idx = term_upper[:, 1] == ""
            term_upper[idx, 1] = term_upper[idx, 0]
            term_upper = np.char.add(
                np.char.add(term_upper[:, 0], " "), term_upper[:, 1]
            )

            linelist["term_lower"] = term_lower
            linelist["term_upper"] = term_upper

            # extract error data
            error = np.array([s[:10].strip() for s in comment])
            error = LineList.parse_line_error(
                error,
                linelist["depth"] if valdtype == "extract_stellar" else None,
            )
            linelist["error"] = error

        # Convert from whatever unit to Angstrom
        factor = self.unit.to(u.AA)
        linelist["wlcent"] *= factor
        self.unit = "Angstrom"
        self.lineformat = fmt

        return linelist

    def parse_valdatmo(self, line):
        """Parse VALD model atmosphere line from a VALD line data file

        Parameters
        ----------
        line : str
            line form the model atmosphere

        Returns
        -------
        atmo : str
            Name of the model atmosphere

        Raises
        ------
        ValdError
            If the line is not from a model atmosphere
        """
        lstr = line.strip()
        if lstr[0] != "'" or lstr[-2:] != "',":
            raise ValdError(f"error parsing model atmosphere: {lstr}")
        return lstr[1:-2]

    def parse_abund(self, lines):
        """Parse VALD abundance lines from a VALD line data file

        Parameters
        ----------
        lines : list of str
            Lines containing the VALD abundance data

        Returns
        -------
        abund : Abund
            Parsed abundance data

        Raises
        ------
        ValdError
            If the data could not be parsed
        """
        abstr = "".join(["".join(line.split()) for line in lines])
        words = [w[1:-1] for w in abstr.split(",")]
        if len(words) != 100 or words[99] != "END":
            raise ValdError(f"Error parsing abundances: {abstr}")
        pattern = [w.split(":") for w in words[:-1]]
        pattern = {el: float(ab) for el, ab in pattern}
        monh = 0
        return Abund(monh, pattern, type="sme")

    def parse_references(self, lines, fmt):
        # Search the linelist data for this pattern, e.g:
        # 1 gf:K14
        # 4 KCN'
        if fmt == "long":
            idiscard = 45
        elif fmt == "short":
            idiscard = 90
        else:
            raise ValueError

        pattern = r"\s\d+ (\w+:)?([\w+]+)[\s']"
        pattern = re.compile(pattern)
        # Discard the initial part of the line
        lines = [l[idiscard:] for l in lines]
        lines = "".join(lines)
        references = [match.group(2) for match in re.finditer(pattern, lines)]
        # We only need each reference ones
        ref = set(references)
        # Multiple references are seperated by '+'
        references = []
        for r in ref:
            references += r.split("+")
        # And make it unique again, if necessary
        references = set(references)
        # some data entries are case sensitive, but bibtex is case insnsitive
        # so remove those and replace them with fixed versions
        if "LWb" in references:
            references.add("LWb2")
            references.remove("LWb")
        if "LGb" in references:
            references.add("LGb2")
            references.remove("LGb")

        # Get references from bibtex file
        # TODO: only load this once? But then again, how often will we do this?
        bibdata = pybtex.database.parse_file(join(dirname(__file__), "VALD3_ref.bib"))
        # DEBUG:
        # pybtex.format_from_string(bibdata.to_string("bibtex"), style="plain", output_backend="plaintext")

        entries = {}
        for r in references:
            try:
                entries[r] = bibdata.entries[r]
            except KeyError as ex:
                logger.warning(f"Could not find citation key: {r}")
                logger.debug(ex)
        bibdata_filtered = pybtex.database.BibliographyData(entries)
        return bibdata_filtered.to_string("bibtex")
