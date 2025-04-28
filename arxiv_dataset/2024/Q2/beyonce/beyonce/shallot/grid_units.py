"""This module contains the Unit enum for grid disk properties"""

from enum import Enum


class Unit(Enum):
    """All the available units for disk properties."""
    
    ECLIPSE_DURATION = "$t_{ecl}$"
    DEGREE = "$^o$"
    NONE = "-"

    def __str__(self) -> str:
        """
        This method serves as the print value for this class

        Returns
        -------
        unit : str
            Human readable string representation of the enum value selected.
        """
        return f"{self.property_unit} ({self.symbol})"

    @property
    def property_unit(self) -> str:
        """ This method is used to parse the unit name in a pretty way."""
        words = self.name.split("_")
        formatted_words = []
        
        for word in words:
            formatted_words.append(word.capitalize())
        
        return " ".join(formatted_words)

    @property
    def symbol(self) -> str:
        """
        This method is used to parse the unit in a pretty way.

        Returns
        -------
        unit : str
            LaTeX formatted version of enum name.
        """
        return self.value