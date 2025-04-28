"""This module contains the Name enum for grid disk properties"""

from enum import Enum, auto


class Name(Enum):
    """All the available property names for disk properties."""
    
    DISK_RADIUS = auto()
    INCLINATION = auto()
    TILT = auto()
    FX_MAP = auto()
    FY_MAP = auto()
    DIAGNOSTIC_MAP = auto()
    GRADIENT = auto()
    GRADIENT_FIT = auto()


    def __str__(self) -> str:
        """
        This method serves as the print value for this class

        Returns
        -------
        property_name : str
            String value of the enum selected.
        """
        return self.property_name


    @property
    def property_name(self) -> str:
        """
        This property parses the name in a pretty way.

        Returns
        -------
        name : str
            Human readable version of enum name.
        """
        words = self.name.split("_")
        formatted_words = []
        
        for word in words:
            formatted_words.append(word.capitalize())
        
        return " ".join(formatted_words)