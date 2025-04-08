"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


from enum import Enum

class EventType(Enum):
    """
    Enum for discrete event simulation
    """
    StartUndetActive = 1
    EndUndetActive = 2
    StartDetActive = 3
    EndDetActive = 4
    StartImmune = 5
    EndImmune = 6
    StartVaccinated = 7
    SwitchUndetDet = 8