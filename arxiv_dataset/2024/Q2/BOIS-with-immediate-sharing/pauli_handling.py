"""
This file is part of the program BOIS-with-immediate-sharing, which implements the variational quantum eigensolver (VQE) algorithm based on Bayesian optimisation (BO) with different types of information sharing allowing "immediate sharing".
Copyright (C) 2024  Milena Röhrs

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy

def twopaulis(list1, list2):
    """
    Compares two lists and creates a new list that combines the unique elements from both lists.
    Additionally, it returns a mapping list that indicates the index of each element in the new list.

        Parameters:
            list1 (list): The first list to be compared.
            list2 (list): The second list to be compared.

        Returns:
            tuple: A tuple containing two elements:
                - list3 (list): The new list that combines the unique elements from both input lists.
                - mapping (list): A list of indices that maps the elements in list2 to their corresponding
                positions in list3.
    """

    mapping = []
    list3 = copy.deepcopy(list1)
    for wl in list2:
        found = False
        indx = 0
        laenge = len(list3)
        while (not found) and (indx < laenge):
            if wl == list3[indx]:
                found = True
                mapping.append(indx)
            else:
                indx += 1

        if (not found):
            list3.append(wl)
            mapping.append(laenge)

    return list3, mapping
