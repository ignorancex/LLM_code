# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class Coefficient:
    """
    Coefficient object

    Parameters
    ----------
        name: str
            name of the operator corresponding to the Wilson coefficient
        minimum : float
            minimum value
        maximum : float
            maximum value
        value : float, optional
            best value. If None set to random between minimum and maximum
        constrain : dict, bool
            - if False, the parameter is free, default option
            - if True, the parameter is fixed to the given value
            - if dict the parameter is fixed to a function of other coefficients

    """

    def __init__(self, name, minimum, maximum, value=None, constrain=False):
        self.name = name
        # Temporary check for deprecated operator
        if name == "Opd":
            raise ValueError(
                "The operator Opd is deprecated and has been renamed. Use OpBox instead."
            )
        self.minimum = minimum
        self.maximum = maximum

        # determine if the parameter is free
        self.is_free = False
        self.constrain = None

        if constrain is False:
            if value is not None:
                raise ValueError(
                    f"Wilson Coefficient {self.name} is free, but a value is specified"
                )
            self.is_free = True
        elif constrain is True:
            if value is None:
                raise ValueError(
                    f"Wilson Coefficient {self.name} is fixed, but no value is specified"
                )
        elif isinstance(constrain, dict):
            value = 0.0
            self.constrain = [self.build_additive_factor_dict(constrain)]
        elif isinstance(constrain, list):
            value = 0.0
            self.constrain = [
                self.build_additive_factor_dict(fact_dict) for fact_dict in constrain
            ]

        # if no value is already there, the parameter is free
        self.value = value
        if value is None:
            self.value = np.random.uniform(low=minimum, high=maximum)

    @staticmethod
    def build_additive_factor_dict(constrain):
        r"""
        Build the dictionary for each additive factor appearing in the constrain:

        .. :math:
             \prod_{i=1} a_{i} c_{i}^{n_{i}}

        Parameters
        ----------
            constrain: dict
                dict object with the form {'c1': a1, 'c2': [a2,n2], ...}

        Returns
        -------
            factor_dict: dict
                dict object with the form {'c1': [a1,1], 'c2': [a2,n2], ...}
        """
        factor_dict = {}
        # loop on free parameters appearing in the factor
        for key, factor in constrain.items():
            if isinstance(factor, (int, float)):
                factor_dict[key] = np.array([factor, 1])
            else:
                factor = np.array(factor)
                # if factor has not 2 elements or is not a number raise error
                if factor.size > 2 or factor.dtype not in [int, float]:
                    raise ValueError(f"Unknown specified constrain {constrain}")
                factor_dict[key] = factor
        return factor_dict

    def __repr__(self):
        return self.name

    def __eq__(self, coeff_other):
        return self.name == coeff_other.name

    def __lt__(self, coeff_other):
        return self.name < coeff_other.name

    def __add__(self, coeff_other):
        self.value += coeff_other.value
        return self

    def update_constrain(self, inv_rotation):
        """Update the constrain when a new basis is chosen.
        Only linear constrain are supported.

        Parameters
        ----------
            inv_rotation: pd.DataFrame
                rotation matrix from the original basis to the new_basis
        """

        # loop on the sum and simplify the constrain
        old_coeffs = [(*factor.keys(),)[0] for factor in self.constrain]
        old_factors = [(*factor.values(),)[0][0] for factor in self.constrain]
        rot = inv_rotation[old_coeffs]
        new_constrain = (rot * old_factors).sum(axis=1)
        new_constrain = new_constrain[new_constrain != 0]
        self.constrain = [{k: v} for k, v in new_constrain.items()]


class CoefficientManager:
    """
    Coefficient objcts manager

    Parameters
    ----------
        input_array: np.ndarray or list
            list of `smefit.coefficients.Coefficient` instances

    """

    def __init__(self, input_array):
        # all the numerical informations are stored into a DataFrame
        self._table = pd.DataFrame(
            np.array(
                [[o.value, o.minimum, o.maximum] for o in input_array], dtype=float
            ),
            columns=["value", "minimum", "maximum"],
        )
        self._table.index = np.array([o.name for o in input_array], dtype=str)
        self.is_free = np.array([o.is_free for o in input_array], dtype=bool)

        # NOTE: this will not be updated.
        self._objlist = input_array

    @property
    def name(self):
        return np.array(self._table.index, dtype=str)

    @property
    def value(self):
        return np.array(self._table.value.values, dtype=float)

    @property
    def minimum(self):
        return self._table.minimum.values

    @property
    def maximum(self):
        return self._table.maximum.values

    @property
    def size(self):
        return self._table.shape[0]

    @classmethod
    def from_dict(cls, coefficient_config):
        """
        Create a coefficientManager from a dictionary

        Parameters
        ----------
            coefficient_config : dict
                coefficients configuration dictionary

        Returns
        -------
            coefficient_manager: `smefit.coefficients.CoefficientManager`
                instance of the class
        """
        elements = []
        for name, property_dict in coefficient_config.items():
            constrain = (
                property_dict["constrain"] if "constrain" in property_dict else False
            )
            elements.append(
                Coefficient(
                    name,
                    property_dict["min"],
                    property_dict["max"],
                    constrain=constrain,
                    value=property_dict.get("value", None),
                )
            )
        # make sure elements are sorted by names
        return cls(np.unique(elements))

    def __getitem__(self, idx):
        # TODO: shall it return the object list element?
        # in that case it has to be updated
        if isinstance(idx, int):
            return self._table.iloc[idx]
        return self._table.loc[idx]

    @property
    def free_parameters(self):
        """Returns the table containing only free parameters"""
        return self._table[self.is_free]

    def set_free_parameters(self, value):
        """Set the values of the free parmaters"""
        self._table.iloc[self.is_free, 0] = value

    def set_constraints(self):
        r"""
        Sets constraints between coefficients according to the
        coefficient.constrain information:

        .. :math:
            c_{m} = \sum_{i=1} \prod_{j=1}^{N_i} a_{i,j} c_{i,j}^{n_{i,j}}
        """

        # loop pn fixed coefficients
        for coefficient_fixed in self._objlist[np.invert(self.is_free)]:
            # skip coefficient fixed to a single value
            if coefficient_fixed.constrain is None:
                continue

            temp = 0.0
            for add_factor_dict in coefficient_fixed.constrain:
                free_dofs = [
                    self._table.at[fixed_name, "value"]
                    for fixed_name in add_factor_dict
                ]

                # matrix with multiplicative factors and exponents
                fact_exp = np.array((*add_factor_dict.values(),), dtype=float)
                temp += np.prod(fact_exp[:, 0] * np.power(free_dofs, fact_exp[:, 1]))
            self._table.at[coefficient_fixed.name, "value"] = temp

    def update_constrain(self, inv_rotation):
        r"""Update the constraints according to rotation matrix.
        Only linear constrain are supported.

        Parameters
        ----------
            inv_rotation: pd.DataFrame
                rotation matrix from the original basis to the new_basis
        """
        for coefficient_fixed in self._objlist[~self.is_free]:
            # skip coefficient fixed to a single value
            if coefficient_fixed.constrain is None:
                continue
            coefficient_fixed.update_constrain(inv_rotation)
