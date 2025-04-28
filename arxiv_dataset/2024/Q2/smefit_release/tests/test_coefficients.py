# -*- coding: utf-8 -*-
"""Test utils module"""
import numpy as np
import pandas as pd

from smefit import coefficients

coeff_dict = {
    "c_b": {
        "min": -1,
        "max": 1,
    },
    "c_a": {
        "min": -2,
        "max": 1,
    },
    "c_c": {  # fixed to 1
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
    "c_d": {  # fixed to -0.1 * c_b
        "constrain": {"c_b": -0.1},
        "min": -3,
        "max": 1,
    },
    "c_e": {  # fixed to -0.1 * c_a + 0.2 * c_b
        "constrain": [{"c_b": 0.2}, {"c_a": -0.1}],
        "min": -4,
        "max": 1,
    },
    "c_f": {  # fixed to -0.2 * c_a * c_b^2 /(2 * c_c^2) -0.1 * c_a
        "constrain": [
            {"c_a": 0.2, "c_b": [1.0, 2.0], "c_c": [2.0, -2.0]},
            {"c_a": -0.1},
        ],
        "min": -5,
        "max": 1,
    },
    "c_g": {  # fixed to -0.2 * c_a * c_b
        "constrain": [{"c_a": [-0.2, 1.0], "c_b": [1.0, 1.0]}],
        "min": -5,
        "max": 1,
    },
}

coeff_dict_free = {
    "c_b": {
        "min": -1,
        "max": 1,
    },
    "c_a": {
        "min": -2,
        "max": 1,
    },
}


class TestCoefficient:
    name = "c_a"
    minimum = -1
    maximum = 1
    c_test = coefficients.Coefficient(name, minimum, maximum)
    constrain_test = {"c_b": np.array([-0.1, 1])}

    def test_init(self):
        assert self.c_test.name == self.name
        assert self.c_test.minimum == self.minimum
        assert self.c_test.maximum == self.maximum
        assert self.c_test.value <= self.maximum and self.c_test.value >= self.minimum

    def test_build_additive_factor_dict(self):
        np.testing.assert_equal(
            self.constrain_test,
            coefficients.Coefficient.build_additive_factor_dict(
                coeff_dict["c_d"]["constrain"]
            ),
        )

    def test_add(self):
        rand_val_2 = np.random.rand()
        rand_val_1 = self.c_test.value
        c_2 = coefficients.Coefficient(
            "c_a", np.random.rand(), np.random.rand(), value=rand_val_2, constrain=True
        )
        self.c_test += c_2
        assert self.c_test.value == rand_val_1 + rand_val_2

    def test_eq(self):
        c_same = coefficients.Coefficient("c_a", np.random.rand(), np.random.rand())
        assert self.c_test == c_same

    def test_lt(self):
        c_same = coefficients.Coefficient("c_a", np.random.rand(), np.random.rand())
        assert not self.c_test < c_same
        c_diff = coefficients.Coefficient("c_c", np.random.rand(), np.random.rand())
        assert self.c_test < c_diff

    def test_repr(self):
        assert repr(self.c_test) == self.name


class TestCoefficientManager:
    c_list = coefficients.CoefficientManager.from_dict(coeff_dict)

    def test_init(self):
        assert self.c_list[0].name == "c_a"
        np.testing.assert_equal(self.c_list.minimum, [-2, -1, -2, -3, -4, -5, -5])

    def test_free_parameters(self):
        c_list_free = coefficients.CoefficientManager.from_dict(coeff_dict_free)
        np.testing.assert_equal(self.c_list.free_parameters.index, c_list_free.name)

    def test_set_free_parameters(self):
        temp = pd.Series(np.random.rand(2))
        self.c_list.set_free_parameters(temp)
        np.testing.assert_equal(self.c_list.free_parameters.value.values, temp.values)

    def test_set_constrains(self):
        c_a = self.c_list["c_a"].value
        c_b = self.c_list["c_b"].value
        c_c = self.c_list["c_c"].value
        c_d = self.c_list["c_d"].value
        c_e = self.c_list["c_e"].value
        c_f = self.c_list["c_f"].value
        c_g = self.c_list["c_g"].value

        np.testing.assert_equal(c_c, 1.0)
        np.testing.assert_equal(c_d, 0.0)
        np.testing.assert_equal(c_e, 0.0)
        np.testing.assert_equal(c_f, 0.0)
        np.testing.assert_equal(c_g, 0.0)
        self.c_list.set_constraints()
        c_d = self.c_list["c_d"].value
        c_e = self.c_list["c_e"].value
        c_f = self.c_list["c_f"].value
        c_g = self.c_list["c_g"].value
        bound_d = coeff_dict["c_d"]["constrain"]["c_b"] * c_b
        np.testing.assert_equal(c_d, bound_d)
        bound_e = (
            coeff_dict["c_e"]["constrain"][0]["c_b"] * c_b
            + coeff_dict["c_e"]["constrain"][1]["c_a"] * c_a
        )
        np.testing.assert_equal(c_e, bound_e)
        bound_f = (
            coeff_dict["c_f"]["constrain"][0]["c_a"]
            * c_a
            * coeff_dict["c_f"]["constrain"][0]["c_b"][0]
            * np.power(c_b, coeff_dict["c_f"]["constrain"][0]["c_b"][1])
            * coeff_dict["c_f"]["constrain"][0]["c_c"][0]
            * np.power(c_c, coeff_dict["c_f"]["constrain"][0]["c_c"][1])
            + coeff_dict["c_f"]["constrain"][1]["c_a"] * c_a
        )
        np.testing.assert_equal(c_f, bound_f)
        bound_g = (
            coeff_dict["c_g"]["constrain"][0]["c_a"][0]
            * np.power(c_a, coeff_dict["c_g"]["constrain"][0]["c_a"][1])
            * coeff_dict["c_g"]["constrain"][0]["c_b"][0]
            * np.power(c_b, coeff_dict["c_g"]["constrain"][0]["c_b"][1])
        )
        np.testing.assert_equal(c_g, bound_g)
