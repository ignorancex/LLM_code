"""
This module contains useful constants
"""

GRAVITY_CONSTANT = 6.67430e-11
""" The GRAVITY CONSTANT G [N * m^2 * kg^-2]"""

GRAVITY_CONSTANT_INVERSE = 1.0 / GRAVITY_CONSTANT
""" The inverse of the GRAVITY CONSTANT G """

SOLAR_CONSTANT = 1361.0
"""The solar constant in [W/m^2]"""

SPEED_OF_LIGHT = 299792458.0
"""The speed of light in [m/s}"""

BODY_MASS = {
    "bennu": 7.329e10,
    "churyumov-gerasimenko": 9.982e12,
    "eros": 6.687e15,
    "itokawa": 3.51e10,
    "torus": 9.982e12,
    "hollow": 9.982e12,
}
"""Contains the mass in [kg] for each body"""

BODY_SEMI_MAJOR_AXIS = {
    "bennu": 1.126,
    "churyumov-gerasimenko": 3.4628,
    "eros": 1.458,
    "itokawa": 1.324
}
"""Contains the bodies semi-major axis in [AU]"""

BODY_METRIC = {
    "bennu": 352.1486930549145,
    "churyumov-gerasimenko": 3126.6064453124995,
    "eros": 20413.864850997925,
    "itokawa": 350.438691675663,
    "torus": 3126.6064453124995,
    "hollow": 3126.6064453124995
}
"""Contains the conversion from unitless normalized body to metric [m] coordinates"""

UNITLESS_TO_METER = {
    "bennu": BODY_METRIC["bennu"],
    "bennu_nu": BODY_METRIC["bennu"],
    "churyumov-gerasimenko": BODY_METRIC["churyumov-gerasimenko"],
    "eros": BODY_METRIC["eros"],
    "itokawa": BODY_METRIC["itokawa"],
    "itokawa_nu": BODY_METRIC["itokawa"],
    "torus": BODY_METRIC["torus"],
    "hollow": BODY_METRIC["hollow"],
    "hollow_nu": BODY_METRIC["hollow"],
    "hollow2": BODY_METRIC["hollow"],
    "hollow2_nu": BODY_METRIC["hollow"]
}
"""Contains the conversion from unitless to meter [m]"""

UNITLESS_TO_ACCELERATION = {
    "bennu":
        BODY_MASS["bennu"] * GRAVITY_CONSTANT / BODY_METRIC["bennu"] ** 2,
    "bennu_nu":
        BODY_MASS["bennu"] * GRAVITY_CONSTANT / BODY_METRIC["bennu"] ** 2,
    "churyumov-gerasimenko":
        BODY_MASS["churyumov-gerasimenko"] * GRAVITY_CONSTANT / BODY_METRIC["churyumov-gerasimenko"] ** 2,
    "eros":
        BODY_MASS["eros"] * GRAVITY_CONSTANT / BODY_METRIC["eros"] ** 2,
    "itokawa":
        BODY_MASS["itokawa"] * GRAVITY_CONSTANT / BODY_METRIC["itokawa"] ** 2,
    "itokawa_nu":
        BODY_MASS["itokawa"] * GRAVITY_CONSTANT / BODY_METRIC["itokawa"] ** 2,
    "torus":
        BODY_MASS["torus"] * GRAVITY_CONSTANT / BODY_METRIC["torus"] ** 2,
    "hollow":
        BODY_MASS["hollow"] * GRAVITY_CONSTANT / BODY_METRIC["hollow"] ** 2,
    "hollow_nu":
        BODY_MASS["hollow"] * GRAVITY_CONSTANT / BODY_METRIC["hollow"] ** 2,
    "hollow2":
        BODY_MASS["hollow"] * GRAVITY_CONSTANT / BODY_METRIC["hollow"] ** 2,
    "hollow2_nu":
        BODY_MASS["hollow"] * GRAVITY_CONSTANT / BODY_METRIC["hollow"] ** 2
}
"""Contains the conversion from unitless to acceleration [m/s^2]"""
