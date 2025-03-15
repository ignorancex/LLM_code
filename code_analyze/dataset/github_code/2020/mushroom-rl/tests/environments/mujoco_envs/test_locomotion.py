import numpy as np

from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"


def test_ant():
    np.random.seed(1)
    mdp = Ant()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.array(
        [
            0.51195905,
            0.96649327,
            0.01502863,
            -0.10029343,
            -0.23580953,
            0.52586297,
            1.22402323,
            0.52585736,
            -0.52127442,
            0.52586083,
            -0.52131149,
            0.5258635,
            1.22399241,
            -0.06224879,
            -0.05603109,
            -0.00547634,
            0.1469538,
            -0.07348908,
            -0.03388605,
            0.01493919,
            0.01516731,
            0.01486824,
            0.01521562,
            0.01500421,
            0.01527666,
            0.01501851,
            0.01518631,
        ]
    )
    # mdp.render()

    assert np.allclose(ns, ns_test)


def test_half_cheetah():
    np.random.seed(1)
    mdp = HalfCheetah()
    mdp.reset()
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])

    ns_test = np.array(
        [
            -0.3523922,
            0.4273692,
            0.04377365,
            -0.03046335,
            0.0271416,
            0.66837943,
            0.55671229,
            0.5101724,
            0.90368074,
            0.5282341,
            4.20814634,
            -3.79190889,
            -4.79708061,
            -3.59628463,
            -0.8683651,
            -1.68162896,
            -0.46546397,
        ]
    )
    # mdp.render()

    assert np.allclose(ns, ns_test)


def test_hopper():
    np.random.seed(1)
    mdp = Hopper()
    mdp.reset()
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])
    ns_test = np.array(
        [
            1.22259864e00,
            1.33571984e-02,
            2.73945687e-03,
            2.78558910e-03,
            2.22120792e-01,
            4.37020357e-01,
            -1.26902951e-01,
            6.36483604e-01,
            -1.24339512e-02,
            -4.45634448e-02,
            5.19363748e00,
        ]
    )
    # mdp.render()

    assert np.allclose(ns, ns_test)


def test_walker_2d():
    np.random.seed(1)
    mdp = Walker2D()
    mdp.reset()
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])

    ns_test = np.array(
        [
            1.20430686,
            -0.00534619,
            0.01462398,
            0.01368397,
            1.04371038,
            0.01421412,
            0.01298897,
            1.04628971,
            -0.2684887,
            -0.0424852,
            -0.77339133,
            -0.21414624,
            -0.15925905,
            -4.93041582,
            -0.24038571,
            -0.16340492,
            -5.00609819,
        ]
    )
    # mdp.render()

    assert np.allclose(ns, ns_test)
