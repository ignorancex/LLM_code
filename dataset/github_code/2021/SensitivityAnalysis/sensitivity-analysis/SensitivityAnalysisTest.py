import unittest
import numpy as np
import math
import tellurium as te
import SensitivityAnalysis as sa
            
class TestRichardsonExtrapolation(unittest.TestCase):
    """
    Unit test framework for richardson_extrapolation_EE function in Sensitivity Analysis
    """
    def test_sine(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given a sine function is close
        to a cosine function evaluated from a range of 0 to 4 pi radians
        """
        r = te.loada (" variable: $s -> $p; sin (parameter); parameter = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from -4 pi to 4 pi
        thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
        for i in thetas:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='parameter', variable='variable', parameter_init=i, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, math.cos(i))

    def test_cubic_polynomial(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given a cubic polynomial (s^3 + 2*s^2 - 3*s) is close
        to the derivative of the function (3*s^2 + 4*s) evaluated from -6 to 6.
        """
        # Loading the model
        r = te.loada("variable: $s -> $p;s^3 + 2*s^2 -3*s; s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from 0 to 4 pi
        xrange = np.arange(-6, 6, 0.01)
        for x in xrange:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, 3*x**2 + 4*x - 3)

    def test_sinc(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given a sinc function is close
        to a cosine function evaluated from a range of 0 to 4 pi radians
        """
        # Loading the model
        r = te.loada("variable: $s -> $p; sin(parameter)/parameter; parameter = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from -4 pi to 4 pi
        thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
        for i in thetas:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='parameter', variable='variable', parameter_init=i, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, math.cos(i)/i - math.sin(i)/i**2)

    def test_exponential(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given an exponential f(x) = exp(x^2) is close
        to the derivative of the function f(x) = 2*x*exp(x^2) evaluated from -3 to 3.
        """
        # Loading the model
        r = te.loada("variable: $s -> $p; exp(s^2); s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from -3 to 3
        xrange = np.arange(-3, 3, 0.01)
        for x in xrange:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, 2*x*math.exp(x**2))
        
    def test_square_root(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given a square root function f(x) = sqrt(x) is close
        to the derivative of the function f'(x) = 1/sqrt(x) evaluated from 10^-10 to 1000.
        """
        # Loading the square root model
        r = te.loada("variable: $s -> $p; sqrt(s); s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from 10^-10 to 1000
        xrange = np.linspace(0.005, 1, 10)
        for x in xrange:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, 1/(2*math.sqrt(x)))

    def test_inverse(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given an inverse function f(x) = 1/x is close
        to the derivative of the function, f'(x) = -1/x^2 evaluated from 0.005 to 1.
        """
        # Loading the inverse function model
        r = te.loada("variable: $s -> $p; 1/s; s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from 10^-10 to 1000
        xrange = np.linspace(0.005, 1, 10)
        for x in xrange:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.001)
            self.assertAlmostEqual(deriv, -1/(x**2))

    def test_log(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given an inverse function f(x) = ln(x + sqrt(1 + x^2)) is close
        to the derivative of the function, f'(x) = -1/x^2 evaluated from 0.005 to 1.
        """
        # Loading the log function model
        r = te.loada("variable: $s -> $p; log(s + sqrt(1 + s^2)); s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from 10^-10 to 1000
        xrange = np.linspace(-2, 5, 10)
        for x in xrange:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, 1/math.sqrt(x**2 + 1))


    def test_sine_x_squared(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given a function f(x) = sin(x^2) is close
        to the derivative of the function, f'(x) = 2x*cos(x^2) evaluated from -4 pi to 4 pi.
        """
        # Loading the log function model
        r = te.loada("variable: $s -> $p; sin(s^2); s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from -4 pi to 4 pi
        thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
        for i in thetas:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=i, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, 2*i*math.cos(i**2))
                
    def test_x_squared_cosine(self):
        """
        Tests if the estimated derivative of richardson_extrapolation_EE function given an inverse function f(x) = x^2*cos(2*x) is close
        to the derivative of the function, f'(x) = -2x(xsin(2x) - cos(2x)) evaluated from -4 pi to 4 pi.
        """
        # Loading the log function model
        r = te.loada("variable: $s -> $p; s^2*cos(2*s); s = 1")
        r.conservedMoietyAnalysis = True
        
        # Testing a range from -4 pi to 4 pi
        thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
        for i in thetas:
            deriv, err = sa.richardson_extrapolation_EE(r=r, parameter='s', variable='variable', parameter_init=i, initial_step_size=0.01)
            self.assertAlmostEqual(deriv, -2*i*(i*math.sin(2*i) - math.cos(2*i)))

class TestGetuCC(unittest.TestCase):
    """
    This unittest test case tests the getuCC function for
        - bad inputs and parameters
    """
    def setUp(self):
        """
        The setUp function is ran before every test case
        """
        model = """
        v1: A -> AP; Vm1*A*K/(Km1 + A)
        v2: AP -> A; Vm2*AP/(Km2 + AP)

        A = 10; AP = 0
        Vm1 = 15; Vm2 = 15
        Km1 = 0.1; Km2 = 0.1

        K = 1"""

        # Loading model
        self.r = te.loada(model)
        self.r.conservedMoietyAnalysis = True
    
    def test_param_type(self):
        # r type
        with self.assertRaises(TypeError):
            sa.getuCC('model', 'K', 'AP')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 4, 'AP')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 0.03)
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', initial_step_size='point one')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', parameter_init="start")
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', reduction_factor='high')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', error_init=3)
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', error_init='big error')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', ntab='five')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', ntab=4.3)
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', safe='safe')
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', safe=0)
        with self.assertRaises(TypeError):
            sa.getuCC(self.r, 'K', 'AP', method=1)

    def test_runtime(self):
        with self.assertRaises(RuntimeError):
            sa.getuCC(self.r, 'Q', 'AP')
        with self.assertRaises(RuntimeError):
            sa.getuCC(self.r, 'K', 'AD')
    
    def test_param_value(self):
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', initial_step_size=-3.4)
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', reduction_factor=0.3)
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', ntab=-2)
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', safe=-3.0)
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', safe=-1.2)
        with self.assertRaises(ValueError):
            sa.getuCC(self.r, 'K', 'AP', method='Richardson')

class TestGetCC(unittest.TestCase):
    """
    This unittest test case tests the getCC function for
        - bad inputs and parameters
    """
    def setUp(self):
        """
        The setUp function is ran before every test case
        """
        model = """
        v1: A -> AP; Vm1*A*K/(Km1 + A)
        v2: AP -> A; Vm2*AP/(Km2 + AP)

        A = 10; AP = 0
        Vm1 = 15; Vm2 = 15
        Km1 = 0.1; Km2 = 0.1

        K = 1"""

        # Loading model
        self.r = te.loada(model)
        self.r.conservedMoietyAnalysis = True
    
    def test_param_type(self):
        # r type
        with self.assertRaises(TypeError):
            sa.getCC('model', 'K', 'AP')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 4, 'AP')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 0.03)
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', initial_step_size='point one')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', parameter_init="start")
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', reduction_factor='high')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', error_init=3)
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', error_init='big error')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', ntab='five')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', ntab=4.3)
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', safe='safe')
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', safe=0)
        with self.assertRaises(TypeError):
            sa.getCC(self.r, 'K', 'AP', method=1)

    def test_runtime(self):
        with self.assertRaises(RuntimeError):
            sa.getCC(self.r, 'Q', 'AP')
        with self.assertRaises(RuntimeError):
            sa.getCC(self.r, 'K', 'AD')
    
    def test_param_value(self):
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', initial_step_size=-3.4)
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', reduction_factor=0.3)
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', ntab=-2)
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', safe=-3.0)
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', safe=-1.2)
        with self.assertRaises(ValueError):
            sa.getCC(self.r, 'K', 'AP', method='Richardson')


class TestGetuEE(unittest.TestCase):
    """
    This unittest test case tests the getuEE function for
        - bad inputs and parameters
    """
    def setUp(self):
        """
        The setUp function is ran before every test case
        """
        model = """
        v1: A -> AP; Vm1*A*K/(Km1 + A)
        v2: AP -> A; Vm2*AP/(Km2 + AP)

        A = 10; AP = 0
        Vm1 = 15; Vm2 = 15
        Km1 = 0.1; Km2 = 0.1

        K = 1"""

        # Loading model
        self.r = te.loada(model)
        self.r.conservedMoietyAnalysis = True
    
    def test_param_type(self):
        # r type
        with self.assertRaises(TypeError):
            sa.getuEE('model', 'AP', 'v2')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', initial_step_size='point one')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', parameter_init="start")
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', reduction_factor='high')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', error_init=3)
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', error_init='big error')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', ntab='five')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', ntab=4.3)
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', safe='safe')
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', safe=0)
        with self.assertRaises(TypeError):
            sa.getuEE(self.r, 'AP', 'v2', method=1)

    def test_runtime(self):
        with self.assertRaises(RuntimeError):
            sa.getuEE(self.r, 'Q', 'AP')
        with self.assertRaises(RuntimeError):
            sa.getuEE(self.r, 'K', 'AD')
        with self.assertRaises(RuntimeError):
            sa.getuEE(self.r, "AP", "K")
    
    def test_param_value(self):
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', initial_step_size=-3.4)
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', reduction_factor=0.3)
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', ntab=-2)
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', safe=-3.0)
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', safe=-1.2)
        with self.assertRaises(ValueError):
            sa.getuEE(self.r, 'AP', 'v2', method='Richardson')

class TestGetEE(unittest.TestCase):
    """
    This unittest test case tests the getEE function for
        - bad inputs and parameters
    """
    def setUp(self):
        """
        The setUp function is ran before every test case
        """
        model = """
        v1: A -> AP; Vm1*A*K/(Km1 + A)
        v2: AP -> A; Vm2*AP/(Km2 + AP)

        A = 10; AP = 0
        Vm1 = 15; Vm2 = 15
        Km1 = 0.1; Km2 = 0.1

        K = 1"""

        # Loading model
        self.r = te.loada(model)
        self.r.conservedMoietyAnalysis = True
    
    def test_param_type(self):
        # r type
        with self.assertRaises(TypeError):
            sa.getEE('model', 'AP', 'v2')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', initial_step_size='point one')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', parameter_init="start")
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', reduction_factor='high')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', error_init=3)
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', error_init='big error')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', ntab='five')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', ntab=4.3)
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', safe='safe')
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', safe=0)
        with self.assertRaises(TypeError):
            sa.getEE(self.r, 'AP', 'v2', method=1)

    def test_runtime(self):
        with self.assertRaises(RuntimeError):
            sa.getEE(self.r, 'Q', 'AP')
        with self.assertRaises(RuntimeError):
            sa.getEE(self.r, 'K', 'AD')
        with self.assertRaises(RuntimeError):
            sa.getEE(self.r, "AP", "K")
    
    def test_param_value(self):
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', initial_step_size=-3.4)
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', reduction_factor=0.3)
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', ntab=-2)
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', safe=-3.0)
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', safe=-1.2)
        with self.assertRaises(ValueError):
            sa.getEE(self.r, 'AP', 'v2', method='Richardson')

if __name__ == '__main__':
    unittest.main()