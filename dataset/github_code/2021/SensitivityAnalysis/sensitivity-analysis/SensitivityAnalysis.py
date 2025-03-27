import tellurium as te
import numpy as np
import math


def getuCC(r, parameter, variable, initial_step_size=0.001, parameter_init="current", reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0, method='RichardsonExtrapolation'):
    """
    Compute the unscaled control coefficients of a roadrunner model given a parameter to perturb and a variable to observe.
    
    Returns the derivative and estimated error values via the default method Richardson Extrapolation or alternatively
    Newton's difference quotient ("OnePoint") or symmetric difference quotient ("TwoPoint")
    
    
    Parameters
    -----------
    r : ExtendedRoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration
    initial_step_size : float
        Initial stepsize, default value is 0.001. Decrease for increased accuracy.
    parameter_init : float
        Initial parameter value to compute control coefficients for, set to "current" as default
    reduction_factor : float
        Stepsize is decreased by reduction_factor each iteration. Default is 1.4
    error_init : float
        the initial starting error of the richardson algorithm, default is 1.0e30
    ntab : int
        Sets the maximum size of Neville tableau, if error is still large increase ntab, default is 10
    safe : float
        Return when error becomes 'safe' worse than the lowest error so far. Default safe is 2.0.
    method : str
        Possible differentiation methods are "RichardsonExtrapolation", "OnePoint" (Newton's difference quotient), 
        and "TwoPoint" (symmetric difference quotient). Note that "RichardsonExtrapolation" utilizes the method of symmetric


    
    Returns
    --------
    deriv : float
        the value of the unscaled control coefficient
    error : float
        Returns an estimation of the error
    """  
    # r exception
    if not isinstance(r, te.roadrunner.extended_roadrunner.ExtendedRoadRunner):
        raise TypeError("r is not ExtendedRoadRunner object")
    # Initial_step_size exception
    if not isinstance(initial_step_size, float):
        raise TypeError("initial_step_size is not type float")
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be nonzero and positive in richardson_extrapolation")
    # parameter_init exception
    if isinstance(parameter_init, str) and (parameter_init != "current"):
        raise TypeError('parameter_init must be a float or set to "current"')
    # reduction_factor exception
    if not isinstance(reduction_factor, float):
        raise TypeError("reduction_factor must be type float, default is 1.4")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be greater than 1")
    # error_init exception
    if not isinstance(error_init, float):
        raise TypeError("error_init must be type float, default is 1.0e30")
    # ntab exception
    if not isinstance(ntab, int):
        raise TypeError("ntab must be type int, default is 10")
    if ntab <= 0:
        raise ValueError("ntab must be an integer greater than 0, default is 10")
    # safe exception
    if not isinstance(safe, float):
        raise TypeError("safe must be a type float, default is 2.0")
    if safe <= 0:
        raise ValueError("safe must be type float greater than 0")
    # method exceptions
    if not isinstance(method, str):
        raise TypeError("method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint")
    possible_methods = ['RichardsonExtrapolation', 'OnePoint', 'TwoPoint', 'FivePoint']
    if method not in possible_methods:
        raise ValueError('method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint')
        
    
    # if parameter_init is set to current, the Control Coefficient is estimated based on
    # the current value of x
    if parameter_init == "current":
        parameter_init = r[parameter] # set parameter_init to current value of x
    else:
        r[parameter] = parameter_init
    
    # Method types
    if method == 'RichardsonExtrapolation':
        deriv, error = richardson_extrapolation(r, parameter, variable, initial_step_size, parameter_init, reduction_factor, error_init, ntab, safe)
    elif method == 'OnePoint':
        deriv = one_point_diff(r, parameter, variable, initial_step_size, parameter_init) ## Need to write this function
        error = 'N/A'
    elif method == 'TwoPoint':
        deriv = two_point_diff(r, parameter, variable, initial_step_size, parameter_init, "Derivative")  ## Need to write this function as well
        error = 'N/A'
    elif method == 'FivePoint':
        deriv = five_point_diff(r, parameter, variable, initial_step_size, parameter_init)
        error = 'N/A'
    return deriv, error

def getCC(r, parameter, variable, initial_step_size=0.001, parameter_init="current", reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0, method='RichardsonExtrapolation'):
    """
    Compute the scaled control coefficients of a roadrunner model given a parameter to perturb and a variable to observe.
    
    Returns the derivative and estimated error values via the default method Richardson Extrapolation or alternatively
    Newton's difference quotient ("OnePoint") or symmetric difference quotient ("TwoPoint")
    
    
    Parameters
    -----------
    r : ExtendedRoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration
    initial_step_size : float
        Initial stepsize, default value is 0.001. Decrease for increased accuracy.
    parameter_init : float
        Initial parameter value to compute control coefficients for, set to "current" as default
    reduction_factor : float
        Stepsize is decreased by reduction_factor each iteration. Default is 1.4
    error_init : float
        the initial starting error of the richardson algorithm, default is 1.0e30
    ntab : int
        Sets the maximum size of Neville tableau, if error is still large increase ntab, default is 10
    safe : float
        Return when error becomes 'safe' worse than the lowest error so far. Default safe is 2.0.
    method : str
        Possible differentiation methods are "RichardsonExtrapolation", "OnePoint" (Newton's difference quotient), 
        and "TwoPoint" (symmetric difference quotient). Note that "RichardsonExtrapolation" utilizes the method of symmetric


    
    Returns
    --------
    deriv : float
        the value of the scaled control coefficient
    error : float
        Returns an estimation of the error
    """
    # r exception
    import tellurium
    if not isinstance(r, tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner):
        raise TypeError("r is not ExtendedRoadRunner object")
    # Initial_step_size exception
    if not isinstance(initial_step_size, float):
        raise TypeError("initial_step_size is not type float")
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be nonzero and positive in richardson_extrapolation")
    # parameter_init exception
    if isinstance(parameter_init, str) and (parameter_init != "current"):
        raise TypeError('parameter_init must be a float or set to "current"')
    # reduction_factor exception
    if not isinstance(reduction_factor, float):
        raise TypeError("reduction_factor must be type float, default is 1.4")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be greater than 1")
    # error_init exception
    if not isinstance(error_init, float):
        raise TypeError("error_init must be type float, default is 1.0e30")
    # ntab exception
    if not isinstance(ntab, int):
        raise TypeError("ntab must be type int, default is 10")
    if ntab <= 0:
        raise ValueError("ntab must be an integer greater than 0, default is 10")
    # safe exception
    if not isinstance(safe, float):
        raise TypeError("safe must be a type float, default is 2.0")
    if safe <= 0:
        raise ValueError("safe must be type float greater than 0")
    # method exceptions
    if not isinstance(method, str):
        raise TypeError("method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint")
    possible_methods = ['RichardsonExtrapolation', 'OnePoint', 'TwoPoint', 'FivePoint']
    if method not in possible_methods:
        raise ValueError('method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint')
        
    
    # if parameter_init is set to current, the Control Coefficient is estimated based on
    # the current value of parameter
    if parameter_init == "current":
        parameter_init = r[parameter] # set parameter_init to current value of x
    else:
        r[parameter] = parameter_init
    # Method types
    if method == 'RichardsonExtrapolation':
        deriv, error = richardson_extrapolation(r, parameter, variable, initial_step_size, parameter_init, reduction_factor, error_init, ntab, safe)
    elif method == 'OnePoint':
        deriv = one_point_diff(r, parameter, variable, initial_step_size, parameter_init) ## Need to write this function
        error = 'N/A'
    elif method == 'TwoPoint':
        deriv = two_point_diff(r, parameter, variable, initial_step_size, parameter_init, "Derivative")  ## Need to write this function as well
        error = 'N/A'
    elif method == 'FivePoint':
        deriv = five_point_diff(r, parameter, variable, initial_step_size, parameter_init)
        error = 'N/A'
        
    # Scaling the derivative
    r[parameter] = parameter_init
    r.steadyState()
    variable_init = r[variable]
    deriv = deriv*parameter_init/variable_init
    return deriv, error

def richardson_extrapolation(r, parameter, variable, initial_step_size, parameter_init, reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0):
    """
    Returns the derivative of a function, f, at a point parameter by Ridders' method of polynomial
    extrapolation. The value h is as an estimated initial step size; it should be an increment
    in parameter over which func changes substantially. An estimate of the error is returned as err.
    """  
    # Initializing
    hh = initial_step_size
    a = np.zeros((ntab,ntab)) # array for storing values
    
    # retrieve current species values
    P1, P2 = two_point_diff(r, parameter, variable, hh, parameter_init, "Points")

    # first value in matrix
    a[0,0] = (P2 - P1)/(2*hh)
    err = error_init
    con2 = reduction_factor**2
    # Successful columns in the Neville tableau will go to smaller stepsizes
    # and high orders of extrapolation
    for i in range(1, ntab):
        hh = hh/reduction_factor
        P1, P2 = two_point_diff(r, parameter, variable, hh, parameter_init, "Points")
        a[0, i] = (P2 - P1)/(2*hh)
        fac = con2 # can probably remove and just use con2 in place of fac
        # Compute extrapolations of various orders, requiring no new function evaluations
        for j in range(1, i + 1):
            a[j, i] = (a[j-1, i]*fac - a[j-1, i-1])/(fac-1.0)
            fac=con2*fac
            errt = np.maximum(np.abs(a[j, i] - a[j-1, i]),np.abs(a[j, i]-a[j-1,i-1])) # current error estimation
            # error strategy is to compare each new extrapolation to one order lower, both
            # at the present stepsize and the previous one.
            if errt <= err:
                err = errt
                ans = a[j, i]
        # If higher order is worse by a significant factor 'safe' then quit early
        if np.abs(a[i,i] - a[i-1,i-1]) >= safe*err:
            break
    return ans, err

def one_point_diff(r, parameter, variable, step, parameter_init):
    """
    Returns the derivative of a function based on the one point
    numerical differentiation method. Also known as Newton's Difference Quotient
    
    Parameters
    -----------
    r : An Extended RoadRunner object
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    
    Returns
    --------
    deriv : double
        The one point derivative of the variable with respect to the parameter
    """
    # Initial value of variable
    var_init = r[variable] 
    # Current state of variable
    P1 = var_init

    # Perturbing the system forward
    r[parameter] = parameter_init + step
    r.steadyState()
    
    # Perturbed state of variable
    P2 = r[variable]

    # Compute the one point differentiation
    deriv = (P2 - P1)/step

    # Restore parameter value
    r[parameter] = parameter_init
    r.steadyState()

    return deriv


def two_point_diff(r, parameter, variable, step, parameter_init, output):
    """
    Returns the derivative of a function based on the two point
    numerical differentiation method also known as the symmetric difference quotient. If output parameter
    is assigned to 'Points', then P1 and P2 are returned.
    
    Parameters
    -----------
    r : An Extended RoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    output: str
        specifies the desired return output, "Points" or "Derivative"
    
    Returns
    --------
    P1 : double
        The first perturbed point P1 (f(x-h))
    P2 : double
        The second perturbed point P2 (f(x+h)).
    deriv : double
        The one point derivative of the variable with respect to the parameter
    """
    # Initial value to determine if r.steadyState() needs to be run
    # var_init = r[variable]
    # Perturbing the system backward
    r[parameter] = parameter_init - step
    r.steadyState()
    
    # Perturbed state of variable f(x-h)
    P1 = r[variable]
    
    # Perturbing the system forward
    r[parameter] = parameter_init + step
    r.steadyState()
    
    # Perturbed state of variable f(x+h)
    P2 = r[variable]

    # Restore the value of parameter
    r[parameter] = parameter_init
    r.steadyState()
    
    if output == "Points":
        return P1, P2
    elif output == "Derivative":
        # Compute the one point differentiation
        deriv = (P2 - P1)/(2*step)
        return deriv

def five_point_diff(r, parameter, variable, step, parameter_init):
    """
    Returns the derivative of a function based on the five point
    numerical differentiation method.
    
    Parameters
    -----------
    r : An Extended RoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    
    Returns
    --------
    deriv : double
        The one point derivative of the variable with respect to the parameter
    """
    # Get points P1 (f(x-2h)), P2 (f(x-h)), P3 (f(x)), P4 (f(x+h)), P5 (f(x+2h))

    # Perturbing the system a steps backward f(x-h)
    r[parameter] = parameter_init - 2*step
    
    # Computing points: 
    # f1: f(x - 2h), f2: f(x - h), f3: f(x), 
    # f4: f(x + h), and f5: f(x + 2h)
    var_vals = {} # dictionary storing the variable values
    labels = ['f1', 'f2', 'f3', 'f4', 'f5']
    # Computing perturbed variable values and storing them in var_vals
    l = 0
    for i in np.arange(-2, 3): # points should be [-2, -1, 0, 1, 2]
        r[parameter] = parameter_init + i*step # perturbing the system forward/backward
        r.steadyState()
        var_vals[labels[l]] = r[variable]
        l += 1
    
    # Restore the value of parameter
    r[parameter] = parameter_init
    r.steadyState()
    deriv = (-var_vals['f5'] + 8*var_vals['f4'] - 8*var_vals['f2'] + var_vals['f1'])/(12*step)
    return deriv

def getuEE(r, parameter, variable, initial_step_size=0.001, parameter_init="current", reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0, method='RichardsonExtrapolation'):
    """
    Compute the single unscaled elasticity coefficient with respect to a global parameter; reaction speed with respect to a global parameter.
    
    Returns the derivative and estimated error values via the default method Richardson Extrapolation or alternatively
    Newton's difference quotient ("OnePoint") or symmetric difference quotient ("TwoPoint")

    Parameters
    -----------
    r : ExtendedRoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration
    initial_step_size : float
        Initial stepsize, default value is 0.001. Decrease for increased accuracy.
    parameter_init : float
        Initial parameter value to compute elasticity coefficient for, set to "current" as default
    reduction_factor : float
        Stepsize is decreased by reduction_factor each iteration. Default is 1.4
    error_init : float
        the initial starting error of the richardson algorithm, default is 1.0e30
    ntab : int
        Sets the maximum size of Neville tableau, if error is still large increase ntab, default is 10
    safe : float
        Return when error becomes 'safe' worse than the lowest error so far. Default safe is 2.0.
    method : str
        Possible differentiation methods are "RichardsonExtrapolation", "OnePoint" (Newton's difference quotient), 
        and "TwoPoint" (symmetric difference quotient). Note that "RichardsonExtrapolation" utilizes the method of symmetric


    
    Returns
    --------
    deriv : float
        the value of the unscaled elasticity coefficient
    error : float
        Returns an estimation of the error
    """
    # r exception
    import tellurium
    if not isinstance(r, tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner):
        raise TypeError("r is not ExtendedRoadRunner object")
    # Variable value
    if variable not in r.getReactionIds():
        raise RuntimeError(f"Unable to locate reactionId [{variable}]")
    # Initial_step_size exception
    if not isinstance(initial_step_size, float):
        raise TypeError("initial_step_size is not type float")
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be nonzero and positive in richardson_extrapolation")
    # parameter_init exception
    if isinstance(parameter_init, str) and (parameter_init != "current"):
        raise TypeError('parameter_init must be a float or set to "current"')
    # reduction_factor exception
    if not isinstance(reduction_factor, float):
        raise TypeError("reduction_factor must be type float, default is 1.4")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be greater than 1")
    # error_init exception
    if not isinstance(error_init, float):
        raise TypeError("error_init must be type float, default is 1.0e30")
    # ntab exception
    if not isinstance(ntab, int):
        raise TypeError("ntab must be type int, default is 10")
    if ntab <= 0:
        raise ValueError("ntab must be an integer greater than 0, default is 10")
    # safe exception
    if not isinstance(safe, float):
        raise TypeError("safe must be a type float, default is 2.0")
    if safe <= 0:
        raise ValueError("safe must be type float greater than 0")
    # method exceptions
    if not isinstance(method, str):
        raise TypeError("method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint")
    possible_methods = ['RichardsonExtrapolation', 'OnePoint', 'TwoPoint', 'FivePoint']
    if method not in possible_methods:
        raise ValueError('method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint, FivePoint')
    
    # if parameter_init is set to current, the elasticity coefficient is estimated based on
    # the current value of x
    if parameter_init == "current":
        parameter_init = r[parameter] # set parameter_init to current value of x
    else:
        r[parameter] = parameter_init
    
    # Method types
    if method == 'RichardsonExtrapolation':
        deriv, error = richardson_extrapolation_EE(r, parameter, variable, initial_step_size, parameter_init, reduction_factor, error_init, ntab, safe)
    elif method == 'OnePoint':
        deriv = one_point_diff_EE(r, parameter, variable, initial_step_size, parameter_init) ## Need to write this function
        error = 'N/A'
    else:
        deriv = two_point_diff_EE(r, parameter, variable, initial_step_size, parameter_init, "Derivative") ## Need to write this function as well
        error = 'N/A'
    return deriv, error

def getEE(r, parameter, variable, initial_step_size=0.001, parameter_init="current", reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0, method='RichardsonExtrapolation'):
    """
    Compute the single scaled elasticity coefficient with respect to a global parameter; reaction speed with respect to a global parameter.
    
    Returns the derivative and estimated error values via the default method Richardson Extrapolation or alternatively
    Newton's difference quotient ("OnePoint") or symmetric difference quotient ("TwoPoint")
    
    
    
    Parameters
    -----------
    r : ExtendedRoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration
    initial_step_size : float
        Initial stepsize, default value is 0.001. Decrease for increased accuracy.
    parameter_init : float
        Initial parameter value to compute elasticity coefficient for, set to "current" as default
    reduction_factor : float
        Stepsize is decreased by reduction_factor each iteration. Default is 1.4
    error_init : float
        the initial starting error of the richardson algorithm, default is 1.0e30
    ntab : int
        Sets the maximum size of Neville tableau, if error is still large increase ntab, default is 10
    safe : float
        Return when error becomes 'safe' worse than the lowest error so far. Default safe is 2.0.
    method : str
        Possible differentiation methods are "RichardsonExtrapolation", "OnePoint" (Newton's difference quotient), 
        and "TwoPoint" (symmetric difference quotient). Note that "RichardsonExtrapolation" utilizes the method of symmetric


    
    Returns
    --------
    deriv : float
        the value of the scaled elasticity coefficient
    error : float
        Returns an estimation of the error
    """
    # r exception
    import tellurium
    if not isinstance(r, tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner):
        raise TypeError("r is not ExtendedRoadRunner object")
    # Variable value
    if variable not in r.getReactionIds():
        raise RuntimeError(f"Unable to locate reactionId [{variable}]")
    # Initial_step_size exception
    if not isinstance(initial_step_size, float):
        raise TypeError("initial_step_size is not type float")
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be nonzero and positive in richardson_extrapolation")
    # parameter_init exception
    if isinstance(parameter_init, str) and (parameter_init != "current"):
        raise TypeError('parameter_init must be a float or set to "current"')
    # reduction_factor exception
    if not isinstance(reduction_factor, float):
        raise TypeError("reduction_factor must be type float, default is 1.4")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be greater than 1")
    # error_init exception
    if not isinstance(error_init, float):
        raise TypeError("error_init must be type float, default is 1.0e30")
    # ntab exception
    if not isinstance(ntab, int):
        raise TypeError("ntab must be type int, default is 10")
    if ntab <= 0:
        raise ValueError("ntab must be an integer greater than 0, default is 10")
    # safe exception
    if not isinstance(safe, float):
        raise TypeError("safe must be a type float, default is 2.0")
    if safe <= 0:
        raise ValueError("safe must be type float greater than 0")
    # method exceptions
    if not isinstance(method, str):
        raise TypeError("method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint")
    possible_methods = ['RichardsonExtrapolation', 'OnePoint', 'TwoPoint', 'FivePoint']
    if method not in possible_methods:
        raise ValueError('method must be a string of one of the following methods: RichardsonExtrapolation, OnePoint, TwoPoint')
    
    # if parameter_init is set to current, the elasticity coefficient is estimated based on
    # the current value of x
    if parameter_init == "current":
        parameter_init = r[parameter] # set parameter_init to current value of x
    else:
        r[parameter] = parameter_init
    
    # Method types
    if method == 'RichardsonExtrapolation':
        deriv, error = richardson_extrapolation_EE(r, parameter, variable, initial_step_size, parameter_init, reduction_factor, error_init, ntab, safe)
    elif method == 'OnePoint':
        deriv = one_point_diff_EE(r, parameter, variable, initial_step_size, parameter_init) ## Need to write this function
        error = 'N/A'
    else:
        deriv = two_point_diff_EE(r, parameter, variable, initial_step_size, parameter_init, "Derivative") ## Need to write this function as well
        error = 'N/A'
    # Resetting system
    r[parameter] = parameter_init
    variable_init = r[variable]
    deriv = deriv*parameter_init/variable_init
    return deriv, error

def richardson_extrapolation_EE(r, parameter, variable, initial_step_size, parameter_init, reduction_factor=1.4, error_init=1.0e30, ntab=10, safe=2.0):
    """
    Returns the derivative of a function, f, at a point parameter by Ridders' method of polynomial
    extrapolation. The value h is as an estimated initial step size; it should be an increment
    in parameter over which func changes substantially. An estimate of the error is returned as err.
    """  
    # Initializing
    hh = initial_step_size
    a = np.zeros((ntab,ntab)) # array for storing values
    
    # retrieve current species values
    P1, P2 = two_point_diff_EE(r, parameter, variable, hh, parameter_init, "Points")

    # first value in matrix
    a[0,0] = (P2 - P1)/(2*hh)
    err = error_init
    con2 = reduction_factor**2
    # Successful columns in the Neville tableau will go to smaller stepsizes
    # and high orders of extrapolation
    for i in range(1, ntab):
        hh = hh/reduction_factor
        P1, P2 = two_point_diff_EE(r, parameter, variable, hh, parameter_init, "Points")
        a[0, i] = (P2 - P1)/(2*hh)
        fac = con2 # can probably remove and just use con2 in place of fac
        # Compute extrapolations of various orders, requiring no new function evaluations
        for j in range(1, i + 1):
            a[j, i] = (a[j-1, i]*fac - a[j-1, i-1])/(fac-1.0)
            fac=con2*fac
            errt = np.maximum(np.abs(a[j, i] - a[j-1, i]),np.abs(a[j, i]-a[j-1,i-1])) # current error estimation
            # error strategy is to compare each new extrapolation to one order lower, both
            # at the present stepsize and the previous one.
            if errt <= err:
                err = errt
                ans = a[j, i]
        # If higher order is worse by a significant factor 'safe' then quit early
        if np.abs(a[i,i] - a[i-1,i-1]) >= safe*err:
            break
    return ans, err

def one_point_diff_EE(r, parameter, variable, step, parameter_init):
    """
    Returns the derivative of a function based on the one point
    numerical differentiation method. Also known as Newton's Difference Quotient
    
    Parameters
    -----------
    r : An Extended RoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    
    Returns
    deriv : double
        the one point derivative of the variable with respect to the parameter
    """
    # Current state of variable
    P1 = r[variable]
    
    # Perturbing the system forward
    r[parameter] = parameter_init + step
    
    # Perturbed state of variable
    P2 = r[variable]
    
    # Compute the one point differentiation
    deriv = (P2 - P1)/step
    return deriv

def two_point_diff_EE(r, parameter, variable, step, parameter_init, output):
    """Returns the derivative of a function based on the two point
    numerical differentiation method also known as the symmetric difference quotient. If output parameter
    is assigned to 'Points', then P1 and P2 are returned.
    
    Parameters
    -----------
    r : An Extended RoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    output: str
        specifies the desired return output, "Points" or "Derivative"
    
    Returns
    --------
    deriv : double
        the two point derivative of the variable with respect to the parameter
    """
    # Perturbing the system backward
    r[parameter] = parameter_init - step
    
    # Perturbed state of variable f(x-h)
    P1 = r[variable]
    
    # Perturbing the system forward
    r[parameter] = parameter_init + step

    # Perturbed state of variable f(x+h)
    P2 = r[variable]
    
    # Resetting
    r[parameter] = parameter_init
    if output == "Points":
        return P1, P2
    elif output == "Derivative":
        # Compute the one point differentiation
        deriv = (P2 - P1)/(2*step)
        return deriv

def five_point_diff_EE(r, parameter, variable, step, parameter_init):
    """
    Returns the derivative of a function based on the five point
    numerical differentiation method.
    
    Parameters
    -----------
    r : An Extended RoadRunner object
        A roadrunner object representing a function or metabolic pathway
    parameter : str
        The id of the independent parameter, for example a kinetic constant or boundary species
    variable : str
        The id of a dependent variable of the coefficient, for example a reaction or species concentration.
    step : a float
        The step size to use to compute Newton's difference quotient.
    parameter_init : a float
        Initial variable value to compute control coefficients for, set to "current" as default
    
    Returns
    --------
    deriv : double
        The five point derivative of the variable with respect to the parameter
    """
    # Get points P1 (f(x-2h)), P2 (f(x-h)), P3 (f(x)), P4 (f(x+h)), P5 (f(x+2h))
    variable_init = r[variable]

    # Computing points: 
    # f1: f(x - 2h), f2: f(x - h), f3: f(x), 
    # f4: f(x + h), and f5: f(x + 2h)
    var_vals = {} # dictionary storing the variable values
    labels = ['f1', 'f2', 'f3', 'f4', 'f5']
    # Computing perturbed variable values and storing them in var_vals
    l = 0
    for i in np.arange(-2, 3): # points should be [-2, -1, 0, 1, 2]
        r[parameter] = parameter_init + i*step # perturbing the system forward/backward
        var_vals[labels[l]] = r[variable]
        l += 1
    # Evaluate the derivative
    deriv = (-var_vals['f5'] + 8*var_vals['f4'] - 8*var_vals['f2'] + var_vals['f1'])/(12*step)
    return deriv


def test_sine():
    """
    Tests if the estimated derivative of richardson_extrapolation function given a sine function is close
    to a cosine function evaluated from a range of 0 to 4 pi radians
    """
    r = te.loada (" variable: $s -> $p; sin (parameter); parameter = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from -4 pi to 4 pi
    thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
    for i in thetas:
        deriv, err = richardson_extrapolation(r=r, parameter='parameter', variable='variable', parameter_init=i, initial_step_size=0.01)
        if not np.isclose(deriv, math.cos(i)):
            raise Exception("test_sine function failed")

def test_cubic_polynomial():
    """
    Tests if the estimated derivative of richardson_extrapolation function given a cubic polynomial (s^3 + 2*s^2 - 3*s) is close
    to the derivative of the function (3*s^2 + 4*s) evaluated from -6 to 6.
    """
    # Loading the model
    r = te.loada("variable: $s -> $p;s^3 + 2*s^2 -3*s; s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from 0 to 4 pi
    xrange = np.arange(-6, 6, 0.01)
    for x in xrange:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
        if not np.isclose(deriv, 3*x**2 + 4*x - 3):
            raise Exception("test_cubic_polynomial function failed")

def test_sinc():
    """
    Tests if the estimated derivative of richardson_extrapolation function given a sinc function is close
    to a cosine function evaluated from a range of 0 to 4 pi radians
    """
    # Loading the model
    r = te.loada("variable: $s -> $p; sin(parameter)/parameter; parameter = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from -4 pi to 4 pi
    thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
    for i in thetas:
        deriv, err = richardson_extrapolation(r=r, parameter='parameter', variable='variable', parameter_init=i, initial_step_size=0.01)
        if not np.isclose(deriv, math.cos(i)/i - math.sin(i)/i**2):
            raise Exception("test_sinc function failed")

def test_exponential():
    """
    Tests if the estimated derivative of richardson_extrapolation function given an exponential f(x) = exp(x^2) is close
    to the derivative of the function f(x) = 2*x*exp(x^2) evaluated from -3 to 3.
    """
    # Loading the model
    r = te.loada("variable: $s -> $p; exp(s^2); s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from -3 to 3
    xrange = np.arange(-3, 3, 0.01)
    for x in xrange:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
        if not np.isclose(deriv, 2*x*math.exp(x**2)):
            raise Exception("test_exponential function failed")
    
def test_square_root():
    """
    Tests if the estimated derivative of richardson_extrapolation function given a square root function f(x) = sqrt(x) is close
    to the derivative of the function f'(x) = 1/sqrt(x) evaluated from 10^-10 to 1000.
    """
    # Loading the square root model
    r = te.loada("variable: $s -> $p; sqrt(s); s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from 10^-10 to 1000
    xrange = np.linspace(0.005, 1, 10)
    for x in xrange:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
        if not np.isclose(deriv, 1/(2*math.sqrt(x))):
            raise Exception("test_square_root function failed")

def test_inverse():
    """
    Tests if the estimated derivative of richardson_extrapolation function given an inverse function f(x) = 1/x is close
    to the derivative of the function, f'(x) = -1/x^2 evaluated from 0.005 to 1.
    """
    # Loading the inverse function model
    r = te.loada("variable: $s -> $p; 1/s; s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from 10^-10 to 1000
    xrange = np.linspace(0.005, 1, 10)
    for x in xrange:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.001)
        if not np.isclose(deriv, -1/(x**2)):
            raise Exception("test_inverse function failed")

def test_log():
    """
    Tests if the estimated derivative of richardson_extrapolation function given an inverse function f(x) = ln(x + sqrt(1 + x^2)) is close
    to the derivative of the function, f'(x) = -1/x^2 evaluated from 0.005 to 1.
    """
    # Loading the log function model
    r = te.loada("variable: $s -> $p; log(s + sqrt(1 + s^2)); s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from 10^-10 to 1000
    xrange = np.linspace(-2, 5, 10)
    for x in xrange:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=x, initial_step_size=0.01)
        if not np.isclose(deriv, 1/math.sqrt(x**2 + 1)):
            raise Exception("test_log function failed")


def test_sine_x_squared():
    """
    Tests if the estimated derivative of richardson_extrapolation function given a function f(x) = sin(x^2) is close
    to the derivative of the function, f'(x) = 2x*cos(x^2) evaluated from -4 pi to 4 pi.
    """
    # Loading the log function model
    r = te.loada("variable: $s -> $p; sin(s^2); s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from -4 pi to 4 pi
    thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
    for i in thetas:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=i, initial_step_size=0.01)
        if not np.isclose(deriv, 2*i*math.cos(i**2)):
            raise Exception("test_sine_x_squared function failed")
            
def test_x_squared_cosine():
    """
    Tests if the estimated derivative of richardson_extrapolation function given an inverse function f(x) = x^2*cos(2*x) is close
    to the derivative of the function, f'(x) = -2x(xsin(2x) - cos(2x)) evaluated from -4 pi to 4 pi.
    """
    # Loading the log function model
    r = te.loada("variable: $s -> $p; s^2*cos(2*s); s = 1")
    r.conservedMoietyAnalysis = True
    
    # Testing a range from -4 pi to 4 pi
    thetas = np.arange(-4*math.pi, 4*math.pi, 0.01)
    for i in thetas:
        deriv, err = richardson_extrapolation(r=r, parameter='s', variable='variable', parameter_init=i, initial_step_size=0.01)
        if not np.isclose(deriv, -2*i*(i*math.sin(2*i) - math.cos(2*i))):
            raise Exception("test_x_squared_cosine function failed")
            
            
def test_richardson_extrapolation():
    test_sine()
    print("test_sine passed")
    test_cubic_polynomial()
    print("test_cubic_polynomial passed")
    test_sinc()
    print("test_sinc passed")
    test_exponential()
    print("test_exponential passed")
    test_square_root()
    print("test_square_root passed")
    test_inverse()
    print("test_inverse passed")
    test_log()
    print("test_log passed")
    test_sine_x_squared()
    print("test_sine_x_squared passed")
    test_x_squared_cosine()
    print("test_x_squared_cosine passed")
    print("All tests passed!")

