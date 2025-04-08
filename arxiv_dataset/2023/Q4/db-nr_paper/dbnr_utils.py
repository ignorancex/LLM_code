# Miscellaneous code for analysis of double-bar and nuclear-ring fractions and sizes

import copy
import math
import random
import numpy as np

import datautils as du

random.seed()


# lower and upper bounds of 68.3% confidence interval:
ONESIGMA_LOWER = 0.1585
ONESIGMA_UPPER = 0.8415

# conversion factor: kpc/arcsec for distance of 1 Mpc
k_kpcperarcsec = 1.0e3 / 206265.0 

MIN_INC = 26    # minimum inclination to apply deprojections for


# location of data files
baseDir_pub = "/Users/erwin/Documents/Working/Papers/Paper-extended-wiyn-survey/public/"




def Read2ColumnProfile( fname ):
	"""Read in the (first) two columns from a simple text file where the columns
	are separated by whitespace and lines beginning with '#' are ignored.
	
	Returns tuple of (x, y), where x and y are numpy 1D arrays corresponding to
	the first and second column
	"""
	dlines = [line for line in open(fname) if len(line) > 1 and line[0] != "#"]
	x = [float(line.split()[0]) for line in dlines]
	y = [float(line.split()[1]) for line in dlines]
	return np.array(x), np.array(y)



def DiffAngle( angle1, angle2, rotationCode, diskPA = -1, i = 0):
    """
    Calculates angle between two components, with positive =
    first component *leads* second.  rotationCode = 1 for
    counter-clockwise rotation, -1 for clockwise.
    """

    # Correct for inclination, if requested:
    if (diskPA >= 0) and (i >= MIN_INC):
        angle1 = deprojectpa(angle1 - diskPA, i) + diskPA
        angle2 = deprojectpa(angle2 - diskPA, i) + diskPA
    if (angle1 >= angle2):
        delPA = angle1 - angle2
        if delPA > 90:
            delPA = -(180 - delPA)
    else:
        delPA = angle2 - angle1
        if delPA > 90:
            delPA = -(180 - delPA)
        delPA = -delPA   # if component2 is leading, then comp1 is trailing
    return delPA * rotationCode



def deprojectr( deltaPA: float, inclination: float, r: float ) -> float:
    """Function to calculate a deprojected length, given an input
    observed position angle (*relative to disk line-of-nodes*, *not*
    straight position angle east of north!) and inclination, both in
    degrees, and an input observed (projected) length r.
    
    Parameters
    ----------
    deltaPA : float
        angle [deg] of measured value (e.g., bar major axis) relative to disk line of nodes
    
    inclination : float
        inclination angle [deg] of disk (i = 0 for face-on, i = 90 for edge-on)
    
    r : float
        length to be deprojected
    
    Returns
    -------
    r_scaled : float
        deprojected length
    """

    deltaPA_rad = np.radians(deltaPA)
    i_rad = np.radians(inclination)
    cosi = np.cos(i_rad)
    sindp = np.sin(deltaPA_rad)
    cosdp = np.cos(deltaPA_rad)
    scale = np.sqrt( (sindp*sindp)/(cosi*cosi) + cosdp*cosdp )
    return ( scale * r )


def RectifyPA( angle: float, maxAngle: float ) -> float:
    """Convert angle to lie between [0,maxAngle) degrees.
    (i.e., angle can be >=0, but must be < maxAngle)

        For maxAngle = 360, this just does standard modular arithmetic
        on angles, mapping the angle into [0,360]
        For maxAngle = 180, this maps the angle into [0,180), assuming
        m = 2 symmetry
        For max = 90, this assumes m=4 symmetry (so -30 --> 60, which
        may not be what you are expecting)
    """
    # map angle into [0,360], in case it's < 0 or > 360 to start with
    angle = angle % 360
    while (angle >= maxAngle):
        angle -= maxAngle
    return angle


def deprojectpa( deltaPA: float, inclination: float ) -> float:
    """Function to calculate a deprojected position angle, given an input
    observed position angle (*relative to disk line-of-nodes*, *not*
    straight position angle east of north!) and an input inclination, both
    in degrees.  Returns the deprojected position angle, relative to disk
    line-of-nodes, in degrees."""

    deltaPA_rad = np.radians(deltaPA)
    i_rad = np.radians(inclination)
    deltaPA_deproj = math.atan( math.tan(deltaPA_rad) / math.cos(i_rad) )
    return np.degrees(deltaPA_deproj)


def deprojectpa_abs( obsPA: float, diskPA: float, inclination: float, 
                    symmetric=True ) -> float:
    """Function to calculate deprojected PA on the sky of a structure, as
    if the galaxy were rotated to face-on orientation.
        obsPA = observed (projected) PA of structure
        diskPA = major axis (line of nodes) of disk
        inclination = inclination of disk (0 = face-on).
    If symmetric=True, then values > 180 are converted to the equivalent
    0--180 values."""

    if (inclination == 0.0):
        dPA = obsPA
    else:
        deltaPA = obsPA - diskPA
        deltaPA_dp = deprojectpa(deltaPA, inclination)
        dPA = deltaPA_dp + diskPA
    if symmetric:
        dPA = RectifyPA(dPA, 180.0)
    return dPA


def kpc_per_arcsec( distMpc ):
    """
    Given an (angular-diameter) distance in Mpc, returns kpc/arcsec scale.
    
    Parameters
    ----------
    distMpc : float
        distance in Mpc (angular-diameter distance)
    
    Returns
    -------
    kpcScale : float
        factor which converts arcsec to kpc
    """
    return k_kpcperarcsec * distMpc

    
def MungeName( gname_full ):
    """
    Given a galaxy name in normal form (e.g., "NGC 936"), returns it in
    'munged' form with leading zeros and no internal spaces (e.g., "NGC0936").
    Only works for IC, NGC, UGC, and PGC names.
    
    Parameters
    ----------
    gname_full : str
        name of galaxy
    
    Returns
    -------
    newname : str
        'munged' version of input galaxy name
    """
    prefix, num = gname_full.split()
    if prefix in ["IC", "NGC"]:
        newname = "{0}{1:04d}".format(prefix, int(num))
    elif prefix == "UGC":
        newname = "{0}{1:05d}".format(prefix, int(num))
    elif prefix == "PGC":
        newname = "{0}{1:06d}".format(prefix, int(num))
    else:
        msg = "Unrecognized galaxy-name prefix (\"{0}\")".format(prefix)
        raise ValueError(msg)
    return newname


def GetDBdict( fname = "table_innerbars.dat", baseDir="" ):
    """
    Returns a dict mapping names of double-barred galaxies to list of observed
    inner-bar measurements.
    
    Parameters
    ----------
    fname : str
        filename holding inner-bar data
    
    baseDir : str
        path to directory containing fname
    
    Returns
    -------
    innerbar_dict : dict
        maps (munged) galaxy name (e.g., "NGC0718") to list of values for inner bar:
            [barPA, amax, Lbar, emax] : all float
    """
    filePath = baseDir + "/" + fname
    lines = open(filePath).readlines()
    dlines = lines[12:]
    dbDict = {}
    for line in dlines:
        gname = MungeName(line[:9])
        barPA = float(line[10:13])
        amax = float(line[15:18])
        lbar = float(line[20:23])
        emax = float(line[25:29])
        dbDict[gname] = [barPA, amax, lbar, emax]

    return dbDict


def GetNRdict( fname = "table_nuclearrings.dat", baseDir="" ):
    """
    Returns a dict mapping names of nuclear-ring galaxies to list of observed
    nuclear-ring measurements.
    
    Parameters
    ----------
    fname : str
        path to filename holding nuclear-ring data
    
    baseDir : str
        path to directory containing fname
    
    Returns
    -------
    nrbar_dict : dict
        maps (munged) galaxy name (e.g., "NGC0718") to list of values for nuclear:
            [nrPA, a_ring, emax_ring, ring-class] : [float, float, float, str]
    """
    filePath = baseDir + "/" + fname
    lines = open(filePath).readlines()
    dlines = lines[13:]
    nrDict = {}
    for line in dlines:
        gname = MungeName(line[:9])
        ringPA = float(line[10:13])
        a_ring = float(line[15:18])
        emax = float(line[20:24])
        ring_class = line[26:38].strip()
        if ring_class == "blue":
            ring_class = "star-forming"
        nrDict[gname] = [ringPA, a_ring, emax, ring_class]

    return nrDict


def DeprojectSizes_kpc( objectSizes, objectPAs, diskPAs, inclinations, distances ):
    """
    Converts observed angular sizes (in arcsec) to deprojected linear sizes (in kpc)
    for features assumed to lie in a galaxy disk.
    
    Parameters
    ----------
    objectSizes : 1D sequence of float
        lengths of objects (e.g., bar radius), arcsec
    
    objectPAs : 1D sequence of float
        position angles of objects (e.g., bar PA), deg E of N

    diskPAs : 1D sequence of float
        position angles of galaxy disks, deg E of N
    
    inclinations : 1D sequence of float
        inclinations of galaxy disks, deg
    
    distances : 1D sequence of float
        distances of galaxies, Mpc
    
    Returns
    -------
    sizes_dp_kpc : list of float
    """
    nObjs = len(objectSizes)
    kpc_scales = [ kpc_per_arcsec(distances[i]) for i in range(nObjs) ]
    sizes_dp_kpc = [ kpc_scales[i] * deprojectr(objectPAs[i] - diskPAs[i], 
                        inclinations[i], objectSizes[i]) for i in range(nObjs) ]
    return sizes_dp_kpc


def ConvertToParsecs( objectSizes, distances ):
    nObjs = len(objectSizes)
    pc_scales = [ 1e3 * kpc_per_arcsec(distances[i]) for i in range(nObjs) ]
    sizes_pc = [ pc_scales[i] * objectSizes[i] for i in range(nObjs) ]
    return sizes_pc


def GetBarredGalaxyData( fname="table_mainsample.dat", baseDir="" ):
    """
    Reads in general data, including outer/single bar measurements for all the barred
    galaxies; also adds inner-bar and nuclear-ring measurements and deprojected
    sizes of all bars and nuclear rings in kpc. (Inner-bar and nuclear-ring size = 0
    for galaxies lacking those features.)
    
    Parameters
    ----------
    fname : str
        path to filename holding barred-galaxy data
    
    baseDir : str
        path to directory containing fname
    
    Returns
    -------
    df = datautils.ListDataFrame
        data frame holding columns of data, with each line = one galaxy
    
    gname_rowdict : dict
        maps (munged) galaxy name to (0-based) row number in data frame
    """
    filePath = baseDir + "/" + fname
    
    lines = open(filePath).readlines()
    dlines = lines[32:]
    
    gnames = []
    htypes = []
    distances = []
    logmstars  = []
    diskPAs = []
    incs = []
    rotCodes = []
    barPAs = []
    amaxs = []
    lbars = []
    emaxs = []
    dbFlags = []
    nrFlags = []
    fwhms = []
    innerBarAmaxs = []
    innerBarLbars = []
    innerBarEmaxs = []
    innerBarPAs = []
    nrAmaxs = []
    nrEmaxs = []
    nrPAs = []
    ringClasses = []
    amax_dp_kpcs = []
    amax2_dp_kpcs = []
    nr_dp_kpcs = []
    gname_rowdict = {}
    i = 0
    for line in dlines:
        gname = MungeName(line[:9])
        gname_rowdict[gname] = i
        T = float(line[37:41])
        dist = float(line[44:48])
        logmstar = float(line[56:62])
        diskPA = float(line[64:67])
        inc = float(line[70:72])
        rotCode = line[75]
        barPA = float(line[80:83])
        amax = float(line[85:88])
        lbar = float(line[89:93])
        emax = float(line[95:99])
        dbStatus = line[100]
        if dbStatus.strip() == "Y":
            dbFlags.append(True)
        else:
            dbFlags.append(False)
        nrStatus = line[102]
        if nrStatus.strip() == "Y":
            nrFlags.append(True)
        else:
            nrFlags.append(False)
        fwhms.append(float(line[104:108]))
        
        gnames.append(gname)
        htypes.append(T)
        distances.append(dist)
        logmstars.append(logmstar)
        diskPAs.append(diskPA)
        incs.append(inc)
        rotCodes.append(rotCode)
        barPAs.append(barPA)
        amaxs.append(amax)
        lbars.append(lbar)
        emaxs.append(emax)
        i += 1
    
    # add DB data
    dbDict = GetDBdict(baseDir=baseDir)
    for gname in gnames:
        if gname in dbDict:
            barPA, amax, Lbar, emax = dbDict[gname]
        else:
            barPA, amax, Lbar, emax = -999.0, 0.0, 0.0, 0.0
        innerBarPAs.append(barPA)
        innerBarAmaxs.append(amax)
        innerBarLbars.append(Lbar)
        innerBarEmaxs.append(emax)
    # add NR data
    nrDict = GetNRdict(baseDir=baseDir)
    for gname in gnames:
        if gname in nrDict:
            ringPA, a_ring, emax, ring_class = nrDict[gname]
        else:
            ringPA, a_ring, emax, ring_class = -999.0, 0.0, 0.0, ""
        nrPAs.append(ringPA)
        nrAmaxs.append(a_ring)
        nrEmaxs.append(emax)
        # treat NGC 718's "blue" ring as SF
        if ring_class == "blue":
            ring_class = "star-forming"
        ringClasses.append(ring_class)

    # deproject amax and a_ring values for bars and NRs, convert to kpc
    amax_dp_kpcs = DeprojectSizes_kpc(amaxs, barPAs, diskPAs, incs, distances)
    amax2_dp_kpcs = DeprojectSizes_kpc(innerBarAmaxs, innerBarPAs, diskPAs, incs, distances)
    nr_dp_kpcs = DeprojectSizes_kpc(nrAmaxs, nrPAs, diskPAs, incs, distances)
    fwhm_pcs = ConvertToParsecs(fwhms, distances)
    
    # assemble output data frame
    dataList = [ np.array(gnames), np.array(distances), np.array(htypes), np.array(logmstars),
                np.array(diskPAs), np.array(incs), np.array(rotCodes), np.array(barPAs), 
                np.array(amaxs), np.array(lbars), np.array(emaxs), np.array(dbFlags), 
                np.array(nrFlags), np.array(innerBarPAs), np.array(amax_dp_kpcs), 
                np.array(amax2_dp_kpcs), np.array(nr_dp_kpcs), np.array(ringClasses), 
                np.array(fwhm_pcs) ]
    colNames = ["name", "dist", "T", "logmstar", "diskPA", "inclination", "rotCode", "barPA", 
                "amax", "Lbar", "emax", "dbFlag", "nrFlag", "bar2PA", "amax_dp_kpc", 
                "amax2_dp_kpc", "nr_dp_kpc", "nr_class", "fwhm_pc"]    
    df = du.ListDataFrame(dataList, colNames)
    
    return (df, gname_rowdict)



def bootstrap_validation( x, y, nIter, fittingFn,  modelFn=None, computeModelFn=None,
							initialParams=None, adjustEstimate=True, errs=None,
							verbose=False ):
	"""
	Uses bootstrap resampling to estimate the accuracy of a model (analogous to
	"leave-k-out" cross-validation).
	
	See Sec. 7.11 of Hastie, Tibshirani, and Friedman 2008, Elements of Statistical
	Learning (2nd Ed.).

	Parameters
	----------
	x : numpy array of independent variable values (predictors)
		Can also be tuple or list of 2 numpy arrays

	y : numpy array of dependent variable values

	nIter : int
		number of bootstrap iterations to run		

	fittingFn : function or callable
		fittingFn(x, y, initialParams=None) fits the model
		specified by modelFn to the data specified by x and y
		
		Returns "fitResult", which will be used by modelFn or computeModelFn
			
		If modelFn is supplied, then we use
		fittingFn(x, y, modelFn, initialParams)

	modelFn : function or callable, quasi-optional
		modelFn(x, params) -- used by fittingFn; computes model which is fit to data
			params = either initialParams or fitResult
		
	computeModelFn : function or callable, quasi-optional
		computeModelFn(x, fitResult) -- computes model which is fit to data;
		meant for cases when modelFn is not needed.
	
	initialParams : any or None, optional
		object passed as optional input to fittingFn
		
	adjustEstimate : bool, optional [default = True]
		If True (default), then the final error estimate is corrected using
		the ".632+ bootstrap estimator" rule (Efron & Tibshirani 1997):
			err = 0.368*err_training + 0.632*err_bootstrap
		where err_training is the mean squared error of the model fit to the
		complete dataset and err_bootstrap is the mean of the mean squared
		errors from bootstrap resampling
		
		If False, then the return value is just err_bootstrap
		
	errs : numpy array of float or None, optional
		array of Gaussian sigmas associated with y
	
	Returns
	---------
	errorEstimate : float
		Approximation to the test error (mean squared error for predictions from the model
	
	Examples
	---------
	Fit a 2nd-order polynomial to data:
		# define wrapper for np.polyval, since that function uses reverse of
		# our normal input ordering
		def nicepolyval( x, p ):
			return np.polyval(p, x)
		
		# use initialParams to set the "deg" parameter for np.polyfit
		bootstrap_validation(x, y, 100, np.polyfit, computeModelFn=nicepolyval,
			initialParams=2)
	"""
	
	if modelFn is None and computeModelFn is None:
		print("ERROR: you must supply at least one of modelFn or computeModelFn!")
		return None
	if computeModelFn is None:
		evaluateModel = modelFn
	else:
		evaluateModel = computeModelFn
	
	nData = len(y)
	# initial fit to all the data ("training")
	fitResult = fittingFn(x, y, initialParams, errs)
	# MSE for fit to all the data
	residuals = y - evaluateModel(x, fitResult)
	errorTraining = np.mean(residuals**2)
	if verbose:
		print(fitResult)
		print("training MSE = %g" % errorTraining)
	
	# Do bootstrap iterations
	indices = np.arange(0, nData)
	nIterSuccess = 0
	individualBootstrapErrors = []
	for b in range(nIter):
		i_bootstrap = np.random.choice(indices, nData, replace=True)
		i_excluded = [i for i in indices if i not in i_bootstrap]
		nExcluded = len(i_excluded)
		if (nExcluded > 0):
			if type(x) in [tuple,list]:
				x_b = (x[0][i_bootstrap], x[1][i_bootstrap])
			else:
				x_b = x[i_bootstrap]
			y_b = y[i_bootstrap]
			try:
				if errs is None:
					fitResult_b = fittingFn(x_b, y_b, initialParams, None)
				else:
					fitResult_b = fittingFn(x_b, y_b, initialParams, errs[i_bootstrap])
				residuals = y - evaluateModel(x, fitResult_b)
				# calculate mean squared prediction error for this sample
				errorB = (1.0/nExcluded) * np.sum(residuals[i_excluded]**2)
				individualBootstrapErrors.append(errorB)
				nIterSuccess += 1
			except RuntimeError:
				# couldn't get a proper fit, so let's discard this sample and try again
				pass
	
	individualBootstrapErrors = np.array(individualBootstrapErrors)
	errorPredict = np.mean(individualBootstrapErrors)
	if verbose:
		print("test MSE = %g (%d successful iterations)" % (errorPredict, nIterSuccess))

	
	if adjustEstimate is True:
		adjustedErrorPredict = 0.368*errorTraining + 0.632*errorPredict
		if verbose:
			print("Adjusted test MSE = %g" % adjustedErrorPredict)
		return adjustedErrorPredict
	else:
		return errorPredict



def AIC( logLikelihood, nParams ):
	"""Calculate the original Akaike Information Criterion for a model fit
	to data, given the ln(likelihood) of the best-fit model and the number of
	model parameters nParams.
	
	Note that this should only be used for large sample sizes; for small
	sample sizes (e.g., nData < 40*nParams), use the corrected AIC function
	AICc [below].
	"""
	
	return -2.0*logLikelihood + 2.0*nParams



def AICc( logLikelihood, nParams, nData, debug=False ):
	"""Calculate the bias-corrected Akaike Information Criterion for a
	model fit to data, given the ln(likelihood) of the best-fit model,
	the number of model parameters nParams, and the number of data points
	nData (the latter is used to correct the 2*nParams part of AIC for small
	sample size).
	
	Formula from Burnham & Anderson, Model selection and multimodel inference: 
	a practical information-theoretic approach (2002), p.66.
	"""
	
	# use corrected form of nParams term
	aic = AIC(logLikelihood, nParams)
	# add bias-correction term
	correctionTerm = 2*nParams*(nParams + 1) / (nData - nParams - 1.0)
	if debug:
		print("AICc: ", aic, correctionTerm)
	return aic + correctionTerm


def ConfidenceInterval( vect ):
	
	nVals = len(vect)
	lower_ind = int(round(ONESIGMA_LOWER*nVals)) - 1
	upper_ind = int(round(ONESIGMA_UPPER*nVals))
	vect_sorted = copy.copy(vect)
	vect_sorted.sort()
	return (vect_sorted[lower_ind], vect_sorted[upper_ind])


def Binomial( n, n_tot, nsigma=1.0, conf_level=None, method="wilson" ):
	"""Computes fraction (aka frequency or rate) of occurances p = (n/n_tot).
	Also computes the lower and upper confidence limits using either the 
	Wilson (1927) or Agresti & Coull (1998) method (method="wilson" or method="agresti");
	default is to use Wilson method.
	Default is to calculate 68.26895% confidence limits (i.e., 1-sigma in the
	Gaussian approximation).
	
	Returns tuple of (p, sigma_minus, sigma_plus).
	"""
	
	p = (1.0 * n) / n_tot
	q = 1.0 - p
	
	if (conf_level is not None):
		print("Alternate values of nsigma or conf_limit not yet supported!")
		alpha = 1.0 - conf_level
		# R code would be the following:
		#z_alpha = qnorm(1.0 - alpha/2.0)
		return None
	else:
		z_alpha = nsigma   # e.g., z_alpha = nsigma = 1.0 for 68.26895% conf. limits
	
	if (method == "wald"):
		# Wald (aka asymptotic) method -- don't use except for testing purposes!
		sigma_minus = sigma_plus = z_alpha * np.sqrt(p*q/n_tot)
	else:
		z_alpha2 = z_alpha**2
		n_tot_mod = n_tot + z_alpha2
		p_mod = (n + 0.5*z_alpha2) / n_tot_mod
		if (method == "wilson"):
			# Wilson (1927) method
			sigma_mod = np.sqrt(z_alpha2 * n_tot * (p*q + z_alpha2/(4.0*n_tot))) / n_tot_mod
		elif (method == "agresti"):
			# Agresti=Coull method
			sigma_mod = np.sqrt(z_alpha2 * p_mod * (1.0 - p_mod) / n_tot_mod)
		else:
			print("ERROR: method \"%s\" not implemented in Binomial!" % method)
			return None
		p_upper = p_mod + sigma_mod
		p_lower = p_mod - sigma_mod
		sigma_minus = p - p_lower
		sigma_plus = p_upper - p
	
	return (p, sigma_minus, sigma_plus)


def bootstrap_validation( x, y, nIter, fittingFn,  modelFn=None, computeModelFn=None,
							initialParams=None, adjustEstimate=True, errs=None,
							verbose=False ):
	"""
	Uses bootstrap resampling to estimate the accuracy of a model (analogous to
	"leave-k-out" cross-validation).
	
	See Sec. 7.11 of Hastie, Tibshirani, and Friedman 2008, Elements of Statistical
	Learning (2nd Ed.).

	Parameters
	----------
	x : numpy array of independent variable values (predictors)
		Can also be tuple or list of 2 numpy arrays

	y : numpy array of dependent variable values

	nIter : int
		number of bootstrap iterations to run		

	fittingFn : function or callable
		fittingFn(x, y, initialParams=None) fits the model
		specified by modelFn to the data specified by x and y
		
		Returns "fitResult", which will be used by modelFn or computeModelFn
			
		If modelFn is supplied, then we use
		fittingFn(x, y, modelFn, initialParams)

	modelFn : function or callable, quasi-optional
		modelFn(x, params) -- used by fittingFn; computes model which is fit to data
			params = either initialParams or fitResult
		
	computeModelFn : function or callable, quasi-optional
		computeModelFn(x, fitResult) -- computes model which is fit to data;
		meant for cases when modelFn is not needed.
	
	initialParams : any or None, optional
		object passed as optional input to fittingFn
		
	adjustEstimate : bool, optional [default = True]
		If True (default), then the final error estimate is corrected using
		the ".632+ bootstrap estimator" rule (Efron & Tibshirani 1997):
			err = 0.368*err_training + 0.632*err_bootstrap
		where err_training is the mean squared error of the model fit to the
		complete dataset and err_bootstrap is the mean of the mean squared
		errors from bootstrap resampling
		
		If False, then the return value is just err_bootstrap
		
	errs : numpy array of float or None, optional
		array of Gaussian sigmas associated with y
	
	Returns
	---------
	errorEstimate : float
		Approximation to the test error (mean squared error for predictions from the model
	
	Examples
	---------
	Fit a 2nd-order polynomial to data:
		# define wrapper for np.polyval, since that function uses reverse of
		# our normal input ordering
		def nicepolyval( x, p ):
			return np.polyval(p, x)
		
		# use initialParams to set the "deg" parameter for np.polyfit
		bootstrap_validation(x, y, 100, np.polyfit, computeModelFn=nicepolyval,
			initialParams=2)
	"""
	
	if modelFn is None and computeModelFn is None:
		print("ERROR: you must supply at least one of modelFn or computeModelFn!")
		return None
	if computeModelFn is None:
		evaluateModel = modelFn
	else:
		evaluateModel = computeModelFn
	
	nData = len(y)
	# initial fit to all the data ("training")
	fitResult = fittingFn(x, y, initialParams, errs)
	# MSE for fit to all the data
	residuals = y - evaluateModel(x, fitResult)
	errorTraining = np.mean(residuals**2)
	if verbose:
		print(fitResult)
		print("training MSE = %g" % errorTraining)
	
	# Do bootstrap iterations
	indices = np.arange(0, nData)
	nIterSuccess = 0
	individualBootstrapErrors = []
	for b in range(nIter):
		i_bootstrap = np.random.choice(indices, nData, replace=True)
		i_excluded = [i for i in indices if i not in i_bootstrap]
		nExcluded = len(i_excluded)
		if (nExcluded > 0):
			if type(x) in [tuple,list]:
				x_b = (x[0][i_bootstrap], x[1][i_bootstrap])
			else:
				x_b = x[i_bootstrap]
			y_b = y[i_bootstrap]
			try:
				if errs is None:
					fitResult_b = fittingFn(x_b, y_b, initialParams, None)
				else:
					fitResult_b = fittingFn(x_b, y_b, initialParams, errs[i_bootstrap])
				residuals = y - evaluateModel(x, fitResult_b)
				# calculate mean squared prediction error for this sample
				errorB = (1.0/nExcluded) * np.sum(residuals[i_excluded]**2)
				individualBootstrapErrors.append(errorB)
				nIterSuccess += 1
			except RuntimeError:
				# couldn't get a proper fit, so let's discard this sample and try again
				pass
	
	individualBootstrapErrors = np.array(individualBootstrapErrors)
	errorPredict = np.mean(individualBootstrapErrors)
	if verbose:
		print("test MSE = %g (%d successful iterations)" % (errorPredict, nIterSuccess))

	
	if adjustEstimate is True:
		adjustedErrorPredict = 0.368*errorTraining + 0.632*errorPredict
		if verbose:
			print("Adjusted test MSE = %g" % adjustedErrorPredict)
		return adjustedErrorPredict
	else:
		return errorPredict


