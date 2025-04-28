import numpy as np

def chi_square(yPred, yData, err):
    chi2 = np.sum((yPred-yData)**2/(err**2))
    return chi2

def AIC(yPred, data, err, k):
    """
    Returns the Akaike information criterion (AIC) for a given function with a
    number of parameters k and a negative log-likelihood value given
    by func(data, params)
    
    Does not take into account the size of the sample because the sample is the same 
    in all of our cases
    
    k is the number of free parameters
    """
    # ll = log_likelihood(yPred, data, err)
    # aic = 2 * k + 2 * ll
    chi2 = chi_square(yPred, data, err)
    aic = 2 * k + chi2 # we leave out the constant because it is the same for
    # both models
    return aic, chi2

def probaic(aicmin, aiclist):
    return np.exp((aicmin-aiclist)/2)

def get_AIC_for_cube(cube, rmsmap, mask, fittedmodels, nfittedmodels=2, nparamsfittedmodels=[3, 6], prob_tolerance=0.05):
    """
    cube: SpectralCube type
    mask: nparray 2 dimensional
    the default case is two models: 1 gaussian and 2 gaussians.
    """
    ylist, xlist = np.where(mask==1) # we select the pixels where to do the evaluation
    totalpix = len(ylist)
    aiccube = np.zeros((nfittedmodels, np.shape(mask)[0], np.shape(mask)[1])) * np.nan
    chisquarecube = np.zeros((nfittedmodels, np.shape(mask)[0], np.shape(mask)[1])) * np.nan
    ncomponentsmap = np.zeros(np.shape(mask)) * np.nan
    probmap = np.zeros(np.shape(mask)) * np.nan
    flag_prob = np.zeros(np.shape(mask)) * np.nan
    deltaaicmap = np.zeros((nfittedmodels-1, np.shape(mask)[0], np.shape(mask)[1])) * np.nan

    for x, y in zip(xlist, ylist):
        spectrum = cube.unmasked_data[:, y, x].value
        rmspix = rmsmap[y, x]
        for ni in range(nfittedmodels):
            ypred = fittedmodels[ni][:, y, x]
            if np.all(np.isnan(ypred)):
                continue
            aicni, chini = AIC(ypred, spectrum, rmspix, nparamsfittedmodels[ni])
            aiccube[ni, y, x] = aicni
            chisquarecube[ni, y, x] = chini
        aiclist = aiccube[:, y, x]
        # choose the minimum AIC
        if np.all(np.isnan(aiclist)):
            #print("All AIC for pixel ({0},{1}) out of {2} are NaN. This should not happen. Check it.".format(x,y,totalpix))
            continue
        minaicindex = np.nanargmin(aiclist)
        ncomponentsmap[y, x] = minaicindex+1 # now the index in ncomponentsmap is the number of the model that is best
  
        # testing if it is not really that much better
        minaic = aiclist[minaicindex]
        aicnew = np.delete(aiclist, minaicindex)
        deltaaicmap[:, y, x] = aicnew - minaic
        prob = probaic(minaic, aicnew)
        probmap[y, x] = np.amax(prob)
        if np.amax(prob) > prob_tolerance:
            flag_prob[y,x] = 1
 
    return ncomponentsmap, flag_prob, chisquarecube, deltaaicmap, probmap, aiccube #aic1Gmap, aic2Gmap, aic3Gmap

def get_aic_filtered_params(paramsnormal, ncomponents, results):
    paramsaic = np.zeros(np.shape(paramsnormal)) * np.nan
    yparamsaic, xparamsaic = np.where(results==ncomponents)
    for x, y in zip(xparamsaic, yparamsaic):
        for i in range(6*ncomponents): # includes uncertainties
            paramsaic[i, y, x] = paramsnormal[i, y, x]
    return paramsaic