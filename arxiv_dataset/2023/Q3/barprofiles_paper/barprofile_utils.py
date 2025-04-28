# Use functions for working with bar profiles

# *** B/P Morphologies

import numpy as np


bpMorphologyFile = "./data/s4gbars_bp-buckling_profiles_checklist.dat"

def GetGalaxyNamesAndDict_morphology( filename=bpMorphologyFile ):
    """
    
    Parameters
    ----------
    filename : str
    
    Returns
    -------
    gnames, bdDict : list of str, dict
        gnames = list of galaxy names
        bpDict = dict mapping galaxy names to B/P morphology classification codes
    """
    dlines = [line for line in open(filename) if line[0] != "#"]
    gnames = []
    bpDict= {}
    for line in dlines:
        if line.find("not barred") < 0 and line[:3] not in ["[-]", "[ ]"]:
            pp = line.split()
            gname = pp[0].split(']')[1].split()[0]
            gnames.append(gname)
            if len(pp) > 1:
                bpStatus = pp[1]
                if bpStatus in ["NO", "NO?", "BUCKLING???"]:
                    bpDict[gname] = "NO"
                else:
                    bpDict[gname] = "B/P"
            else:
                bpDict[gname] = "NO"
    return gnames, bpDict




# *** Profile Classifications

scrambleMap = "./data/scrambled_map.txt"
classificationsFile_pe = "./data/classifications_pe.txt"
classificationsFile_vd2 = "./data/classifications_vd_revised.txt"


def GetDescrambleDict( filename=scrambleMap ):
    """Returns a dict which maps scrambled galaxy numbers to galaxy names.
    """
    dlines = [line for line in open(filename) if len(line) > 1 and line[0] != '#']
    descrambleDict = {}
    for line in dlines:
        pp = line.split()
        i_scrambled = int(pp[0])
        gname = pp[2]
        descrambleDict[i_scrambled] = gname
    return descrambleDict



# Get all classifications from one person
def GetClassifications( classif_filename, descrambling_filename=scrambleMap, classifier="PE" ):
    """Returns a dict mapping galaxy names to the corresponding profile classification.
    
    Parameters
    ----------
    classif_filename : str
        filename with scrambled numbers and corresponding profile classifications.
    
    descrambling_filename : str
        filename mapping scrambled numbers to galaxy names
    
    classifier : str
        "PE" or "VD"
    
    Returns
    -------
    classsifDict : dict mapping str: str
        dict that maps galaxy names to profile classifications
    
    Currently, profiles with no classification ("?") are ignored.
    """
    descrambledDict = GetDescrambleDict(descrambling_filename)
    dlines = [line for line in open(classif_filename) if len(line) > 1 and line[0] != '#']
    classifDict = {}
    for line in dlines:
        pp = line.split()
        if len(pp) > 1:
            gnum = int(pp[0])
            classif = pp[1]
            gname = descrambledDict[gnum]
            if classif != "?":
                classifDict[gname] = classif.rstrip("?")
    return classifDict


def GetValuesAndIndices( classifDict, valuesDict ):
    """
    Returns lists of indices into vectors of possible values.
    First list is for galaxies with BP profiles, second is for galaxies
    with different profiles.
    
    Parameters
    ----------
    classifDict : dict mapping str: str
        galaxy_name: profile classification
    
    valuesDict : dict mapping str: float
        galaxy_name: parameter value (stellar mass, Hubble type, etc.)
    
    Returns
    -------
    values, i_bp, i_non_bp : tuple of (1D ndarray of float, list of int, list of int)
        values = array of galaxy values
        i_bp = list of indices into values, for galaxies with BP-profiles
        i_non_bp = list of indices into values, for galaxies without BP-profiles
    """

    values = []
    ii_bp = []
    ii_nonbp = []
    i = 0
    
    # go through first user-supplied data set
    for gname,classif in classifDict.items():
        classif = classif.rstrip("?")
        values.append(valuesDict[gname])
        if classif == 'BP':
            ii_bp.append(i)
        else:
            ii_nonbp.append(i)
        i += 1
    
    return (np.array(values), ii_bp, ii_nonbp)


def MakeValuesDict( classifDict, mainValuesDict, faceon_names ):
    """
    Returns tuple of dicts mapping profile classifications ("BP", "Exp", "FT", "2S") to
    lists of galaxy parameters (e.g., stellar masses): 
    all galaxies, low-inc galaxies, modinc galaxies
    
    We assume that classifDict maps galaxy names to classifications.
    """
    valsDict = {}
    valsDict_lowinc = {}
    valsDict_modinc = {}

    for gname,classif in classifDict.items():
        value = mainValuesDict[gname]
        classif = classif.rstrip("?")
        if classif in ["Exp", "Exp(N)"]:
            classif = "Exp"
        elif classif in ["FT", "FT(N)"]:
            classif = "FT"
        if classif not in list(valsDict.keys()):
            valsDict[classif] = []
            valsDict_lowinc[classif] = []
            valsDict_modinc[classif] = []
        valsDict[classif].append(value)
        if gname in faceon_names:
            valsDict_lowinc[classif].append(value)
        else:
            valsDict_modinc[classif].append(value)
    
    return (valsDict, valsDict_lowinc, valsDict_modinc)



def GetS4gIndices( classifDict1, s4gdata, classifDict2=None, profType='BP' ):
    """
    Returns lists of indices into s4gdata corresponding to galaxies with and without
    (by default) P+Sh (aka "BP") profiles. 
    If classifDict2 is supplied, then only galaxies classified by *both* classifies as 
    having profType profiles are counted as P+Sh/BP.
    
    Parameters
    ----------
    classifDict1 : dict mapping str: str
        galaxy_name: profile classification from first classifier

    s4gdict : datautils.ListDataFrame object
        dataFrame containing 'name' attribute

    classifDict2 : dict mapping str: str, optional
        galaxy_name: profile classification from second classifier
    
    profType : str, optional
        the primary selected profile type (default = 'BP' = 'P+Sh')
        
    Returns
    -------
    ii_type, ii_nontype : 2-tuple of list of int
        ii_type = indices into s4gdata corresponding to profType profile
        ii_nontype = indices into s4gdata corresponding to galaxies without profType profile
    """
    
    nGals = len(s4gdata.name)
    # names of galaxies in sample
    sampleNames = list(classifDict1.keys())
    if classifDict2 is not None:
        bothClassifiers = True
    else:
        bothClassifiers = False
    
    ii_type = []   # profType profiles
    ii_nontype = []
    for i in range(nGals):
        gname = s4gdata.name[i]
        if gname in sampleNames:
            # this gets tricky because there are two modes:
            #    1. single-classifier --> galaxy is either valid profType1 or not
            #       valid profType1 (valid non-profType1)
            #    2. both-classifiers --> galaxy can be valid profType1 (both classifiers
            #       agreed), valid non-profType1 (both classifiers agreed it was
            #       something other than profType1), *or* ambiguous (one classifier
            #       said profType1, other said something else)
            valid_type = False   # = this galaxy is accepted as unambiguously profType1
            valid_nontype = False   # this galaxy is accepted as unambiguously *not* profType1
            classif1 = classifDict1[gname].rstrip("?")
            if bothClassifiers:
                classif2 = classifDict2[gname].rstrip("?")
                if classif1 == profType and classif2 == profType:
                    valid_type = True
                elif classif1 != profType and classif2 != profType:
                    valid_nontype = True
            else:
                if classif1 == profType:
                    valid_type = True
                else:
                    valid_nontype = True

            if valid_type:
                ii_type.append(i)
            elif valid_nontype:
                ii_nontype.append(i)

    return (ii_type, ii_nontype)



