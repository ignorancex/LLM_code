import h5py
import functools
import numpy as np
import inspect
import sys
import os
import zlib

def h5py_get_type(file, path):
    if type(file[path]) == h5py._hl.dataset.Dataset:
        return "data"
    elif type(file[path]) == h5py._hl.group.Group:
        if "type" in file[path].attrs:
            return file[path].attrs["type"]
        else:
            return "group"

def h5py_write_list(file, path, data):
    file.create_group(path)
    file[path].attrs["len"] = len(data)
    if type(data) == tuple:
        file[path].attrs["type"] = "tuple"
    elif type(data) == list:
        file[path].attrs["type"] = "list"
    else:
        raise ValueError("Unknown type: ", type(data))
    for ind,item in enumerate(data):
        file.create_dataset(path + "/%d" % ind, data=item)

def h5py_read_list(file, path):
    assert file[path].attrs["type"] in ("list", "tuple")
    num = file[path].attrs["len"]
    res = []
    for ind in range(0, num):
        res.append(np.array(file[path + "/%d" % ind]))

    if file[path].attrs["type"] == "tuple":
        return tuple(res)
    else:
        return res


def h5py_write_dict(file, datadict, overwrite=False, verbose=True):
    """ This function can write recursive dicts as hdf5 file """

    import h5py

    if type(file) is str: # Filename was provided -> open as hdf5 file
        with h5py.File(file, "a") as myfile:
            return h5py_write_dict(myfile, datadict, overwrite=overwrite, verbose=verbose)

    for dictkey in datadict.keys():
        if type(dictkey) == type(1):
            key = "_int_%d" % dictkey
        elif type(dictkey) == type(1.2):
            key = "_float_%g" % dictkey
        else:
            key = dictkey

        if type(datadict[dictkey]) is dict:
            file.require_group(key)
            h5py_write_dict(file[key], datadict[dictkey], overwrite=overwrite, verbose=verbose)
        else:
            if (key in file) and not overwrite:
                if verbose:
                    print("Ignoring existing %s (overwrite=False) " % (file[key].name, ))
            else:
                if key in file:
                    if verbose:
                        print("Overwriting %s " % (file[key].name,))
                    del file[key]
                file.create_dataset(key, data=datadict[dictkey])

def h5py_read_dict(file, datadict=None, verbose=True):
    """ This function can read recursive dicts from hdf5 files """
    import h5py

    if datadict is None:
        datadict = {}

    if type(file) is str: # Filename was provided -> open as hdf5 file
        with h5py.File(file, "r") as myfile:
            return h5py_read_dict(myfile, datadict, verbose=verbose)

    for key in file.keys():
        if key[0:5] == "_int_":
            dictkey = int(key[5:])
        elif key[0:7] == "_float_":
            dictkey = float(key[7:])
        else:
            dictkey = key

        if type(file[key]) is h5py._hl.group.Group:
            datadict[dictkey] = {}
            h5py_read_dict(file[key], datadict[dictkey], verbose=verbose)
        elif type(file[key]) is h5py._hl.dataset.Dataset:
            datadict[dictkey] = np.array(file[key])
        else:
            print("Ignoring unhandled type of %s :" % file[key].name, type(file[key]))

    return datadict

def h5py_write_any(file, path, data):
    if (type(data) == tuple) | (type(data) == list):
        h5py_write_list(file, path, data)
    elif type(data) == dict:
        file.create_group(path)
        file[path].attrs["type"] = "dict"
        h5py_write_dict(file[path], data, overwrite=False, verbose=True)
    else:
        file.create_dataset(path, data=data)

def h5py_read_any(file, path):
    mytype = h5py_get_type(file, path)
    if ((mytype == "list") | (mytype == "tuple")):
        return h5py_read_list(file, path)
    elif mytype == "dict":
        return h5py_read_dict(file[path])
    elif mytype == "data":
        return np.array(file[path])
    else:
        print("Problems with %s mytype = %s", path, mytype)

def args_and_kwargs_to_dict(argspec, args, kwargs, nargsskip=0):
    """This function unifies all arguments (keyword or non keyword) into a dict

    argspec must be an instance of inspect.getargspec(func) (py2) or inspect.getfullargspec(func) (py3)

    nargsskip is a hack in the case the call signature has been hacked, by
    not passing all args which are expected from the signature"""
    mydict = {}

    # append all arguments
    numargs = len(args)
    ilastarg = 0
    for i,arg in enumerate(args):
        argname = argspec.args[i]
        mydict[argname] = arg
        ilastarg += 1
    
    if argspec.defaults is None:
        num_pure_args = len(argspec.args)
    elif argspec.args is None:
        num_pure_args = 0
    else:
        num_pure_args = len(argspec.args)-len(argspec.defaults)
    num_pure_args_as_kw = num_pure_args - ilastarg

    # go through default keyword arguments
    for j,argname in enumerate(argspec[0][numargs+nargsskip:]):
        if argname in kwargs:
            mydict[argname] = kwargs[argname]
        else:
            mydict[argname] = argspec.defaults[j-num_pure_args_as_kw]

    # now append any possible variable keyword arguments that are not listed
    # in argspec, because they might have been passed through **kwargs
    for argname in kwargs:
        if argname not in argspec.args:
            mydict[argname] = kwargs[argname]

    return mydict

def val_to_str(val):
    if type(val) == np.ndarray:
        if val.size >= 4:
            mystr =  "hash" + str(zlib.adler32(val.data.tobytes()))
        else:
            mystr = str(val)
    else:
        mystr = str(val)
        
    # remove "/" to not create confusion with hdf5 paths
    mystr = mystr.replace("/", "_")
            
    return mystr

def args_to_path(argspec, *args, ignoreargs=[], nargsskip=0, **kwargs):
    myargdict = args_and_kwargs_to_dict(argspec, args, kwargs, nargsskip=nargsskip)

    res = "/"
    for argname in myargdict:
        if argname in ignoreargs:
            continue
        else:
            res += ("%s=%s|" % (argname, val_to_str(myargdict[argname])))

    if res == "/":
        res += "noargs"
    return res

def signature_as_string(argspec, ignoreargs=()):
    filteredargs = []
    for arg in argspec.args:
        if not arg in ignoreargs:
            filteredargs.append(arg)
    return str(filteredargs)

def check_or_create_signature_block(file, signature, blockname="/_signature", verbose=False):
    import os

    if os.path.exists(file):
        with h5py.File(file, "r") as f:
            if blockname in f:
                s1 = np.array(f[blockname])
                s2 = signature
                if  s1 == s2:
                    if verbose:
                        print("Signature is correct: %s" % signature)
                    need_create_signature = False
                elif  s1 == bytes(s2, encoding = "utf-8"):
                    if verbose:
                        print("Signature is correct: %s" % signature)
                    need_create_signature = False
                else:
                    need_create_signature = False
                    raise ValueError(
                    """The Signature in %s differs:
                    %s (previous)
                    %s (current)
                    Signatures should not differ, probably you have changed the functions arguments
                    and should reset the cache by passing resetcache=True to the decorator
                    or to a function call""" % (file, str(np.array(f[blockname])), signature))
            else:
                print("Did not find the signature block (%s) in %s. I will write one now" % (blockname, file))
                print("If you are not sure why this happened, consider resetting the cache")
                need_create_signature = True
    else:
        need_create_signature = True

    if need_create_signature:
        with h5py.File(file, "a") as f:
            f.create_dataset(blockname, data=signature)

def find_in_files(files, mypath, verbose=False):
    for myfile in files:
        if not os.path.exists(myfile):
            continue
        with h5py.File(myfile, "r") as f:
            if mypath in f:
                needwrite = False
                res = h5py_read_any(f, mypath)
                if verbose:
                    print("read ", mypath, " in ", myfile)
                return res, True
    return None, False


def get_argspec(func):
    """Keep a cache of previous function calls"""
    if sys.version_info >= (3, 0):
        argspec = inspect.getfullargspec(func)
    else:
        import warnings
        warnings.warn("h5cache: Keyword arguments will not work with python 2 (but py3 yes)", DeprecationWarning)
        argspec = inspect.getargspec(func)

    return argspec

def h5cache(_func=None, file=None, ignoreargs=(), verbose=False, resetcache=False, readfiles=None, mpicomm=None, nfiles=None):
    """This a decorator, use it as @h5cache or @h5cache(file=..., ...)
       It stores in file the output from the calls of the decorated function
       in an hdf5 file. If a call has the same input arguments as a current one
       it will return the previous output (so make sure to delete the cache file
       if you change the function)

       You can delete the previous cache by passing resetcache=True to the function call
       or the decorator
       """
    if mpicomm is not None:
        assert readfiles is None

        if nfiles is None:
            nfiles = mpicomm.Get_size()

        readfiles = ["%s_%i.hdf5" % (file, i) for i in range(0,nfiles)]
        readfiles = readfiles + ["%s.hdf5" % (file,)]
        file = "%s_%i.hdf5" % (file, mpicomm.Get_rank())

        #print("readfiles: ", mpicomm.Get_rank(), readfiles)

    if readfiles is None:
        readfiles = []

    if type(readfiles) == type(""):
        readfiles = [readfiles]

    def decorator_cache(func):
        argspec = get_argspec(func)

        if file is None:
            myfile = "cache_%s.hdf5" % func.__name__
        else:
            myfile = file

        if resetcache:
            if os.path.exists(myfile):
                if verbose:
                    print("Deleting previous cache in %s" % myfile)
                os.remove(myfile)

        signature = signature_as_string(argspec, ignoreargs=ignoreargs)

        check_or_create_signature_block(myfile, signature, verbose=verbose)

        @functools.wraps(func)
        def wrapper_cache(*args, resetcache=False, **kwargs):
            if resetcache:
                if os.path.exists(myfile):
                    if verbose:
                        print("Deleting previous cache in %s" % myfile)
                    os.remove(myfile)
                    check_or_create_signature_block(myfile, signature, verbose=verbose)

            mypath = args_to_path(argspec, *args, ignoreargs=ignoreargs, **kwargs)

            res, found = find_in_files(readfiles + [myfile], mypath, verbose=verbose)

            if not found:
                if verbose:
                    print("calculating ", mypath)

                res = func(*args, **kwargs)
                with h5py.File(myfile, "a") as f:
                    h5py_write_any(f, mypath, res)

                if verbose:
                    print("saved ", mypath)

            return res
        return wrapper_cache

    if _func is None:
        return decorator_cache
    else:
        return decorator_cache(_func)
