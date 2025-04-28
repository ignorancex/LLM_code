import os, sys, logging, collections, numpy as np
from . import logging as utils_logging
LOGGER = utils_logging.get_logger(__file__)

def rewrite(a):

    names = [f for f in a.dtype.fields]
    types = [a.dtype.fields[f][0] for f in names]
    a_new = np.empty(len(a), dtype=dict(names=names, formats=types))
    for f in names:
        a_new[f] = a[f]

    return a_new

def arr_to_rec(arr, cols, dtype=np.float64):

    arr = np.atleast_2d(arr)
    rec = np.empty(len(arr), dtype=np.dtype([(c,dtype) for c in cols]))
    for i, p in enumerate(rec.dtype.names):
        rec[p] = arr[:,i]
    return rec

def rec_to_arr(rec, cols=None, dtype=np.float64):

    if cols == None:
        cols = rec.dtype.names

    return np.array([rec[c] for c in cols], dtype=dtype).T

def arrstr(arr, fmt=': 2.3f'):

    assert arr.ndim<=2, "arrstr works with arrays up to 2 dims"
    arr = np.atleast_2d(arr)
    s = '['
    templ = '{%s}' % fmt
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            s += templ.format(val)
            if j <len(row)-1:
                s += ' '
        if i < len(arr)-1:
            s+='\n'
    s+=']'
    return s

def get_dtype(columns, main='f8', shapes=None):

    list_name = []
    list_dtype = []

    if shapes is None:
        shapes = [() for _ in columns]

    for col in columns:
        if ':' in col:
            name, dtype = col.split(':')
        else:
            name, dtype = col, main

        list_name.append(str(name))
        list_dtype.append(str(dtype))

    dtype = np.dtype(list(zip(list_name, list_dtype, shapes)))

    return dtype
    
def add_cols(rec, names, shapes=None, data=0, overwrite=False):

    # check if new data should be sliced
    slice_data = isinstance(data, np.ndarray) and data.ndim == 2

    # create new recarray
    names = [str(name) for name in names]
    new_names = list(set(names)-set(rec.dtype.names))
    extra_dtype = get_dtype(new_names, shapes=shapes)
    newdtype = np.dtype(rec.dtype.descr + extra_dtype.descr)
    newdtype = np.dtype([newdtype.descr[i] for i in np.argsort(newdtype.names)])
    newrec = np.empty(rec.shape, dtype=newdtype)

    # add data to new recarray
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
        
    for ni, name in enumerate(names):

        if not overwrite and (name in rec.dtype.names):
            continue

        if slice_data:
            newrec[name] = data[ni]
        else:
            newrec[name] = data

    return newrec
