from __future__ import print_function
import numpy as np
import sys
import copy
from scipy import interpolate
import os
import errno
# tools module

def takeRatio(d0,d1,logit=False):

    print ('d1'," ",len(d1),'\n')
    print ('d0',' ',len(d0),'\n')
    dif = len(d0)-len(d1)
    d0x=d0[:,0]
    d0y=d0[:,1]
    d1x=d1[:,0]
    d1y=d1[:,1]
    if logit==True:
        d0x=np.log10(d0x)
        d0y=np.log10(d0y)
        d1x=np.log10(d1x)
        d1y=np.log10(d1x)
    if dif > 0:
        ratioDat=np.zeros((len(d1),2))
        fInterp = interpolate.pchip(d0x,d0y)   #!!!Important pchip = monotonic interpolation
        condition=d1x<=max(d0x)
        ratio=d1y[condition]/fInterp(d1x[condition])
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d1x[condition],ratio])
        print (str(dif)+' elements deleted')
    elif dif < 0:
        ratioDat=np.zeros((len(d0),2))
        fInterp = interpolate.pchip(d1x,d1y)
        condition=d0x<=max(d1x)
        print (d0x[condition],'\n',d1x)
        ratio=fInterp(d0x[condition])/d0y[condition]
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d0x[condition],ratio])
        print (str(dif)+' elements deleted')
    else:
        ratioDat=np.zeros((len(d0),2))
        fInterp = interpolate.pchip(d1x,d1y)
        condition=d0x<=max(d1x)

        ratio=fInterp(d0x[condition])/d0y[condition] #in all cases data d1 is divided through data d0
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d0x[condition],ratio])
        print ('equal size')
    #print ratioDat, '\n'
    #print darray1, '\n'
    return ratioDat

def derivatives(x,y,order=1):
    yderi=np.diff(y)/np.diff(x)
    xderi=(x[1:]+x[:-1])/2
    if order==2:
        xderi2, yderi2 = derivatives(xderi,yderi,1)
        return xderi,yderi,xderi2,yderi2
    elif order==1:
        return xderi,yderi
    else:
        print ("error, function does not return "+str(order)+"order derivatives")
        return None

def calcMaxLim(x,y,xmin,maxArra,minArra):
    arra = np.transpose([x,y])
    arra = arra[arra[:,0]>=xmin]
    maxY = max(arra[:,1])
    minY = min(arra[:,1])
    maxArra.append(maxY)
    minArra.append(minY)
    return maxArra, minArra

def mkdirp(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def fileexists(filename):
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    try:
        file_handle = os.open(filename, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            return True
        else:  # Something unexpected went wrong so reraise the exception.
            raise
    else:
        return False
