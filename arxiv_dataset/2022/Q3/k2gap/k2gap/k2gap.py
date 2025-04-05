#Copyright (c) 2021 Sanjib Sharma
"""
A module for K2GAP target selection function


Example:

import k2gap
insurvey=k2gap.sf(cno,ra,dec,jmag,hmag,kmag)
or
insurvey=k2gap.sf(cno,ra,dec,jmag,hmag,kmag,simulate_onsilicon=True)

"""


import numpy as np
import json
import os

def read_json(filename):
    d=None
    with open(filename,'r') as infile:
        dt=json.load(infile)
        d={key:np.array(dt[key]) for key in dt}
    return d

def jk2vmag(j,ks):
    """
    Convert J and K band mag to V mag
    """
    return ks+2.0*((j-ks)+0.14)+0.382*np.exp((j-ks-0.2)*2)

def lbr2xyz(l,b,r=1.0):
    """
    Convert (longitude, latitude, radius) to cartesian coordinates (x,y,z)
    Arguments
    ---------
    l : (float) array_like 
        Longitude [degree]

    b : (float) array_like 
        Latitude [degree]

    r : (float) array_like 
        Radius [arbitrary units]

    """
    l=np.radians(l)
    b=np.radians(b)
    return [r*np.cos(b)*np.cos(l),r*np.cos(b)*np.sin(l),r*np.sin(b)]

def angsep(l1,b1,l2,b2):
    """
    Angular distance between points (l1, b1) and (l2, b2) on a sphere

    Arguments
    ---------
    l1 : (float) array_like 
        Longitude [degree]
       
    b1 : (float) array_like 
        Latitude [degree]

    l2 : (float) array_like 
        Longitude [degree]
       
    b2 : (float) array_like 
        Latitude [degree]

    Returns d: (float) array_like 
        Angular distance [degree]  
    
    """
    x1,y1,z1=lbr2xyz(l1,b1)
    x2,y2,z2=lbr2xyz(l2,b2)
    return np.degrees(2*np.arcsin(np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)/2.0))

def circ_id(cno,ra,dec,radius=1.0):
    """

    """
    cno=np.zeros(ra.size,dtype=np.int64)+cno
    circ_id=np.zeros(ra.size,dtype=np.float64)-1
    circ_sep=np.zeros(ra.size,dtype=np.float64)+radius
    datafile=os.path.join(os.path.dirname(__file__),'k2circles.json')    
    fields=read_json(datafile)
    for cno1 in np.unique(cno):
        indd=np.where(cno==cno1)[0]
        if (cno1 >= 0)and(cno1 < 21):
            indf=np.where(fields['circ_id']//100==cno1)[0]
            for i in indf:
                fsep=angsep(ra[indd],dec[indd],fields['ra'][i],fields['dec'][i])
                ind=np.where(fsep<circ_sep[indd])[0]
                circ_id[indd[ind]]=fields['circ_id'][i]
                circ_sep[indd[ind]]=fsep[ind]
    ind=np.where((cno<0)|(cno>20))[0]
    if ind.size> 0:
        circ_id[ind]=8    
    return circ_id

def sf(cno,ra,dec,jmag,hmag,kmag,radius=1.75,simulate_onsilicon=False):
    """
    Implements the K2GAP selection function. For a set of observables x 
    returns the function y(x) which is True if it satisfies the selection 
    criteria and False otherwise.

    Arguments
    ---------
    cno : (int) array_like 
        campaign number

    ra : (float) array_like 
        Right ascension [degree]

    dec : [float] array_like
        Declination  [degree]

    jmag : [float] array_like
        2MASS J band magnitude 

    hmag : [float] array_like
        2MASS H band magnitude 

    kmag : [float] array_like
        2MASS Ks band magnitude 

    simulate_onsilicon : [boolean], optional keyword
        If True it emulates the effect of 
        K2fov.K2onSilicon.onSiliconCheck (https://github.com/KeplerGO/K2fov) 
        by setting radius=1.4. The actual computation of onsilicion test 
        is computationally expensive. This is useful when working with mock 
        data. See description of keyword radius for more details.

    radius : [float] array_like, optional keyword
        The distance of a star in [degree] from the center of its nearest 
        ccd module. The field of view of each campaign is divided up into 
        21 ccd modules with gaps between them. Circles defined by this radius and 
        centered on each module are non-overalping till a radius of 1.6 degrees.
        Use radius=1.75 to encapsulate the whole ccd module when working with 
        data for which K2fov.K2onSilicon.onSiliconCheck (https://github.com/KeplerGO/K2fov) 
        has been applied, e.g. observational data from K2. 
        Use radius=1.4, when working with 
        data for which K2fov.K2onSilicon.onSiliconCheck has not been applied, 
        e.g. a mock surveys. This has an area similar to that of a ccd module. 
        So using radius=1.4 is effectively similar to using radius=1.75 
        but with K2fov.K2onSilicon.onSiliconCheck.
        This is to avoid running K2fov.K2onSilicon.onSiliconCheck which is slow. 
        Note, for radius<=1.75 the four guiding ccds outside the science field of view 
        are not included in any of the circles.

    Returns
    -------
    y : (boolean) array_like
        True if selection function is satisfied otherwise False   

    """
    
    cno1=cno; ra1=ra;    dec1=dec;    jmag1=jmag;    hmag1=hmag; kmag1=kmag
    if simulate_onsilicon:
        radius=1.4
    
    cno1=np.zeros(ra1.size,dtype=np.int64)+cno1
    col_jk1=jmag1-kmag1
    vmag1=jk2vmag(jmag1,kmag1)
    ccd1=circ_id(cno1,ra1,dec1,radius=1.0)%100
    ccd1[ccd1>20]=-1
    ccd2=circ_id(cno1,ra1,dec1,radius=radius)%100
    ccd2[ccd2>20]=-1


    result1=np.zeros(ra1.size,dtype=np.bool)

    for cno in np.unique(cno1):
        indd=np.where(cno1==cno)[0]
        ra=ra1[indd]
        dec=dec1[indd]
        jmag=jmag1[indd]
        hmag=hmag1[indd]
        kmag=kmag1[indd]
        col_jk=col_jk1[indd]
        vmag=vmag1[indd]
        ccd=ccd1[indd]
        ccdb=ccd2[indd]

        if cno == 1:
            result=(col_jk >= 0.5)&(hmag>=7.0)&(hmag<12.927)&(ccdb>=0)
        elif cno ==2:
            ccdset=[17,12,6,14,10]
            result=(col_jk >= 0.5)&(hmag>=7.0)&(hmag<11.5)&(np.isin(ccd,ccdset))
        elif cno ==3:
            temp1=(col_jk >= 0.5)&(hmag>=7.0)&(hmag<12.0)&(ccd>=0)
            temp2=(col_jk >= 0.5)&(hmag>=7.0)&(hmag<10.929)&(ccd<0)
            result=(temp1|temp2)
        elif cno ==4:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<13.447)&(ccdb>=0)
        elif cno ==5:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccdb>=0)
        elif cno ==6:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccdb>=0)
        elif cno ==7:
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccd==17)
            temp2=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccd==6)
            temp3=(col_jk >= 0.5)&(vmag>=14.276)&(vmag<14.5)&(ccd==14)
            result=(temp1|temp2|temp3)
        elif cno == 8:
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccdb>=0)
            temp2=(col_jk >= 0.5)&(col_jk < 0.7)&(vmag>=14.5)&(vmag<14.580)&(ccdb>=0)
            result=(temp1|temp2)
        elif cno == 10:
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccdb>=0)&(ccdb!=1)
            temp2=(col_jk >= 0.5)&(col_jk < 0.7)&(vmag>=14.5)&(vmag<15.577)&(ccdb>=0)&(ccdb!=1)
            result=(temp1|temp2)
        elif cno == 11:
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccd==3)
            temp2=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccd==2)
            temp3=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.175)&(ccd==8)
            result=(temp1|temp2|temp3)
        elif cno == 12:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<16.0)&(ccdb>=0)&(ccdb!=1)
        elif cno == 13:
            ccdset=[0,13,7,9,4,12]
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccd==3)
            temp2=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(ccd==8)
            temp3=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.0)&(np.isin(ccd,ccdset))
            temp4=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<12.818)&(ccd==14)
            result=(temp1|temp2|temp3|temp4)            
        elif cno == 14:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccdb>=0)&(ccdb!=1)
        elif cno == 15:
            ccdset=[13,7,9,4,12,14,2]
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.5)&(ccd==3)
            temp2=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccd==8)
            temp3=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<14.5)&(np.isin(ccd,ccdset))
            temp4=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<13.838)&(ccd==5)
            result=(temp1|temp2|temp3|temp4)            
        elif cno == 16:
            result=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<15.0)&(ccdb>=0)&(ccdb!=1)
        elif cno == 17:
            temp1=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<16.0)&((ccd==8)|(ccd==3))
            temp2=(col_jk >= 0.5)&(vmag>=9.0)&(vmag<12.414)&(ccd!=8)&(ccd!=3)&(ccdb!=1)
            result=(temp1|temp2)
        elif cno == 18:
            result=np.zeros(vmag.size,dtype=np.bool)
        elif cno == 19:
            result=(col_jk >= 0.5)&(vmag>=9)&(vmag<14.8)&(ccdb>=0)&(ccdb!=1)
        else:
            result=np.zeros(vmag.size,dtype=np.bool)
        result1[indd]=result&(ccdb>=0)&(jmag<15)
    return result1


