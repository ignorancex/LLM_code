import numpy as np
import ephem

def observe(data,oparams):
    cotm=Cot(**oparams)
    cotm.gc2helio(data)


def equ2gal(*args,epoch='2000'):
    al = args[0]
    de = args[1]
    al = np.radians(al) 
    de = np.radians(de)
    
    g = ephem.Galactic(0.0,np.pi/2.0,epoch=str(epoch))
    [al_gp,de_gp] = g.to_radec()
    g.from_radec(0.0,np.pi/2.0)
    l_cp = g.lon.znorm

    l = l_cp-np.arctan2(np.cos(de)*np.sin(al-al_gp),np.cos(de_gp)*np.sin(de)-np.sin(de_gp)*np.cos(de)*np.cos(al-al_gp))
    b = np.arcsin(np.sin(de_gp)*np.sin(de)+np.cos(de_gp)*np.cos(de)*np.cos(al - al_gp))

    if len(args) == 2:
        l = np.degrees(l)%360.0
        b = np.degrees(b)      
        return [l,b]
    else:    
        mua = args[2]
        mud = args[3]
    
        c1 = np.sin(de_gp)*np.cos(de)-np.cos(de_gp)*np.sin(de)*np.cos(al-al_gp)
        c2 = np.cos(de_gp)*np.sin(al-al_gp)
        cosb = np.sqrt(c1*c1+c2*c2)
        mul = (c1*mua+c2*mud)/cosb
        mub = (-c2*mua+c1*mud)/cosb   

        l = np.degrees(l)%360.0
        b = np.degrees(b)      
        return [l,b,mul,mub]


def gal2equ(*args, epoch='2000'):
    l = args[0]
    b = args[1]
    l = np.radians(l) 
    b = np.radians(b)
    
    g=ephem.Galactic(0.0,np.pi/2.0,epoch=str(epoch))
    [al_gp,de_gp]=g.to_radec()
    g.from_radec(0.0,np.pi/2.0)
    l_cp=g.lon.znorm   

    al = al_gp + np.arctan2(np.cos(b) * np.sin(l_cp - l), np.cos(de_gp) * np.sin(b) - np.sin(de_gp) * np.cos(b) * np.cos(l_cp - l))   
    de=np.arcsin(np.sin(de_gp)*np.sin(b)+np.cos(de_gp)*np.cos(b)*np.cos(l_cp - l))
   
    if len(args) == 2:
        al = np.degrees(al)%360.0    
        de = np.degrees(de)      
        return [al,de,mua,mud]
    else:
        mul = args[2]
        mub = args[3]
    
        c1=np.sin(de_gp)*np.cos(de)-np.cos(de_gp)*np.sin(de)*np.cos(al-al_gp)
        c2=np.cos(de_gp)*np.sin(al-al_gp)
        cosb=np.sqrt(c1*c1+c2*c2)
        mua=(c1*mul-c2*mub)/cosb
        mud=(c2*mul+c1*mub)/cosb   

        al = np.degrees(al)%360.0    
        de = np.degrees(de)
        return [al, de, mua, mud]
        


def ecl2equ(l, b,epoch='2000'):   
   g=ephem.Ecliptic(0.0,np.pi/2.0,epoch=str(epoch))
   [al_gp,de_gp]=g.to_radec()
   g.from_radec(0.0,np.pi/2.0)
   l_cp=g.lon.znorm   
   l = np.radians(l) 
   b = np.radians(b)
   
   al = al_gp + np.arctan2(np.cos(b) * np.sin(l_cp - l), np.cos(de_gp) * np.sin(b) - np.sin(de_gp) * np.cos(b) * np.cos(l_cp - l))   
   de=np.arcsin(np.sin(de_gp)*np.sin(b)+np.cos(de_gp)*np.cos(b)*np.cos(l_cp - l))
   de = np.degrees(de)      
   al = np.degrees(al)%360.0    
   return [al,de]


def equ2ecl(al, de,epoch='2000'):   
   g=ephem.Ecliptic(0.0,np.pi/2.0,epoch=str(epoch))
   [al_gp,de_gp]=g.to_radec()
   g.from_radec(0.0,np.pi/2.0)
   l_cp=g.lon.znorm
   al = np.radians(al) 
   de = np.radians(de)

   l=l_cp-np.arctan2(np.cos(de)*np.sin(al-al_gp),np.cos(de_gp)*np.sin(de)-np.sin(de_gp)*np.cos(de)*np.cos(al-al_gp))
   b=np.arcsin(np.sin(de_gp)*np.sin(de)+np.cos(de_gp)*np.cos(de)*np.cos(al - al_gp))
   b = np.degrees(b)      
   l=np.degrees(l)%360.0   
   return [l,b]
    

def lbr2xyz(*args):
    l=np.radians(args[0]);    b=np.radians(args[1]);    r=args[2]
    px=r*np.cos(b)*np.cos(l)
    py=r*np.cos(b)*np.sin(l)
    pz=r*np.sin(b)
    if len(args) == 3:
        return [px, py, pz]
    
    vl=np.radians(args[3]);    vb=np.radians(args[4]);    vr=args[5]
    tm_00=-np.sin(l) ; tm_01=-np.sin(b)*np.cos(l) ; tm_02= np.cos(b)*np.cos(l)
    tm_10= np.cos(l) ; tm_11=-np.sin(b)*np.sin(l) ; tm_12= np.cos(b)*np.sin(l)
    tm_20= 0.0       ; tm_21= np.cos(b)           ; tm_22= np.sin(b)
    vx=vl*tm_00+vb*tm_01+vr*tm_02
    vy=vl*tm_10+vb*tm_11+vr*tm_12
    vz=vl*tm_20+vb*tm_21+vr*tm_22
    return [px, py, pz, vx, vy, vz]

def xyz2lbr(*args):
    px=args[0];    py=args[1];    pz=args[2]
    rc2=px*px+py*py
    l=np.degrees(np.arctan2(py,px))
    b=np.degrees(np.arctan(pz/np.sqrt(rc2)))
    r1=np.sqrt(rc2+pz*pz)
    if len(args) == 3:
        return l, b, r

    vx=args[3];    vy=args[4];    vz=args[5]
    r=np.sqrt(rc2+pz*pz)
    ind=np.where(r == 0.0)[0]
    r[ind]=1.0
    px=px/r
    py=py/r
    pz=pz/r
    rc=np.sqrt(px*px+py*py)
    ind=np.where(rc == 0.0)[0]
    rc[ind]=1.0
    tm_00=-py/rc; tm_01=-pz*px/rc; tm_02= rc*px/rc
    tm_10= px/rc; tm_11=-pz*py/rc; tm_12= rc*py/rc
    tm_20= 0.0  ; tm_21= rc  ; tm_22= pz
    vl=(vx*tm_00+vy*tm_10+vz*tm_20)
    vb=(vx*tm_01+vy*tm_11+vz*tm_21)
    vr=(vx*tm_02+vy*tm_12+vz*tm_22)
    return [l, b, r1, vl, vb, vr]

    
def lzr2xyz(*args):
    l=args[0]; pz=args[1]; r=args[2]
    l=np.radians(l)
    px=r*np.cos(l)
    py=r*np.sin(l)
    pz=pz*1.0
    if len(args) == 3:
        return [px, py, pz]

    vl=args[3]; vz=args[4]; vr=args[5]
    tm_00=-np.sin(l) ; tm_02= np.cos(l)
    tm_10= np.cos(l) ; tm_12= np.sin(l)
    vx=vl*tm_00+vr*tm_02
    vy=vl*tm_10+vr*tm_12
    vz=vz*1.0
    return [px, py, pz, vx, vy,vz]


def xyz2lzr(*args):
    px=args[0];    py=args[1];    pz=args[2]    
    l=np.degrees(np.arctan2(py,px))
    z=np.array(pz)
    rc1=np.sqrt(px*px+py*py)
    if len(args) == 3:
        return [l,z,rc1]
    
    vx=args[3];    vy=args[4];    vz=args[5]    
    r=np.sqrt(px*px+py*py+pz*pz)
    ind=np.where(r == 0.0)[0]
    r[ind]=1.0
    px=px/r
    py=py/r
    pz=pz/r
    rc=np.sqrt(px*px+py*py)
    ind=np.where(rc == 0.0)[0]
    rc[ind]=1.0
    tm_00=-py/rc; tm_01=0.0; tm_02= px/rc
    tm_10= px/rc; tm_11=0.0; tm_12= py/rc
    tm_20= 0.0  ; tm_21=1.0; tm_22= 0.0
    vl=(vx*tm_00+vy*tm_10+vz*tm_20)
    vz=(vx*tm_01+vy*tm_11+vz*tm_21)
    vr=(vx*tm_02+vy*tm_12+vz*tm_22)
    return [l, z, rc1, vl, vz, vr]


def rotate(th,x,y):
    """
    Transformation of coordinates under rotation of coordinate sysytem 
    by an angle th.
    Alternatively, new coordinates after rotation the body defined by (x,y) 
    by an angle th about the existing coordinate system.
    Parameters
    ----------
    th- angle in degree
    (x,y)- coordinatesto be transformed 

    Returns
    ----------
    (x,y) - coordinates of (x,y) in new rotated coordinate system.

    """
    x1=x*1.0 
    y1=y*1.0
    th=np.radians(th)
    x=x1*np.cos(th)+y1*np.sin(th)
    y=-x1*np.sin(th)+y1*np.cos(th)
    return [x,y]


class Cot():
    def  __init__(self,theta_zx=0.0,theta_yz=0.0,l=0.0,b=0.0,distance=8.0):
        """
        Transform coordinate of a  galaxies from internal 
        reference frame to observational and vice versa. 
        We assume the center of the new Galaxy to be at (l, b, s). Following 
        operations are done to move and orient the Galaxy to the desired 
        location. 
        rotate_zx(-theta_zx), rotate_yz(-theta_yz) translate_x(distance), 
        rotate_zx(b), rotate_xy(-l) 

        Parameters
        ------------
        theta_zx- inclination angle [degree], rotate around y axis z to x
        theta_yz- orientation angle [degree], rotate around x axis y to z
        l- Galactic long [degree]
        b- Galactic lat [degree]
        distance-  [kpc]

        """
        self.glon=l
        self.glat=b
        self.distance=distance
        self.theta_zx=theta_zx
        self.theta_yz=theta_yz
            
    def helio2gc(self,data):
        """
        Transform from observers reference frame to a galaxies internal 
        reference.  Without any loss of genrality 
        
        Parameters
        ----------
        data- dict of np arrays containing data to be transformed    

        Returns
        -------
        The data is modified inplace. 
        (lgc,pzgc,rcgc,vlgc,vzgc,vrcgc)
        """
        d=data
       
        if 'px' not in d:
            print(d.keys())
            if ('ra' in d) and ('l' not in d):
                if 'pmra' in d:
                    d['l'],d['b'],d['vl'],d['vb']=equ2gal(d['ra'],d['dec'],d['pmra'],d['pmdec'])
                    d['vl']=d['vl']*4.74e3*d['r']
                    d['vb']=d['vb']*4.74e3*d['r']
                else:
                    d['l'],d['b']=equ2gal(d['ra'],d['dec'])
           
            if 'l' in d:
                if 'vl' in d:
                    d['px'],d['py'],d['pz'],d['vx'],d['vy'],d['vz']=lbr2xyz(d['l'],d['b'],d['r'],d['vl'],d['vb'],d['vr'])
                else:
                    d['px'],d['py'],d['pz']=lbr2xyz(d['l'],d['b'],d['r'])
            else:
                raise RuntimeError('should have glon glat')
           
        if self.glon != 0.0:
            d['px'],d['py']=rotate(self.glon,d['px'],d['py'])
            if 'vx' in d:
                d['vx'],d['vy']=rotate(self.glon,d['vx'],d['vy'])
        if self.glat != 0.0:
            d['pz'],d['px']=rotate(-self.glat,d['pz'],d['px'])
            if 'vx' in d:
                d['vz'],d['vx']=rotate(-self.glat,d['vz'],d['vx'])
           
           
        d['px']=d['px']-self.distance
        if self.theta_yz != 0.0: 
            d['py'],d['pz']=rotate(self.theta_yz,d['py'],d['pz'])
            if 'vz' in d:
                d['vy'],d['vz']=rotate(self.theta_yz,d['vy'],d['vz'])
                
        if self.theta_zx != 0.0: 
            d['pz'],d['px']=rotate(self.theta_zx,d['pz'],d['px'])
            if 'vz' in d:
                d['vz'],d['vx']=rotate(self.theta_zx,d['vz'],d['vx'])
           
        if 'l' in d:
            if 'vz' in d:
                d['lgc'],d['pzgc'],d['rcgc'],d['vlgc'],d['vzgc'],d['vrcgc']=Cot.xyz2lzr(d['px'],d['py'],d['pz'],d['vx'],d['vy'],d['vz'])
            else:
                d['lgc'],d['pzgc'],d['rcgc']=xyz2lzr(d['px'],d['py'],d['pz'])
           
       
    def gc2helio(self, data):
       """
       Transform from a galaxies internal reference frame to observers 
       reference frame. Without any loss of genrality we 
       assume the center of the new Galaxy to be at (s,0,0). Theta is the 
       inclination of the z axis of the galaxy with that of observers.

       Rotate around y axis by angle theta (z to x) a Galaxy, then  
       translate in x by distance s.
       
       Parameters
       ----------
       data- dict of np arrays containing data to be transformed    
       should contain ['lgc','pzgc','rcgc'] and optionally 
       ['vlgc','vzgc','vrcgc']
       Returns
       -------
       The data is modified inplace. 
       (l, b, r, vl, vb, vr)
       (ra, dec, pmra, pmdec)
       """
       d=data
       if 'lgc' in d:
           if 'vlgc' in d:
               d['px'],d['py'],d['pz'],d['vx'],d['vy'],d['vz']=lzr2xyz(d['lgc'],d['pzgc'],d['rcgc'],d['vlgc'],d['vzgc'],d['vrcgc'])
           else:
               d['px'],d['py'],d['pz']=lzr2xyz(d['lgc'],d['pzgc'],d['rcgc'])
       else:
           raise RuntimeError('should have lgc ')

       if self.theta_zx != 0.0: 
           d['pz'],d['px']=rotate(-self.theta_zx,d['pz'],d['px'])
           if 'vz' in d:
               d['vz'],d['vx']=rotate(-self.theta_zx,d['vz'],d['vx'])
       if self.theta_yz != 0.0: 
           d['py'],d['pz']=rotate(-self.theta_yz,d['py'],d['pz'])
           if 'vz' in d:
               d['vy'],d['vz']=rotate(-self.theta_yz,d['vy'],d['vz'])
       d['px']=d['px']+self.distance

       if self.glat != 0.0:
           d['pz'],d['px']=rotate(self.glat,d['pz'],d['px'])
           if 'vx' in d:
               d['vz'],d['vx']=rotate(self.glat,d['vz'],d['vx'])
           
       if self.glon != 0.0:
           d['px'],d['py']=rotate(-self.glon,d['px'],d['py'])
           if 'vx' in d:
               d['vx'],d['vy']=rotate(-self.glon,d['vx'],d['vy'])
       
       if 'vz' in d:
           d['l'],d['b'],d['r'],d['vl'],d['vb'],d['vr']=xyz2lbr(d['px'],d['py'],d['pz'],d['vx'],d['vy'],d['vz'])
           d['ra'],d['dec'],d['pmra'],d['pmdec']=gal2equ(d['l'],d['b'],d['vl']/(4.74e3*d['r']),d['vb']/(4.74e3*d['r']))
       else:
           d['l'],d['b'],d['r']=xyz2lbr(d['px'],d['py'],d['pz'])
           d['ra'],d['dec']=gal2equ(d['l'],d['b'])


           
