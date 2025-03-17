
try:
    import numpy as np
except ImportError:
    print('Failed to import numpy')
    traceback.format_exc()


def check_particle_type(partype):
    """
    Check if the input particle type is correctly formatted

    Parameters
    ----------
    partype : str
    """
    
    if partype not in ['grs','CALP','CAMP','CAHP','FA19','FA15','FA13','FA11']:
        msg='Incorrect particle type. The input value is '+partype+', while available values are '\
                "['grs','CALP','CAMP','CAHP','FA19','FA15','FA13','FA11']"
        raise ValueError(msg)


def check_particle_size(partype,amon,size):
    """
    Check if the input particle size is correctly formatted

    Parameters
    ----------
    partype : str
    amon    : str (optional)
    size    : str
    """

    nl=get_sizelist(partype,amon=amon)
    # check consistency between monomer radius and number of monomers
    if size not in nl:
        msg='Incorrect particle size. The particle model that was attempted to be read is '+partype
        if partype != 'grs':
            msg=msg+' with the monomer radius of '+amon+'.'

        if not nl:
            msg=msg+'---> The input monomer radius is not available.'
        else:
            msg=msg+' Input size option is '+size+\
                    ', while available size options for this particle model is '+str(nl)
        
        raise ValueError(msg)

def check_particle_composition(comp):
    """
    Check if the input particle composition is correctly formatted

    Parameters
    ----------
    comp : str
    """

    if comp not in ['org','amc']:
        msg='Incorrect particle composition. The input composition option is '+comp+\
                ", while available values are 'org' or 'amc'"
        raise ValueError(msg)


def get_sizelist(partype,amon=None):
    """
    Get a list of available ``size`` arguments.

    Parameters
    ----------
    partype : str

    Returns
    ----------
    nlist   : list of size arguments

    Examples
    ----------
        >>> print(get_sizelist('FA19'))
        Available amon and size for FA19
        amon='100nm' | '8','16','32','64','128','256','512','1024','2048','4096'
        amon='150nm' | '8','16','32','64','128','256','512'
        amon='200nm' | '8','16','32','64','128','256','512'
        amon='300nm' | '8','16','32','64','128','256'
        amon='400nm' | '8','16','32','64','128'
        
        >>> print(get_sizelist('FA19',amon='100nm'))
        ['8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096']

        >>> print(get_sizelist('CAHP',amon='400nm'))
        ['8', '16', '32', '64']

        >>> print(get_sizelist('grs'))
        ['0_2000', '0_2520', '0_3175', '0_4000', '0_5040', '0_6350', '0_8000', '1_0079', '1_2699', '1_6000']
    """
    
    # check particle type
    check_particle_type(partype)

    #
    # output all info
    #
    if partype != 'grs' and amon==None:
        print(" Available amon and size for",partype)
        if partype in ['CALP','CAMP','CAHP']:
            print(" amon='100nm' | size='8','16','32','64','128','256','512','1024','2048','4096'")
            print(" amon='200nm' | size='8','16','32','64','128','256','512'")
            print(" amon='400nm' | size='8','16','32','64'")
        elif partype=='FA19':
            print(" amon='100nm' | '8','16','32','64','128','256','512','1024','2048','4096'")
            print(" amon='150nm' | '8','16','32','64','128','256','512'")
            print(" amon='200nm' | '8','16','32','64','128','256','512'")
            print(" amon='300nm' | '8','16','32','64','128','256'")
            print(" amon='400nm' | '8','16','32','64','128'")
        elif partype=='FA15':
            print(" amon='100nm' | '8','16','32','64','128','256','512','1024'")
            print(" amon='150nm' | '8','16','32','64','128','256'")
            print(" amon='200nm' | '8','16','32','64','128','256'")
            print(" amon='300nm' | '8','16','32','64','128''")
            print(" amon='400nm' | '8','16','32','64'")
        elif partype=='FA13':
            print(" amon='100nm' | '8','16','32','64','128','256','512'")
            print(" amon='150nm' | '8','16','32','64','128'")
            print(" amon='200nm' | '8','16','32','64','128'")
            print(" amon='300nm' | '8','16','32','64'")
            print(" amon='400nm' | '8','16','32'")
        elif partype=='FA11':
            print(" amon='100nm' | '8','16','32','64','128','256'")
            print(" amon='150nm' | '8','16','32','64'")
            print(" amon='200nm' | '8','16','32','64'")
            print(" amon='300nm' | '8','16','32'")
            print(" amon='400nm' | '8','16'")
        #exit()
        return

    # make a monomer size list
    if partype in ['CALP','CAMP','CAHP']:
        amon_list=['100nm','200nm','400nm']
    elif partype in ['FA11','FA13','FA15','FA19']:
        amon_list=['100nm','150nm','200nm','300nm','400nm']
    elif partype=='grs':
        amon=None

    # check input amon value
    if amon is not None and amon not in amon_list:
        msg='error in get_sizelist: incorrect monomer radius'\
            ' input value     = '+str(amon)+\
            ' while available values are = '+str(amon_list)
        raise ValueError(msg)

    #
    nlist=[]
    if partype=='FA11':
        if amon=='100nm':
            nlist=['8','16','32','64','128','256']
        elif amon=='150nm':
            nlist=['8','16','32','64']
        elif amon=='200nm':
            nlist=['8','16','32','64']
        elif amon=='300nm':
            nlist=['8','16','32']
        elif amon=='400nm':
            nlist=['8','16']
    
    elif partype=='FA13':
        if amon=='100nm':
            nlist=['8','16','32','64','128','256','512']
        elif amon=='150nm':
            nlist=['8','16','32','64','128']
        elif amon=='200nm':
            nlist=['8','16','32','64','128']
        elif amon=='300nm':
            nlist=['8','16','32','64']
        elif amon=='400nm':
            nlist=['8','16','32']

    elif partype=='FA15':
        if amon=='100nm':
            nlist=['8','16','32','64','128','256','512','1024']
        elif amon=='150nm':
            nlist=['8','16','32','64','128','256']
        elif amon=='200nm':
            nlist=['8','16','32','64','128','256']
        elif amon=='300nm':
            nlist=['8','16','32','64','128']
        elif amon=='400nm':
            nlist=['8','16','32','64']

    elif partype=='FA19':
        if amon=='100nm':
            nlist=['8','16','32','64','128','256','512','1024','2048','4096']
        elif amon=='150nm':
            nlist=['8','16','32','64','128','256','512']
        elif amon=='200nm':
            nlist=['8','16','32','64','128','256','512']
        elif amon=='300nm':
            nlist=['8','16','32','64','128','256']
        elif amon=='400nm':
            nlist=['8','16','32','64','128']

    elif partype=='CAHP' or partype=='CAMP' or partype=='CALP':
        if amon=='100nm':
            nlist=['8','16','32','64','128','256','512','1024','2048','4096']
        elif amon=='200nm':
            nlist=['8','16','32','64','128','256','512']
        elif amon=='400nm':
            nlist=['8','16','32','64']

    elif partype=='grs':
        nlist=['0_2000','0_2520','0_3175','0_4000','0_5040','0_6350','0_8000','1_0079','1_2699','1_6000']

    return nlist




# volume equivalent radius of grs particles in unit of microns
def grs_voleq(sizename):
    """
    Get a value of the volume-equivalent radius of GRS particles.

    Parameters
    ----------
    sizename : str
               a string to specify the size of GRS particles    

    Returns
    ----------
    av       : float
               the grain radius in units of microns
    """

    if sizename=='0_2000':
        av = 0.2000e0
    elif sizename=='0_2520':
        av = 0.2520e0
    elif sizename=='0_3175':
        av = 0.3175e0
    elif sizename=='0_4000':
        av = 0.4000e0
    elif sizename=='0_5040':
        av = 0.5040e0
    elif sizename=='0_6350':
        av = 0.6350e0
    elif sizename=='0_8000':
        av = 0.8000e0
    elif sizename=='1_0079':
        av = 1.0079e0
    elif sizename=='1_2699':
        av = 1.2699e0
    elif sizename=='1_6000':
        av = 1.6000e0
    return av


# calculate mass
def calculate_dust_mass(self):
    """
    Calculate the dust mass (or the distribution-averaged value when the size distribution is present).

    Parameters
    ----------
    qind    : float
              power-law index of the size distribution
    sizemin : str (optional)
              a string for specifying the minimum dust particle
    sizemax : str (optional)
              a string for specifying the maximum dust particle

    Returns
    ----------
    mass     : float
               dust mass in units of gram
    """

    # calculate (distribution-averaged) mass 
    mic2cm=1.e-4
    cm2mic=1.e4
    if self.dist:
        # power-law index of size distribution
        #qind=3.5    # DO NOT CHANGE !!!
        tot1 = 0.0
        tot2 = 0.0
        if self.aggop:
            imin=int(np.log2(self.npmin))
            imax=int(np.log2(self.npmax))
            for i in range(imin,imax+1):
                av = self.amon*2.0**(float(i)/3.0)
                av = av * mic2cm
                #tot1 = tot1 + av ** (4.0 - qind)
                #tot2 = tot2 + av ** (1.0 - qind)
                tot1 = tot1 + av ** (4.0 - self.pow)
                tot2 = tot2 + av ** (1.0 - self.pow)
        else:
            grs_list=get_sizelist('grs')
            #imin=grs_list.index(sizemin)
            #imax=grs_list.index(sizemax)
            imin=int(3*np.log2(self.avmin/0.2))
            imax=int(3*np.log2(self.avmax/0.2))
            for i in range(imin,imax+1):
                av = grs_voleq(grs_list[i])
                av = av * mic2cm
                #tot1 = tot1 + av ** (4.0 - qind)
                #tot2 = tot2 + av ** (1.0 - qind)
                tot1 = tot1 + av ** (4.0 - self.pow)
                tot2 = tot2 + av ** (1.0 - self.pow)

        # calculate distribution-averaged mass
        mass = 4.0*np.pi*self.rhomat*tot1/tot2/3.0
    else:   
        mass = 4.0*np.pi*self.rhomat*(self.av*mic2cm)**3.0/3.0

    return mass


# 
def get_radius_and_porosity(partype,amon,np):
    """
    Get the characteristic radius and porosity of an aggregate
    
    Examples
    ---------
        >>> ac,por=get_radius_and_porosity('FA19','100nm','4096')
        >>> print('%-21s'%"characteristic radius",'= %5.3f um'%ac)
        >>> print('%-21s'%"porosity",'= %5.2f %%'%por)
        characteristic radius = 10.04 um
        porosity              = 99.60 %

        >>> ac,por=get_radius_and_porosity('CAHP','400nm','64')
        >>> print('%-21s'%"characteristic radius",'= %5.3f um'%ac)
        >>> print('%-21s'%"porosity",'= %5.2f %%'%por)
        characteristic radius =  3.12 um
        porosity              = 86.47 %
    """

    #
    # safety checks
    #
    # check partype
    partype_list=['CALP','CAMP','CAHP','FA19','FA15','FA13','FA11']
    if partype not in partype_list:
        print('error in get_radius_and_poroty: incorrect partype type')
        print(' inputted value     = ',partype)
        print(' while available values are= ',partype_list)
        exit()

    # check amon
    amon_list=['100nm','150nm','200nm','300nm','400nm']
    if amon not in amon_list:
        print('error in get_radius_and_poroty: incorrect monomer radius')
        print(' inputted value     = ',amon)
        print(' while available values are= ',amon_list)
        exit()

    # check np
    np_list=get_sizelist(partype,amon=amon)
    if np not in np_list:
        print('error in get_radius_and_poroty: incorrect number of monomers.')
        print('        inputted value     = ',np)
        print('        while available values are= ',np_list)
        exit()

    #
    # first store aggregate radius (normalized to amon=100 nm)
    #
    if partype=='FA11':
        if np=='8':
            rc=0.51170114E+01
            po=94.03
        elif np=='16':
            rc=0.98256942E+01
            po=98.31
        elif np=='32':
            rc=0.18565017E+02
            po=99.50
        elif np=='64':
            rc=0.34922949E+02
            po=99.85
        elif np=='128':
            rc=0.65612469E+02
            po=99.95
        elif np=='256':
            rc=0.12322839E+03
            po=99.99

    elif partype=='FA13':

        if np=='8':
            rc=0.44973857E+01
            po=91.21
        elif np=='16':
            rc=0.78695278E+01
            po=96.72
        elif np=='32':
            rc=0.13530396E+02
            po=98.71
        elif np=='64':
            rc=0.23129327E+02
            po=99.48
        elif np=='128':
            rc=0.39461053E+02
            po=99.79
        elif np=='256':
            rc=0.67279401E+02
            po=99.92
        elif np=='512':
            rc=0.11468214E+03
            po=99.97

    elif partype=='FA15':

        if np=='8':
            rc=0.39191880E+01
            po=86.71
        elif np=='16':
            rc=0.64216775E+01
            po=93.96
        elif np=='32':
            rc=0.10317279E+02
            po=97.09
        elif np=='64':
            rc=0.16454757E+02
            po=98.56
        elif np=='128':
            rc=0.26168845E+02
            po=99.29
        elif np=='256':
            rc=0.41570925E+02
            po=99.64
        elif np=='512':
            rc=0.66008933E+02
            po=99.82
        elif np=='1024':
            rc=0.10479467E+03
            po=99.91

    elif partype=='FA19':

        if np=='8':
            rc=0.35921630E+01
            po=82.74
        elif np=='16':
            rc=0.55369074E+01
            po=90.57
        elif np=='32':
            rc=0.76808037E+01
            po=92.94
        elif np=='64':
            rc=0.10549461E+02
            po=94.55
        elif np=='128':
            rc=0.15686970E+02
            po=96.68
        elif np=='256':
            rc=0.21807930E+02
            po=97.53
        elif np=='512':
            rc=0.32392439E+02
            po=98.49
        elif np=='1024':
            rc=0.49487722E+02
            po=99.16
        elif np=='2048':
            rc=0.67223888E+02
            po=99.33
        elif np=='4096':
            rc=0.10043553E+03
            po=99.60

    elif partype=='CAHP':

        if np=='8':
            rc=0.32514276E+01
            po=76.73
        elif np=='16':
            rc=0.44484377E+01
            po=81.82
        elif np=='32':
            rc=0.59977666E+01
            po=85.17
        elif np=='64':
            rc=0.77906738E+01
            po=86.47
        elif np=='128':
            rc=0.98686230E+01
            po=86.68
        elif np=='256':
            rc=0.12394360E+02
            po=86.55
        elif np=='512':
            rc=0.15560679E+02
            po=86.41
        elif np=='1024':
            rc=0.19615370E+02
            po=86.43
        elif np=='2048':
            rc=0.24804457E+02
            po=86.58
        elif np=='4096':
            rc=0.31440169E+02
            po=86.82
    
    elif partype=='CAMP':

        if np=='8':
            rc=0.25707121E+01
            po=52.91
        elif np=='16':
            rc=0.34233881E+01
            po=60.12
        elif np=='32':
            rc=0.46841991E+01
            po=68.87
        elif np=='64':
            rc=0.61717829E+01
            po=72.78
        elif np=='128':
            rc=0.79544462E+01
            po=74.57
        elif np=='256':
            rc=0.10207683E+02
            po=75.93
        elif np=='512':
            rc=0.12913911E+02
            po=76.23
        elif np=='1024':
            rc=0.16479447E+02
            po=77.12
        elif np=='2048':
            rc=0.20918822E+02
            po=77.63
        elif np=='4096':
            rc=0.26428819E+02
            po=77.81
    
    elif partype=='CALP':
        if np=='8':
            rc=0.22291601E+01
            po=27.78
        elif np=='16':
            rc=0.29659565E+01
            po=38.68
        elif np=='32':
            rc=0.39164124E+01
            po=46.73
        elif np=='64':
            rc=0.52142878E+01
            po=54.86
        elif np=='128':
            rc=0.67492996E+01
            po=58.37
        elif np=='256':
            rc=0.87215615E+01
            po=61.41
        elif np=='512':
            rc=0.11219171E+02
            po=63.74
        elif np=='1024':
            rc=0.14317935E+02
            po=65.11
        elif np=='2048':
            rc=0.18286614E+02
            po=66.51
        elif np=='4096':
            rc=0.23404664E+02
            po=68.05


    # convert str --> int
    a0=int(amon[0:3])/100
    
    # 0.1 is to make rc in units of microns
    rc = rc * a0 * 0.1

    return rc,po

if __name__ == '__main__':

    print(get_sizelist('FA19'))

    print(get_sizelist('FA11','300nm'))
    print(get_sizelist('grs'))

    ac,por=get_radius_and_porosity('FA19','100nm','4096')
    print('%-21s'%"characteristic radius",'= %5.2f um'%ac)
    print('%-21s'%"porosity",'= %5.2f %%'%por)
   
    ac,por=get_radius_and_porosity('CAHP','400nm','64')
    print('%-21s'%"characteristic radius",'= %5.2f um'%ac)
    print('%-21s'%"porosity",'= %5.2f %%'%por)
