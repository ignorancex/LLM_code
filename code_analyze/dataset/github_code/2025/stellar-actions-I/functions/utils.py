import numpy as np
import yt
import yt.units as u

hydrogenMassFraction=0.76


##derivative function- gives first and second derivative
def der(y,x):
    h = np.diff(x)
    return (-y[4:]+8*y[3:-1]-8*y[1:-3]+y[:-4])/(12*h[1:-2]),(-y[4:]+16*y[3:-1]-30*y[2:-2]+16*y[1:-3]-y[:-4])/(12*h[1:-2]**2)
def der_diff(y,x):
    return np.diff(y)/np.diff(x),np.diff(np.diff(y)/np.diff(x))/np.diff(x)[1:]**2

def fourth_order_derivatives(x, y):
    n = len(y)
    if n < 5:
        raise ValueError("Array must have at least 5 elements for 4th order accuracy.")
    
    dy_dx = np.zeros(n)
    d2y_dx2 = np.zeros(n)

    # For internal points using 4th order central differences
    for i in range(2, n-2):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        h3 = x[i+2] - x[i+1]

        dy_dx[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (6 * (h2 + h1))

        h1 = x[i-1] - x[i-2]
        h2 = x[i] - x[i-1]
        h3 = x[i+1] - x[i]
        h4 = x[i+2] - x[i+1]

        term1 = -y[i+2] / (h3 * h4 * (h3 + h4))
        term2 = 16 * y[i+1] / (h2 * h3 * (h2 + h3))
        term3 = -30 * y[i] / (h1 * h2 * (h1 + h2))
        term4 = 16 * y[i-1] / (h1 * h2 * (h1 + h2))
        term5 = -y[i-2] / (h1 * h2 * (h1 + h2))

        d2y_dx2[i] = (term1 + term2 + term3 + term4 + term5) / (12)

    # For boundary points using lower order differences
    dy_dx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy_dx[1] = (y[2] - y[0]) / (x[2] - x[0])
    dy_dx[-2] = (y[-1] - y[-3]) / (x[-1] - x[-3])
    dy_dx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    d2y_dx2[0] = (y[2] - 2*y[1] + y[0]) / (x[2] - x[0])**2
    d2y_dx2[1] = (y[3] - 2*y[2] + y[1]) / (x[3] - x[1])**2
    d2y_dx2[-2] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-3])**2
    d2y_dx2[-1] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-2])**2

    return dy_dx, d2y_dx2


def weighted_median(data, weights):
    sorted_data = np.argsort(data)
    sorted_weights = weights[sorted_data]
    sorted_data = data[sorted_data]
    
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = cumulative_weight[-1] / 2.0
    
    return sorted_data[np.where(cumulative_weight >= cutoff)[0][0]]

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def _HII(field, data):
    T = data["PartType0","temperature"].in_units('K')
    HII = data["PartType0","GrackleHII"]
    HII[(T > 10766.8) & (T<10766.9)] = 0.745
    return HII

yt.add_field(
    name=("PartType0","HII"),
    function=_HII,
    sampling_type = 'local',
    units="dimensionless"
)

def meanWeight(field, data):
    mw = 1/(hydrogenMassFraction + data["PartType0","HII"]+(1-hydrogenMassFraction)/4 + data["PartType0","GrackleHeII"]/4 + data["PartType0","GrackleHeIII"]/2)
    return mw
yt.add_field(name=("PartType0","meanWeight"), function = meanWeight, sampling_type = 'local', units="dimensionless", force_override=True)
def Tg(field, data):
    return(data["PartType0","InternalEnergy"]*data["PartType0","meanWeight"]*u.mh*2/3/u.kb)
yt.add_field(name=("PartType0","Tg"), function=Tg, sampling_type = 'local', units="K", force_override=True)

##function to get CoM of cold gas given the filename
def CoM(fileName,mean_median=None):
    """takes filename then reads it using yt, finds the temperature and then takes gas particles colder than 10^4 K and finds their CoM.
    If mean_median=0 or None, it uses mean to find centre of mass. If it is 1, weighted median is used """
    hydrogenMassFraction=0.76
    bbox_lim = 10000 #kpc
    bbox = [[-bbox_lim,bbox_lim],
            [-bbox_lim,bbox_lim],
            [-bbox_lim,bbox_lim]]
    
    unit_base = {'UnitLength_in_cm'         : 3.08568e+21, #1    kpc
                 'UnitMass_in_g'            :   1.989e+43, #1e10 solarmass
                 'UnitVelocity_in_cm_per_s' :      100000} #1    km/s
    
    ds2 = yt.load(fileName,unit_base=unit_base, bounding_box=bbox)
    ad=ds2.all_data()
    T_g = np.array(ad['PartType0','Tg'])
    C_g = np.array(ad['PartType0','Coordinates'])*1000 #pc
    M_g = np.array(ad['PartType0','Masses'])*1e10 #M_sun
    if mean_median==None or mean_median==0:
        return np.average(C_g[T_g<1e4],axis=0,weights=M_g[T_g<1e4])
    if mean_median==1:
        return np.array((weighted_median(C_g[T_g<1e4][:,0],weights=M_g[T_g<1e4]),weighted_median(C_g[T_g<1e4][:,1],weights=M_g[T_g<1e4]),weighted_median(C_g[T_g<1e4][:,2],weights=M_g[T_g<1e4])))
    
