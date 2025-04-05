import numpy as np
import scipy.stats

"""
    This function takes a tuple that describes how many bins there are in the 2D data image.
    It then returns a 2D array where each element corresponds to a bin in the image and contains a list of the vertices of that bin, eg:
    [
        [
            [ <-- Each of these elements corresponds to a bin
                [0.0,0.0], <-- Each of these elements is a vertex
                [0.5,0.0],
                [0.0,0.5],
                [0.5,0.5]
            ], ...
        ], ...
    ]
    The other two results of this function are a list of the bin edges in the x-axis and the y-axis respectively.
"""
def create_image_coords(image_shape):
    # We choose the bins to be linearly spaced between zero and one.
    # This means the coordinates of the image go from zero to one in each axis.
    x_edges, y_edges = np.linspace(0, 1, image_shape[0]+1), np.linspace(0, 1, image_shape[1]+1)
    # This creates a list of 2D arrays where each element of each array is an edge of a bin. 
    # The first 2D array in this list are the bin edges in the x-axis, the second is the bin edges in the y-axis.
    edges = np.meshgrid(x_edges, y_edges, indexing='ij')
    # Stacking these bin edges along the last axis (axis=-1) gives a 2D array where each element is a vertex of the bins.
    edges = np.stack(edges, axis=-1)
    # Each bin has 4 vertices. For bin (0,0) these vertices can be found at (0,0), (0, 1), (1, 0), and (1, 1) in the array.
    # So for each bin we pull out these four vertices, and then stack them to create a new array that contains all vertices for all bins.
    edges_array = np.stack([edges[:-1,:-1], edges[:-1, 1:], edges[1:, :-1], edges[1:, 1:]], axis=-1)
    return edges_array, x_edges, y_edges

"""
    This function creates an "effective area function" that is centered at 'loc' with shape defined by the covariance matrix 'cov'.
    The "effective area function" itself takes a coordinate 'coords' and returns the effective area at 'coords'.
    The value of the "effective area function" is defined so that it is equal to one at 'loc':  eff_area(loc) = 1.
"""
def create_eff_area_fuction(loc, cov):
    def eff_area(coords):
        return scipy.stats.multivariate_normal.pdf(coords, loc, cov)/scipy.stats.multivariate_normal.pdf(loc, loc, cov)
    return eff_area

# def create_eff_area_map(image_coords, eff_area):
#     return eff_area(np.array(image_coords))


"""
    This function creates a "point spread function" (psf) that has a shape defined by the covariance matrix 'cov'.
    This function returns two functions: the psf itself, and a function that draws random samples from the psf.
"""
def psf_function(cov):
    # This function defines the PSF. using the supplied covariance matrix 'cov', it creates a multivariate normal distribution centered on 'source_loc'.
    # The 'source_loc' parameter should be the location of the point-source, while 'x' should be the location of the photon/event detection.
    # This function returns a probability density at the location 'x'.
    def psf(x, source_loc):
        return scipy.stats.multivariate_normal.pdf(x, source_loc, cov)
    # This function will generate photons/events according to the psf centered on the location provided by 'source_loc'.
    def psf_draw(source_loc, **kw_args):
        return np.random.multivariate_normal(source_loc, cov, **kw_args)
    return psf, psf_draw

"""
    This function creates a "template" -- the spatial probability distribution of point-sources -- using a list of modes.
    The list of modes is provided by the parameter 'modes', and has the following format: each element is a tuple of relative normalisation, mean, and covariance matrix.
    Eg:
    modes = [
        (0.2, [0.5, 0.5], [[1.0, 0], [0, 1.0]]), ...
    ]
    The mean and covariance matrix are used to construct a normal distribution, so that we get one normal distribution for each element of the list 'modes'.
    Then these distributions are "mixed" together by multiplying each by its respective relative normalisation and adding them together. 
    The result is that the template can be defined by the sum (in latex notation)

        template(x) = \sum_i^len(modes) modes[i, 0] p(x | modes[i, 1], modes[i, 2])

    where p(x | m, c) is a normal distribution with mean 'm' and covariance matrix 'c'. As such, the relative normalisations must sum to 1.
    The function creates and returns two subfunctions: one defines the template distribution itself, the other draws random source locations from the template.
"""
def create_template_function(modes):
    # This function computes the probaility density of a source being location at location 'x'.
    def template(x):
        # Here we unpack the relative normalisation 'norm', the mean 'loc', and covariance matrix 'cov' from the list of modes.
        # We then create a new list, where each element is the probability density of that specific mode multiplied by the relative normalisation.
        distributions = [ 
            norm*scipy.stats.multivariate_normal.pdf(x, loc, cov) 
            for norm, loc, cov in modes ]
        # Finally, we sum all these elements together 
        return np.sum(distributions, axis=0)

    # To draw elements from the template, we need to do a little post-processing first.
    coeffs, locs, covs = map(np.array, zip(*modes)) # In this line, we take the transpose of 'modes'. This means that instead of having a list of tuples, we now have a tuple of lists. 'coeffs' just contains the relative normalisations, 'locs' just contains the means, and 'covs' just contains the covariance matrices.
    
    # if we are given just numbers instead of full matrices for the covariance matrices, then "promote" them to full matrices by multiplying by the 2x2 indentity matrix
    if len(covs.shape) == 1:
        covs = np.array([[1, 0], [0, 1]])[None, :, :] * covs[:, None, None]

    def template_draw(size=None):
        # To draw a random sample from a mixture, we must first select one element of the mixture at random.
        # The probabilities of each element are given by the relative normalisations, we we could draw a random element like so
        #   > mode = np.random.choice(len(modes), p=coeffs)
        # Then we could draw the location within the template using the mean and covariance matrix of that mode:
        #   > x = np.random.multivariate_normal(locs[mode], covs[mode])
        # However, this would give us only one sample. If we wanted more, we could do the above in a loop, but there is a more efficience way below:
        # First, the number of samples we want is defined by 'size'. Drawing multiple samples from a series of discrete choices is equivalent to a multinomial distribution,
        # so we can draw samples from this multinomial distribution with probabilties defined by the relative normalisations.
        # We end up with an array 'mode_cnts' where 'mode_cnts[i]' is the number of samples we need to draw from mode 'i'.
        mode_cnts = np.random.multinomial(size if size else 1, pvals=coeffs)
        # Now we have how many samples we need to draw for each mode, we loop over each mode and construct a list.
        mode_samples = [ # This loop goes over mode number 'i' and also the number of samples for the that mode 'n'.
            np.random.multivariate_normal(locs[i], covs[i], size=n) # We then draw 'n' samples from the normal distribution for that mode.
            for i, n in enumerate(mode_cnts) ]
        # Finally, we pack all these samples together so that we end up with just a single array of samples.
        return np.concatenate(mode_samples, axis=0)
    return template, template_draw


# def create_template_map(image_coords, template):
#     template_map = template(np.array(image_coords))
#     return template_map 

"""
    This function is intended to work like np.digitize, but for 2D bin coordinates, rather than just 1D as in the case of np.digitize.
    It takes a tuple 'image_coords' that contains the bin edges, then it digitises the locations in the parameter 'locs' to find which bins they land in.
"""
def digitise_map(image_coords, locs):
    map_edges, x_edges, y_edges = image_coords # Unpack the tuple to get the bin edges in the x and y axes.
    x_idx = np.digitize(locs[0], x_edges) - 1 # Digitise the x-axis of the coordinates to find the x-axis bin index.
    y_idx = np.digitize(locs[1], y_edges) - 1 # And do the same for the y-axis.
    # np.digitize returns 0 or len(edges_array) if the coordinates lie outside the binning,
    # after subtracting one, this is -1 or len(edges_array) - 1, so we create a mask to
    # filter out any coordinates that lie outside the chosen binning.
    x_mask = np.logical_and(x_idx >= 0, x_idx < len(x_edges) - 1) 
    y_mask = np.logical_and(y_idx >= 0, y_idx < len(y_edges) - 1)
    mask = np.logical_and(x_mask, y_mask)
    # We do a little bookkeeping here depending on if 'locs' is one coordinate or many.
    if len(np.array(locs).shape) == 1:
        # If 'locs' is one coodinate, then 'mask' tells us if this coordinate lies inside the binning or not.
        if mask:
            # If it does, we return the bin location
            return (x_idx, y_idx)
        else:
            # Otherwise, we turn None to signal no valid bin was found.
            return None
    else:
        # If 'locs' contains many coordinates, then we stack the x and y axis bin locations into one array, and then filter it using 'mask'.
        # This means the array could be empty if no coordinates fell inside the bins.
        return np.stack([x_idx, y_idx], axis=-1)[mask]


"""
    Here we calculate mu(epsilon) in one of three ways: in this case by direct integration.
    the function mu(epsilon) is defined by

        mu(eps) = \integrate dx T(x) delta( eps - kappa(x) \integrate_bin dy eta(y) psf(y | x)  )

    where '\integrate dx' integrates over the whole space of source coordinates, 'x'; 'T(x)' is the template at source location 'x'; delta( ... ) is the direct delta distribution;
    'kappa(x)' is the effective area at source location 'x'; '\integrate_bin dy' integrates over the extent of a single bin using photon/event locations 'y'; 
    'eta(y)' is the detection probability at event location 'y'; and 'psf(y | x)' is the probability density of detection an event/photon at location 'y' given a source at location 'x'.

    Here we make the simplifying assumption that eta(y) = 1 everywhere. 
    Now, if we want to evaluate this function numerically, we will need to replace the integrations by sums, and this also means replacing the dirac delta as well. 
    Let's start with the dirac delta. We will be moving from a continous function 'mu(eps)' to a binned, discrete function 'mu(i)'.
    Let us denote 'mu(i)' as the i-th bin of mu, and we say that a value, 'eps', lies in the bin for 'mu(i)' if eps \in E_i, where E_i is a set that defines bin i.
    Eg, if bin i has bin edges of eps_edges[i] and eps_edges[i+1] then E_i = [eps_edges[i], eps_edges[i+1]] and so 'eps \in E_i' == 'eps_edges[i] <= eps <= eps_edges[i+1]'.
    With this language, we can define the "indicator function"
    
        ind(i; v) = 1 if (v \in E_i) else 0

    that is, if the value 'v' lies in bin E_i, then ind(i; v) = 1 otherwise it is zero. We can use this to discretise the dirac delta function:

        mu(i) = \integrate dx T(x) ind(i; kappa(x) \integrate_bin dy eta(y) psf(y | x)  )

    Now, lets move to a sum over 'x':

        mu(i) = \sum_j dx_j T(x_j) ind(i; kappa(x_j) \integrate_bin dy eta(y) psf(y | x_j)  )

    where 'j' is the index of spatial coordinates, 'dx_j' is the width between spatial coodinates, and 'x_j' is the coordinates of index j.
    What this expression is telling us, is that we need to loop over the spatial coordaintes, then we calcuate the value of the indicator function and add it to the appropriate bin weighted by 'T(x_j)', eg (in psuodocode)

        mu = zeros(number of eps bins)
        for j in range(number of spatial coodinates):
            v = kappa(x_j) \integrate_bin dy eta(y) psf(y | x_j) 
            i = digitize(v, eps_bin_edges)
            mu[i] += T(x_j) * dx_j

    From here, we just need to replace the other integral by a sum:

        v = kappa(x_j) \sum_k dy_k eta(y_k) psf(y_k | x_j) 

    This just means we need a sum:

        mu = zeros(number of eps bins)
        for j in range(number of spatial coodinates):
            v = kappa(x_j) * sum( [dy_k * eta(y_k) * psf(y_k | x_j) for k in range(number of coordinates in bin)] ) 
            i = digitize(v, eps_bin_edges)
            mu[i] += T(x_j) * dx_j

    This gives us mu(eps) for just one bin. To calculate mu(eps) for all bins, we just need one more loop over bins using bin index 'b':

        mu_map = zeros(number of bins, number of eps bins)
        for b in range(number of bins):
            for j in range(number of spatial coodinates):
                v = kappa(x_j) * sum( [dy_k * eta(y_k) * psf(y_k | x_j) for k in range(number of coordinates in bin b)] ) 
                i = digitize(v, eps_bin_edges)
                mu[b, i] += T(x_j) * dx_j

    Here we have two loops over the somewhat poorly defined 'number of spatial coordinates'. In this specific function, we'll sample coordinates randomly within a bin
    (we could instead create a grid within each bin of coordinates and use that, but random locations reduce aliasing effects).
"""
def integrate_mu_eps(image_coords, # 'image_coords' is a tuple that contains the bin edges for our 2D image 
                     template, # this is a function that describes the template 'T(x)'
                     eff_area, # this is a function that desribes the effective area 'kappa(x)'
                     psf, # this is a function describes the psf such that psf(y, x) == 'psf(y | x)'
                     eps_bin_edges): # this is an array that gives the bin edges for our binned mu(epsilon)
    map_edges, x_edges, y_edges = image_coords
    flat_edges = flatten_map(map_edges)

    N_samples = 40 # This is the number of coordinates we will randomly sample within each bin to produce the spatial coordinates.

    flat_eps_map = []

    for dest_bin_edges in flat_edges: # Here we loop over bins, this essentially corresponds to 'for b in range(number of bins):' above.
        # Now we randomly sample locations uniformly from within the bin.
        dest_bin_extents_x = dest_bin_edges[0, 0], dest_bin_edges[0, 3]
        dest_bin_extents_y = dest_bin_edges[1, 0], dest_bin_edges[1, 3]
        antialias_samples_dest = np.stack([
            np.random.uniform(*dest_bin_extents_x, size=N_samples), # x values
            np.random.uniform(*dest_bin_extents_y, size=N_samples) # y values
        ], axis=-1)
        # The locations are not contained within 'antialias_samples_dest'

        # And here we calculate the bin size. Because each bin contains N_samples coordinates, we will use dy_k = dest_bin_size/N_samples as the "effective" coordinate spacing.
        dest_bin_size = (dest_bin_extents_x[1] - dest_bin_extents_x[0])*(dest_bin_extents_y[1] - dest_bin_extents_y[0])

        eps_ary = np.zeros(len(eps_bin_edges)-1) # This is mu(i)
        
        # Here is where we would loop over 'x_i', the spatial coordinates for the template.
        # What we will actually do, is loop over the bin edges and then randomly sample locations uniformly within them to generate the 'x_i' values on the fly.
        for source_bin_edges in flat_edges:
            # Now we randomly sample locations uniformly from within the bin.
            source_bin_extents_x = source_bin_edges[0, 0], source_bin_edges[0, 3]
            source_bin_extents_y = source_bin_edges[1, 0], source_bin_edges[1, 3]
            antialias_samples_source = np.stack([
                np.random.uniform(*source_bin_extents_x, size=N_samples), # x values
                np.random.uniform(*source_bin_extents_y, size=N_samples) # y values
            ], axis=-1)
            # The x_j locations are contained in 'antialias_samples_source'

            # Again, we need to bin size, and again we will use dx_i = source_bin_size/N_samples
            source_bin_size = (source_bin_extents_x[1] - source_bin_extents_x[0])*(source_bin_extents_y[1] - source_bin_extents_y[0])

            # Now we loop over our spatial source locations, this is the equivalent to the 'for j in range(number of spatial coodinates):' above.
            for sample in antialias_samples_source:
                # if source_bin_extents_x[1] == dest_bin_extents_x[1] and source_bin_extents_y[1] == dest_bin_extents_y[1]:
                #     print(np.sum(psf(antialias_samples_dest, sample))*dest_bin_size/N_samples)
                # We calcuate the 'v', the value of eps that we will bin
                v = eff_area(sample) * np.sum(psf(antialias_samples_dest, sample))*dest_bin_size/N_samples
                # While weight is 'T(x_i) * dx_i'.
                weight = template(sample)*source_bin_size/N_samples
                # finally, we locate the bin that 'v' lies in (if it exists) and we add the weight to it.
                eps_bin_idx = np.digitize(v, eps_bin_edges) - 1
                if eps_bin_idx >= 0 and eps_bin_idx < len(eps_bin_edges) - 1:
                    eps_ary[eps_bin_idx] += weight

        flat_eps_map.append(eps_ary)

    d_eps_map = np.array(flat_eps_map)
    return d_eps_map.reshape(map_edges.shape[0:2] + (d_eps_map.shape[1],))


"""
    In the previous function we saw that we could write the binned mu(epsilon) function as
    
        mu(i) = \integrate dx T(x) ind(i; kappa(x) \integrate_bin dy eta(y) psf(y | x)  )

    In this function, we will compute mu(eps) through monte-carlo sampling.
    If we draw random samples, x_0, x_1, ... etc from T(x), then we can approximate it using

        T(x) = \sum_j delta(x - x_j) / N

    where 'N' is the number of random samples.

    Lets substitute this approximation into our binned mu(epsilon):

        mu(i) = \sum_j ind(i; kappa(x_j) \integrate_bin dy eta(y) psf(y | x_j)  ) / N

    Now, we can do the same for the integration over 'y'. 
    _For each_ x_j we generate a random sample of 'y' values from the psf, so that we may approximate it at

        psf(y | x_j) = \sum_k delta(y - y_k) / M

    where 'M' is the number of random samples.
    Putting this back into equation, we find

        mu(i) = \sum_j ind(i; kappa(x_j) \integrate_bin dy eta(y) \sum_k delta(y - y_k) / M  ) / N

    Note that when we evaluate \integrate_bin delta(y - y_k) we will get 1 if y_k is in the bin and 0 if it is not. So we get something like

        mu(i) = \sum_j ind(i; kappa(x_j) \sum_k ind(b; y_k) eta(y_k) / M  ) / N

    where 'ind(k; y_k)' is that indicator function for if 'y_k' is located in our bin 'b'.
    This gives the Monte-Carlo estimate for one bin, for multiple bins we would need to loop over this bin index 'b'.
    That would result in three loops, one over the bin index 'b', one for the sum '\sum_j', and one for the sum '\sum_k'.
    We can do this more efficiently with two loops by reusing the same set of random samples for all bins.
    The end result is the algorithm below.
"""
def sample_mu_eps(image_coords, # 'image_coords' is a tuple that contains the bin edges for our 2D image 
                  template_draw, # this function draws random samples from our template 'T(x)'
                  eff_area, # this function describes our effective area 'kappa(x)'
                  psf_draw, # this function draws random samples from our psf conditioned on a point-source location
                  eps_bin_edges): # this is an array that gives the bin edges for our binned mu(epsilon)
    map_edges, x_edges, y_edges = image_coords

    N_sources = 100000 # How many x_j samples to produce from T(x)
    N_events = 1000 # How many y_k samples to produce from psf(y | x_j)

    # This is the full 2D array of binned mu(i) functions for each bin.
    # The shape of the first two dimensions come from the image shape, and the length of the last dimension is determined by the number of epsilon bins.
    mu_eps_map = np.zeros(map_edges.shape[0:2] + (len(eps_bin_edges)-1,))

    for _ in range(N_sources): # This is our loop over x_j
        source_loc = template_draw()[0] # Get the value of x_j by drawing it randomly from T(x)
        source_eff_area = eff_area(source_loc) # and this is k(x_j)
        # Shortly we will see that we compute 'k(x_j) \sum_k ind(b; y_k) eta(y_k) / M' for each bin, the value of this expression gets stored in the following array for each bin
        eps_map = np.zeros(map_edges.shape[0:2]) 

        for _ in range(N_events): # This is our loop over y_k
            event_locs = psf_draw(source_loc) # Get the value of y_k by drawing it randomly from psf(y | x_j)
            event_idxs = digitise_map(image_coords, event_locs) # Here we find which bin index 'b' our x_j lies in.

            # Note that 'k(x_j) ind(b; y_k) eta(y_k) / M' just collapses to 'k(x_j) / M' when ind(b ; y_k)=1 and eta(y_k)=1,
            # So the value to be summed it just the following:
            event_eps = source_eff_area/N_events
            
            if event_idxs is not None: # If x_j actually falls in a bin
                event_x_idx, event_y_idx = event_idxs
                # Then add the effective area contribution to that bin.
                eps_map[event_x_idx, event_y_idx] += event_eps
        # The loop we just completed has calcuated 'k(x_j) \sum_k ind(b; y_k) eta(y_k) / M' for each bin 'b'.
        # Now we need to find what epsilon bin these values belong to
        
        # Note that 'ind(i; k(x_j) \sum_k ind(b; y_k) eta(y_k) / M  ) / N' is just '1/N' when the indicator is 1.
        weight = (1.0/N_sources) 

        # Loop over all 2D bin indicies.
        for x_idx in range(eps_map.shape[0]):
            for y_idx in range(eps_map.shape[1]):
                eps_total = eps_map[x_idx, y_idx] # this is the value of 'k(x_j) \sum_k ind(b; y_k) eta(y_k) / M' in 'ind(i; k(x_j) \sum_k ind(b; y_k) eta(y_k) / M  )'
                eps_idx = np.digitize(eps_total, eps_bin_edges) - 1 # Find the value of 'b' that this value belongs to by binning it.

                if eps_idx >= 0 and eps_idx < len(eps_bin_edges) - 1: # If the value lies in our binning range
                    # Then add the contribution to the binned mu(i)
                    mu_eps_map[x_idx, y_idx, eps_idx] += weight

    return mu_eps_map

"""
    In the previous example, we assumed we were given a function, 'k(x)' (or 'eff_area(x)') that calcuates the effective area at specific spatial location.
    In reality, such a function might not be available. This is because, for some experiments, effective area is calculated through a Monte-Carlo simulation
    of the detector equipment. In this example, we will see how to calculate mu(epsilon) using an "event library" that is generated from one of these detector simulations.

    The first function, shown below, creates an event library that matches a provided effective area function (so that we may make a direct comparison). In reality,
    such an "event library" is usually generated in the following way:

        1. Some grid of "spatial location coordinates" ('x' in our language) is chosen. From the detector's frame of reference, these correspond to directions away from the detector.
        2. For each of these choices of 'x':
            - Create an emission surface (such as a large square), that is normal to 'x'
            - Emit photons/neutrinos/etc from this surface such that all are normal to this surface (they all are colinear with 'x')
            - Simulate how these particles interact with the detector using ray optics/particle physics/etc.
            - Record the direction, 'y', that the detector determined the particles came from
                = The difference between 'x' and 'y' is caued by imperfect focussing, diffraction, scattering, algorithmic reconstruction dificiencies etc
                = The probability distribution of 'y' given 'x' is precisely what we call the psf.
            - Any particle that did not interact/was not detected in the simulation is discarded from the event library.
                = The fraction of detected particles is how we determine effective area for this 'x'.
                = The flux of incoming particles is F = N/A where 'N' is the number of generated particles, and 'A' is the area of the generation surface.
                = The number of detected particles is M=F*eps where 'F' is the flux above and 'eps' is the effective area.
                = Thus, the effective area is eps=A*(M/N) or the area of the generating surface reduced by the fraction of detected particles.
            - To assist with this and other calculations we also record
                = The generation flux (as described above)
                = The fraction (or probability density) of events in the library that were generated with direction 'x'.
                = The total number of events that where generated with direction 'x'.
"""
def create_event_library(template, # this is a function that describes our template 'T(x)' 
                         template_draw, # and this is the corresponding function that draws random samples from said template
                         psf_draw, # this is a function that draws random samples from the psf conditioned on a source location
                         N_source, # this is the number of different directions 'x_j' to generate photons/events for
                         N_psf, # this is number of photons/events 'y_k' to generate for each direction
                         target_eff_area): # this is a function that describes an effective area (like 'kappa(x)') and the generated events will be created to simulate this effective area
    event_source_locs = template_draw(size=N_source)
    
    event_detection_locs = np.stack(list(map(lambda source_loc: psf_draw(source_loc, size=N_psf), event_source_locs)), axis=0)
    # source_plane_size should be the size of the emission plane in our simulation; ie, a fixed constant.
    # However, we have a target effective area in mind, so we'll fake this effective area by varying source_plane_size to match it.
    # This gives us our desired effective area because in this simplistic model of a simulation, 100% of events are detected.
    source_plane_size = target_eff_area(event_source_locs)
    event_source_fluxes = np.ones_like(source_plane_size) * N_psf / source_plane_size
    #source_prob = 1/np.prod(map_edges.shape[0:2])
    event_source_probs = template(event_source_locs) # np.ones_like(event_detection_locs) * source_prob
    event_source_N = np.ones_like(source_plane_size) * N_source

    library = np.recarray((N_source, N_psf), dtype=[
        ('source_locs', event_source_locs.dtype, 2),
        ('source_fluxes', event_source_fluxes.dtype),
        ('source_probs', event_source_probs.dtype),
        ('source_N', event_source_N.dtype),
        ('detection_locs', event_detection_locs.dtype, 2)
    ])

    library.source_locs[:] = event_source_locs[:,None]
    library.source_fluxes[:] = event_source_fluxes[:,None]
    library.source_probs[:] = event_source_probs[:,None]
    library.source_N[:] = event_source_N[:,None]
    library.detection_locs[:] = event_detection_locs

    return library

"""
    With the event library as described above, we can calculated a binned mu(epsilon).
    This is quite similar in operation to the 'sample_mu_eps' function. Recall that the binned mu(epsilon) function could be calculated as

        mu(i) = \integrate dx T(x) ind(i; k(x) \integrate_bin dy eta(y) psf(y | x)  )

    In 'sample_mu_eps' we approximated 'T(x)' using samples from 'T' as

        T(x) = \sum_j delta(x - x_j) / N

    In this function, we will approximate it using the samples from the event library. This means we need to re-weight the samples from the event library to match our template:

        T(x) = \sum_j (T(x_j)/p_j) * delta(x - x_j) / N

    Where 'T(x_j)' is our desired template probability (density) at source location 'x_j' and 'p_j' is the ratio (or probaility density) of event generated from direction 'x_j' (this is 'source_probs' in our example event library).
    In 'sample_mu_eps' we approximated the psf using samples from the psf as
    
        psf(y | x_j) = \sum_k delta(y - y_k) / M

    In this function, we will approximate the product of effective area, psf and detection probability as

        k(x_j) eta(y) psf(y | x_j) = \sum_k delta(y - y_k) / F_k

    where 'F_k' is the generated flux for that simulated particles detected at 'y_k' generated in the direction of 'x_j' (this is 'source_fluxes' in our example event library).
    Combining these together we find that

        mu(i) = \sum_j (T(x_j)/p_j) ind(i; \integrate_bin dy k(x_j) eta(y) psf(y | x_j)  ) / N
              = \sum_j (T(x_j)/p_j) ind(i; \sum_k 1/F_k ) / N

    From here, we can follow a similar algorithm as 'sample_mu_eps' just with a few modifications.
"""
def simulate_mu_eps(image_coords, # 'image_coords' is a tuple that contains the bin edges for our 2D image 
                    template, # this is a function that describes our template 'T(x)'
                    event_library, # this is the library of events we generated with the previous function
                    eps_bin_edges): # this is an array that gives the bin edges for our binned mu(epsilon) 
    map_edges, x_edges, y_edges = image_coords

    mu_eps_map = np.zeros(map_edges.shape[0:2] + (len(eps_bin_edges)-1,))

    for source_library in event_library: # This is essentially our '\sum_j'
        eps_map = np.zeros(map_edges.shape[0:2])

        for event in source_library: # This is essentially our '\sum_k'
            _, source_flux, _, _, detection_loc = event
            
            eps = 1/source_flux # Here is the '1/F_k' term.
            
            detection_bin_idxs = digitise_map(image_coords, detection_loc)

            if detection_bin_idxs is not None:
                detection_x_idx, detection_y_idx = detection_bin_idxs
                eps_map[detection_x_idx, detection_y_idx] += eps # This is how we find '\sum_k 1/F_k'

        source_loc, _, source_prob, source_N, _ = source_library[0]
        weight = (1/source_N) * (template(source_loc) / source_prob) # This is the reweighting term '(T(x_j)/p_j)' multiplied by '1/N'

        for x_idx in range(eps_map.shape[0]):
            for y_idx in range(eps_map.shape[1]):
                eps_total = eps_map[x_idx, y_idx]
                eps_idx = np.digitize(eps_total, eps_bin_edges) - 1 # Now we find what epsilon bin each of the '\sum_k 1/F_k' falls into.

                if eps_idx >= 0 and eps_idx < len(eps_bin_edges) - 1:
                    mu_eps_map[x_idx, y_idx, eps_idx] += weight # And we sum up the weights, completing the calculation.

    return mu_eps_map

"""
    Now that we have calcuated our mu(epsilon) function, we need to test it against some (fake) data.
    To generate this data, we will need to simulate some point sources within the field of view defined by our binning.
    Each of these sources will need a flux, and we will use the following function to randomly sample the fluxes from a broken power law distribution.
"""
def create_flux_function(Fb, # The location of the break in the broken power law 
                         n1, # the index of the power law after the break (high flux side)
                         n2): # the index od the power law before the break (low flux side)
    # This function was constructed by finding the cumulative distribution function (CDF) of a broken power law distribution,
    # then the CDF was inverted. the result is that a random flux is given by inverseCDF(u) where u is a uniform random number between 0 and 1.
    def draw_flux(size=None):
        cut = (n1 - 1) / (n1 - n2)
        u = np.random.random_sample(size)

        # if u < cut:
        result = Fb * (u * (n1 - n2)/(n1 - 1))**(1.0/(1-n2))
        # else:
        mask = u >= cut
        oneMinusU = (1-cut)*np.random.random_sample(mask.sum()) # potential numerical instability fix
        result[mask] = Fb * ((1 - n2) / ((n1 - n2) * oneMinusU))**(1.0/(n1-1))
        return result

        # The above is a vectorised version of the following...
        # if u < cut:
        #     return Fb * (u * (n1 - n2)/(n1 - 1))**(1.0/(1-n2))
        # else:
        #     # potential numerical instability fix
        #     oneMinusU = (1-cut)*random.random_sample()
        #     return Fb * ((1 - n2) / ((n1 - n2) * oneMinusU))**(1.0/(n1-1))
    return draw_flux

"""
    In this function, we simulate some point-sources on an image.
    This function takes a list of point-source models 'models' and uses it to generate an image by sampling sources from each model.
    The list 'models' should be formatted as a list of tuples like so:
        models = [
            (
                template_draw, <-- This is the template ('T(x)') for the model, in the form of a function that draws random samples from the template.
                N, <-- This is the mean number of sources for the model.
                flux_draw <-- This is the flux distribution in the form of a function that draws random samples from the flux distribution (like the function above)
            ), ...
        ]
"""
def create_test_data(image_coords, # 'image_coords' is a tuple that contains the bin edges for our 2D image 
                     eff_area, # a function that desribes the effective area 'kappa(x)'
                     psf_draw, # a function that draws random samples from the psf conditioned on the source location
                     models): # this list of point-source population models to generate data for
    map_edges, x_edges, y_edges = image_coords
    # We first create an empty image which we will then populate with photons/events
    data = np.zeros(map_edges.shape[0:2], dtype=int)

    for template_draw, N, flux_draw in models: # For each model
        N_sources = np.random.poisson(N) # Sample the number of sources from the mean parameter N
        source_locs = template_draw(size=N_sources) # Then, sample the locations of each of these sources by drawing random samples from the template
        fluxes = flux_draw(N_sources) # Now draw an equal number of random samples from the flux distribution
        mean_events = eff_area(source_locs) * fluxes # The mean number of photons/events for each source is the product of the effective area and the flux
        N_events = np.random.poisson(mean_events) # Draw the actual number of photons/events using the mean that we just calcuated


        for n, source_loc in zip(N_events, source_locs): # Loop over each source
            event_locs = psf_draw(source_loc, size=n) # Draw the photon/event locations from the psf conditioned on the location of the point-source. 
            event_idxs = digitise_map(image_coords, event_locs.T) # Find which bins these photon/events land in
            # Now we want to add these as counts to our image 'data'. We want to do something like
            #   data[event_idxs] += 1
            # however there are two problems here. This first is that numpy does not support 2D indexing in this way, we need to convert our 2D bin indices into flat indices.
            # We can do this using the 'np.ravel_multi_index' function, but even if we do this the following
            #   data.flat[np.ravel_multi_index(event_idxs.T, data.shape)] += 1
            # will not work, this is because the '+= 1' doesn't work correctly when we have repeated indices in our list (which we expect because multiple photon/events could land in a single bin).
            # Instead we can get around this using the 'np.bincount' function, we counts the multiplicity of repeated indices, then constructs an array that contains those multiplicities.
            data.flat += np.bincount(np.ravel_multi_index(event_idxs.T, data.shape), minlength=len(data.flat))

    return data

"""
    In this function, we set up all the moving parts required to calculate mu(epsilon) using the three different methods.
"""
def create_d_eps_map():
    image_shape = (30, 20) # The number of bins in the x axis and the y axis of the desired image
    image_coords = create_image_coords(image_shape) # create the tuple describing the bin edges

    eff_area = create_eff_area_fuction(loc=(0, 0), cov = np.array([[1, 0], [0, 1]])) # create the function that describes the effective area
    psf, psf_draw = psf_function(cov = 0.05*0.05 * np.array([[1, 0], [0, 1]])) # create the functions that describe the psf, and draw random samples from the psf
    template, template_draw = create_template_function([ # create the functions that describe the template, and draw random samples from the template
        (0.25, [0.25, 0.25], 0.3*0.3), # Recall this function takes a list of modes, this first tuple contains the relative normalisation, location and covariance matrix of the first mode
        (0.75, [0.55, 0.55], 0.5*0.5) # And the same for the second mode, note that the relative normalisations of both modes add to 1
    ]) 
    flux_draw = create_flux_function(100, 3, -3) # create the function that allows us to draw random samples from the flux distribution
    data_map = create_test_data(image_coords, eff_area, psf_draw, [(template_draw, 100, flux_draw)]) # create the (fake) data to test our mu(epsilon) functions on

    # We will use 30 bins for mu(epsilon), these bins will be logarithmically spaced between 10^-5 and 10^0.5
    # In theory, because our effective area function has a maximum of 1 we shouldn't need any bins above 10^0
    eps_bin_edges = np.logspace(-5, 0.5, 30)
    d_eps_map = {}
    # Now we create mu(epsilon) maps three different ways
    library = create_event_library(template, template_draw, psf_draw, 100000, 1000, eff_area)
    d_eps_map['sim'] = simulate_mu_eps(image_coords, template, library, eps_bin_edges)
    d_eps_map['sample'] = sample_mu_eps(image_coords, template_draw, eff_area, psf_draw, eps_bin_edges)
    d_eps_map['integrate'] = integrate_mu_eps(image_coords, template, eff_area, psf, eps_bin_edges)

    # We declare the centers of each epsilon bin to be the average of the bin edges.
    eps_bin_centers = (eps_bin_edges[:-1] + eps_bin_edges[1:])/2
    return data_map, eps_bin_centers, d_eps_map

import cpg_likelihood

def flatten_map(map2d):
    return map2d.reshape((map2d.shape[0]*map2d.shape[1],)+map2d.shape[2:])

def simple_model(eps, d_eps_map):
    """
    A simple model that just includes a single point source model.
    """
    d_eps_map = flatten_map(d_eps_map)

    # First, create prior objects that represent both the parameters for the models and the priors for each of these parameters.
    N_prior = cpg_likelihood.priors.Log10UniformPrior( # A uniform prior in log space (base 10)
        0, # Start of the prior range in log10 (so this is 1.0 in linear space)
        5, # End of the prior range in log10 (so this is 100.0 in linear space)
        'N', # Simple name of the parameter, referred to as the 'ID', should be unique for easy access later
        r"$\log_{10}{N}$", # A 'display' name for the parameter, I use this for the TeX version of the parameter name
        display_log=True) # Asking for the 'display' values will produce a log10 value instead of the linear version of the parameter (useful for corner plots)
    F_total_prior = cpg_likelihood.priors.Log10UniformPrior(-3+0, 3+5, 'F', r"$\log_{10}{F}$", display_log=True)
    beta_priors = [] # Right now, code only supports one break so this has to be empty
    # These use the Tan priors discussed in the paper. The start and end ranges are specified in the slope of the line (so they correspond to the usual values of n1 and n2)
    n_priors = [ cpg_likelihood.TanUniformPrior(2, 5, 'n1', r"$n_1$"), cpg_likelihood.priors.LinearUniformPrior(-5, 0, 'n2', r"$n_2$") ]

    # Now the actual model object is created
    model = cpg_likelihood.models.NaturalPSModel(
        eps, 
        d_eps_map, # As mu(eps) depends on the template, the actual mu(eps) used for this model must be provided here (with the associated eps values above) 
                   # Note that this should be a 'flat' map, so that d_mu_eps is a 2D array with dimensions of [map_bin, epsilon_bins]
        N_prior, # From here on, the prior objects (which also have the parameter names) are provided for each of the parameters of the model
        F_total_prior, 
        beta_priors, 
        n_priors)

    return model

def run(out_fname):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    data_map, eps, d_eps_maps = create_d_eps_map()

    burn_in = 0.7

    with PdfPages(out_fname) as pdf:

        for mu_eps_type, d_eps_map in d_eps_maps.items():

            model = simple_model(eps, d_eps_map)
            data = flatten_map(data_map)
            d_eps_flat = flatten_map(d_eps_map)

            plt.title(mu_eps_type)
            plt.imshow(data_map)
            plt.colorbar()
            pdf.savefig()
            plt.close()
            
            plt.title(mu_eps_type)
            plt.imshow((eps[None,None,:] * d_eps_map).sum(axis=2))
            plt.colorbar()
            pdf.savefig()
            plt.close()

            for d_eps in d_eps_flat:
                plt.plot(eps, d_eps)
            
            plt.title(mu_eps_type)
            plt.xscale('log')
            #plt.yscale('log')

            pdf.savefig()
            plt.close()

            # After definition of the model, the JointDistrubiton object defines the probability distribtion to sample from
            joint = cpg_likelihood.models.JointDistribution(
                data, # provided data should match the epsilon maps used to define the model. It should be one dimensional.
                eps, # Same epsilon bins used to define the model. All epsilon maps must share the same binning as defined here.
                model, 
                # The number of threads used to evalulate the likelihood. What you use here will depend on your parallelisation strategy.
                # If you have many CPU cores, Python's multiprocessing pool cannot keep up, and so increasing the thread count here 
                # will allow most cores to be utilised.
                # MPI parallelisation may scale better, in which case it might be best to keep the thread count lower.
                threads=1
            )

            for sampler_name in ['emcee', 'dynesty']:

                import multiprocessing
                # In my experience, going above a pool size of 16 is counter-productive, instead of going over 16 you may want to consider
                # increasing the thread count instead.
                #with multiprocessing.Pool(64) as pool:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                    if sampler_name == 'dynesty':
                        # This helper function sets up and runs dynesty
                        raw_samples, raw_weights = cpg_likelihood.mcmc.run_dynesty(
                            joint, # Joint distribution to sample
                            sampler_opts = dict(
                                rstate=np.random.default_rng(), # random number generator used to draw the initial locations
                                pool=pool,
                                queue_size=16
                            ),
                            n_effective=30000, # Number of samples to aim for
                        )
                    if sampler_name == 'emcee':
                        raw_samples = cpg_likelihood.mcmc.run_emcee(
                            joint, # Joint distribution to sample
                            32, # Number of emcee walkers
                            30000, # Number of samples to take (1k is used for this test, try to keep this over 10k if possible.)
                            np.random, # random number generator used to draw the initial locations
                            pool, 
                            progress=True # Show a progress bar using tqdm
                        )
                        raw_weights = np.ones(raw_samples.shape[0])

                        burn_in_idx = int(burn_in*len(raw_samples))
                        raw_samples, raw_weights = raw_samples[burn_in_idx:], raw_weights[burn_in_idx:]

                # raw_samples is an array of values between 0 and 1, that is, the parameter values before the prior transformations.
                # get_params_flat returns a dictionary of all parameters defined in the model.
                # The keys of this dictionary are the string IDs provided when defining the priors.
                # As such, this method can only be used when all of these IDs are unique.
                samples = joint.get_params_flat(
                    raw_samples, 
                    # Causes the prior transformation to be slightly different, depending on if any priors were defined with a display_x=True parameters.
                    # Notably, for Log10UniformPrior, if display_log is true, then values in log10 will be returned instead of the linear values of the parameter.
                    display_value=True 
                )
                # In order to get the display names, this method will return a dictionary where the keys are the parameter IDs and values are the display
                # names specified when defining the priors.
                display_names_map = joint.get_id_display_map()

                samples_cols = samples.keys() # Get all the parameter IDs.
                display_names = [ display_names_map[id_] for id_ in samples_cols ] # And all the parameter display names
                # Create a single numpy array, where the order of columns now corresponds to the order of IDs in samples_cols.
                # To control the order, you could define your own array of IDs and use that instead (make sure to use it for the display names as well).
                samples_ary = np.array([ samples[id_] for id_ in samples_cols ]).T

                import corner
                corner.corner(samples_ary, weights=raw_weights, labels=display_names)
                plt.suptitle(mu_eps_type + ' ' + sampler_name)

                pdf.savefig(plt.gcf())

                plt.close()

if __name__ == '__main__':
    import sys
    run(sys.argv[1])