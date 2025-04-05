import numpy as np
import scipy.stats

def create_test_data(bg_flux):
    """
    This function creates a synthetic image and mu(epsilon) function for the purpose of the functions that construct models.
    """
    image_shape = (10,100)

    eps_bins = np.logspace(-3, 3, 100)
    eps = (eps_bins[:-1] + eps_bins[1:])/2.0
    d_eps_map = scipy.stats.norm.sf(eps,)
    d_eps_map = np.zeros(image_shape + (len(eps_bins)-1,)) + scipy.stats.norm.pdf(eps, 1, 0.5)[None,None,:]
    d_eps_map /= np.prod(image_shape) * d_eps_map.sum(axis=2)[:,:,None] # As this is synthetic, need to ensure that it's normalised correctly

    N = 5000
    F = 10
    poiss_flux = 20.0 if bg_flux else 0.0

    image_N_sources = scipy.stats.poisson.rvs(N/np.prod(image_shape), size=image_shape)
    image_expected_counts = np.array(list(map(lambda n, m: list(map(lambda N_sources, mu_eps: (F * np.random.choice(eps, size=N_sources, p=mu_eps/mu_eps.sum())).sum(), n, m)), image_N_sources, d_eps_map)))
    image = scipy.stats.poisson.rvs(image_expected_counts + poiss_flux)

    return image, eps, d_eps_map

import cpg_likelihood

def flatten_map(map2d):
    return map2d.reshape((map2d.shape[0]*map2d.shape[1],)+map2d.shape[2:])

def simple_model():
    """
    A simple model that just includes a single point source model.
    """
    image, eps, d_eps_map = create_test_data(False)
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
    n_priors = [ cpg_likelihood.TanUniformPrior(2, 5, 'n1', r"$n_1$"), cpg_likelihood.priors.TanUniformPrior(-5, 0, 'n2', r"$n_2$") ]

    # Now the actual model object is created
    model = cpg_likelihood.models.NaturalPSModel(
        eps, 
        d_eps_map, # As mu(eps) depends on the template, the actual mu(eps) used for this model must be provided here (with the associated eps values above) 
                   # Note that this should be a 'flat' map, so that d_mu_eps is a 2D array with dimensions of [map_bin, epsilon_bins]
        N_prior, # From here on, the prior objects (which also have the parameter names) are provided for each of the parameters of the model
        F_total_prior, 
        beta_priors, 
        n_priors)

    return flatten_map(image), eps, model

def simple_model_with_bg():
    """
    A model that includes a single point source model with a background.
    """
    image, eps, d_eps_map = create_test_data(True)
    d_eps_map = flatten_map(d_eps_map)

    N_prior = cpg_likelihood.priors.Log10UniformPrior(0, 5, 'N', r"$\log_{10}{N}$", display_log=True)
    F_total_prior = cpg_likelihood.priors.Log10UniformPrior(-3+0, 3+5, 'F', r"$\log_{10}{F}$", display_log=True)
    beta_priors = []
    n_priors = [ cpg_likelihood.priors.TanUniformPrior(2, 5, 'n1', r"$n_1$"), cpg_likelihood.priors.TanUniformPrior(-5, 0, 'n2', r"$n_2$") ]
    # The above is similar to the simple model
    # In addition, a prior on omega (the flux fraction parameter) is needed.
    omega_prior = cpg_likelihood.priors.UnitLinearUniformPrior('omega', r"$\omega$") # UnitLinearUniform refers to a uniform prior between 0 and 1.

    # Creating a model out of two components that share a flux parameters works as follows:
    # The models are arranged into a tree, where each component is a leaf of the tree.
    # For this relatively simple example, the tree will be quite trivial

    # First, the leaf corresponding to the point source component is created.
    # This is quite similar to the simple model case, but no prior for the flux is provided (there is one less function parameter here) 
    PS_leaf = cpg_likelihood.models.NaturalPSLeaf(eps, d_eps_map, N_prior, beta_priors, n_priors)

    # Next, the leaf for a Poisson component is created.
    # This just requires an epsilon map (kind of like an effective area map), which is easy to find given and mu(eps) map:
    eps_map = (eps*d_eps_map).sum(axis=1)
    # Then, create the Poisson component leaf. Note there is no parameter for the flux here as well.
    poiss_leaf = cpg_likelihood.models.PoissLeaf(eps_map)

    # Following, the tree node that represents the flux fraction, omega, is created.
    node = cpg_likelihood.models.OmegaNode(
        omega_prior, # Prior for the flux fraction parameter.
        # Each leaf is provided with a string giving the leaf's name. The names can be used to refer to the components later if needed, but if the defined parameter IDs
        # (given when creating the prior objects) are unique, then these are usually not needed (and are just used for internal bookkeeping).
        'PS', PS_leaf,
        'bg', poiss_leaf
    )

    # Finally, the tree is created using a FluxRoot object
    model = cpg_likelihood.models.FluxRoot(
        F_total_prior, # Prior for the total flux parameter, as this is shared between both components. 
        node)

    return flatten_map(image), eps, model

def complex_model():
    """
    A complex that includes a two point source models with three poisson backgrounds.
    Each point source model has an associated background, while the third background
    is provided outside of the shared flux system (such as for an instrument background).
    """
    image, eps, d_eps_map = create_test_data(True)
    eps_A, d_eps_map_A, eps_B, d_eps_map_B = eps, flatten_map(d_eps_map), eps, flatten_map(d_eps_map)

    # The priors for point source model A (and associated poisson background) are created much like the previous example
    N_prior_A = cpg_likelihood.priors.Log10UniformPrior(0, 5, 'N_A', r"$\log_{10}{N_A}$", display_log=True)
    beta_priors = []
    n_priors_A = [ cpg_likelihood.priors.TanUniformPrior(2, 5, 'n1_A', r"$n_{1,A}$"), cpg_likelihood.priors.TanUniformPrior(-5, 0, 'n2_A', r"$n_{2,A}$") ]
    # Like in the previous example, and epsilon map is created for the poisson background term.
    eps_map_A = (eps_A*d_eps_map_A).sum(axis=1)

    # A second point source model (model B) is also created, usually the template for this model would be different, and so the provided d_mu_eps_B would be
    # different to the d_mu_eps_A above (for this example it is the same, though).
    N_prior_B = cpg_likelihood.priors.Log10UniformPrior(0, 5, 'N_B', r"$\log_{10}{N_B}$", display_log=True)
    n_priors_B = [ cpg_likelihood.priors.TanUniformPrior(2, 5, 'n1_B', r"$n_{1,B}$"), cpg_likelihood.priors.TanUniformPrior(-5, 0, 'n2_B', r"$n_{2,B}$") ]
    eps_map_B = (eps_B*d_eps_map_B).sum(axis=1)

    # Now the flux tree will be defined.  
    F_total_prior = cpg_likelihood.priors.Log10UniformPrior(-3+0, 3+5, 'F', r"$\log_{10}{F}$", display_log=True)
    # Here, all nodes and leaves are created inline so as to show the tree structure.
    flux_tree = cpg_likelihood.models.FluxRoot(
        F_total_prior,

        # An overall 'omega' parameter defines the fraction of flux between models A and models B
        cpg_likelihood.models.OmegaNode(
            cpg_likelihood.priors.UnitLinearUniformPrior('omega', r"$\omega$"),

            # The A models have their own flux fraction parameter, that defines how flux is shared between the sources and background
            'group_A', cpg_likelihood.models.OmegaNode(
                cpg_likelihood.priors.UnitLinearUniformPrior('omega_A', r"$\omega_A$"),
                'PS', cpg_likelihood.models.NaturalPSLeaf(eps_A, d_eps_map_A, N_prior_A, beta_priors, n_priors_A),
                'bg', cpg_likelihood.models.PoissLeaf(eps_map_A)
            ),

            # Same for the B models
            'group_B', cpg_likelihood.models.OmegaNode( 
                cpg_likelihood.priors.UnitLinearUniformPrior('omega_B', r"$\omega_B$"),
                'PS', cpg_likelihood.models.NaturalPSLeaf(eps_B, d_eps_map_B, N_prior_B, beta_priors, n_priors_B),
                'bg', cpg_likelihood.models.PoissLeaf(eps_map_B)
            )
        )
    )

    # Now the instrument backgorund will be defined seperately, as it is specified in units of counts and can't be compared to an astrophysical flux
    instrument_background = cpg_likelihood.models.PoissModel(
        np.ones_like(eps_map_A), # Unit will be in counts, so the 'epsilon map' for this component is just all ones.
        cpg_likelihood.priors.LinearUniformPrior( # The prior defined here will multiply the 'epsilon map', so here it should be in units of counts.
            2.35, 2.45, # Typically an instrument background would be very tightly defined around a known value.
            'bg_inst', r"$S_{\mathrm{bg}}$") 
        )

    # Now the tree and instrument background need to be combined into a single model, this is done using a group.
    model = cpg_likelihood.models.ModelGroup(
        # The group components are given as keyword arguments.
        # The key and value for these arguments are like the names and leaves/nodes in the OmegaNode object.
        # They can be used to refer to these components later, but are mostly for bookkeeping, 
        # especially if all parameter IDs are unique for everything inside this model.
        astro = flux_tree,
        inst = instrument_background
    )

    return flatten_map(image), eps, model

def run(out_fname):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_fname) as pdf:

        for (data, eps, model) in [ 
                simple_model(), 
                simple_model_with_bg(),
                complex_model()
            ]:

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

            import multiprocessing
            # In my experience, going above a pool size of 16 is counter-productive, instead of going over 16 you may want to consider
            # increasing the thread count instead.
            #with multiprocessing.Pool(64) as pool:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                # This helper function sets up and runs emcee
                raw_samples = cpg_likelihood.mcmc.run_emcee(
                    joint, # Joint distribution to sample 
                    32, # Number of emcee walkers
                    1000, # Number of samples to take (1k is used for this test, try to keep this over 10k if possible.)
                    np.random, # random number generator used to draw the initial locations
                    pool, 
                    progress=True # Show a progress bar using tqdm
                )

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

            import matplotlib.pyplot as plt

            import corner
            corner.corner(samples_ary, labels=display_names)

            pdf.savefig()

            plt.close()

if __name__ == '__main__':
    import sys
    run(sys.argv[1])