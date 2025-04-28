################################################################################
#
# Delaunay density diangostic for MSD and grad-MSD rates
#   as described in the paper
#   Algorithm XXXX: The Delaunay Density Diagnostic
#       under review at ACM Transactions on Mathematical Software
#       (original title: Data-driven geometric scale detection via Delaunay interpolation)
#   Algorithm XXXX: The Delaunay Density Diagnostic
#       under review at ACM Transactions on Mathematical Software
#       (original title: Data-driven geometric scale detection via Delaunay interpolation)
#   by Andrew Gillette and Eugene Kur
#   Version 2.1, September 2024
#
# For usage information, run:
# python delaunay_density_diagnostic.py --help
#
################################################################################



#==================================================================================================#
# Load packages.  Set random state and validation split.
#==================================================================================================#

# from matplotlib.pyplot import legend
# import torch
import pandas as pd     
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.utils.data as Data
# from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import numpy.ma as ma
import numpy.ma as ma

from numpy.random import rand, default_rng
from numpy import arccos, array, degrees, absolute
from numpy.linalg import norm

from optparse import OptionParser

from sys import exit
import os.path

import copy

from delsparse import delaunaysparsep as dsp


#==================================================================================================#
# Define the test function 
#==================================================================================================#

def tf_gwk(X): # Griewank function, arbitrary dimension input
    X = X.T
    term_1 = (1. / 4000.) * sum(X ** 2)
    term_2 = 1.0
    for i, x in enumerate(X):
        term_2 *= np.cos(x) / np.sqrt(i + 1)
    return 1. + term_1 - term_2


def tf_pbd(X): # Paraboloid, arbitrary dimension input
    X = X.T
    return (7/20_000) *  ( X[0]**2 + 0.5*(X[1]**2) )

def tf_pbd(X): # Paraboloid, arbitrary dimension input
    X = X.T
    return (7/20_000) *  ( X[0]**2 + 0.5*(X[1]**2) )

#==================================================================================================#
# Make query point lattice in R^dim
#==================================================================================================#
def make_test_data_grid(rng, static_data=False):

    num_samples_per_dim = options.numtestperdim

    x = np.linspace(options.queryleftbound, options.queryrightbound, num_samples_per_dim)

    print("===> Test coordinates for each dimension = ", x)
    mg_in = []
    for i in range(options.dim):
        mg_in.append(x)
    grid_pts = np.array(np.meshgrid(*mg_in))

    grid_pts = grid_pts.reshape(options.dim, num_samples_per_dim ** options.dim)
    grid_pts = grid_pts.T

    if options.fn_name == 'griewank':
        outputs_on_grid = tf_gwk(grid_pts)
    elif options.fn_name == 'paraboloid':
        outputs_on_grid = tf_pbd(grid_pts)
    else:
        print("ERROR: function name not supported.")
        exit()
    if options.fn_name == 'griewank':
        outputs_on_grid = tf_gwk(grid_pts)
    elif options.fn_name == 'paraboloid':
        outputs_on_grid = tf_pbd(grid_pts)
    else:
        print("ERROR: function name not supported.")
        exit()

    data_test_inputs  = pd.DataFrame(grid_pts)
    data_test_outputs = pd.DataFrame(outputs_on_grid)

    return data_test_inputs, data_test_outputs



#==================================================================================================#
# Collect random sample from bounding box
#==================================================================================================#
def make_random_training_in_box(rng):

    train_set_size = options.numtrainpts
    # print("==> Generating ", train_set_size, " random points.")
       
    rand_pts_n = rng.random((train_set_size, options.dim))
        
    train_box_scale_vector = np.full(options.dim, (options.bboxrightbound - options.bboxleftbound) )
    train_box_shift_vector = np.full(options.dim, options.bboxleftbound )

    # do scaling in each dim first
    for i in range(options.dim):
        rand_pts_n[:,i] *= train_box_scale_vector[i]
    # then do shifts
    for i in range(options.dim):
        rand_pts_n[:,i] += train_box_shift_vector[i]


    if options.fn_name == 'griewank':
        outputs_on_rand_n = tf_gwk(rand_pts_n)
    elif options.fn_name == 'paraboloid':
        outputs_on_rand_n = tf_pbd(rand_pts_n)
    else:
        print("ERROR: function name not supported.")
        exit()

    if options.fn_name == 'griewank':
        outputs_on_rand_n = tf_gwk(rand_pts_n)
    elif options.fn_name == 'paraboloid':
        outputs_on_rand_n = tf_pbd(rand_pts_n)
    else:
        print("ERROR: function name not supported.")
        exit()

    data_train_inputs  = pd.DataFrame(rand_pts_n)
    data_train_outputs = pd.DataFrame(outputs_on_rand_n)

    return data_train_inputs, data_train_outputs



#==================================================================================================#
# Function to compute DelaunaySparse 
#==================================================================================================#
def compute_DS_only(data_train_inputs, data_train_outputs, data_test_inputs, data_test_outputs):

    # # note: data_test_outputs is only converted to numpy if needed, transposed, 
    #         and returned as actual_test_vals

    # # WARNING: deepcopy here may be inefficient at scale
    pts_in = copy.deepcopy(data_train_inputs)
    q      = copy.deepcopy(data_test_inputs)

    interp_in = data_train_outputs
    
    actual_test_vals = data_test_outputs
    if not isinstance(actual_test_vals, np.ndarray):
        actual_test_vals = actual_test_vals.to_numpy()
    actual_test_vals = actual_test_vals.T

    actual_train_vals = data_train_outputs
    if not isinstance(actual_train_vals, np.ndarray):
        actual_train_vals = actual_train_vals.to_numpy()
    actual_train_vals = actual_train_vals.T

    interp_in_n = interp_in

    if not isinstance(interp_in_n, np.ndarray):
        interp_in_n = interp_in_n.to_numpy()
    interp_in_n = interp_in_n.T

    if not isinstance(pts_in, np.ndarray):
        pts_in = pts_in.to_numpy()
    pts_in = pts_in.T
    pts_in = np.require(pts_in, dtype=np.float64, requirements=['F'])

    if not isinstance(q, np.ndarray):
        q = q.to_numpy()
    p_in = np.asarray(q.T, dtype=np.float64, order="F")

    ir=interp_in_n.shape[0]

    interp_in_n = np.require(interp_in_n, 
                dtype=np.float64, requirements=['F'])
    simp_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                dtype=np.int32, order="F")
    weights_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                dtype=np.float64, order="F")
    error_out = np.ones(shape=(p_in.shape[1],), 
                dtype=np.int32, order="F")
    interp_out_n = np.zeros([interp_in_n.shape[0],p_in.shape[1]])
    interp_out_n = np.require(interp_out_n, 
                dtype=np.float64, requirements=['F'])
    rnorm_n = np.zeros(p_in.shape[1])
    rnorm_n = np.require(rnorm_n, dtype=np.float64, requirements=['F'])

    # From delsparse.py documenation:
    #   Setting EXTRAP=0 will cause all extrapolation points to be
    #   ignored without ever computing a projection. By default, EXTRAP=0.1
    #   (extrapolate by up to 10% of the diameter of PTS).
    dsp(pts_in.shape[0], pts_in.shape[1],
        pts_in, p_in.shape[1], p_in, simp_out,
        weights_out, error_out, 
        extrap=options.extrap_thresh,
        rnorm=rnorm_n,
        pmode=1,
        interp_in=interp_in_n, interp_out=interp_out_n)


    if (options.computeGrad):
            
        # # arbitrary number of outputs, as determind by interp_in_n.shape[0]
        grad_est_DS = np.zeros([interp_in_n.shape[0], simp_out.shape[1], options.dim])
        grad_est_DS.fill(999)
        
        for j in range(simp_out.shape[1]):
            # note: the value of simp_out.shape[1] should equal the number of interpolation outputs
            #       extrapolation points don't get a simp_out entry, I think?
            #
            #       mutliple test points may lie in the same simplex
            #       but that just means you might duplicate effort
            #       if you already saw a simplex and comptued the gradient(s)

            # this presumes pts_in was deep copied from data_train_inputs at start of compute_DS_only(...)
            unscaled_inputs = data_train_inputs.to_numpy().T
            # can try using scaled points instead:
            # # unscaled_inputs = pts_in

            for outputdim in range(interp_in_n.shape[0]):
                # in extrapolation cases, the "enclosing simplex" provided by
                #   DelaunaySparse is stored as all zeros.  In this case,
                #   we set grad out to be nans.  Later, we mask for these
                #   nans when computing the gradient improvement rate. 
                # in extrapolation cases, the "enclosing simplex" provided by
                #   DelaunaySparse is stored as all zeros.  In this case,
                #   we set grad out to be nans.  Later, we mask for these
                #   nans when computing the gradient improvement rate. 

                simp_out_np = np.array(simp_out[:,j])
                if np.all((simp_out_np == 0)):
                    grad_out = np.zeros(options.dim)
                    grad_out[:] = np.nan
                else:
                    matrixA = np.zeros([options.dim+1, options.dim+1])
                    for i in range(options.dim+1):
                        matrixA[i] = np.append(unscaled_inputs[:,simp_out[:,j][i]-1], interp_in_n[outputdim][simp_out[:,j][i]-1])
                simp_out_np = np.array(simp_out[:,j])
                if np.all((simp_out_np == 0)):
                    grad_out = np.zeros(options.dim)
                    grad_out[:] = np.nan
                else:
                    matrixA = np.zeros([options.dim+1, options.dim+1])
                    for i in range(options.dim+1):
                        matrixA[i] = np.append(unscaled_inputs[:,simp_out[:,j][i]-1], interp_in_n[outputdim][simp_out[:,j][i]-1])

                    coords = matrixA
                    coords = matrixA

                    G = coords.sum(axis=0) / coords.shape[0]
                    G = coords.sum(axis=0) / coords.shape[0]

                    # run SVD
                    u, s, vh = np.linalg.svd(coords - G)
                    
                    # unitary normal vector
                    hyper_sfc_normal = vh[options.dim, :]
                    # run SVD
                    u, s, vh = np.linalg.svd(coords - G)
                    
                    # unitary normal vector
                    hyper_sfc_normal = vh[options.dim, :]

                    # approx grad as normal scaled by vertical component, times -1
                    grad_out = hyper_sfc_normal/hyper_sfc_normal[options.dim]
                    grad_out = -grad_out[:-1]
                # end if/else for simplex all zero indices
                    # approx grad as normal scaled by vertical component, times -1
                    grad_out = hyper_sfc_normal/hyper_sfc_normal[options.dim]
                    grad_out = -grad_out[:-1]
                # end if/else for simplex all zero indices
                grad_est_DS[outputdim][j] = grad_out
            # end loop over output dimns 
        # end loop over simplices
        # end loop over simplices
    # end if computeGrad
    else:
        grad_est_DS = []


    allow_extrapolation=True
    print_errors=True
    # note: error code 1= sucessful extrap; 2 = extrap beyond threshold
    extrap_indices = np.where((error_out == 1) | (error_out == 2))
    extrap_indices = np.array(extrap_indices[0])
    # print("extrap indices = ", extrap_indices)
    # print("rnorm = ", rnorm_n)
    # print("rnorm[indices] = ", rnorm_n[extrap_indices])
    # print("type = ", type(extrap_indices))
    # print("e i [0]:",extrap_indices[0])
    
    #==============================================================================#
    # Check for errors in DelaunaySparse run
    #==============================================================================#
    if allow_extrapolation: 
        # print("Extrapolation occured at ", np.where(error_out == 1))
        # Next line replaces error code 1 (successful extrapolation) 
        #               with error code 0 (successful interpolation)
        error_out = np.where(error_out == 1, 0, error_out)
    else:
        if 1 in error_out:
            class Extrapolation(Exception): pass
            raise(Extrapolation("Encountered extrapolation point (beyond threshold) when making Delaunay prediction."))
    # Handle any errors that may have occurred.
    if (sum(error_out) != 0):
        if print_errors:
            unique_errors = sorted(np.unique(error_out))
            if unique_errors == [0,2]: # only extrapolation errors
                pass
            else:
                print(" [Delaunay errors:",end="")
                for e in unique_errors:
                    if (e == 0): continue
                    indices = tuple(str(i) for i in range(len(error_out))
                                    if (error_out[i] == e))
                    if (len(indices) > 5): indices = indices[:2] + ('...',) + indices[-2:]
                    print(" %3i"%e,"at","{"+",".join(indices)+"}", end=";")
                print("] ")
            if unique_errors == [0,2]: # only extrapolation errors
                pass
            else:
                print(" [Delaunay errors:",end="")
                for e in unique_errors:
                    if (e == 0): continue
                    indices = tuple(str(i) for i in range(len(error_out))
                                    if (error_out[i] == e))
                    if (len(indices) > 5): indices = indices[:2] + ('...',) + indices[-2:]
                    print(" %3i"%e,"at","{"+",".join(indices)+"}", end=";")
                print("] ")
        # Reset the errors to simplex of 1s (to be 0) and weights of 0s.
        bad_indices = (error_out > (1 if allow_extrapolation else 0))
        simp_out[:,bad_indices] = 1
        weights_out[:,bad_indices] = 0

    return interp_out_n, actual_test_vals, actual_train_vals, extrap_indices, grad_est_DS



#==================================================================================================#
# Main section, includes some bad input checks
#==================================================================================================#
if __name__ == '__main__':

    #==================================================================================================#
    # Provide help screen documentation. Let the user define options. Also define defaults.            #
    #==================================================================================================#

    usage = "%prog [options]"
    parser = OptionParser(usage)
    parser.add_option( "--jobid", help="Job ID.", 
        dest="jobid", type=int, default=999999)   
    parser.add_option( "--staticdatapath", help="Path to static data set from which samples will be drawn. " +
        "If no path is provided, code uses --fn option to sample data.", 
        dest="data_path", type=str, default=None)  
    parser.add_option( "--fn", help="Test function to use.  Version 2.0 of the code supports the Griewank "+
        "function and paraboloid used in the paper (in any dimension).  " +
        "Additional functions can be added.  Default 'griewank'.", 
        dest="fn_name", type=str, default="griewank")  
    parser.add_option( "--dim", dest="dim", type=int, default=2, 
        help="Dimension of input space.  Default 2.")
    parser.add_option("--extrap", dest="extrap_thresh", type=float, default=0.0,
        help="Extrapolation threshold parameter passed to DelaunaySparse.  Default 0.0.")
    parser.add_option("--maxsamp", dest="max_samp", type=int, default=20_000,
        help="Max number of samples to draw.  Default = 20,000.")
    parser.add_option("--numtrainpts", dest="numtrainpts", type=int, default=850,
        help="Initial number of samples points (n_0 in the paper).  Default = 850.")
    parser.add_option("--numtestperdim", dest="numtestperdim", type=int, default=999,
        help="Number of test points per dimension. Default = 999 (invokes heuristic for static data).")
    parser.add_option("--logbase", dest="log_base", type=float, default=1.4641,
        help="Upsampling factor b; also the base of the logarithm in rate computation.  Default 1.4641.")
    parser.add_option("--zoomctr", dest="zoom_ctr", type=float, default=0.0,
        help="Zoom modality: used only in conjunction with zoomexp option - see below. " +\
        "Default=0.0.  Use 999.0 in zoomctr or zoomexp to manually specify left/right bounds (not implemented in Version 1.0).")
    parser.add_option("--zoomexp", dest="zoom_exp", type=float, default=1.0,
        help="Zoom modality: set query bounds and bounding box such that (1) center is (x,x,...,x) where x=zoomctr"+\
        " (2) length of query grid is 10e[zoomexp] in each dimension and (3) bounding box determined from testbdsc."+\
        " Default=0.0.  Use 999.0 in zoomctr or zoomexp to manually specify left/right bounds (not implemented in Version 1.0).")
    parser.add_option("--queryleftbd", dest="queryleftbound", type=float, default=0.0,
        help="Left bound of interval used to build query point domain [a, b]^dim. Overwritten if zoom modality is used (see above). Default 0.0")
    parser.add_option("--queryrightbd", dest="queryrightbound", type=float, default=1.0,
        help="Right bound of interval used to build query point domain [a, b]^dim. Overwritten if zoom modality is used (see above). Default 1.0")
    parser.add_option("--bboxleftbd", dest="bboxleftbound", type=float, default=0.0,
        help="Left bound of interval used to build bounding box [a, b]^dim. Overwritten if zoom modality is used (see above). Default 0.0")
    parser.add_option("--bboxrightbd", dest="bboxrightbound", type=float, default=1.0,
        help="Right bound of interval used to build bounding box [a, b]^dim. Overwritten if zoom modality is used (see above). Default 1.0")
    parser.add_option("--testbdsc", dest="tb_scale", type=float, default=0.8,
        help="Query points dimension fraction (qpdf), defined as the side length of the query lattice "
        + "divided by the side length of the bounding box. Default=0.8")
    parser.add_option("--grad", dest="computeGrad", action="store_true", default=True,
	    help="Compute gradients within subroutine that calls DelaunaySparse. Default True.")
    parser.add_option("--outc", dest="out_cor", type=int, default=-1,
        help="Output coordinate to assess.  Default -1 avoids this modality and takes the first output coordinate.")
    parser.add_option("--seed", dest="spec_seed", type=int, default=0,
        help="Value passed as global seed to random number generator.  Default 0.")
    parser.add_option("--itmax", dest="it_max", type=int, default=100,
        help="Max number of iterations.  More robust to use --maxsamp to set threshold.  Default = 100.")
    parser.add_option("--numrates", dest="num_rates", type=int, default=3,
	    help="Target number of rates to compute (i.e. number of points on final figure).  Default = 3.")
    parser.add_option("--minpctile", dest="min_pctile", type=int, default=999,
	    help="For static data, sample on a grid from minpctile to 100-minpctile in each dim.  Default = use built-in heuristic.")
    parser.add_option("--save2static", dest="saveTF", action="store_true", default=False,
	    help="Alternate modality for creating static datasets from test function. If True, sample and evaluate the test function and save to csv file. Default False.")
    parser.add_option("--removeClosePts", dest="removeClose", action="store_true", default=False,
	    help="Alternate modality for removing close points in a data set. If True, load static file and save a new file with close poitns removed. Default False.")
    
    
    (options, args) = parser.parse_args()
    

    def echo_options(options):
        print("Selected options:")
        print()
        print("Job ID:      ", options.jobid) 
        print("Global seed for randomization: ", options.spec_seed)
        print()

        if options.data_path == None: # use test function to acquire new data
            print("Function:    ", options.fn_name)
            print("Dimension:   ", options.dim)
            print()
            print("Query points per dim:", options.numtestperdim)
            print("Total number of query points:", options.numtestperdim ** options.dim)
        print("Global seed for randomization: ", options.spec_seed)
        print()

        if options.data_path == None: # use test function to acquire new data
            print("Function:    ", options.fn_name)
            print("Dimension:   ", options.dim)
            print()
            print("Query points per dim:", options.numtestperdim)
            print("Total number of query points:", options.numtestperdim ** options.dim)

            # Set bounding box left/right bounds based on zoom center, zoom exponent, and scale factor qpdf
            if (options.zoom_ctr != 999 and options.zoom_exp != 999):
                options.bboxleftbound  = np.round(options.zoom_ctr - (10 ** (options.zoom_exp))/options.tb_scale,2)
                options.bboxrightbound = np.round(options.zoom_ctr + (10 ** (options.zoom_exp))/options.tb_scale,2)
            # Set bounding box left/right bounds based on zoom center, zoom exponent, and scale factor qpdf
            if (options.zoom_ctr != 999 and options.zoom_exp != 999):
                options.bboxleftbound  = np.round(options.zoom_ctr - (10 ** (options.zoom_exp))/options.tb_scale,2)
                options.bboxrightbound = np.round(options.zoom_ctr + (10 ** (options.zoom_exp))/options.tb_scale,2)

            # Set query lattice left/right bounds based on bounding box bounds and scale factor qpdf
            tg_scale_fac = (1.0-options.tb_scale)/2
            interval_width = options.bboxrightbound - options.bboxleftbound
            options.queryleftbound  = options.bboxleftbound  + tg_scale_fac * interval_width
            options.queryrightbound = options.bboxrightbound - tg_scale_fac * interval_width
            # Set query lattice left/right bounds based on bounding box bounds and scale factor qpdf
            tg_scale_fac = (1.0-options.tb_scale)/2
            interval_width = options.bboxrightbound - options.bboxleftbound
            options.queryleftbound  = options.bboxleftbound  + tg_scale_fac * interval_width
            options.queryrightbound = options.bboxrightbound - tg_scale_fac * interval_width

            print("Query point bounds in each dim: ", "[", options.queryleftbound, ", ", options.queryrightbound, "]")
            print("Query points dimension fraction (qpdf): ", options.tb_scale)
            print("Bounding box bounds in each dim: ", "[", options.bboxleftbound, ", ", options.bboxrightbound, "]")
            print()
            print("Initial sample size:", options.numtrainpts)
            print("Maximum sample size:", options.max_samp)
            print("Upsampling factor b: ", options.log_base)
            print()
            print("Using gradients? : ", options.computeGrad)
            print("Extrapolation threshold: ", options.extrap_thresh)
            # print("Output cor : ", options.out_cor)
            if options.fn_name not in ['griewank','paraboloid']:
                print("==> ERROR: Requested function ", options.fn_name)
                print("Only the functions 'griewank' and 'paraboloid' are currently included in the code.")
                exit()
        else: # static data path provided
            print("Path to static data: ", options.data_path)
            options.fn_name = 'static'
            options.zoom_ctr = 999.0
            options.zoom_exp = 999.0
        
        if (options.bboxrightbound <= options.bboxleftbound):
            print("Right bound must be larger than left bound")
            exit()
        if (options.tb_scale < 0.0001):
            print("Test bound scale must be > 0")
            exit()
        if options.log_base <= 1:
            print("Log base must be > 1.  Default is 2.0.")
            exit()
        if (options.numtestperdim ** options.dim > 10000) and (options.data_path == None):
            print()
            print("==> WARNING: number of query points = (query pts per dim)^(dim) =",options.numtestperdim ** options.dim,"is very large.")
            print("Exiting.")
            exit()
            print("==> WARNING: number of query points = (query pts per dim)^(dim) =",options.numtestperdim ** options.dim,"is very large.")
            print("Exiting.")
            exit()
        if (options.extrap_thresh < 0 or options.extrap_thresh > 0.5):
            print()
            print("==> Set extrapolation threshold in [0,0.5]")
            exit()
        if (options.min_pctile < 0 or options.min_pctile > 49) and options.min_pctile != 999:
            print()
            print("==> Minimum percentile for static data must be in [0,49]")
        if (options.min_pctile < 0 or options.min_pctile > 49) and options.min_pctile != 999:
            print()
            print("==> Minimum percentile for static data must be in [0,49]")
            exit()

    echo_options(options)

    globalseed = options.spec_seed
    rng = np.random.default_rng(globalseed)  


    if options.saveTF:
        # ALTERNATE MODALITY: 
        #
        # Randomly sample numtrainpts points with specified parameters
        # Evaluate the given test function on each point
        # Save into a csv file
        # Exit
        #

        data_train_inputs, data_train_outputs = make_random_training_in_box(rng)
        all_data_train = pd.concat([data_train_inputs, data_train_outputs],axis=1)
        outfname = 'temp_generated_data.csv'
        all_data_train.to_csv(outfname, index=False, header=False)    
        print(all_data_train)
        print("Saved samples from above dataframe to file", outfname)
        print("Exiting.")
        exit()
    # end saveTF modality

    # torch.manual_seed(globalseed)

    if options.data_path == None: # use test function to acquire new data
        if options.removeClose:
            print("==> ERROR: The removeClosePoints modality is intended for use with static data.")
            print("    Remove this check if use with synthetic data is desired.  Exiting.")
            exit()
        data_train_inputs, data_train_outputs = make_random_training_in_box(rng)    
        data_test_inputs, data_test_outputs = make_test_data_grid(rng)
    else:
        try:
            
            if options.data_path[-4:] == '.csv':
                full_dataset = pd.read_csv(options.data_path, header=None, index_col=None)
            elif options.data_path[-4:] == '.npy':
                full_dataset = pd.DataFrame(np.load(options.data_path))
            dfrowct = full_dataset.shape[0]
            dfcolct = full_dataset.shape[1]
            print("Read in data from path.  Interpreted as",dfrowct,"points in R^",dfcolct-1,"with one output value per point.\n")
            options.dim = dfcolct-1
            print("==> Set dimension based on number of input columns to",dfcolct-1)
            
            options.max_samp = dfrowct
            print("==> Set max sample size to", dfrowct, ", the amount of data points.")
                        
            if options.removeClose:
                # ALTERNATE MODALITY: (for use when DelaunaySparse gives Error Code 30)
                #
                # Group points that are within a specied tolerance of each other in the input space using a kd tree
                # For each group, store the average of each coordinate and output independently, then delete the group
                # Write the result to a file
                # Exit
                #
                # adapted from: 
                # https://www.tutorialguruji.com/python/finding-duplicates-with-tolerance-and-assign-to-a-set-in-pandas/

                import networkx as nx
                from scipy.spatial import KDTree

                def group_neighbors(df, tol, p=np.inf, show=False):
                    r = np.linalg.norm(np.ones(len(tol)), p)
                    print("==> Making kd tree")
                    kd = KDTree(df[tol.index] / tol)
                    print("==> Done making kd tree")
                    g = nx.Graph([
                        (i, j)
                        for i, neighbors in enumerate(kd.query_ball_tree(kd, r=r, p=p))
                        for j in neighbors
                    ])
                    print("==> Done making neighbor graph")
                    if show:
                        nx.draw_networkx(g)
                    ix, id_ = np.array([
                        (j, i)
                        for i, s in enumerate(nx.connected_components(g))
                        for j in s
                    ]).T
                    id_[ix] = id_.copy()
                    return df.assign(set_id=id_)

                inputdf = full_dataset.iloc[:,0:options.dim]
                # for uniform_tol in [1e-7,1e-6,1e-5,1e-4,1e-3]:
                uniform_tol = 1e-5
                array_tol = uniform_tol * np.ones(options.dim)
                tol = pd.Series(array_tol, index=inputdf.columns)
                print("==> Starting group neighbors routine")
                gndf = group_neighbors(inputdf, tol, p=2)
                full_dataset['set_id'] = gndf['set_id']
                full_dataset = full_dataset.groupby(['set_id']).mean().reset_index(drop=True)
                grouped_data_name = options.data_path[:-4]+"_grouped"
                print("==> For tolerance = ", array_tol, " there are ", len(full_dataset), " distinct inputs.")
                np.save(grouped_data_name, full_dataset.to_numpy())
                print("==> SAVED full_dataset with close points average to",grouped_data_name+".npy") 
                print("==> reduced dataset has shape:")
                print(full_dataset.shape)
                print("==> reduced dataset has description:")
                print(full_dataset.describe())
                print("Exiting; re-run using",grouped_data_name+".npy","and remove flag --removeClosePts")
                exit()

            # end removeClose modality



            # options.numtrainpts = int(0.1*dfrowct)
            # print("==> Set initial sample size to", int(0.1*dfrowct), ", roughly 10 percent of data.")
            # options.numtrainpts = int(0.025*dfrowct)
            # print("==> Set initial sample size to", int(0.25*dfrowct), ", roughly 2.5 percent of data.")
            options.numtrainpts = int(0.01*dfrowct)
            print("==> Set initial sample size to", int(0.01*dfrowct), ", roughly 1 percent of data.")
            # options.numtrainpts = 5
            # print("+++++++++ WARNING HARDCODED NUM TRAINTPTS HERE +++++++++")

            options.log_base = 10 ** (1 / (options.dim * (options.num_rates +1)))
            print("==> Requested number of rates to compute (command line option --numrates) =",options.num_rates)
            print("==> Set upsampling factor b to",options.log_base,"(see heuristic in code)")
            # Heuristic:  assuming n_0 = 0.1 * (number of data points), as above,
            #             let n_c = target number of rates to be computed (n_c must be \geq 1) 
            #             set b = 10^[1/(dim *(n_c+2))]
            #             this will use close to all the data in the final iteration.
            if options.min_pctile == 999:
                print("==> Using heuristic to set percentile bounds for query point grid (comment in code)")
                # Heuristic based on experimental studies for 10,000 points collected by Latin Hypercube sampling
                # You can adjust the percentile by the command line option --minpctile
                #   ideally it should be as low as possible while still having little to no extrapolation
                # 
                #          dim 2       3       4       5       6      7+
                # pctile range [5,95] [10,90] [15,85] [20,80] [25,75] [25,75]
                #
                pct_map = np.array([0,0,5,10,15,20,25])
                if options.dim < 7:
                    options.min_pctile = pct_map[options.dim]
                else:
                    options.min_pctile = 25
            print("==> Set grid of query points from",options.min_pctile,"th-",100-options.min_pctile,"th percentiles.")
            
            max_query_pts = 10000
            side_length_pctle_box = 0.02*(50-options.min_pctile)
            est_num_pts_in_pctle_box = options.max_samp * (side_length_pctle_box ** options.dim)
            target_num_query_pts = np.min([est_num_pts_in_pctle_box/(2**options.dim),max_query_pts])
            if options.numtestperdim == 999:
                options.numtestperdim = max(int(target_num_query_pts ** (1/options.dim)),2)
            print("==> Set the number of query points per dimension to",options.numtestperdim)
            print("     (aiming for roughly",np.round(est_num_pts_in_pctle_box/(2**options.dim)),"query points or 10,000, whichever is smaller.)")
            
            print()
            print("Initial sample size:", options.numtrainpts)
            print("Query points per dim:", options.numtestperdim)
            print("Total number of query points:", options.numtestperdim ** options.dim)
            print("Upsampling factor b: ", options.log_base)
            print()
        except:
            if options.removeClose:
                exit()
            else:
                print("\n Error reading in data.  To debug, check that:",
                        "\n  (1) path printed above is correct,",
                        "\n  (2) file format is .csv of numerical data where each row is",
                                "the input coordinates followed by 1 output,"
                        "\n  (3) sample files from ddd/staticdata/examples/ load sucessfully")
                exit()
            
        ########################
        # shuffle dataset and make input/output datasets
        ########################
        shuffle_seed = rng.integers(low=0, high=1_000_000, size=1)
        shuffle_seed = shuffle_seed[0] # converts array of size 1 to an integer
        full_dataset = full_dataset.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        print("==> Shuffled dataset based on provided seed.")
        full_dataset_inputs = full_dataset.iloc[:,0:options.dim]  # inputs  = all but last column
        full_dataset_outputs = full_dataset.iloc[:,-1:] # outputs = last column

        ########################
        # create first batch of sample points 
        ########################
        data_train_inputs  = full_dataset_inputs.iloc[0:options.numtrainpts, :]  
        data_train_outputs = full_dataset_outputs.iloc[0:options.numtrainpts, :]  
        
        ########################
        # create grid of query points by looking at percentiles of full data set
        ########################

        cols = full_dataset_inputs.columns
        num_cols = len(cols)
        
        x = np.zeros([num_cols, options.numtestperdim])

        print("==> Description of inputs from full dataset:")
        print(full_dataset_inputs.describe())
        
        i = 0
        for col in cols: 
            # set coordinate mins and maxes according to 25-75 percentiles
            cor_min = full_dataset_inputs.quantile(q=0.01*options.min_pctile)[col]
            cor_max = full_dataset_inputs.quantile(q=1-0.01*options.min_pctile)[col]
            x[i] = np.linspace(cor_min, cor_max, options.numtestperdim)
            i += 1

        mg_in = []
        for i in range(num_cols):
            mg_in.append(x[i])
        grid_pts = np.array(np.meshgrid(*mg_in))
        grid_pts = grid_pts.reshape(options.dim, options.numtestperdim ** options.dim)
        grid_pts = grid_pts.T

        outputs_on_grid = []
        data_test_inputs  = pd.DataFrame(grid_pts)
        data_test_outputs = pd.DataFrame(outputs_on_grid)  # intentionally empty df
        print("==> Query point grid has",data_test_inputs.shape[0],"points in R^"+str(data_test_inputs.shape[1]))

    # end else; now data_{train,test}_{inputs,outputs} are set in either case
    
    if options.data_path == None: # use test function to acquire new data
        outfname = 'zz-' + str(options.jobid) + "-" + str(options.fn_name) + "-d" + str(options.dim) + "-tpd" + str(options.numtestperdim) + "-lb" + str(options.bboxleftbound) + "-rb" + str(options.bboxrightbound) + "-tb" + str(options.tb_scale) + "-log" + str(options.log_base) +".csv"
        if (options.zoom_ctr != 999.0 and options.zoom_exp != 999.0): # add in -zoom[exponent value] before csv
            outfname = outfname[:-4] + "-zoom" + str(options.zoom_exp) + ".csv"
        if (options.spec_seed != 0): # add in -seed[seed value] before csv
            outfname = outfname[:-4] + "-seed" + str(options.spec_seed) + ".csv"
    else:
        outfname = 'zz-' + str(options.jobid) + "-" + str(options.fn_name) + "-d" + str(options.dim) + "-tpd" + str(options.numtestperdim) + "-lb" + str(options.min_pctile) + "pctle" + "-log" + str(options.log_base) +".csv"
        if (options.spec_seed != 0): # add in -seed[seed value] before csv
            outfname = outfname[:-4] + "-seed" + str(options.spec_seed) + ".csv"
    print("==> Output will be stored in file ",outfname)

    results_df = []
    all_pts_in  = copy.deepcopy(data_train_inputs)
    all_pts_out = copy.deepcopy(data_train_outputs)

    if (options.out_cor == -1): # default
        out_coord = 0 # this means we will only measure error in 0th component of output; no problem if codomain is R^1
    else:
        out_coord = options.out_cor

    print("")
    print("==============================================")
    # print("For output coordinate ", out_coord,": ")
    print("===  results for", options.fn_name, "dim", options.dim, "seed", options.spec_seed, " ===")
    print("==============================================")
    print("==============================================")
    # print("For output coordinate ", out_coord,": ")
    print("===  results for", options.fn_name, "dim", options.dim, "seed", options.spec_seed, " ===")
    print("==============================================")
    print("samples | density | prop extrap | MSD diff | MSD rate | grad diff | grad rate | analytic diff | analytic rate ") 
    print("")

    prev_error = 999999
    prev_vals_at_test = []
    prev_extrap_indices = []
    prev_extrap_indices = []
    prev_diff = 999999

    ########################################################################
    # create list of number of samples for each update step; 
    #    have to do in advance to avoid rounding issues
    #    can also help in future applications to know sampling rate calculation a priori
    #######################################################################

    quitloop = False
    num_samples_to_add   = np.zeros(options.it_max+1)
    total_samples_so_far = np.zeros(options.it_max+1)
    total_samples_so_far[0] = all_pts_in.shape[0]
    
    for i in range(options.it_max): # i = number of "refinements" of interpolant
        if quitloop:
            break
        #
        # upsampling rule:
        # update number of samples by replacing 2 points per unit per dimension with (logbase + 1) points per unit per dimension      
        #   ALSO: round to the nearst integer and cast as an integer - this is essential for the static data case
        #         otherwise you may add the same sample point more than once, causing an error for DS
        

        total_samples_so_far[i+1] = int(np.round(np.power((options.log_base*np.power(total_samples_so_far[i], 1/options.dim) - (options.log_base - 1)),options.dim)))
        num_samples_to_add[i]     = int(total_samples_so_far[i+1] - total_samples_so_far[i])
        if (total_samples_so_far[i+1] > options.max_samp):
            quitloop = True

    ########################################################################
    # do iterative improvement according to upsampling schedule saved in num_samples_to_add
    #######################################################################

    quitloop = False
    print_extrap_warning = False
    print_extrap_warning = False

    for i in range(options.it_max): # i = number of "refinements" of interpolant

        if quitloop:
            break

        # use the accumulated sample points as the training and the fixed test data sets as the test data
        interp_out_n, actual_test_vals, actual_train_vals, extrap_indices, grad_est_DS = compute_DS_only(all_pts_in, all_pts_out, data_test_inputs, data_test_outputs)

        prop_extrap_iterate = len(extrap_indices)/interp_out_n.shape[1]
        # print('====> proportion extrapolated = %1.2f' % prop_extrap_iterate)
        density_of_sample = all_pts_in.shape[0] ** (1/options.dim)

        # Make mask for extrapolation points for current and prev interpolation values
        #   mask according to previous iteration, 
        #   whose extrap pts are a superset of extrap pts in current iteration
        #   (This computation is irrelevant in the case i=0)

        if i==0:
            interp_out_int_only = interp_out_n
            prev_vals_int_only = prev_vals_at_test
        else:
            mask_extrap = np.zeros_like(interp_out_n, dtype=int)
            mask_extrap[:,prev_extrap_indices] = 1
            interp_out_int_only = ma.masked_array(interp_out_n, mask_extrap)
            prev_vals_int_only  = ma.masked_array(prev_vals_at_test, mask_extrap)


        if options.data_path == None: # use test function to acquire new data 
            # for analytical functions, we can compute the "actual" rate of convergence, for reference
            ds_vs_actual_at_test = np.sqrt(((interp_out_n[out_coord,:]-actual_test_vals[out_coord,:]) ** 2).mean())
            if (i == 0): 
                error_rate = 0
            else:
                error_rate =  (np.log(prev_error/ds_vs_actual_at_test))/np.log(options.log_base)
        else: # have static data; actual_test_vals = [] which causes an error below
            actual_test_vals = np.zeros_like(interp_out_n)
            ds_vs_actual_at_test = 0
            error_rate = 0
        # Make mask for extrapolation points for current and prev interpolation values
        #   mask according to previous iteration, 
        #   whose extrap pts are a superset of extrap pts in current iteration
        #   (This computation is irrelevant in the case i=0)

        if i==0:
            interp_out_int_only = interp_out_n
            prev_vals_int_only = prev_vals_at_test
        else:
            mask_extrap = np.zeros_like(interp_out_n, dtype=int)
            mask_extrap[:,prev_extrap_indices] = 1
            interp_out_int_only = ma.masked_array(interp_out_n, mask_extrap)
            prev_vals_int_only  = ma.masked_array(prev_vals_at_test, mask_extrap)


        if options.data_path == None: # use test function to acquire new data 
            # for analytical functions, we can compute the "actual" rate of convergence, for reference
            ds_vs_actual_at_test = np.sqrt(((interp_out_n[out_coord,:]-actual_test_vals[out_coord,:]) ** 2).mean())
            if (i == 0): 
                error_rate = 0
            else:
                error_rate =  (np.log(prev_error/ds_vs_actual_at_test))/np.log(options.log_base)
        else: # have static data; actual_test_vals = [] which causes an error below
            actual_test_vals = np.zeros_like(interp_out_n)
            ds_vs_actual_at_test = 0
            error_rate = 0

        # difference and rate computation steps for Algorithms 3.1 and 3.2
        if (i == 0):
            new_vs_prev_at_test = 0
            diff_rate = 0
            prev_diff = 0
            if (options.computeGrad):
                grad_new_vs_prev_at_test = 0
                grad_diff_rate = 0
                grad_prev_diff = 0
        elif (i == 1):
            new_vs_prev_at_test = np.sqrt(((interp_out_n[out_coord,:]-prev_vals_at_test[out_coord,:]) ** 2).mean())
            diff_rate = 0
            prev_diff = new_vs_prev_at_test
            if (options.computeGrad):
                ## grad_est_DS may have nans as a flag for extrapolation (look for grad_out[:] = np.nan above)
                ## so we mask for nans, compress to remove nans, and then compute norm
                temp = grad_est_DS - grad_prev_vals_at_test
                temp = np.ma.array(temp, mask=np.isnan(temp))
                grad_new_vs_prev_at_test = np.linalg.norm(temp.compressed())
                ## grad_est_DS may have nans as a flag for extrapolation (look for grad_out[:] = np.nan above)
                ## so we mask for nans, compress to remove nans, and then compute norm
                temp = grad_est_DS - grad_prev_vals_at_test
                temp = np.ma.array(temp, mask=np.isnan(temp))
                grad_new_vs_prev_at_test = np.linalg.norm(temp.compressed())
                grad_diff_rate = 0
                grad_prev_diff = grad_new_vs_prev_at_test
        else: # i > 1
            new_vs_prev_at_test = np.sqrt(((interp_out_n[out_coord,:]-prev_vals_at_test[out_coord,:]) ** 2).mean())
            # computation of r_k for MSD rate
            diff_rate = np.log(prev_diff/new_vs_prev_at_test)/np.log(options.log_base) 
            prev_diff = new_vs_prev_at_test
            if (options.computeGrad):
                ## grad_est_DS may have nans as a flag for extrapolation (look for grad_out[:] = np.nan above)
                ## so we mask for nans, compress to remove nans, and then compute norm
                temp = grad_est_DS - grad_prev_vals_at_test
                temp = np.ma.array(temp, mask=np.isnan(temp))
                grad_new_vs_prev_at_test = np.linalg.norm(temp.compressed())

                ## original line - does not check for nans:
                ## grad_new_vs_prev_at_test = np.linalg.norm(grad_est_DS - grad_prev_vals_at_test)
                
                ## grad_est_DS may have nans as a flag for extrapolation (look for grad_out[:] = np.nan above)
                ## so we mask for nans, compress to remove nans, and then compute norm
                temp = grad_est_DS - grad_prev_vals_at_test
                temp = np.ma.array(temp, mask=np.isnan(temp))
                grad_new_vs_prev_at_test = np.linalg.norm(temp.compressed())

                ## original line - does not check for nans:
                ## grad_new_vs_prev_at_test = np.linalg.norm(grad_est_DS - grad_prev_vals_at_test)
                
                grad_diff_rate = -np.log(grad_new_vs_prev_at_test/grad_prev_diff)/np.log(options.log_base)
                grad_prev_diff = grad_new_vs_prev_at_test

        if (options.computeGrad == False): 
            grad_new_vs_prev_at_test = 0.0
            grad_diff_rate = 0.0

        if (i == 0):
            print(("% 6i & %3.2f & %1.2f & -- & -- & -- & -- & %5.5f & -- \\\\") % (all_pts_in.shape[0], density_of_sample,  prop_extrap_iterate, ds_vs_actual_at_test), flush=True)
            if prop_extrap_iterate > 0:
                print_extrap_warning = True
            if prop_extrap_iterate > 0:
                print_extrap_warning = True
        elif (i == 1):
            print(("% 6i & %3.2f & %1.2f & %5.5f & -- & %5.5f & -- & %5.5f & %5.2f \\\\") % (all_pts_in.shape[0], density_of_sample,  prop_extrap_iterate, new_vs_prev_at_test, grad_new_vs_prev_at_test, ds_vs_actual_at_test, error_rate), flush=True)
        else:
            print(("% 6i & %3.2f & %1.2f & %5.5f & %5.2f & %5.5f & %5.2f & %5.5f & %5.2f \\\\") % (all_pts_in.shape[0], density_of_sample,  prop_extrap_iterate, new_vs_prev_at_test, diff_rate, grad_new_vs_prev_at_test, grad_diff_rate, ds_vs_actual_at_test, error_rate), flush=True)
        
        # note all_pts_in.shape[0] should be identical to total_samples_so_far[i]
        results_df.append({
            "dim of intput"    : options.dim,
            "function name"    : options.fn_name,
            "num test points"  : options.numtestperdim ** options.dim,
            "left bound"       : options.bboxleftbound,
            "right bound"      : options.bboxrightbound,
            "test grid scale"  : options.tb_scale,
            "iterate"          : i, 
            "samples"          : all_pts_in.shape[0], 
            "density"          : density_of_sample,  
            "prop extrap"      : prop_extrap_iterate, 
            "iterate diff"     : new_vs_prev_at_test,
            "iterate rate"     : diff_rate, 
            "grad diff"        : grad_new_vs_prev_at_test, 
            "grad rate"        : grad_diff_rate, 
            "actual diff"      : ds_vs_actual_at_test, 
            "actual rate"      : error_rate,
            "log base"         : options.log_base,
            "seed"             : options.spec_seed,
        })

        prev_error = ds_vs_actual_at_test
        prev_vals_at_test = interp_out_n
        prev_extrap_indices = extrap_indices
        prev_extrap_indices = extrap_indices
        if (options.computeGrad):
            grad_prev_vals_at_test = grad_est_DS


            

        ##########################################
        # get more samples for next iteration
        ##########################################
        
        # # now int(round(...)) is done at creation of num_samples_to_add
        # options.numtrainpts = int(np.round(num_samples_to_add[i]))
        options.numtrainpts = int(num_samples_to_add[i])

        # check if we will go over max samples
        if (total_samples_so_far[i] + options.numtrainpts > options.max_samp):
            print("")
            print("==> Next step would go over ", options.max_samp, " samples; breaking loop.")
            print("==> Next step would go over ", options.max_samp, " samples; breaking loop.")
            # print("====>    ALSO: setting numtrainpts to 1 to avoid generating unused points ")
            options.numtrainpts = 1
            quitloop = True
            if print_extrap_warning:
                print("====> WARNING: Extrapolation occured in some cases - see prop extrap column above.")
                print("====>          If the proportion of extrapoation does not go to zero quickly in most")
                print("====>          trials, increase the value of --minpctile closer to 50.")
            if print_extrap_warning:
                print("====> WARNING: Extrapolation occured in some cases - see prop extrap column above.")
                print("====>          If the proportion of extrapoation does not go to zero quickly in most")
                print("====>          trials, increase the value of --minpctile closer to 50.")

        if options.data_path == None: # use test function to acquire new data 
            new_sample_inputs, new_sample_outputs = make_random_training_in_box(rng)
        else: # have static data set
            start = int(total_samples_so_far[i])                        # index to start next sample batch
            stop  = int(total_samples_so_far[i]) + options.numtrainpts  # index to end   next sample batch
            new_sample_inputs  = full_dataset.iloc[start:stop, 0:options.dim]
            new_sample_outputs = full_dataset.iloc[start:stop,-1:] # NOTE: check selection of output coord

        if options.data_path == None: # use test function to acquire new data 
            new_sample_inputs, new_sample_outputs = make_random_training_in_box(rng)
        else: # have static data set
            start = int(total_samples_so_far[i])                        # index to start next sample batch
            stop  = int(total_samples_so_far[i]) + options.numtrainpts  # index to end   next sample batch
            new_sample_inputs  = full_dataset.iloc[start:stop, 0:options.dim]
            new_sample_outputs = full_dataset.iloc[start:stop,-1:] # NOTE: check selection of output coord


        ##########################################
        # update sample set for next iteration
        ##########################################

        all_pts_in = pd.concat([all_pts_in, new_sample_inputs])
        all_pts_out = pd.concat([all_pts_out, new_sample_outputs])
        
        columns = [ "dim of intput"    ,
                    "function name"    ,
                    "num test points"  ,
                    "left bound"       ,
                    "right bound"      ,
                    "test grid scale"  ,
                    "iterate"          ,
                    "samples"          ,
                    "density"          ,
                    "prop extrap"      ,
                    "iterate diff"     ,
                    "iterate rate"     ,
                    "grad diff"        ,
                    "grad rate"        ,
                    "actual diff"      ,
                    "actual rate"      ,
                    "log base",
                    "seed",
                ]
        
        df = pd.DataFrame(results_df, columns=columns)
        df.to_csv(outfname)
        
    
    # print("====> final all_pts_out shape was ", all_pts_out.shape)
    # print("===> done with output coord ", out_coord)
    print("==> Results saved in ", outfname)
    print("==> Results saved in ", outfname)

