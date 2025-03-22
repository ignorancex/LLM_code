"""
    Reconstruct the MalConv architecture from the Flush+Reload results
"""
# basic
import os
import json
import argparse
from math import ceil
from copy import deepcopy
from itertools import product

# externals (numpy, networkx, matplotlib)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# custom utils
from utils.loaders import load_from_csv
from utils.computations import _COMPUTE_UNARY_PT, _COMPUTE_BINARY_PT, _check_computation
from utils.architectures import _same_inout_dimensions, _sane_narrow_connection


# ------------------------------------------------------------------------------
#   Global variables
# ------------------------------------------------------------------------------
_save_dir = os.path.join('results', 'reconstruct', 'malconv')
_datafile = 'dataset'
_zerocase = 1       # the margin that we allow when the processing time is '0'
_lin_case = 0.1     # the margin that we allow (for the linear layers)
_convcase = 0.05    #           "              (for the conv layers)


# ------------------------------------------------------------------------------
#   Reconstruction code
# ------------------------------------------------------------------------------
def reconstruct_malconv( \
    csvfile, indim, outdim, \
    timer, dataloc, resolution=2000, verbose=True):

    # load the data from the csvfile
    csv_data = load_from_csv(csvfile)
    assert csv_data, 'Error: cannot read the content of [{}]'.format(csvfile)
    csv_data = [ \
        (each_data[0], float(each_data[1]), float(each_data[2])) for each_data in csv_data]
    print ('[reconstruction] load the total [{}] events'.format(len(csv_data)))

    # reconstruction of the computational graphs
    #  based on the architecture characteristics
    computational_graphs = reconstruct_computational_graphs(csv_data, stdout=verbose)
    print ('[reconstruction] identified ' + \
            '[{}] computational graphs'.format(len(computational_graphs)))

    # decide the reconstructed computational graphs
    #  with the possible attribute parameter combinations
    architecture_graphs = \
        reconstruct_attribute_parameters( \
            computational_graphs, csv_data, timer, dataloc, resolution)
    print ('[reconstruction] we decided the ' + \
            'candidates into [{}]'.format(len(architecture_graphs)))

    # prune with more rules
    architecture_graphs = \
        prune_reconstructed_architectures(architecture_graphs)
    print ('[reconstruction] we pruned the ' + \
            'candidates into [{}]'.format(len(architecture_graphs)))

    # returns the architectures
    return architecture_graphs


def reconstruct_computational_graphs(fcalls, stdout=False):
    # output the function calls (input)
    if stdout:
        print ('----- Func. calls -----')
        for ecall in fcalls: print (ecall)
        print ('-----------------------')

    # recursively create the graph
    extract_fnames = [fcall[0] for _, fcall in enumerate(fcalls)]
    compute_graphs = _recursive_reconstruction_graphs(extract_fnames)

    # check if the store location exists
    store_loc = os.path.join(_save_dir, 'computational_graphs')
    if not os.path.exists(store_loc): os.makedirs(store_loc)

    # store only the graphs
    compute_graphs = [each_graph for (_, each_graph) in compute_graphs]

    # store the visualizations
    options = {
        'node_color': 'red',
        'node_size': 40,
        'width': 1,
        'alpha': 0.8,
        'arrowstyle': '-|>',
        'arrowsize': 8,
        'font_size': 10,
    }
    for cidx, each_graph in enumerate(compute_graphs):
        nx.draw_networkx(each_graph, arrows=True, **options)
        plt.savefig(os.path.join(store_loc, 'arch_{}.pdf'.format(cidx)))
        plt.clf()

    return compute_graphs

def _recursive_reconstruction_graphs(efcalls):
    # estimation starts from the back...
    last_fcall = efcalls.pop()

    # ----------------------------------------
    # Base case: when there is no more calls
    # ----------------------------------------
    if not efcalls:
        # : create a graph and return with the last element
        newG = nx.DiGraph()
        newG.add_node(last_fcall)
        new_data = (last_fcall, newG)
        return [new_data]

    # ----------------------------------------
    # Recursion: perform DFSes
    # ----------------------------------------
    else:
        if _check_computation(last_fcall, _COMPUTE_UNARY_PT):
            # :: create an edge for each graph in the list
            prevGs = _recursive_reconstruction_graphs(efcalls.copy())
            uretGs = []
            for (prev_fcall, prevG) in prevGs:
                prevG.add_edge(prev_fcall, last_fcall)
                uretGs.append((last_fcall, prevG))
            return uretGs

        elif _check_computation(last_fcall, _COMPUTE_BINARY_PT):
            """
                For each preceding element except the previous one,
                we assume a branch started from there and split the
                elements in between into two sets.
                ex. a -> b -> c -> e -> f
                         b -> d -> e
                    we assumed 'b' is the branch start, then:
                    split the [c, d] into two sets, i.e.,
                    ([], [c,d]), ([c], [d]) -- candidates.
            """
            bretGs = []

            # :: assume, for each preceding element, a branch exists
            for bidx in range(1, len(efcalls)-1):
                fcalls_between = efcalls[bidx:len(efcalls)]
                fcalls_remains = efcalls[:bidx]

                # ::: Operations:
                #  - split the f-calls in between into two lists
                #  - recusively run the reconstructions for each
                for sidx in range(len(fcalls_between)):
                    # - split
                    fcalls_branch1 = (fcalls_remains + fcalls_between[:sidx])
                    fcalls_branch2 = (fcalls_remains + fcalls_between[sidx:])

                    # - recursive reconstructions
                    prevGs_branch1 = _recursive_reconstruction_graphs(fcalls_branch1.copy())
                    prevGs_branch2 = _recursive_reconstruction_graphs(fcalls_branch2.copy())

                    # - combine them
                    prevGs = []
                    for (prev_fcall1, prevG1), (prev_fcall2, prevG2) \
                        in product(prevGs_branch1, prevGs_branch2):
                        bretG = nx.compose(prevG1, prevG2)
                        bretG.add_edge(prev_fcall1, last_fcall)
                        bretG.add_edge(prev_fcall2, last_fcall)
                        bretGs.append((last_fcall, bretG))
                    # end for (prev...
                # end for sidx...
            # end for bidx...
            return bretGs
        else:
            assert False, ('[_recursive_reconstruction_graphs] ' + \
                            'Unknown layer - [{}]'.format(last_fcall))
    # done (the function will not be reached at this point...)

def reconstruct_attribute_parameters( \
    compute_graphs, fcalls, timer, dataloc, resolution):
    # choose the parameter candidates
    #  for the (convolutional and linear) operations
    parameter_database = \
        load_parameter_database(fcalls, timer, dataloc, resolution)

    # data-holders
    architecture_graphs = []
    compute_start = fcalls[len(fcalls)-1][0]   # ex. '[12] Sigmoid', '[0] Embedding'
    compute_termi = fcalls[0][0]
    start_dimension = (1,)
    termi_dimension = (2*1000*1000, 8)

    # loop through the candidates
    for cidx, each_compute_graph in enumerate(compute_graphs):
        """
            Note: this recursion doesn't work for the different structures of
                  computational graphs: at the initial call, we provide the only
                  one structure in the list argument, ex. [each_compute_graph].
        """
        # : do reconstruction of parameters
        cur_arch_graphs = \
            _recursive_reconstruction_parameters( \
                [each_compute_graph], parameter_database, \
                compute_start, start_dimension, \
                compute_termi, termi_dimension, \
                reverse=True, verbose=False)
        # : store when it's not None
        if cur_arch_graphs:
            # :: reversing the results
            cur_arch_graphs = [ \
                each_graph.reverse() for each_graph in cur_arch_graphs]
            # :: store to the list
            architecture_graphs += cur_arch_graphs
    # end for cidx...

    # check if the store location exists
    store_loc = os.path.join(_save_dir, 'architecture_candidates')
    if not os.path.exists(store_loc): os.makedirs(store_loc)

    # store the visualizations
    options = {
        'node_color': 'red',
        'node_size': 40,
        'width': 1,
        'alpha': 0.8,
        'arrowstyle': '-|>',
        'arrowsize': 8,
        'font_size': 10,
    }
    for cidx, each_agraph in enumerate(architecture_graphs):
        # : relabel the node names to include the attribute parameters
        new_alabels = {}
        for node, data in each_agraph.nodes(data=True):
            new_attr = '{}'.format(node)
            if ('attr_param' in data) \
                and data['attr_param']:
                new_attr += ' - {}'.format(data['attr_param'])
            new_alabels[node] = new_attr
        new_agraph = nx.relabel_nodes(each_agraph, new_alabels, copy=True)

        # : networkx - draw the graphs
        nx.draw_networkx(new_agraph, arrows=True, **options)
        plt.savefig(os.path.join(store_loc, 'arch_params_{}.pdf'.format(cidx)))
        plt.clf()

    return architecture_graphs

def load_parameter_database( \
    efcalls, ctimer, dataloc, resolution):
    # load the data from the stored locations
    profile_datasets = {
        'conv': _load_dataset('conv', dataloc, ctimer),
        'fc'  : _load_dataset('fc', dataloc, ctimer),
    }

    # data-holders
    candidate_params = {}

    # loop through the extracted calls
    for (cname, cwhen, ctime) in efcalls:
        # : estimate the candidate parameters
        #   based on the linear computation profiles...
        if 'FC' in cname:
            if not ctime:
                # :: set the lower/upper bounds
                lower_bound, upper_bound = 0, _zerocase
                if ctimer == 'tsc':
                    lower_bound, upper_bound = \
                        lower_bound*resolution, upper_bound*resolution

                # :: collect the candidates
                cur_candidates = []
                for each_profile in profile_datasets['fc']:
                    """
                        Profile data: in, out, comp, time
                    """
                    if lower_bound <= each_profile[3] <= upper_bound:
                        cur_candidates.append(tuple(each_profile))
                cur_candidates = list(set(cur_candidates))

                # :: add to the data-holder
                candidate_params[cname] = cur_candidates

            else:
                # :: set the lower/upper bounds
                lower_bound, upper_bound = \
                    ctime * (1. - _lin_case), ctime * (1. + _lin_case)
                if ctimer == 'tsc':
                    lower_bound, upper_bound = \
                        lower_bound*resolution, upper_bound*resolution

                # :: collect the candidates
                cur_candidates = []
                for each_profile in profile_datasets['fc']:
                    """
                        Profile data: in, out, comp, time
                    """
                    if lower_bound <= each_profile[3] <= upper_bound:
                        cur_candidates.append(tuple(each_profile))
                cur_candidates = list(set(cur_candidates))

                # :: add to the data-holder
                candidate_params[cname] = cur_candidates

        # : estimate the candidate parameters
        #   based on the linear computation profiles...
        elif 'Convolution' in cname:
            if not ctime:
                # :: set the lower/upper bounds
                lower_bound, upper_bound = 0, _zerocase
                if ctimer == 'tsc':
                    lower_bound, upper_bound = \
                        lower_bound*resolution, upper_bound*resolution

                # :: collect the candidates
                cur_candidates = []
                for each_profile in profile_datasets['conv']:
                    """
                        Profile data: data, in, out, kern, stride, comp, time
                    """
                    if lower_bound <= each_profile[6] <= upper_bound:
                        cur_candidates.append(tuple(each_profile))
                cur_candidates = list(set(cur_candidates))

                # :: add to the data-holder
                candidate_params[cname] = cur_candidates

            else:
                # :: set the lower/upper bounds
                lower_bound, upper_bound = \
                    ctime * (1. - _convcase), ctime * (1. + _convcase)
                if ctimer == 'tsc':
                    lower_bound, upper_bound = \
                        lower_bound*resolution, upper_bound*resolution

                # :: collect the candidates
                cur_candidates = []
                for each_profile in profile_datasets['conv']:
                    """
                        Profile data: data, in, out, kern, stride, comp, time
                    """
                    if lower_bound <= each_profile[6] <= upper_bound:
                        cur_candidates.append(tuple(each_profile))
                cur_candidates = list(set(cur_candidates))

                # :: add to the data-holder
                candidate_params[cname] = cur_candidates

        else:
            continue
    # end for (cname...

    # return the candidates
    return candidate_params

def _load_dataset(dataset, dataloc, ctimer):
    # compose the datafile to use
    if ('tsc' == ctimer) or ('schannel' == ctimer):
        datafname = os.path.join( \
            dataloc, dataset, '{}.{}.npy'.format(_datafile, ctimer))
    else:
        assert False, ('[_load_dataset] Error - undefined timer - {}, abort'.format(ctimer))

    # read the numpy data
    profile_dataset = np.load(datafname)
    return profile_dataset

def _recursive_reconstruction_parameters( \
    compute_graphs, parameter_database, \
    compute_curr, curr_dout, compute_term, term_dout, \
    reverse=True, verbose=False):
    # reverse the connection (only at the first call)
    if reverse:
        compute_graphs = [ \
            compute_graph.reverse() \
            for compute_graph in compute_graphs]

    # print-out the status
    if verbose:
        print ('[_recursive_recon_params] ' + \
                '\'{} {}\''.format(compute_curr, curr_dout) + \
                ' to end \'{} {}\''.format(compute_term, term_dout))

    # ----------------------------------------
    # Ops: Set the output dimension of a node
    # ----------------------------------------
    for compute_graph in compute_graphs:
        # : set the node attribute
        nx.set_node_attributes( \
            compute_graph, { compute_curr: { 'out_dim': curr_dout } })

    # constructed architectures (to return)
    candidate_architectures = []

    # ----------------------------------------
    # Base case: when we reached the terminal
    # ----------------------------------------
    if compute_curr == compute_term:
        # : check if we have the dimension as we expected
        if curr_dout != term_dout:
            if verbose:
                print ('[_recursive_recon_params] base, ' + \
                    '{} != {}, fail.'.format(curr_dout, term_dout))
            # return nothing, empty architectures
            return candidate_architectures
        else:
            # :: print-out the status, base case
            if verbose:
                print ('[_recursive_recon_params] base, ' + \
                    '{} == {}, success.'.format(curr_dout, term_dout))

            # return the architecture as a list: to compute...
            candidate_architectures += compute_graphs
            return candidate_architectures

    # ----------------------------------------
    # Recursion
    # ----------------------------------------
    else:
        """
            Estimate the candidate parameters
             1. 'conv, linear': choose based on the profile databases
             2. 'same' in/out: use the same dimensions
             3. 'transpose':
                (1) 1-dim: use the same dimension
                (2) 2-dim: swap the two axises
             4. 'narrow': list of possible candidates (4 -> 4, 5, 6, 7, 8)
             5. TBD...
        """
        # : estimate candidate parameters
        if ('FC' in compute_curr):
            parameter_candidates = \
                _search_linear_database(parameter_database, compute_curr, curr_dout)

        elif ('Convolution' in compute_curr):
            # :: search the candidates....
            parameter_candidates = \
                _search_conv1d_database(parameter_database, compute_curr, curr_dout)

        # : when it have the same in/out dimensions
        elif _same_inout_dimensions(compute_curr):
            # :: no specific info, by pass the information
            parameter_candidates = [(curr_dout, None)]

        # : when it is the transpose operation
        elif 'transpose' in compute_curr:
            # :: based on the output dimensions
            #   (1D - no op, 2D - swap, 3D - swap any two)
            parameter_candidates = []
            if len(curr_dout) == 1:
                parameter_candidates.append((curr_dout, None))
            elif len(curr_dout) == 2:
                swap_dout = tuple(reversed(curr_dout))
                parameter_candidates.append((swap_dout, None))
            else:
                assert False, \
                    ('[_recursive_reconstruction_parameters] ' + \
                        'transpose with {}-dims is undefined'.format(len(curr_dout)))

        # : MaxPool1d: list all the factors of the dimensions
        elif 'MaxPool1d' in compute_curr:
            # :: consider all the factors of a dimension
            #    (under the assumption of kernel == dimension)
            parameter_candidates = []
            for each_factor in _compute_factors(2*1000*1000):
                # [Note] that max-pool only decreases the dimension-size
                if curr_dout[0] < each_factor:
                    parameter_candidates.append( \
                        ((curr_dout[0], each_factor), None))

        # : 'narrow', splits the dimension into two intervals
        elif 'narrow' in compute_curr:
            # :: data-holders
            expand_curr = int(curr_dout[0])
            expand_term = int(term_dout[1])

            # :: compute the candidates: split a dimension
            parameter_candidates = []
            for each_dim in range(expand_curr+1, expand_term+1):
                parameter_candidates.append(
                    ((each_dim, curr_dout[1]), None))

        # : 'view', usually used to linearize a multi-dimensional tensor
        #       into one dimensional tensor, before the linear operation
        elif 'view' in compute_curr:
            raise NotImplementedError

        # : undefined cases...
        else:
            assert False, \
                ('[_recursive_reconstruction_parameters] ' + \
                    'undefined computation - {}'.format(compute_curr))

        """
            Recursive computations:
        """
        for each_pcandidate in parameter_candidates:
            cur_candidate_indim = each_pcandidate[0]
            cur_candidate_pinfo = each_pcandidate[1]

            # --------------------------------------------------
            # Ops: Set the input dimension and info, at the node
            # --------------------------------------------------
            for compute_graph in compute_graphs:
                # ::: set the node attribute
                nx.set_node_attributes( \
                    compute_graph, { \
                        compute_curr: {
                            'in_dim': cur_candidate_indim,
                            'attr_param': cur_candidate_pinfo,
                        }})

            # :: copy the entire compute graphs
            cur_compute_graphs = deepcopy(compute_graphs)

            # :: loop over the each computational graph
            for each_compute_graph in cur_compute_graphs:

                # ::: data containers
                #  (list of computational graphs from successors)
                list_of_compute_graphs_from_successors = []

                # ::: loop over the multiple successors
                for each_successor in each_compute_graph.successors(compute_curr):

                    # :::: recursively call for each successors
                    each_successor_compute_graphs = \
                        _recursive_reconstruction_parameters( \
                            [each_compute_graph], parameter_database, \
                            each_successor, cur_candidate_indim, \
                            compute_term, term_dout, reverse=False)

                    # :::: store them
                    list_of_compute_graphs_from_successors.append( \
                        each_successor_compute_graphs)


                # ::: end for each successor

                # ::: error check if the list from any successor is empty, skip
                if not all(list_of_compute_graphs_from_successors): continue

                """
                    Post-process based on the number of successors
                     - combine the multiple successors from the list
                """
                for chosen_compute_graphs \
                    in product(*list_of_compute_graphs_from_successors):

                    # :::: combine the multiple graphs into one
                    merged_compute_graph = None
                    for chosen_idx, chosen_compute_graph \
                        in enumerate(chosen_compute_graphs):
                        # -> initially assign to the data-holder
                        if not chosen_idx:
                            merged_compute_graph = chosen_compute_graph
                        else:
                            merged_compute_graph = nx.compose( \
                                merged_compute_graph, chosen_compute_graph)
                        # end if ...
                    # end for chosen...

                    # :::: store to the estimated architectures
                    if merged_compute_graph:
                        candidate_architectures.append(merged_compute_graph)

                # ::: end for (compute...)

            # :: end for each_compute...
        # : end for each_p....

    # end if compute_curr...

    # return the estimated architectures...
    return candidate_architectures

def _search_linear_database(database, computation, outdim):
    # output channel dimension: ex. from (1)-tuple to 1-int
    out_chdim = outdim[0]

    # data-holders
    param_info = []

    # search over the database
    for (each_cin, each_cout, each_tot, each_time) in database[computation]:
        if each_cout == out_chdim:
            # : conver the attributes into floats
            each_cin = int(each_cin)
            each_cout = int(each_cout)

            # : store
            #   - channel input dimension: tuple
            #   - attributes             : tuple
            cur_chin = (each_cin,)
            cur_attr = (each_cin, each_cout)
            param_info.append((cur_chin, cur_attr))

    # reduce the duplicates,
    #  and convert into the list of tuples
    param_info = list(set(param_info))

    # return candidates
    return param_info

def _search_conv1d_database(database, computation, outdim):
    # 1D, output dimension: ex. (8, 4000) -> 8
    out_chdim, out_datdim = outdim

    # data-holders
    param_info = []

    # search over the database
    for (each_dat, each_cin, each_cout, \
         each_kern, each_str, each_com, each_time) in database[computation]:
        # : when the output channel matches, consider the details
        if each_cout == out_chdim:

            # :: store the dimension,
            #   when the computed out-dim
            #   is the same as the current out-dim
            compute_outdim = ceil((each_dat - each_kern)/each_str + 1)
            if compute_outdim == out_datdim:
                # : conver the attributes into floats
                each_dat = int(each_dat)
                each_cin = int(each_cin)
                each_cout = int(each_cout)
                each_kern = int(each_kern)
                each_str = int(each_str)

                # :: store
                #   - channel input dimension: tuple
                #   - attributes             : tuple
                cur_chin = (each_cin, each_dat)
                cur_attr = (each_cin, each_cout, each_kern, each_str)
                param_info.append((cur_chin, cur_attr))

    # reduce the duplicates,
    #  and convert into the list of tuples
    param_info = list(set(param_info))

    # return param_info
    return param_info

def _compute_factors(number):
    factors = []
    for each_num in range(1, number+1):
        if number % each_num == 0:
            factors.append(each_num)
    return factors

def prune_reconstructed_architectures(archs):
    tot_archs = []

    # remove the architecture doesn't make any sense
    for each_arch in archs:
        """
            1: Check if the narrow connection is sane
        """
        if not _sane_narrow_connection( \
            each_arch, '[0] Embedding', '[12] Sigmoid'): continue

        # add the survived ones
        tot_archs.append(each_arch)
    # end for each...

    # check if the store location exists
    store_loc = os.path.join(_save_dir, 'architectures')
    if not os.path.exists(store_loc): os.makedirs(store_loc)

    # store the architecture as a graph and data
    options = {
        'node_color': 'red',
        'node_size': 40,
        'width': 1,
        'alpha': 0.8,
        'arrowstyle': '-|>',
        'arrowsize': 8,
        'font_size': 10,
    }
    for aidx, each_arch in enumerate(tot_archs):
        # : write the edgelists to a YAML file
        nx.write_yaml( \
            each_arch, \
            os.path.join(store_loc, 'architecture_{}.yaml'.format(aidx)))

        # : relabel the node names to include the attribute parameters
        new_nodes = {}
        for each_node, each_data in each_arch.nodes(data=True):
            each_attr = '{}'.format(each_node)
            if ('attr_param' in each_data) \
                and each_data['attr_param']:
                each_attr += ' - {}'.format(each_data['attr_param'])
            new_nodes[each_node] = each_attr
        new_each_arch = nx.relabel_nodes(each_arch, new_nodes, copy=True)

        # : networkx - draw the graphs
        nx.draw_networkx(new_each_arch, arrows=True, **options)
        plt.savefig(os.path.join(store_loc, 'architecture_{}.pdf'.format(aidx)))
        plt.clf()
    # end for aidx...

    return tot_archs


# ------------------------------------------------------------------------------
# Main (for the command line compatibility)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # command line
    parser = argparse.ArgumentParser( \
        description='Reconstruct the MalConv architecture from the Flush+Reload trace.')

    # load arguments
    parser.add_argument('--in-dims', type=int, default=2000000,
                        help='the input dimension (default: 2000000)')
    parser.add_argument('--outdims', type=int, default=1,
                        help='the output dimension (default: 1)')

    # arguments about the profiled data
    parser.add_argument('--c-timer', type=str, default='schannel',
                        help='the timer used to measure tsc/schannel (default: tsc)')
    parser.add_argument('--frcycle', type=int, default=2000,
                        help='Flush+Reload attack resolution (default: 2000 cycles)')
    parser.add_argument('--dataloc', type=str, default='datasets/profile/pytorch',
                        help='the location where the dataset is (default: dataset/profile/pytorch)')

    # arguments about the processed traces
    parser.add_argument('--tr-file', type=str, default='',
                        help='output file (csv data) location')
    parser.add_argument('--verbose', action='store_true',
                        help='display debug messages (default: false)')

    # load inputs
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))


    # do reconstruction
    reconstruct_malconv( \
        args.tr_file, args.in_dims, args.outdims, \
        args.c_timer, args.dataloc, resolution=args.frcycle, \
        verbose=args.verbose)
    # Fin.
