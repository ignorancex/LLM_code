"""
    Characteristics of architecture parameters:
    used for the reconstruction of the architecture in/out dimensions.
"""

# ------------------------------------------------------------------------------
# Rules for the candidate parameters
# ------------------------------------------------------------------------------
_NARROW_COUNT      = 0
_OPS_NO_DIM_CHANGE = [
    # tensor operations
    '* (multiplication)',
    # layers
    'Sigmoid',
]

# ------------------------------------------------------------------------------
# Supporting functions
# ------------------------------------------------------------------------------
def _same_inout_dimensions(operation):
    for each_ops in _OPS_NO_DIM_CHANGE:
        if each_ops in operation: return True
    return False

def _sane_narrow_connection( \
    arch, cnode, enode, narrow=0):
    """
        Recursively check if the architecture doesn't include more than
        one 'narrow' computation in a short path (subpath by a branch)
    """
    # base condition 1:
    #  the narrow appear twice, then the arch. is messed-up
    if (narrow > 1): return False

    # base condition 2:
    #  the search has reached its end...
    elif (cnode == enode): return True

    # recursive search, first, check the successors
    #  one successor: do the recursive search
    #  two successor: reset the counter
    else:
        successors = list(arch.successors(cnode))
        sane_iness = True
        if len(successors) > 1:
            for each_successor in successors:
                # : set to one since a branch starts
                if 'narrow' in cnode: narrow = 1
                sane_iness = (sane_iness and \
                    _sane_narrow_connection( \
                        arch, each_successor, enode, narrow=narrow))
        else:
            for each_successor in successors:
                # : add one to the counter
                if 'narrow' in cnode: narrow += 1
                sane_iness = (sane_iness and \
                    _sane_narrow_connection( \
                        arch, each_successor, enode, narrow=narrow))
        return sane_iness
    # done.
