import qiskit

def duplicate_xss(xss):
    return [xs.copy() for xs in xss for _ in range(2)]

def update_slot(slots, B):
    for b in B:
        slots[b] += 1
    return slots

def get_qubit_indices(qargs):
    return [q.index for q in qargs]

def create_zip_chain(num_operators, num_xoperators, is_cx_first):
    """Create list 0,1,0,1,...
    If is_cx_first is True, then 1 is first, else 0 is first
    Args:
        n (_type_): _description_
        m (_type_): _description_
        is_cx_first (bool): _description_

    Returns:
        _type_: _description_
    """
    result = []
    while num_operators > 0 or num_xoperators > 0:
        if is_cx_first:
            if num_xoperators > 0:
                result.append(1)
                num_xoperators -= 1
            if num_operators > 0:
                result.append(0)
                num_operators -= 1
        else:   
            if num_operators > 0:
                result.append(0)
                num_operators -= 1
            if num_xoperators > 0:
                result.append(1)
                num_xoperators -= 1
    return result
