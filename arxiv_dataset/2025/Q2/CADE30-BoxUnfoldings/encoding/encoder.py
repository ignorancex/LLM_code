from pysat.card import *


def get_dim(idx, dims):
    ##Returns the dimesions of the i-th face
    idx = idx // 2
    assert (len(dims) == 3)
    assert (idx <= 2)
    if idx == 0:
        return [dims[0], dims[1]]
    if idx == 1:
        return [dims[2], dims[1]]
    if idx == 2:
        return [dims[2], dims[0]]


def get_neigh(face, row, col, dims):
    ##Given the position of a square on the box of dimension dims, returns the positions of the four neighbours
    neigh = []
    a, b, c = dims[0], dims[1], dims[2]
    face += 1

    ##Order given by: (col + 1, col - 1, row + 1, row - 1)
    if face == 1:
        if col < b - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([6, 0, row])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([5, 0, row])
        if row < a - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([4, 0, col])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([3, 0, col])
    if face == 2:
        if col < b - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([6, c - 1, row])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([5, c - 1, row])
        if row < a - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([4, c - 1, col])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([3, c - 1, col])
    if face == 3:
        if col < b - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([6, row, 0])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([5, row, 0])
        if row < c - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([2, 0, col])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([1, 0, col])
    if face == 4:
        if col < b - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([6, row, a - 1])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([5, row, a - 1])
        if row < c - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([2, a - 1, col])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([1, a - 1, col])
    if face == 5:
        if col < a - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([4, row, 0])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([3, row, 0])
        if row < c - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([2, col, 0])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([1, col, 0])
    if face == 6:
        if col < a - 1:
            neigh.append([face, row, col + 1])
        else:
            neigh.append([4, row, b - 1])
        if col > 0:
            neigh.append([face, row, col - 1])
        else:
            neigh.append([3, row, b - 1])
        if row < c - 1:
            neigh.append([face, row + 1, col])
        else:
            neigh.append([2, col, b - 1])
        if row > 0:
            neigh.append([face, row - 1, col])
        else:
            neigh.append([1, col, b - 1])
    for nei in neigh:
        nei[0] -= 1
    return neigh


def append_eq(cnf, lits, k, id):
    ##Encodes sum x \in lits x = k
    ##Returns the highest identifier used.
    if k == 1:
        eq = CardEnc.equals(lits, k, id, encoding=EncType.pairwise)
    else:
        eq = CardEnc.equals(lits, k, id, encoding=EncType.kmtotalizer)
    for e in eq:
        for l in e:
            id = max(id, abs(l))
        cnf.append(e)
    return id


def add_o(o1, o2):
    ##Adds orientations
    return (o1 + o2) % 4


def get_ori(ori, neigh, face):
    ##Given the current orientation, reorders the list of neighbours into [right, left, up, down]
    # Each element of neigh is given as a (face, row, col) tuple
    # ori is value in [0, 1, 2, 3], corresponding to orienting 90*ori degrees counter-clockwise
    # returns:
    # neigh: List of neighbours ordered with [col + 1, col - 1, row + 1, row - 1]
    # orients: List of orientations of neighbours in neigh
    assert (ori in [0, 1, 2, 3])
    assert (face in [0, 1, 2, 3, 4, 5])

    ##First pre-process faces
    if face in [1, 2, 5]:
        neigh = [neigh[1], neigh[0], neigh[2], neigh[3]]
    orients = [0, 0, 0, 0]

    ##Case works:

    # Right:
    if neigh[0][0] != face:
        if face == 0:
            orients[0] = 3
        if face == 1:
            orients[0] = 1
    # Left:
    if neigh[1][0] != face:
        if face == 0:
            orients[1] = 1
        if face == 1:
            orients[1] = 3
    # Up:
    if neigh[2][0] != face:
        if face == 1 or face == 3:
            orients[2] = 2
        if face == 4:
            orients[2] = 3
        if face == 5:
            orients[2] = 1
    # Down:
    if neigh[3][0] != face:
        if face == 0 or face == 2:
            orients[3] = 2
        if face == 4:
            orients[3] = 3
        if face == 5:
            orients[3] = 1
    for i in range(4):
        orients[i] = add_o(orients[i], ori)
    n1, n2, n3, n4 = neigh[0], neigh[1], neigh[2], neigh[3]
    o1, o2, o3, o4 = orients[0], orients[1], orients[2], orients[3]
    if ori == 1:
        neigh = [n4, n3, n1, n2]
        orients = [o4, o3, o1, o2]
    if ori == 2:
        neigh = [n2, n1, n4, n3]
        orients = [o2, o1, o4, o3]
    if ori == 3:
        neigh = [n3, n4, n2, n1]
        orients = [o3, o4, o2, o1]
    return neigh, orients


def append_am(cnf, lits, k, id):
    ##Encodes sum x \in lits x <= k
    ##Returns the highest identifier used.
    if k == 1:
        eq = CardEnc.atmost(lits, k, id, encoding=EncType.pairwise)
    else:
        eq = CardEnc.atmost(lits, k, id, encoding=EncType.kmtotalizer)
    for e in eq:
        for l in e:
            id = max(id, abs(l))
        cnf.append(e)
    return id


def order(t1, t2):
    return tuple(sorted((t1, t2)))


def get_neigh_edges(cell, dim):
    ##Returns a (size 4) list of neighbouring edges
    neigh = get_neigh(cell[0], cell[1], cell[2], dim)
    edge_list = []
    for nei in neigh:
        edge_list.append(order(cell, tuple(nei)))
    return edge_list

def get_out_orient(cell, edge):
    assert len(cell) == 3, "Incorrect cell format"
    return 0 if edge[0] == cell else 1

def opp_cell(cell, dims):
    ##returns the opposite cell on a box of dimension dims
    opp = list(cell)
    if cell[0] in [0, 2, 4]:
        opp[0] += 1
    else:
        opp[0] -= 1

    r, c = get_dim(opp[0], dims)
    opp[1] = r - 1 - opp[1]
    opp[2] = c - 1 - opp[2]
    return tuple(opp)

def encode_subgraph_connected_bfs(max_time, dim, edges, sink, cnf, id):
    cells = get_cells(dim)
    s = 2 * (dim[0] * dim[1] + dim[0]* dim[2] + dim[1]*dim[2])
    act_times = {}
    for cell in cells:
        act_times[cell] = [id + i for i in range(max_time)]
        id += max_time

    ##Only sink is activated at time 0
    cnf.append([act_times[sink][0]])
    for cell in cells:
        if cell != sink:
            cnf.append([-act_times[cell][0]])

    ##Every cell has an activation time
    for cell in cells:
        cnf.append(act_times[cell][1:])

    for t in range(1, max_time):
        for cell in cells:
            neigh = get_neigh(cell[0], cell[1], cell[2], dim)
            for subset in range(1, 2**4):
                cur = []
                for i in range(4):
                    if ((1 << i) & subset) != 0:
                        cur.append(act_times[tuple(neigh[i])][t - 1])
                    else:
                        cur.append(-edges[order(tuple(neigh[i]), cell)])
                cnf.append([-act_times[cell][t]] + cur)
    return id, act_times

def encode_subgraph_connected(dim, edges, sink, cnf, id):
    ##Encodes that the subgraph generated by edges is connected.
    s = 2 * (dim[0] * dim[1] + dim[0] * dim[2] + dim[1] * dim[2])
    cells = get_cells(dim)

    ##Orientation of edges, edge e has orientation 0 if it's oriented (e[0], e[1]) and orientation 1 if pointed (e[1], e[0])
    edge_orients = {}
    for e in edges:
        edge_orients[e] = [id, id + 1]
        id += 2
        ##An edge can have at most one orientation
        cnf.append([-edge_orients[e][0], -edge_orients[e][1]])
        ##If an edge is not cut, it must have some orientation
        cnf.append([edges[e], edge_orients[e][0], edge_orients[e][1]])
        ##If an edge is oriented, it must not be cut
        cnf.append([-edge_orients[e][0], -edges[e]])
        cnf.append([-edge_orients[e][1], -edges[e]])

    ## (0), (1)
    for cell in cells:
        neigh_edges = get_neigh_edges(cell, dim)
        if cell == sink:
            current_in = []
            for ne in neigh_edges:
                outgoing_orient = 0 if ne[0] == cell else 1
                cnf.append([-edge_orients[ne][outgoing_orient]])
                current_in.append(edge_orients[ne][1 - outgoing_orient])
            ##Has at least one incoming edge
            cnf.append(current_in)
        else:
            ## (1): Each square has at least one outgoing edge
            current_out = []
            current_in = []
            for ne in neigh_edges:
                outgoing_orient = 0 if ne[0] == cell else 1
                current_out.append(edge_orients[ne][outgoing_orient])
                current_in.append(edge_orients[ne][1 - outgoing_orient])
            cnf.append(current_out)

    ## (2) For every three squares in a line, the middle square cannot have two outgoing edges
    for cell in cells:
        neigh_edges = get_neigh_edges(cell, dim)
        lines = [[neigh_edges[0], neigh_edges[1]], [neigh_edges[2], neigh_edges[3]]]
        flips = []
        for i in range(2):
            flips.append([get_out_orient(cell, lines[i][0]), get_out_orient(cell, lines[i][1])])
        for i in range(2):
            cnf.append([-edge_orients[lines[i][0]][flips[i][0]], -edge_orients[lines[i][1]][flips[i][1]]])

    surface_vertices = get_squares(dim)
    assert len(surface_vertices) == s - 6, "Incorrect number of surface vertices"


    ## (3) pairs of parallel edges have the same orientation
    for sq in surface_vertices:

        sq[2], sq[3] = sq[3], sq[2] ##TODO: Fix this hack

        par_edges = []
        all_edges = []
        cycle_orient = []
        for i in range(4):
            par_edges.append(order(sq[i], sq[(i + 1) % 4]))
            assert par_edges[-1] in edges, "Incorrect labeling of edges for surface vertex"
            all_edges.append(edges[par_edges[-1]])
            if par_edges[-1][0] == sq[i]:
                cycle_orient.append(0)
            else:
                cycle_orient.append(1)
        par_edges = [[par_edges[0], par_edges[2]], [par_edges[1], par_edges[3]]]
        cycle_orient = [[cycle_orient[0], cycle_orient[2]], [cycle_orient[1], cycle_orient[3]]]

        for i in range(2):
            cur_pair = par_edges[i]
            orth_pair = par_edges[1 - i]
            cur_ori = cycle_orient[i]
            for ori in range(2):
                for orth_edge in orth_pair:
                    cnf.append([-edge_orients[cur_pair[0]][(cur_ori[0] + ori) % 2], edges[orth_edge], -edge_orients[cur_pair[1]][(cur_ori[1] + ori) % 2]])
                    cnf.append([-edge_orients[cur_pair[0]][(cur_ori[0] + 1 - ori) % 2], edges[orth_edge], -edge_orients[cur_pair[1]][(cur_ori[1] + 1 - ori) % 2]])

    ## (4) Extra encoding to prevent not enough edges being cut

    for cell in cells:
        neigh = get_neigh(cell[0], cell[1], cell[2], dim)
        ori_neigh, orients = get_ori(0, neigh, cell[0])
        for idx in range(4):
            if idx == 0 or idx == 1:
                next = idx + 2
            elif idx == 2:
                next = 1
            else:
                next = 0

            cur_edge = order(cell, tuple(ori_neigh[idx]))
            cur_neigh = ori_neigh[idx]
            cur_neigh_next = get_neigh(cur_neigh[0], cur_neigh[1], cur_neigh[2], dim)
            cur_neigh_next_oriented, cur_neigh_next_orients = get_ori(orients[idx], cur_neigh_next, cur_neigh[0])
            cur_edge_next = order(tuple(cur_neigh), tuple(cur_neigh_next_oriented[next]))
            cur_edge_ori = get_out_orient(cell, cur_edge)
            cur_edge_ori_next = get_out_orient(tuple(cur_neigh), cur_edge_next)

            adj_edge = order(cell, tuple(ori_neigh[next]))
            adj_neigh = ori_neigh[next]
            adj_neigh_next = get_neigh(adj_neigh[0], adj_neigh[1], adj_neigh[2], dim)
            adj_neigh_next_oriented, adj_neigh_next_orients = get_ori(orients[next], adj_neigh_next, adj_neigh[0])
            adj_edge_next = order(tuple(adj_neigh), tuple(adj_neigh_next_oriented[idx]))
            adj_edge_ori = get_out_orient(cell, adj_edge)
            adj_edge_ori_next = get_out_orient(tuple(adj_neigh), adj_edge_next)

            cnf.append([-edge_orients[cur_edge][cur_edge_ori], -edge_orients[adj_edge][adj_edge_ori],
                        edge_orients[cur_edge_next][cur_edge_ori_next]])
            cnf.append([-edge_orients[cur_edge][cur_edge_ori], -edge_orients[adj_edge][adj_edge_ori], edge_orients[adj_edge_next][adj_edge_ori_next]])

    return id, edge_orients




def encode_box_1(dims, cnf, id, starting_cell, sinks = 1):
    edges = {}  ##Bad implementation of edges, each edge is a tuple (square_1, square_2). Each square is encoded by (face, row, col)
    # print(f"Starting edge id: {id}")
    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                neigh = get_neigh(i, j, k, dims)
                cur = (i, j, k)
                for nei in neigh:
                    con = order(tuple(nei), cur)
                    if not con in edges:
                        edges[con] = id  ##Avoids duplicates
                        id += 1
    # print(f"Finishing edge id: {id}")
    ##dirs has dimensions faces*row_on_face*col_on_face*{0, 1, 2, 3}
    ##Indicating the orientation of the square on (face, row, col)
    dirs = [[[[] for kb in range(get_dim(i, dims)[1])] for ka in range(get_dim(i, dims)[0])] for i in range(6)]

    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                for d in range(4):
                    dirs[i][j][k].append(id)
                    id += 1
                id = append_eq(cnf, dirs[i][j][k], 1,
                               id - 1) + 1  ##Populate directions, each square has exactly one direction

    # ##Resulting subgraph needs to be connected

    # id, edge_orients = encode_subgraph_connected_bfs(20, dims, edges, opp_cell(starting_cell, dims), cnf, id)

    if sinks == 1:
        id, edge_orients = encode_subgraph_connected(dims, edges, opp_cell(starting_cell, dims), cnf, id)
    elif sinks == 2:
        id, edge_orients_1 = encode_subgraph_connected(dims, edges, (0, 0, 0), cnf, id)
        id, edge_orients_2 = encode_subgraph_connected(dims, edges, opp_cell((0, 0, 0), dims), cnf, id)
        edge_orients = [edge_orients_1, edge_orients_2]
    elif sinks == 6:
        edge_orients = []
        for face in [0, 2, 4]:
            id, current_orient = encode_subgraph_connected(dims, edges, (face, 0, 0), cnf, id)
            edge_orients.append(current_orient)
            id, current_orient = encode_subgraph_connected(dims, edges, opp_cell((face, 0, 0), dims), cnf, id)
            edge_orients.append(current_orient)
    else:
        assert sinks in [1, 2, 6], "Incorrect number of sinks"

    # #
    # # ##Orientation preserving
    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                neigh = get_neigh(i, j, k, dims)
                current = (i, j, k)
                for d in range(4):
                    current_neigh, orients = get_ori(d, neigh, i)
                    for idx in range(4):
                        nei = current_neigh[idx]
                        ni, nj, nk = nei[0], nei[1], nei[2]
                        e = edges[order(current, (ni, nj, nk))]

                        ##If the edge e isn't cut, then this forces the orientation of the neighbour
                        cnf.append([-dirs[i][j][k][d], e,
                                    dirs[ni][nj][nk][orients[idx]]])
    return dirs, edges, edge_orients, id


def cpt_amo(lits, cnf, id):
    ##Compact encoding of x_1 + x_2 + ... + x_n <= 1
    ##cnf: CNF object
    ##id: first available identifier
    if len(lits) <= 4:
        id = append_am(cnf, lits, 1, id - 1) + 1
        return id
    else:
        pref = lits[0:3] + [-id]
        suff = [id] + lits[3:]
        id = cpt_amo(pref, cnf, id + 1)
        id = cpt_amo(suff, cnf, id + 1)
        return id

def encode_box_2(dims, cnf, dirs, edges, edge_orients, id, prev_dim, prev_edges, prev_edge_orients, prev_dirs, preserve_ori = 0):
    ##Encodes a common unfolding of the next box, by encoding a mapping m:box -> prev_box
    ##A box has 6 faces, ordering: a * b, a * b, b * c, b * c, a * c, a * c

    prev_cells = get_cells(prev_dim)
    cur_cells = get_cells(dims)
    assert len(prev_cells) == len(cur_cells), "Boxes have inconsistent surface area"

    mapping = {cell: dict() for cell in cur_cells}  ##mapping from secondary box to primary box

    for cur_c in cur_cells:
        current = []
        for prev_c in prev_cells:
            mapping[cur_c][prev_c] = id
            current.append(id)
            id += 1
        cnf.append(current)
        id = cpt_amo(current, cnf, id)
        # id = append_eq(cnf, current, 1, id - 1) + 1

    ##Mapping should be surjective (therefore bijective), each cell on the secondary box is only mapped by one cell in the primary box
    for prev_c in prev_cells:
        current = []
        for cur_c in cur_cells:
            current.append(mapping[cur_c][prev_c])
        cnf.append(current)
        id = cpt_amo(current, cnf, id)
        # id = append_eq(cnf, current, 1, id - 1) + 1

    ## Literals denoting change in orientation
    ## mapping_ori[s][s'][r] is true if s is mapped to s' and the orientation of s' + r mod 4 is the orientation of s

    mapping_ori = {}

    for prev_c in prev_cells:
        mapping_ori[prev_c] = [id, id + 1, id + 2, id + 3]
        id += 4

    for cur_c in cur_cells:
        for prev_c in prev_cells:
            for r in range(4):
                for d in range(4):
                    ## r - relative orientation
                    ## d - orientation of cur_c
                    cnf.append([-mapping[cur_c][prev_c], -dirs[cur_c[0]][cur_c[1]][cur_c[2]][(d + r) % 4], -prev_dirs[prev_c[0]][prev_c[1]][prev_c[2]][d], mapping_ori[prev_c][r]])
                    # cnf.append([-mapping_ori[cur_c][prev_c][r], -prev_dirs[prev_c[0]][prev_c[1]][prev_c[2]][d], dirs[cur_c[0]][cur_c[1]][cur_c[2]][(d + r) % 4]])
                    # cnf.append([-mapping_ori[cur_c][prev_c][r], mapping[cur_c][prev_c]])
                    # id = append_am(cnf, mapping_ori[cur_c][prev_c], 1, id - 1) + 1

    ##Coherence using relative orientations
    for cur_c in cur_cells:
        for prev_c in prev_cells:
            prev_neigh = get_neigh(prev_c[0], prev_c[1], prev_c[2], prev_dim)
            prev_oriented_neigh, prev_orients = get_ori(0, prev_neigh, prev_c[0])
            for r in range(4):
                cur_neigh = get_neigh(cur_c[0], cur_c[1], cur_c[2], dims)
                cur_oriented_neigh, cur_orients = get_ori(r, cur_neigh, cur_c[0])
                for idx in range(4):
                    prev_e = order(prev_c, tuple(prev_oriented_neigh[idx]))
                    cnf.append([-mapping_ori[prev_c][r], -mapping[cur_c][prev_c], prev_edges[prev_e], mapping[tuple(cur_oriented_neigh[idx])][tuple(prev_oriented_neigh[idx])]])
                    cur_e = order(cur_c, tuple(cur_oriented_neigh[idx]))
                    cnf.append([-mapping_ori[prev_c][r], -mapping[cur_c][prev_c], prev_edges[prev_e], -edges[cur_e]])
                    cnf.append([-mapping_ori[prev_c][r], -mapping[cur_c][prev_c], -prev_edges[prev_e], edges[cur_e]])
#                    cnf.append([-mapping_ori[prev_c][r], -mapping[cur_c][prev_c], edges[order(cur_c,tuple(cur_oriented_neigh[idx]))], mapping[tuple(cur_oriented_neigh[idx])][tuple(prev_oriented_neigh[idx])]])
    return mapping, id



def break_corners(edges, dims, cnf, id):
    ##Enforces that every corner needs to have at least one edge cut
    corners = get_corners(dims)
    for corner in corners:
        assert (len(corner) == 3)
        to_cut = []
        for i in range(3):
            for j in range(i + 1, 3):
                to_cut.append(edges[order(corner[i], corner[j])])
        cnf.append(to_cut)
    return id


def get_cells(dims):
    ##returns a list of cells on the box with dimension dims
    cells = []
    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                cells.append((i, j, k))
    return cells


def get_squares(dims):
    ##Returns a list of 4-cells representing edge-to-edge squares
    ##(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)

    a, b, c = dims[0], dims[1], dims[2]
    squares = []
    cells = get_cells(dims)
    for cell in cells:
        ci, cj, ck = cell[0], cell[1], cell[2]
        rv, cv = get_dim(ci, dims)
        if cj < rv - 1 and ck < cv - 1:
            squares.append([cell, (ci, cj + 1, ck), (ci, cj, ck + 1), (ci, cj + 1, ck + 1)])

    for j in range(c - 1):
        squares.append([(2, j, 0), (2, j + 1, 0), (4, j, 0), (4, j + 1, 0)])
        squares.append([(2, j, b - 1), (2, j + 1, b - 1), (5, j, 0), (5, j + 1, 0)])
        squares.append([(3, j, 0), (3, j + 1, 0), (4, j, a - 1), (4, j + 1, a - 1)])
        squares.append([(3, j, b - 1), (3, j + 1, b - 1), (5, j, a - 1), (5, j + 1, a - 1)])

    for k in range(a - 1):
        squares.append([(4, 0, k), (4, 0, k + 1), (0, k, 0), (0, k + 1, 0)])
        squares.append([(4, c - 1, k), (4, c - 1, k + 1), (1, k, 0), (1, k + 1, 0)])
        squares.append([(5, 0, k), (5, 0, k + 1), (0, k, b - 1), (0, k + 1, b - 1)])
        squares.append([(5, c - 1, k), (5, c - 1, k + 1), (1, k, b - 1), (1, k + 1, b - 1)])

    for k in range(b - 1):
        squares.append([(0, 0, k), (0, 0, k + 1), (2, 0, k), (2, 0, k + 1)])
        squares.append([(0, a - 1, k), (0, a - 1, k + 1), (3, 0, k), (3, 0, k + 1)])
        squares.append([(1, 0, k), (1, 0, k + 1), (2, c - 1, k), (2, c - 1, k + 1)])
        squares.append([(1, a - 1, k), (1, a - 1, k + 1), (3, c - 1, k), (3, c - 1, k + 1)])

    return squares


def encode_squares(edges, dims, cnf, id):
    squares = get_squares(dims)
    square_edges = []
    for s in squares:
        assert (len(s) == 4)
        current_edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                if order(s[i], s[j]) in edges:
                    current_edges.append(edges[order(s[i], s[j])])
        assert (len(current_edges) == 4)  ##A square should always have 4 edges
        square_edges.append(current_edges)
    for se in square_edges:
        for i in range(4):
            se[i] *= -1
            cnf.append(se)
            se[i] *= -1
    return id


def encode_fix_face(dims, edges, face, reduce, fix_prob, cnf, id):
    # # (1) Fix all edges on the target face provided

    edges_to_consider = []
    r, c = get_dim(face, dims)
    assert 2 * reduce < r and 2 * reduce < c, "Reduction parameter too large"

    ##Row, Col bounds for reduced rectangle
    r_l, r_u = reduce, r - reduce - 1
    c_l, c_u = reduce, c - reduce - 1

    for e in edges.keys():
        if e[0][0] == e[1][0] == face and r_l <= e[0][1] <= r_u and c_l <= e[0][2] <= c_u and r_l <= e[1][
            1] <= r_u and c_l <= e[1][2] <= c_u:
            edges_to_consider.append(edges[e])
            # print(f"Adding edge: {e}")
    ## Randomized edge fixing
    import random
    for e in edges_to_consider:
        rand = random.random()
        if rand <= fix_prob:
            cnf.append([-e])  ##This edge is kept
    return id


def encode_neighbour_edges(dims, edges, cnf, id):
    ##Enforces the constraint that every cell has at least one edge preserved

    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                neigh = get_neigh(i, j, k, dims)
                current = []
                for nei in neigh:
                    current.append(-edges[order((i, j, k), tuple(nei))])
                cnf.append(current)
    return id


def encode_heuristics(dims, dirs, edges, cnf, id, fix_ori=None):
    ##Adds in heuristics for box_2

    ##(1) Break corners
    id = break_corners(edges, dims, cnf, id)

    ##(2) Force square edges
    id = encode_squares(edges, dims, cnf, id)

    ##(4) Fix orientation on center cell of given face
    if fix_ori != None:
        cnf.append([dirs[fix_ori[0]][fix_ori[1]][fix_ori[2]][0]])
    #(5) Every cell should have at least one of its edges preserved
    id = encode_neighbour_edges(dims, edges, cnf, id)

    return id


def get_corners(dims):
    ##Returns a list of 8 triples, representing the 8 corner vertices of the box with dimension dims
    a, b, c = dims[0], dims[1], dims[2]
    corners = [[(1, 0, 0), (2, c - 1, 0), (4, c - 1, 0)], [(1, a - 1, 0), (4, c - 1, a - 1), (3, c - 1, 0)],
               [(1, a - 1, b - 1), (5, c - 1, a - 1), (3, c - 1, b - 1)],
               [(1, 0, b - 1), (2, c - 1, b - 1), (5, c - 1, 0)], [(0, a - 1, b - 1), (5, 0, a - 1), (3, 0, b - 1)],
               [(0, 0, b - 1), (5, 0, 0), (2, 0, b - 1)], [(0, 0, 0), (2, 0, 0), (4, 0, 0)],
               [(0, a - 1, 0), (4, 0, a - 1), (3, 0, 0)]]

    return corners

def get_representatives(dims):
    ##returns a list of representatives
    reps = []
    for i in [0, 2, 4]:
        r, c  = get_dim(i, dims)
        for j in range((r + 1)//2):
            for k in range((c + 1)//2):
                reps.append((i, j, k))
    return reps

def iso_rep(cell, dims):
    ##returns the unique representative isomorphic to cell
    rep = list(cell)
    if cell[0] in [1, 3, 5]:
        rep[0] = cell[0] - 1
    r, c = get_dim(rep[0], dims)
    if rep[1] >= (r + 1)//2:
        rep[1] = r - rep[1] - 1
    if rep[2] >= (c + 1)//2:
        rep[2] = c - rep[2] - 1
    return tuple(rep)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Encoder for box folding')

    # Add arguments

    ##Required dimensions
    parser.add_argument('-d', '--dimensions', nargs='+',
                        help='Dimensions of faces of boxes, should be a list of 3*n numbers', required=True)
    ##Optional
    parser.add_argument('-n', '--boxes', type=int, help='Number of common boxes desired, default = 2', default=2)
    parser.add_argument('-f', '--fix', type=int,
                        help='Face to be fixed on the first box (0, ..., 5), -1 to not fix any face, default = -1',
                        default=-1)
    parser.add_argument('-p', '--fix_prob', type=float,
                        help='Probability of an edge being fixed on the given face. Has no effect if -f = 1, default of p = 1',
                        default=1)
    parser.add_argument('-o', '--output', type=str, help='Path to output encoding, default: encoding.cnf',
                        default='common_unfolding.cnf')
    parser.add_argument('-r', '--reduce', type=int, help='Size reduction for face fixing, default: 0', default=0)
    parser.add_argument( '--orient2', type=int, help='Fixing orientation of s2.', default=-1)
    parser.add_argument( '--orient3', type=int, help='Fixing orientation of s3.', default=-1)
    parser.add_argument('-m', '--mapping', type=int,
                        help='Number of center cells fixed by mapping. Default = 0, at most n - 1', default=0)
    parser.add_argument('-s1', '--start1', type=int, nargs=3, help='Coordinates for starting cell of 1st box (assumed to have orientation 0)', default = [2, 0, 0])
    parser.add_argument('-s2', '--start2', type=int, nargs=3, help='Coordinates for starting cell of 2nd box', default = [2, 0, 0])
    parser.add_argument('-s3', '--start3', type=int, nargs=3, help='Coordinates for starting cell of 3rd box', default = [2, 0, 0])

    parser.add_argument('-t1', '--target1', type=int, nargs=3, help='Coordinates for target cell of 1st box (experimental)', default = [-1, -1, -1])
    parser.add_argument('-t2', '--target2', type=int, nargs=3, help='Coordinates for target cell of 2nd box (experimental)', default = [-1, -1, -1])
    parser.add_argument('-t3', '--target3', type=int, nargs=3, help='Coordinates for target cell of 3rd box (experimental)', default = [-1, -1, -1])

    parser.add_argument('--sinks', type=int, help='Number of sinks. Should be in [1, 2, 6]. Increasing decreases the likelihood of disconnected solutions', default = 1)

    args = parser.parse_args()

    dims = args.dimensions
    boxes = args.boxes
    fix_face = args.fix
    out_path = args.output
    fix_prob = args.fix_prob
    reduce = args.reduce
    map_h = args.mapping
    start1 = args.start1
    start2 = args.start2
    start3 = args.start3
    starting_cells = [tuple(start1), tuple(start2), tuple(start3)]
    target1 = args.target1
    target2 = args.target2
    target3 = args.target3
    sinks = args.sinks

    orient2 = args.orient2
    orient3 = args.orient3

    target_cells = [tuple(target1), tuple(target2), tuple(target3)]

    assert 0 <= map_h <= boxes - 1, "Number of cells fixed by mapping is outside range"
    assert len(
        dims) == 3 * boxes, f"Number of dimensions: {len(args.dimensions)} does not match number of boxes: {args.boxes}"
    assert fix_face in [-1, 0, 1, 2, 3, 4,
                        5], f"Incorrect face to fix. Entered: {fix_face}, should be a value between -1 and 5"
    assert 0 <= fix_prob <= 1, f"Incorrect probability of face being fixed, should be between 0 and 1"
    assert orient2 in [-1, 0, 1, 2, 3] and orient3 in [-1, 0, 1, 2, 3], "Incorrect orientation of s2/s3"
    assert sinks in [1, 2, 6], "Incorrect number of sinks"

    cnf = CNF()
    dimension_list = [[int(dims[i * 3]), int(dims[i * 3 + 1]), int(dims[i * 3 + 2])] for i in range(boxes)]


    for i in range(boxes):
        cell_cur = get_cells(dimension_list[i])
        assert starting_cells[i] in cell_cur, "Invalid starting cell"

    ##Idea: Unfold both primary and secondary boxes
    ##And then ensure mapping equivalence

    orientations = []
    edges = []
    edge_orients = []
    mappings = []
    id = 1
    for i in range(boxes):
        cur_ori, cur_edges, cur_edge_orient, id = encode_box_1(dimension_list[i], cnf, id, starting_cells[i], sinks)
        orientations.append(cur_ori)
        edge_orients.append(cur_edge_orient)
        edges.append(cur_edges)
    print(f"Total number of variables after unfolding all boxes: {cnf.nv}, clauses: {len(cnf.clauses)}")

    ##Encode mappings to the first box
    for i in range(1, boxes):
        cur_mapping, id = encode_box_2(dimension_list[i], cnf, orientations[i], edges[i], edge_orients[i], id,
                                       dimension_list[0], edges[0], edge_orients[0], orientations[0])
        mappings.append(cur_mapping)
        print(f"Total number of variables after matching box_{i + 1}: {cnf.nv}, clauses: {len(cnf.clauses)}")

    ##Heuristics

    ##Seen success by just using heuristics on box_1
    ##Can solve common 2-box unfoldings at surface area 150

    ##If face is required to be fixed
    if fix_face != -1:
        id = encode_fix_face(dimension_list[0], edges[0], fix_face, reduce, fix_prob, cnf, id)

    ##Fix orientation of the first cell

    for i in range(boxes):
        fix_ori = starting_cells[i] if i == 0 else None
        id = encode_heuristics(dimension_list[i], orientations[i], edges[i], cnf, id, fix_ori)

        if 0 < i <= map_h:
            cnf.append([mappings[i - 1][starting_cells[i]][starting_cells[0]]])

    ##Experimental - Fixing pairs
    if target1[0] != -1 and target2[0] != -1:
        cnf.append([mappings[0][target_cells[1]][target_cells[0]]])
    if target1[0] != -1 and target3[0] != -1:
        cnf.append([mappings[1][target_cells[2]][target_cells[0]]])

    if orient2 != -1:
        cnf.append([orientations[1][start2[0]][start2[1]][start2[2]][orient2]])
    if orient3 != -1:
        cnf.append([orientations[2][start3[0]][start3[1]][start3[2]][orient3]])
    print(f"Total number of variables after adding in heuristics: {cnf.nv}, clauses: {len(cnf.clauses)}")
    cnf.to_file(out_path)