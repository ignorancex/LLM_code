from pysat.card import *


def print_grid(grid):
    n, m = len(grid), len(grid[0])
    for i in range(n):
        for j in range(m):
            print(grid[i][j], end="")
        print()
    print()

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
    return id

def add_o(o1, o2):
    ##Adds orientations
    return (o1 + o2)%4


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

def encode_subgraph_connected(dim, edges, starting_cell, cnf, id):
    ##Encodes that the subgraph generated by edges is connected.
    s = 2*(dim[0]*dim[1] + dim[0]*dim[2] + dim[1]*dim[2])

    ## false -> (e[1], e[0]), true -> (e[0], e[1])

    edge_orients = {}
    for e in edges:
        edge_orients[e] = [id, id + 1]
        id += 2
    #
    # ##Every cell other than the starting cell should have one outgoing edge
    #
    # id += 4*(s - 1)

    ##Literals indicating if a surface vertex has at most one cut
    # non_cut = [id + i for i in range(s - 6)]
    # id += s - 6
    # id = append_am(cnf, list(edges.values()) + non_cut, s + 1, id - 1) + 1

    return edge_orients, id



def encode_box_1(dims, cnf, id, starting_cell, sinks = 1):

    edges = {}  ##Bad implementation of edges, each edge is a tuple (square_1, square_2). Each square is encoded by (face, row, col)
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
    edge_orients = []
    for s in range(sinks):
        cur_ori, id = encode_subgraph_connected(dims, edges, starting_cell, cnf, id)
        edge_orients.append(cur_ori)


    # max_activation = 20
    # s = 2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])
    # cells = get_cells(dims)
    # id += max_activation * s

    # for cell in cells:
    #     for t in range(max_activation):
    #         if id + t in model:
    #             print(f"Cell: {cell} activation: {t} id: {id}")
    #             break
    #     id += max_activation
    # print("End of box")

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


def encode_box_2(dims, cnf, dirs, edges, id, prev_dim, prev_edges, prev_dirs):
    ##Encodes a common unfolding of the next box, by encoding a mapping m:box -> prev_box


    s = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])

    ##A box has 6 faces, ordering: a * b, a * b, b * c, b * c, a * c, a * c


    ##List of all possible cells on the previous box
    prev_cells = []
    for i in range(6):
        r, c = get_dim(i, prev_dim)
        for j in range(r):
            for k in range(c):
                prev_cells.append((i, j, k))
    assert (len(prev_cells) == s) ##Both boxes should have the same area

    cur_cells = []
    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                cur_cells.append((i, j, k))

    mapping = {cell : dict() for cell in cur_cells} ##mapping from secondary box to primary box

    for cur_c in cur_cells:
        current = []
        for prev_c in prev_cells:
            mapping[cur_c][prev_c] = id
            current.append(id)
            id += 1
        cnf.append(current)
        id = cpt_amo(current, cnf, id)
        # id = append_am(cnf, current, 1, id - 1) + 1 ##Each cell on the primary box is mapped to at most one cell
    for prev_c in prev_cells:
        current = []
        for cur_c in cur_cells:
            current.append(mapping[cur_c][prev_c])
        cnf.append(current)
        id = cpt_amo(current, cnf, id)
    id += 4*s

    # print(f"After encoding box-to-box mapping is well-defined, nv: {cnf.nv}, clauses: {len(cnf.clauses)}")
    return mapping, id

def get_model(file):
    ##Parses .cnf output file
    ##Ignores first row, ignores first column of every row
    f = open(file, "r")
    vals = f.readlines()
    models = set()
    for v in vals:
        if len(v) == 0 or v[0] != 'v':
            continue
        cur_line = v.split()
        for val in cur_line:
            try:
                if -int(val) not in models:
                    models.add(int(val))
            except ValueError:
                pass
    # print(models)
    return models
    
def draw_decode(grid, orientations, save_path, show_fig):
    ##Prints the decoded grid in grids
    grid, orientations = format_grid(grid), format_grid(orientations)

    n, m = len(grid), len(grid[0])
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    # make a figure + axes
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # make color map
    cmap = matplotlib.colors.ListedColormap(['w', 'r', 'g', 'b', 'y', 'orange', 'lawngreen'])
    # draw the boxes
    ax.imshow(grid, cmap=cmap)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, m, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    for i in range(n):
        for j in range(m):
            if orientations[i][j] == -1:
                continue
            ax.text(j, i, orientations[i][j], ha="center", va="center", color="black")
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    if save_path != "":
        fig.savefig(save_path)
    if show_fig != 0:
        plt.show()
    plt.close(fig)

def format_grid(grid):
    ##Given an n*n grid, resizes it by trimming away the unused entries
    n, m = len(grid), len(grid[0])
    min_r, max_r = n + 1, -1
    min_c, max_c = m + 1, -1
    faces = [x for x in range(6)]
    for i in range(n):
        for j in range(m):
            if grid[i][j] in faces:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
    for j in range(m):
        for i in range(n):
            if grid[i][j] in faces:
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    trimmed = [[grid[i][j] for j in range(min_c, max_c + 1)] for i in range(min_r, max_r + 1)]
    return trimmed

def decode_grid(model, dim, starting_cell, dirs, edges, N, save_path, show_fig):
    grid = [[-1 for j in range(N)] for i in range(N)]
    orientations_2d = [[-1 for j in range(N)] for i in range(N)]
    surface = 2*(dim[0] * dim[1] + dim[1] * dim[2] + dim[0] * dim[2])
    box_to_grid = {}  ##Maps tuples (i, j, k) to the 2d coords
    deltas = [(0, 1), (0, -1), (-1, 0), (1, 0)]  ##2d changes corresponding to the orientations
    start_r, start_c = N // 2, N // 2  ##Start drawing at the center
    orientation = {}  ##Maps each cell to its chosen orientation

    for i in range(6):
        r, c = get_dim(i, dim)
        for j in range(r):
            for k in range(c):
                for d in range(4):
                    if dirs[i][j][k][d] in model:
                        orientation[(i, j, k)] = d  ##Populate orientations of the cells
    box_to_grid[starting_cell] = (start_r, start_c)

    assert (len(box_to_grid) == 1)  ##One unique starting cell with activation time 0
    prev_len = -1
    while len(box_to_grid) != prev_len:
        # print(f"Drawn cells: {len(box_to_grid)}")
        prev_len = len(box_to_grid)
        ##Just try to greedily extend current drawn cells
        to_add = []
        # print(box_to_grid)
        for cell in box_to_grid.keys():
            ##Iterate over potential neighbours
            cur_r, cur_c = box_to_grid[cell][0], box_to_grid[cell][1]
            neigh = get_neigh(cell[0], cell[1], cell[2], dim)
            current_neigh, orients = get_ori(orientation[cell], neigh, cell[0])
            # print(f"current cell:{cell} oriented neighbours: {current_neigh} orientations: {orients}")
            for idx in range(4):
                ni, nj, nk = current_neigh[idx][0], current_neigh[idx][1], current_neigh[idx][2]
                e = edges[order(cell, (ni, nj, nk))]
                if -e in model:
                    ##If this edge isn't cut and the neighbour hasn't been visited
                    to_add.append([(ni, nj, nk), (cur_r + deltas[idx][0], cur_c + deltas[idx][1])])
        for a in to_add:
            if a[0] in box_to_grid and a[1] != box_to_grid[a[0]]:
                print("overwritten - not enough edges are cut")
            else:
                orientations_2d[a[1][0]][a[1][1]] = orientation[a[0]]
                box_to_grid[a[0]] = a[1]
    if len(box_to_grid) != surface:
        print("Insufficient area: Disconnected solution")

    for cell in box_to_grid:
        if grid[box_to_grid[cell][0]][box_to_grid[cell][1]] != -1:
            print("overlap? - most likely a cycle")
        grid[box_to_grid[cell][0]][box_to_grid[cell][1]] = cell[0]
    draw_decode(grid, orientations_2d, save_path, show_fig)
    # print_grid(grid)
    return box_to_grid, grid, orientations_2d

def compute_radius(grid):
    ##Returns radius of the connected component starting at the center of the grid
    cell_to_time = {}
    N = len(grid)
    assert N == len(grid[0])
    to_visit = [(N//2, N//2)]
    cell_to_time[(N//2, N//2)] = 0
    deltas = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    while len(to_visit) > 0:
        cur = to_visit.pop(0)
        for d in range(4):
            nx, ny = cur[0] + deltas[d][0], cur[1] + deltas[d][1]
            if grid[nx][ny] != -1 and (nx, ny) not in cell_to_time:
                cell_to_time[(nx, ny)] = cell_to_time[cur] + 1
                to_visit.append((nx, ny))
    return max(cell_to_time.values())

def count_edges(grid):
    ##Returns a map count of edges for 6 faces
    edge_counts = {i : 0 for i in range(6)}
    N = len(grid)
    for i in range(N):
        for j in range(N):
            if grid[i][j] == -1:
                continue
            cur_f = grid[i][j]
            if i > 0 and grid[i - 1][j] == cur_f:
                edge_counts[cur_f] += 1
            if i < N - 1 and grid[i + 1][j] == cur_f:
                edge_counts[cur_f] += 1
            if j > 0 and grid[i][j - 1] == cur_f:
                edge_counts[cur_f] += 1
            if j < N - 1 and grid[i][j + 1] == cur_f:
                edge_counts[cur_f] += 1
    for i in range(6):
        edge_counts[i] //= 2
    return edge_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Decoder for box folding')

    # Add arguments

    ##Required dimensions
    parser.add_argument('-d', '--dimensions', nargs='+',
                        help='Dimensions of faces of boxes, should be a list of 3*n numbers', required=True)
    ##Optional
    parser.add_argument('-n', '--boxes', type=int, help='Number of common boxes desired, default = 2', default=2)
    parser.add_argument('-g', '--grid', type=int, help='Size of square grid canvas, default = 30', default=30)
    parser.add_argument('-a', '--activation', type=int, help='Max activation time allowed (radius of the unfolded polygon), default = 12', default=12)
    parser.add_argument('-o', '--output', type=str, help='Path to output decoding, pictures not saved if empty. Default is empty', default='')
    parser.add_argument('-s', '--show', action='store_true', help='Flag to show the decoded grid')
    parser.add_argument('-i', '--input', type=str, help='Path to input solution', required=True)
    parser.add_argument('-e', '--edge', type=str, help='Path to file outputting disconnected edges of the first box. Default = "" no output', default="")
    parser.add_argument('-ec', '--edge_connected', type=str, help='Path to file outputting connected edges of the first box. Default = "" no output', default="")
    parser.add_argument('-v', '--verbose', type=int, help='Verbosity value. 0 = no stats, 1 = edge stats', default=0)
    parser.add_argument('-fb', '--first_box', type=int, help='Flag to print grid of first box. 0 = do not print, 1 = print (default)', default=1)
    parser.add_argument('--sinks', type=int, help='Number of sinks. Should be in [1, 2, 6]. Increasing decreases the likelihood of disconnected solutions', default = 1)


    args = parser.parse_args()

    dims = args.dimensions
    boxes = args.boxes
    max_activation = args.activation
    show_fig = args.show
    out_path = args.output
    in_path = args.input
    verbosity = args.verbose
    N = args.grid
    edge_out = args.edge
    con_edge_out = args.edge_connected
    output_grid = args.first_box
    sinks = args.sinks

    assert sinks in [1, 2, 6], "Incorrect number of sinks"
    assert len(dims) == 3 * boxes, f"Number of dimensions: {len(args.dimensions)} does not match number of boxes: {args.boxes}"

    cnf = CNF()
    dimension_list = [[int(dims[i*3]), int(dims[i*3 + 1]), int(dims[i*3 + 2])] for i in range(boxes)]
    model = get_model(in_path)
    ##Idea: Unfold both primary and secondary boxes
    ##And then ensure mapping equivalence

    orientations = []
    edges = []
    mappings = []
    edge_orientations = []
    center_faces = [2 for i in range(boxes)] ##TODO: Add this to command line arguments
    id = 1
    for i in range(boxes):
        r, c = get_dim(center_faces[i], dimension_list[i])
        starting_cell = (center_faces[i], r // 2, c // 2)
        cur_ori, cur_edges, cur_edge_ori, id = encode_box_1(dimension_list[i], cnf, id, starting_cell, sinks)
        orientations.append(cur_ori)
        edges.append(cur_edges)
        edge_orientations.append(cur_edge_ori)
    # print(f"total literals after unfolding: {id}")
    ##Encode mappings to the first box
    for i in range(1, boxes):
        cur_mapping, id = encode_box_2(dimension_list[i], cnf, orientations[i], edges[i], id, dimension_list[0],
                                       edges[0], orientations[0])
        mappings.append(cur_mapping)


    grids = []


    save_path = "" if out_path == "" else out_path + "_box_0.png"
    r, c = get_dim(center_faces[0], dimension_list[0])
    box_to_grid_1, grid_1, orientations_2d_1 = decode_grid(model, dimension_list[0], (center_faces[0], r//2, c//2), orientations[0], edges[0], N, save_path, show_fig)
    grids.append(grid_1)

    for b in range(1, boxes):
        grid = [[-1 for _ in range(N)] for _ in range(N)]
        cur_ori_2d = [[-1 for _ in range(N)] for _ in range(N)]
        cur_ori = orientations[b]
        cell_to_ori = {}
        for i in range(6):
            r, c = get_dim(i, dimension_list[b])
            for j in range(r):
                for k in range(c):
                    for d in range(4):
                        if cur_ori[i][j][k][d] in model:
                            cell_to_ori[(i, j, k)] = d  ##Populate orientations of the cells
                    drawn = False
                    for cell in box_to_grid_1.keys():
                        if mappings[b - 1][(i, j, k)][cell] in model:
                            grid[box_to_grid_1[cell][0]][box_to_grid_1[cell][1]] = i
                            cur_ori_2d[box_to_grid_1[cell][0]][box_to_grid_1[cell][1]] = cell_to_ori[(i, j, k)]
                            drawn = True
                    if not drawn:
                        print(f"Empty map at cell: {i, j, k}")
        save_path = "" if out_path == "" else out_path + f"_box_{b}.png"
        draw_decode(grid, cur_ori_2d, save_path, show_fig)
        grids.append(grid)
    if verbosity != 0:
        for i in range(boxes):
            print(f"c Connected edges for Box: {i + 1}")
            cur_edges = edges[i]
            for e in cur_edges:
                if -cur_edges[e] in model:
                    print(" ".join(str(x) for x in list(e[0]) + list(e[1])))

        if verbosity >= 2:
            rad = compute_radius(grids[0])
            print(f"Radius: {rad}")
            for b in range(boxes):
                print(f"Edge counts for Box: {b + 1}")
                edge_cnts = count_edges(grids[b])
                for i in range(6):
                    r, c = get_dim(i, dimension_list[b])
                    if r == 1 and c == 1:
                        print(
                            f"Face: {i} Total edges: {(r - 1) * c + (c - 1) * r} Remaining edges: {edge_cnts[i]} Percentage: ----")
                    else:
                        print(f"Face: {i} Total edges: {(r - 1)*c + (c - 1)*r} Remaining edges: {edge_cnts[i]} Percentage: {100*edge_cnts[i]/((r - 1)*c + (c - 1)*r):.2f}%")
                print("="*50)

    if edge_out != "":
        def next_to(p1, p2):
             return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1
        edges_cut = []
        for e in edges[0]:
            s, t = e[0], e[1]
            if not next_to(box_to_grid_1[s], box_to_grid_1[t]):
                edges_cut.append(-edges[0][e])
        with open(edge_out, "a") as f:
            f.write(" ".join(map(str, edges_cut)) + "\n")
        print(f"Edges avoided: {len(edges_cut)}")

    if con_edge_out != "":
        def next_to(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1
        connected_edges = []
        for e in edges[0]:
            s, t = e[0], e[1]
            if next_to(box_to_grid_1[s], box_to_grid_1[t]):
                connected_edges.append(-edges[0][e])
        with open(con_edge_out, "a") as f:
            f.write(" ".join(map(str, connected_edges)) + "\n")
        print(f"Edges connected: {len(connected_edges)}")

    if output_grid == 1:
        for b in range(boxes):
            cur_grid = [['*' if grids[b][i][j] == -1 else grids[b][i][j] for j in range(N)] for i in range(N)]
            print_grid(format_grid(cur_grid))