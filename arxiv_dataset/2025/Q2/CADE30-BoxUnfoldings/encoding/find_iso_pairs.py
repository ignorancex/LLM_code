##Script to compute pairs of representatives used in symmetry breaking
from scipy.cluster.hierarchy import DisjointSet
from itertools import chain, combinations


def get_cells(dims):
    ##returns a list of cells on the box with dimension dims
    cells = []
    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                cells.append((i, j, k))
    return cells


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


def reflect_xy(dims):
    ##Returns a dictionary mapping M: cells \to cells
    ##where M[c] is the cell mapped to after reflecting along the xy plane (i.e. faces 0, 1 swapped)
    ##Fixed faces: [2, 3, 4, 5]

    ##dims: dimensions of original box
    mapping = {}

    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                if i == 0:
                    mapping[(i, j, k)] = (1, j, k)
                elif i == 1:
                    mapping[(i, j, k)] = (0, j, k)
                else:
                    mapping[(i, j, k)] = (i, r - 1 - j, k)
    return mapping
def rotate_z_90(dims):
    ### Rotates around the z axis \pi/2 radians clockwise ###
    mapping = {}
    cells = get_cells(dims)
    for cell in cells:
        face = cell[0]
        if face == 0 or face == 1:
            ##faces fixed
            mapping[cell] = (face, cell[2], dims[0] - 1 - cell[1])
        elif face == 2:
            face = 5
            mapping[cell] = (face, cell[1], cell[2])
        elif face == 5:
            face = 3
            mapping[cell] = (face, cell[1], dims[0] - 1 - cell[2])
        elif face == 3:
            face = 4
            mapping[cell] = (face, cell[1], cell[2])
        elif face == 4:
            face = 2
            mapping[cell] = (face, cell[1], dims[0] - 1 - cell[2])
    return mapping, [dims[1], dims[0], dims[2]]

def rotate_x_90(dims):
    ### Rotates around the x axis \pi/2 radians clockwise ###
    mapping = {}
    cells = get_cells(dims)
    for cell in cells:
        face = cell[0]
        if face == 4 or face == 5:
            ##faces fixed
            mapping[cell] = (cell[0], dims[0] - 1 - cell[2], cell[1])
        elif cell[0] == 0:
            face = 2
            mapping[cell] = (face, dims[0] - 1 - cell[1], cell[2])
        elif cell[0] == 2:
            face = 1
            mapping[cell] = (face, cell[1], cell[2])
        elif cell[0] == 1:
            face = 3
            mapping[cell] = (face, dims[0] - 1 - cell[1], cell[2])
        elif cell[0] == 3:
            face = 0
            mapping[cell] = (face, cell[1], cell[2])
    return mapping, [dims[2], dims[1], dims[0]]

def reflect_xz(dims):
    ## gives a function mapping between cells such that
    ## mapping[c] is the cell you get after 'reflecting' about the xz plane
    ## Fixed faces: [0, 1, 4, 5]

    mapping = {}

    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                if i == 0 or i == 1:
                    mapping[(i, j, k)] = (i, r - 1 - j, k)
                elif i == 2:
                    mapping[(i, j, k)] = (3, j, k)
                elif i == 3:
                    mapping[(i, j, k)] = (2, j, k)
                else:
                    mapping[(i, j, k)] = (i, j, c - 1 - k)
    return mapping


def reflect_yz(dims):
    ## gives a function mapping between cells such that
    ## mapping[c] is the cell you get after 'reflecting' about the yz plane
    ## Fixed faces: [0, 1, 2, 3]

    mapping = {}

    for i in range(6):
        r, c = get_dim(i, dims)
        for j in range(r):
            for k in range(c):
                if i == 4:
                    mapping[(i, j, k)] = (5, j, k)
                elif i == 5:
                    mapping[(i, j, k)] = (4, j, k)
                else:
                    mapping[(i, j, k)] = (i, j, c - 1 - k)
    return mapping


def powerset(items):
    ##Does not include the empty set
    s = list(items)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def compose(m_1, m_2):
    ##Returns the compositions m_1 \circ m_2
    ##Requires range(m_2) \subseteq domain(m_1)
    ##Note that m_2 is applied FIRST

    m_comp = {}
    for k in m_2.keys():
        assert m_2[k] in m_1, "Domain/range mismatch between composed mappings"
        m_comp[k] = m_1[m_2[k]]
    return m_comp

def get_symmetries(dims):
    ##Computes all symmetries
    ##First finds 24 rotational symmetries following: https://stackoverflow.com/questions/16452383/how-to-get-all-24-rotations-of-a-3-dimensional-array?rq=3
    ##Then precomposes with reflect_xy to obtain 48 symmetries of a cube
    ##Then checks how many preserve dimensions

    symmetries = []
    cells = get_cells(dims)
    current_map = {cell : cell for cell in cells}
    current_dims = dims

    for cycle in range(2):
        for step in range(3):
            rot, current_dims = rotate_x_90(current_dims)
            current_map = compose(rot, current_map)
            symmetries.append([current_map, current_dims])
            for i in range(3):
                ##Do 3 turns
                turn, current_dims = rotate_z_90(current_dims)
                current_map = compose(turn, current_map)
                symmetries.append([current_map, current_dims])
        ##Do RTR
        rot, current_dims = rotate_x_90(current_dims)
        current_map = compose(rot, current_map)
        turn, current_dims = rotate_z_90(current_dims)
        current_map = compose(turn, current_map)
        rot, current_dims = rotate_x_90(current_dims)
        current_map = compose(rot, current_map)
    ##Prune symmetries that don't match the current type
    correct_syms = []
    refl = reflect_xy(dims)
    for transformation in symmetries:
        if transformation[1] == dims:
            correct_syms.append(transformation[0])
            correct_syms.append(compose(transformation[0], refl))
    # print(f"Found: {len(correct_syms)} symmetries")
    return correct_syms

def find_pairs(dims):
    cells = get_cells(dims)
    equiv_classes = DisjointSet(cells)
    symmetries = get_symmetries(dims)
    for sym in symmetries:
        ##Considers generators of this symmetry
        for cell in cells:
            equiv_classes.merge(sym[cell], cell)
    reps = []
    roots = set()
    for cell in cells:
        root = equiv_classes.__getitem__(cell)
        if root not in roots:
            roots.add(root)
            reps.append([equiv_classes.subset_size(cell), cell])

    reps = [x for _, x in sorted(reps)]
    removed_roots = set()
    pairs = {}
    ##Generate pairs
    for rep in reps:
        ##Consider all fixed vals
        pairs[rep] = []
        cur_roots = set()
        cur_cells = get_cells(dims)
        cur_equiv_class = DisjointSet(cur_cells)
        for sym in symmetries:
            if sym[rep] != rep:
                ##Only look at symmetries that fix the current representative
                continue
            for cell in cur_cells:
                cur_equiv_class.merge(cell, sym[cell])
        for cell in cur_cells:
            if equiv_classes.__getitem__(cell) in removed_roots or cur_equiv_class.__getitem__(
                    cell) in cur_roots or cell == rep:
                continue
            else:
                pairs[rep].append(cell)
                cur_roots.add(cur_equiv_class.__getitem__(cell))
        removed_roots.add(equiv_classes.__getitem__(rep))
    return reps, pairs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generator for pairs of representatives used in symmetry breaking')

    ##Required dimensions
    parser.add_argument('-d', '--dimensions', nargs=3, type=int,
                        help='Dimension of second box', required=True)
    parser.add_argument('-f', '--format', type=int,
                        help='Flag dictating if output is formatted, default is true (1)', default=1)
    args = parser.parse_args()
    to_format = args.format
    dims = args.dimensions

    ##Currently only have reflections implemented
    reps, pairs = find_pairs(dims)
    output = []
    for rep in reps:
        for p in pairs[rep]:
            output.append([rep, p])
    if to_format == 0:
        print(len(output))
    else:
        print(f"Total pairs: {len(output)}")
    for out in output:
        if to_format == 0:
            print(f"{out[0][0]} {out[0][1]} {out[0][2]} {out[1][0]} {out[1][1]} {out[1][2]}")
        else:
            print(out[0], out[1])
