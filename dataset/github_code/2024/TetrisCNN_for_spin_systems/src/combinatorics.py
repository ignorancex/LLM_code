import argparse
import pickle
import time
from itertools import combinations, product
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Define the 3D Grid Graph


def create_3d_grid_graph(H, W, D):
    G = nx.grid_graph(dim=[range(H), range(W), range(D)])
    return G


def plot_graph(G):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the graph
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=8,
        ax=ax,
    )

    # Add edge labels, and round to 2 decimal points
    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.set_title("Graph of connections between nodes")

    # Show the plot
    fig.show()


# Step 2: Generate Combinations


def generate_combinations_efficiently(H, W, D):
    all_cubes = list(product(range(H), range(W), range(D)))
    for k in range(1, H * W * D + 1):
        for comb in combinations(all_cubes, k):
            yield comb


# Step 3: Check symmetries and transform the combinations


# Helper functions
def normalize_coordinates(cubes, H, W, D):
    """
    Normalize coordinates to ensure they fall within the (H, W, D) bounds.
    """
    min_x = min(cube[0] for cube in cubes)
    min_y = min(cube[1] for cube in cubes)
    min_z = min(cube[2] for cube in cubes)

    # Translate to ensure all coordinates are non-negative
    translated_cubes = [(x - min_x, y - min_y, z - min_z) for x, y, z in cubes]

    # Check if translation brought coordinates within bounds
    max_x = max(cube[0] for cube in translated_cubes)
    max_y = max(cube[1] for cube in translated_cubes)
    max_z = max(cube[2] for cube in translated_cubes)

    if max_x >= H or max_y >= W or max_z >= D:
        return None

    return translated_cubes


def ensure_origin_cube(cubes, H, W, D):
    """
    Ensure that the cube (0, 0, 0) is within the shape. If not, shift all the coordinates accordingly,
    preferring to keep the nearest neighbors of the origin node if possible.
    """
    if (0, 0, 0) in cubes:
        return cubes

    # Initialize shifts to large values
    min_shifts = {"H": float("inf"), "W": float("inf"), "D": float("inf")}

    # Check for nearest neighbors in all directions, up to a distance of H, W, D in respective directions
    for z in range(0, D + 1):
        for y in range(0, W + 1):
            for x in range(0, H + 1):
                if (x, y, z) in cubes:
                    # Calculate the shift required to bring this cube to the origin
                    shift_H = x
                    shift_W = y
                    shift_D = z
                    if abs(shift_H) + abs(shift_W) + abs(shift_D) < abs(
                        min_shifts["H"]
                    ) + abs(min_shifts["W"]) + abs(min_shifts["D"]):
                        min_shifts = {"H": shift_H, "W": shift_W, "D": shift_D}

    # Shift all cubes by the minimum shifts
    shifted_cubes = [
        (H - min_shifts["H"], W - min_shifts["W"], D - min_shifts["D"])
        for H, W, D in cubes
    ]

    return shifted_cubes


def rotate(cubes, axis):
    rotation_matrices = {
        "x": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "y": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "z": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    }
    matrix = rotation_matrices[axis]
    return [tuple(matrix.dot(np.array(cube))) for cube in cubes]


def reflect(cubes, axis):
    reflection_matrices = {
        "x": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "y": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        "z": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    }
    matrix = reflection_matrices[axis]
    return [tuple(matrix.dot(np.array(cube))) for cube in cubes]


def generate_symmetric_variants(cubes, H, W, D):
    """
    Generate symmetric variants of a given combination of cubes within given bounds (H, W, D).

    Args:
        cubes (list): List of cubes representing the combination.
        H (int): Height of the 3D space.
        W (int): Width of the 3D space.
        D (int): Depth of the 3D space.

    Returns:
        set: Set of frozenset symmetric variants of the given combination.
    """

    axes = ["x", "y", "z"]
    variants = set()

    # Normalize the original combination and add it if valid
    normalized = normalize_coordinates(cubes, H, W, D)
    if normalized:
        variants.add(frozenset(normalized))

    # Generate all reflected and rotated versions
    for axis in axes:
        reflected = reflect(cubes, axis)
        normalized = normalize_coordinates(reflected, H, W, D)
        if normalized:
            variants.add(frozenset(normalized))

        for _ in range(3):
            reflected = rotate(reflected, axis)
            normalized = normalize_coordinates(reflected, H, W, D)
            if normalized:
                variants.add(frozenset(normalized))

        rotated = rotate(cubes, axis)
        for _ in range(3):
            normalized = normalize_coordinates(rotated, H, W, D)
            if normalized:
                variants.add(frozenset(normalized))
            rotated = rotate(rotated, axis)

    return variants


# The main symmetry check function
def reduce_symmetry(combinations, H, W, D, old_combs=None, full=True):
    if old_combs is None:
        unique_combinations = set()
    else:
        unique_combinations = old_combs
    for combination in combinations:
        symmetries = generate_symmetric_variants(combination, H, W, D)
        if full:
            if not any(variant in unique_combinations for variant in symmetries):
                unique_combinations.add(frozenset(combination))
        else:
            for variant in symmetries:
                if variant not in unique_combinations:
                    unique_combinations.add(variant)
    return unique_combinations


# Step 4: Wrapper functions


def load_precomputed(
    H, W, D, just_new=False, just_topologically_distinct=False, which=None
):
    """
    Load precomputed combinations from pickle files.

    This function loads precomputed combinations from pickle files based on the provided dimensions (H, W, D).
    It provides the flexibility to load only new combinations or only topologically distinct combinations,
    depending on the values of the optional arguments just_new and just_topologically_distinct.

    Args:
        H (int): Height of the combinations.
        W (int): Width of the combinations.
        D (int): Depth of the combinations.
        just_new (bool, optional): If True, load only new combinations. Defaults to False.
        just_topologically_distinct (bool, optional): If True, load only topologically distinct combinations. Defaults to False.

    Returns:
        dict: A dictionary containing the loaded combinations, bounds, axis, and loaded_full_for_just_new flag.
            - 'combinations' (set): The loaded combinations.
            - 'bounds' (list): The bounds of the combinations.
            - 'axis' (str): The axis of rotation.
            - 'loaded_full_for_just_new' (bool): Flag indicating if the full combinations were loaded for just_new=True.
    """

    if which is None:
        which = "combinations"
    elif which == "shifts":
        pass
    else:
        raise ValueError(
            "Invalid value for 'which'. Must be either 'combinations' or 'shifts'."
        )

    # Create a folder to store the combinations if it doesn't exist
    load_folder = Path(f"./precomputed_corrs/{which}")
    load_folder.mkdir(exist_ok=True, parents=True)

    # Set the bounds based on the provided dimensions
    bounds = (H, W, D)

    # Define the possible axes of rotation
    axes = [None, "x", "y", "z"]

    # Initialize variables
    found = False
    loaded_full_for_just_new = False

    # Iterate over the axes
    for axis in axes:
        if axis is not None:
            bounds_tmp = [abs(el) for el in rotate([bounds], axis)[0]]
        else:
            bounds_tmp = bounds

        # Generate the file name based on the bounds and the optional arguments
        file_name_base = "{}_{}_{}.pkl".format(*bounds_tmp)
        file_name = f"{'just_new_' if just_new else ''}{'unique_' if just_topologically_distinct else ''}{file_name_base}"
        file_path = load_folder / file_name

        # Check if the file exists
        if file_path.exists():
            # Load the combinations from the pickle file
            with open(file_path, "rb") as f:
                combinations = pickle.load(f)
            found = True
        # If just_new is True and the file doesn't exist, check if the full combinations file exists
        elif just_new and not file_path.exists():
            full_combs_path = load_folder / file_name_base
            if full_combs_path.exists():
                # Load the combinations from the full combinations file
                with open(full_combs_path, "rb") as f:
                    combinations = pickle.load(f)
                found = True
                loaded_full_for_just_new = True

        if found:
            combinations_tmp = set()
            if axis is not None:
                # Normalize and rotate each combination
                for combination in combinations:
                    cubes = normalize_coordinates(rotate(combination, axis), *bounds)
                    if cubes is not None:
                        combinations_tmp.add(frozenset(cubes))
            else:
                combinations_tmp = combinations
            return {
                "combinations": combinations_tmp,
                "bounds": bounds_tmp,
                "axis": axis,
                "loaded_full_for_just_new": loaded_full_for_just_new,
            }

    return {
        "combinations": None,
        "bounds": bounds,
        "axis": None,
        "loaded_full_for_just_new": False,
    }


def find_combinations(
    H,
    W,
    D,
    sort=False,
    just_new=False,
    serialize=True,
    just_topologically_distinct=False,
):
    """
    Find combinations within a cube of dimensions H, W, and D.

    Parameters:
    - H (int): The height dimension.
    - W (int): The width dimension.
    - D (int): The depth dimension.
    - sort (bool): Whether to sort the combinations by volume (default: False).
    - just_new (bool): Whether to find only new combinations (default: False).
    - serialize (bool): Whether to serialize and save the combinations (default: True).
    - just_topologically_distinct (bool): Whether to find only topologically distinct combinations (default: False).

    Returns:
    - combinations (list): A list of combinations of dimensions H, W, and D.

    Process:
    1. Check if the input dimensions are valid and minimal.
    2. Define the save and load folder.
    3. Load precomputed combinations if available.
    4. If precomputed combinations are not available, generate new combinations.
    5. If just_new is True, find old combinations recursively.
    6. Reduce symmetry in the combinations.
    7. Serialize and save the combinations if required.
    8. Sort the combinations by volume if required.
    9. Return the combinations.
    """

    # Base case to handle invalid or minimal inputs
    if H < 1 or W < 1 or D < 1:
        return set()

    # Define the save and load folder
    bounds = (H, W, D)
    load_folder = Path("./precomputed_corrs/combinations")
    load_folder.mkdir(exist_ok=True, parents=True)
    file_name_base = "{}_{}_{}.pkl".format(*bounds)
    file_name = f"{'just_new_' if just_new else ''}{'unique_' if just_topologically_distinct else ''}{file_name_base}"
    file_path = load_folder / file_name

    # Load precomputed combinations if available
    combinations_dict = load_precomputed(
        H,
        W,
        D,
        just_new=just_new,
        just_topologically_distinct=just_topologically_distinct,
    )

    if combinations_dict["combinations"] is not None:
        combinations = combinations_dict["combinations"]
        generate = False
    else:
        generate = True

    loaded_full_for_just_new = combinations_dict["loaded_full_for_just_new"]
    rotation_axis = combinations_dict["axis"]
    loaded_from_rotation = rotation_axis is not None

    old_combs = set()
    if just_new:
        # Recursive calls for just_new logic
        old_combs_H = (
            find_combinations(
                H - 1,
                W,
                D,
                just_topologically_distinct=just_topologically_distinct,
                serialize=serialize,
            )
            if H > 1
            else set()
        )

        old_combs_W = (
            find_combinations(
                H,
                W - 1,
                D,
                just_topologically_distinct=just_topologically_distinct,
                serialize=serialize,
            )
            if W > 1
            else set()
        )

        old_combs_D = (
            find_combinations(
                H,
                W,
                D - 1,
                just_topologically_distinct=just_topologically_distinct,
                serialize=serialize,
            )
            if D > 1
            else set()
        )

        old_combs = old_combs_H | old_combs_W | old_combs_D

    if generate:
        # Generate new combinations
        G = create_3d_grid_graph(H, W, D)
        combinations = list(generate_combinations_efficiently(H, W, D))
        combinations = reduce_symmetry(
            combinations, H, W, D, old_combs, full=just_topologically_distinct
        )

    if just_new:
        combinations = combinations - old_combs_H - old_combs_W - old_combs_D

    if (
        (generate and serialize)
        or (loaded_full_for_just_new and serialize)
        or (serialize and rotation_axis is not None)
    ):
        # Save the unique combinations
        with open(file_path, "wb") as f:
            pickle.dump(combinations, f)

    if sort:
        # Sort the combinations by volume
        combinations = sort_combinations_by_volume(combinations)

    return combinations


def combinations_to_shifts_dicts(
    combinations, H, W, D, dilation=1, include_origin=False
):
    """
    Convert combinations of cubes in the bounds into a list of shift dictionaries,
    that can be passed to the correlation calculation function.

    Args:
        combinations (list): List of combinations of shifts.
        H (int): Height of the cube.
        W (int): Width of the cube.
        D (int): Depth of the cube.
        dilation (int or tuple, optional): Dilation factor for each dimension. Defaults to 1.

    Returns:
        list: List of shift dictionaries.

    """
    # Check if dilation is an integer or a tuple
    if type(dilation) == int:
        dilation_H = dilation
        dilation_W = dilation
        dilations_D = 1
    else:
        dilation_H, dilation_W, dilations_D = dilation

    shifts_dicts_list = []
    for comb in combinations:
        shifts_dict = {"j": [], "i": [], "channel": []}

        # Ensure that the origin cube is included in the combination
        comb_origin = ensure_origin_cube(comb, H, W, D)
        comb_origin_wo_origin = set(comb_origin) - {(0, 0, 0)}
        if include_origin or comb_origin_wo_origin == set():
            comb_processed = [(0, 0, 0)] + list(comb_origin_wo_origin)
        else:
            comb_processed = list(comb_origin_wo_origin)

        for cube in comb_processed:
            # Calculate the shifts for each dimension based on the dilation factors
            shifts_dict["j"].append(cube[0] * dilation_H)
            shifts_dict["i"].append(cube[1] * dilation_W)
            shifts_dict["channel"].append(cube[2] * dilations_D)

        for key in shifts_dict.keys():
            shifts_dict[key] = np.array(shifts_dict[key])

        shifts_dicts_list.append(shifts_dict)

    return shifts_dicts_list


def shifts_from_bounds(H, W, D=1, dilation=1, just_new=False, include_origin=False):
    """
    Generate a list of shifts based on the given bounds of a kernel.
    The origin is considered to be the lower-left corner of the lowest level in the kernel.

    Args:
        H (int): Height bound.
        W (int): Width bound.
        D (int, optional): Depth bound. Defaults to 1.
        dilation (int or tuple, optional): Dilation factor for each dimension. Defaults to 1.
        just_new (bool, optional): Flag to indicate whether to generate only new shifts. Defaults to False.

    Returns:
        list: List of shifts.

    """
    # Find all combinations of shifts
    combinations = find_combinations(H, W, D, just_new=just_new)

    # Convert combinations to shifts dictionaries
    shifts_list = combinations_to_shifts_dicts(
        combinations, H, W, D, dilation=dilation, include_origin=include_origin
    )

    shifts_list = sorted(shifts_list, key=lambda x: len(x["j"]))

    return shifts_list


# Step 5: Visualization


def create_mesh_from_cubes(combination):
    # Create an empty mesh
    vertices = []
    faces = []
    face_id = 0

    # Iterate through each cube in the combination
    for x, y, z in combination:
        # Define the vertices for the cube
        cube_vertices = np.array(
            [
                [x, y, z],
                [x + 1, y, z],
                [x + 1, y + 1, z],
                [x, y + 1, z],
                [x, y, z + 1],
                [x + 1, y, z + 1],
                [x + 1, y + 1, z + 1],
                [x, y + 1, z + 1],
            ]
        )

        # Add the cube's vertices to the list
        vertices.extend(cube_vertices)

        # Define the faces for the cube (6 faces, each with 4 vertices)
        cube_faces = [
            [face_id, face_id + 1, face_id + 2, face_id + 3],  # Bottom face
            [face_id + 4, face_id + 5, face_id + 6, face_id + 7],  # Top face
            [face_id, face_id + 1, face_id + 5, face_id + 4],  # Front face
            [face_id + 2, face_id + 3, face_id + 7, face_id + 6],  # Back face
            [face_id, face_id + 3, face_id + 7, face_id + 4],  # Left face
            [face_id + 1, face_id + 2, face_id + 6, face_id + 5],  # Right face
        ]

        # Add the faces to the list
        faces.extend(cube_faces)

        # Update the face ID for the next cube
        face_id += 8

    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh


def plot_mesh(mesh, xlim=None, ylim=None, zlim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract the faces and vertices from the mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a collection for the faces
    face_collection = Poly3DCollection(vertices[faces], edgecolor="k")
    face_collection.set_facecolor([0.5, 0.5, 1, 0.8])  # RGBA

    # Add the collection to the plot
    ax.add_collection3d(face_collection)

    # Set the aspect ratio and limits
    ax.set_aspect("auto")
    if xlim is not None:
        ax.set_xlim(0, xlim)
    else:
        ax.set_xlim(np.min(vertices[:, 0]), np.max(vertices[:, 0]))
    if ylim is not None:
        ax.set_ylim(0, ylim)
    else:
        ax.set_ylim(np.min(vertices[:, 1]), np.max(vertices[:, 1]))
    if zlim is not None:
        ax.set_zlim(0, zlim)
    else:
        ax.set_zlim(np.min(vertices[:, 2]), np.max(vertices[:, 2]))

    # Add labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the viewing angle
    ax.view_init(elev=45, azim=-135)

    # Show the plot
    plt.show()


def plot_combination(combination, shape=None):
    # Create the mesh from the combination
    mesh = create_mesh_from_cubes(combination)

    # Plot the mesh
    if shape is not None:
        plot_mesh(mesh, xlim=shape[0], ylim=shape[1], zlim=shape[2])
    else:
        plot_mesh(mesh)


def sort_combinations_by_volume(combinations):
    return sorted(combinations, key=lambda x: len(x))


def main(
    H,
    W,
    D,
    save_to_file=None,
    just_new=False,
    serialize=True,
    unique=False,
    shifts=False,
):
    with profile_memory_and_time(H, W, D, save_to_file):
        print(f"Finding unique combinations for a {H}x{W}x{D} grid...")
        combinations = find_combinations(
            H,
            W,
            D,
            just_new=just_new,
            serialize=serialize,
            just_topologically_distinct=unique,
        )
        if shifts:
            shifts_list = combinations_to_shifts_dicts(combinations, H, W, D)
        if save_to_file is not None:
            with open(save_to_file, "a") as f:
                print(f"{len(combinations)}", file=f, end=",")
        print(
            f"Found {len(combinations)} {'pruned ' if just_new else ''}{'unique' if unique else ''} combinations."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--W", type=int, default=2)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--just_new", action="store_true")
    parser.add_argument("--shifts", action="store_true")
    args = parser.parse_args()

    H, W, D = args.H, args.W, args.D

    main(
        H,
        W,
        D,
        just_new=args.just_new,
        serialize=True,
        unique=False,
        shifts=args.shifts,
    )

    print("Done.")
