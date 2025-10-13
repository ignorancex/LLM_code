import numpy as np
from soar.render import alt_color_scheme_name,color_scheme_name
from collections import Counter, defaultdict
from typing import Optional

# Most of representation from https://github.com/rgreenblatt/arc_draw_more_samples_pub but only numpy is used (could be cool to see which one is better after SFT) 
spreadsheet_col_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AB",
    "AC",
    "AD",
]

from typing import List


def grid_formatting(grid: List[List[int]],mode="ascii") -> str:
    """
    Take a grid and return a string representation
    mode: str
        "ascii": return the grid in ascii format
                        0|7|7
                        7|7|7
                        0|7|7
        "spreadsheet": return the grid in spreadsheet format
                         |A|B|C
                        1|0|7|7
                        2|7|7|7
                        3|0|7|7
        "spreadsheet_color_location": return the grid in spreadsheet with weird format
                        0 A1|7 B1|7 C1
                        7 A2|7 B2|7 C2
                        0 A3|7 B3|7 C3
        "numpy": return the grid as a numpy array style
                        [[0 7 7]
                         [7 7 7]
                         [0 7 7]]

    """
    if mode == "ascii":
        return ascii_grid(np.array(grid))
    elif mode == "spreadsheet":
        return ascii_grid(np.array(grid),spreadsheet_ascii=True)
    elif mode == "spreadsheet_color_location":
        return spreadsheet_ascii_grid_as_color_by_location(np.array(grid))
    elif mode == "numpy":
        return str(np.array(grid))
    elif mode == "colors":
        return "[\n"+ "\n".join(
                            " " * 4
                            + "["
                            + ", ".join(f'"{alt_color_scheme_name[x]}"' for x in row)
                            + "],"
                            for row in grid
                        )+ "\n]"
    else:
        raise ValueError(f"Invalid mode {mode}")
    
def ascii_grid(grid: np.ndarray, separator: str = "|", spreadsheet_ascii: bool = False):
    if spreadsheet_ascii:
        return spreadsheet_ascii_grid(grid, separator=separator)

    return "\n".join(separator.join(str(x) for x in row) for row in grid)


def spreadsheet_ascii_grid(grid: np.ndarray, separator: str = "|"):
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    cols_header_line = separator.join([" "] + spreadsheet_col_labels[:cols])
    rest = "\n".join(
        separator.join([str(i + 1)] + [str(x) for x in row])
        for i, row in enumerate(grid)
    )

    return f"{cols_header_line}\n{rest}"


def get_spreadsheet_notation_str(i, j, quote: bool = True):
    out = f"{spreadsheet_col_labels[j]}{i+1}"
    if quote:
        out = f'"{out}"'
    return out


def spreadsheet_ascii_grid_as_color_by_location(grid: np.ndarray):
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    out = "\n".join(
        "|".join(
            f"{grid[i, j]} {get_spreadsheet_notation_str(i, j,quote=False)}"
            for j in range(cols)
        )
        for i in range(rows)
    )

    return out

## stuff to check:

def spreadsheet_ascii_grid_by_color_diffs(
    grid_input: np.ndarray,
    grid_output: np.ndarray,
    use_alt_color_scheme: bool = True,
    use_expected_vs_got: bool = False,
):
    assert grid_input.shape == grid_output.shape
    grid_differs_x, grid_differs_y = (grid_input != grid_output).nonzero()
    differences_by_color_pairs: dict[tuple[int, int], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist()):
        differences_by_color_pairs[(grid_input[x, y], grid_output[x, y])].append(
            (int(x), int(y))
        )

    out = ""
    for (color_input, color_output), differing_locs in sorted(
        differences_by_color_pairs.items(), key=lambda x: x[0]
    ):
        color_str = get_spreadsheet_notation_support_runs(differing_locs)

        scheme = alt_color_scheme_name if use_alt_color_scheme else color_scheme_name

        if use_expected_vs_got:
            out += (
                f"Expected {scheme[color_input]} ({color_input}) but got {scheme[color_output]} ({color_output}):{color_str}"
            ) + "\n"

        else:
            out += (
                f"{scheme[color_input]} ({color_input}) to {scheme[color_output]} ({color_output}):{color_str}"
            ) + "\n"
    return out

def spreadsheet_ascii_grid_by_color_contiguous_normalized(
    shapes_by_color,
    use_alt_color_scheme: bool = True,
    omit_by_color: Optional[dict[int, bool]] = None,
    disable_absolute_in_normalized_ascii: bool = False,
):
    # TODO: support alt color scheme
    out = ""

    for color in range(11):
        contiguous_shapes = shapes_by_color[color]
        if len(contiguous_shapes) == 0:
            continue

        shape_strs: list[str] = []
        for shape in contiguous_shapes:
            min_i = min(i for i, j in shape)
            min_j = min(j for i, j in shape)
            # basic = ",".join(
            #     get_spreadsheet_notation_str(i - min_i, j - min_j, quote=False)
            #     for i, j in
            # )

            normalized = [
                (i - min_i, j - min_j)
                for i, j in sorted(shape, key=lambda x: (int(x[0]), int(x[1])))
            ]

            basic_shape_str = get_spreadsheet_notation_support_runs(normalized)

            if len(shape) > 2 and not disable_absolute_in_normalized_ascii:
                shape_str = (
                    " [Absolute: "
                    + get_spreadsheet_notation_str(
                        shape[0][0], shape[0][1], quote=False
                    )
                    + "]"
                    + basic_shape_str
                )
            else:
                shape_str = basic_shape_str

            shape_strs.append(shape_str)

        color_str = "|".join(shape_strs)

        if omit_by_color is not None and omit_by_color.get(color, False):
            color_str = " [OMITTED DUE TO EXCESSIVE LENGTH]"

        out += (
            f"{(alt_color_scheme_name if use_alt_color_scheme else color_scheme_name)[color]} ({color}):{color_str}"
        ) + "\n"

    return out


def spreadsheet_ascii_grid_by_color_contiguous_absolute_small_shapes(
    overall_rows: int,
    overall_cols: int,
    shapes_by_color,
    use_alt_color_scheme: bool = True,
    separator: str = "|",
):
    overall_out = ""
    any_ever_used = False
    for color in range(11):
        contiguous_shapes = shapes_by_color[color]
        if len(contiguous_shapes) == 0:
            continue
        this_str = f"Color: {color}\n\n"

        any_used = False
        for shape_idx, shape in enumerate(contiguous_shapes):
            min_i = min(i for i, j in shape)
            min_j = min(j for i, j in shape)

            absolute_shifted_shape = [(i - min_i, j - min_j) for i, j in shape]

            n_rows = max(i for i, j in absolute_shifted_shape) + 1
            n_cols = max(j for i, j in absolute_shifted_shape) + 1

            if (
                (n_rows > overall_rows // 2 and n_cols > overall_cols // 2)
                or n_rows * n_cols > 50
                or n_rows * n_cols == 1
            ):
                continue

            any_used = True
            any_ever_used = True

            assert n_rows <= 30
            assert n_rows <= 30

            cols_header_line = separator.join([" "] + spreadsheet_col_labels[:n_cols])

            grid_labels = np.full((n_rows, n_cols), fill_value="O", dtype=object)

            for i, j in absolute_shifted_shape:
                grid_labels[i, j] = "X"

            rest = "\n".join(
                separator.join([str(i)] + [str(x) for x in row])
                for i, row in enumerate(grid_labels)
            )

            this_str += f'"shape_{shape_idx}_with_color_{(alt_color_scheme_name if use_alt_color_scheme else color_scheme_name)[color]}_{color}":\n\n'
            this_str += f"Bounding box shape: {n_rows} by {n_cols}\n\n"

            this_str += f"{cols_header_line}\n{rest}\n\n"

            this_str += (
                f"Normalized locations: ["
                + ", ".join(
                    get_spreadsheet_notation_str(i, j)
                    for i, j in absolute_shifted_shape
                )
                + "]\n\n"
            )

        if any_used:
            overall_out += this_str

    if not any_ever_used:
        return None

    return overall_out



def get_spreadsheet_notation_support_runs(rows_cols: list[tuple[int, int]]):
    row_cols_v = np.array(sorted(rows_cols, key=lambda x: (x[0], x[1])))

    running_str = ""

    idx = 0
    while idx < len(row_cols_v):
        r, c = row_cols_v[idx]

        count_in_a_row = 0
        for checking_idx, (n_r, n_c) in enumerate(row_cols_v[idx:]):
            if n_r == r and n_c == c + checking_idx:
                count_in_a_row += 1
            else:
                break

        if count_in_a_row > 4:
            start = get_spreadsheet_notation_str(r, c, quote=False)
            c_end = c + count_in_a_row - 1

            assert np.array_equal(row_cols_v[idx + count_in_a_row - 1], (r, c_end)), (
                row_cols_v[idx + count_in_a_row - 1],
                (r, c_end),
            )

            end = get_spreadsheet_notation_str(r, c_end, quote=False)

            running_str += f" {start} ... {end}"
            idx += count_in_a_row
        else:
            running_str += " " + get_spreadsheet_notation_str(r, c, quote=False)
            idx += 1

    return running_str
