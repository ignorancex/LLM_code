from shapely.ops import unary_union, polygonize
from shapely.geometry import Polygon, LineString, Point
from typing import Tuple, Dict, List

planning_area_gen_area_dict = {
    1: "17", 2: "18", 3: "19", 4: "25", 5: "21", 6: "34", 7: "22", 8: "29", 9: "44", 10: "53", 11: "55", 12: "20",
    13: "23",
    14: "24", 15: "26", 16: "40", 17: "27", 18: "28", 19: "33", 20: "56", 21: "60", 22: "31", 23: "30", 24: "38",
    25: "35", 26: "39",
    27: "42", 28: "36", 29: "57", 30: "6", 31: "46", 32: "49", 33: "45", 34: "43", 35: "47", 36: "48", 37: "37",
    38: "32", 39: "13", 40: "4", 41: "52", 42: "54",
}


def find_closed_regions(border_df):
    edges = []
    for _, row in border_df.iterrows():
        start = (row["Start_X"], row["Start_Y"])
        end = (row["End_X"], row["End_Y"])
        if start != end:
            line = LineString([start, end])
            if line.length > 0:
                edges.append(line)
    if len(edges) == 0:
        print("No valid local_edges found in border_df!")
        return []
    try:
        merged_lines = unary_union(edges)
    except Exception as e:
        print(f"Error in unary_union: {e}")
        return []
    polygons = [p for p in polygonize(merged_lines) if isinstance(p, Polygon)]
    return polygons


def find_closed_regions_with_substations(polygons: List[Polygon],
                                         substations_to_coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, List[str]]:
    regions_with_substations = {}
    for substation in substations_to_coordinates:
        coordinate_x, coordinate_y = substations_to_coordinates[substation]
        for idx, polygon in enumerate(polygons, start=1):
            if polygon.contains(Point(coordinate_x, coordinate_y)):
                if planning_area_gen_area_dict[idx] in regions_with_substations:
                    regions_with_substations[planning_area_gen_area_dict[idx]].append(substation)
                else:
                    regions_with_substations[planning_area_gen_area_dict[idx]] = [substation]

    return regions_with_substations
