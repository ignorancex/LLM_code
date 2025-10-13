from shapely.geometry import Point

planning_area_gen_area_dict = {
    1: "17", 2: "18", 3: "19", 4: "25", 5: "21", 6: "34", 7: "22", 8: "29", 9: "44", 10: "53", 11: "55", 12: "20",
    13: "23", 14: "24", 15: "26", 16: "40", 17: "27", 18: "28", 19: "33", 20: "56", 21: "60", 22: "31", 23: "30",
    24: "38", 25: "35", 26: "39", 27: "42", 28: "36", 29: "57", 30: "6", 31: "46", 32: "49", 33: "45", 34: "43",
    35: "47", 36: "48", 37: "37", 38: "32", 39: "13", 40: "4", 41: "52", 42: "54",
}


def map_cities_to_planning_areas(closed_regions, city_to_coordinates):
    planning_area_to_cities = {planning_area_gen_area_dict[idx + 1]: [] for idx in range(len(closed_regions))}

    for name, (x, y) in city_to_coordinates.items():
        city_point = Point(x, y)  # Create a point object for the city

        for idx, polygon in enumerate(closed_regions):
            if polygon.contains(city_point):  # Check if the polygon contains the point
                real_planning_area_id = planning_area_gen_area_dict[idx + 1]  # Get the real planning area identifier
                planning_area_to_cities[real_planning_area_id].append(
                    name)  # Map city to the corresponding planning area
                break  # Once a city is assigned to a planning area, no need to check further

    return planning_area_to_cities


def map_stations_to_planning_areas(closed_regions, substations_to_coordinates):
    planning_area_to_stations = {planning_area_gen_area_dict[idx + 1]: [] for idx in range(len(closed_regions))}

    for name, (x, y) in substations_to_coordinates.items():
        station_point = Point(x, y)

        for idx, polygon in enumerate(closed_regions):
            if polygon.contains(station_point):
                real_planning_area_id = planning_area_gen_area_dict[idx + 1]
                planning_area_to_stations[real_planning_area_id].append(name)
                break

    return planning_area_to_stations


def get_substations_in_city_areas(closed_regions, substations_to_coordinates):
    substation_in_cities = set()

    for name, (x, y) in substations_to_coordinates.items():
        station_point = Point(x, y)

        for idx, polygon in enumerate(closed_regions):
            if polygon.contains(station_point):
                substation_in_cities.add(name)
                break

    return list(substation_in_cities)
