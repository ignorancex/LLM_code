import networkx as nx


def policy_helper_substation_voltage_compare(station_x, station_y, substations_to_voltage, lines_to_voltage, lines):
    if station_x not in substations_to_voltage or station_y not in substations_to_voltage:
        # if not line.startswith("T"): # Policy 0.9: Line name starts with "T" means that's internal line between two adjacent substations
        print(f"Missing values for station_x: {station_x}, station_y: {station_y}")
        return None

    # Policy 0.95: Substation can't flow into lines with higher voltage
    # If two substations are connected by a line with a higher voltage, then an invalid line is detected
    for line in lines:
        if line not in lines_to_voltage:
            print(f"Missing values for line: {line}")
            return None

        if lines_to_voltage[line] > substations_to_voltage[station_x] and lines_to_voltage[line] > \
                substations_to_voltage[station_y]:
            print(f"Policy: Line {line} has a higher voltage than both substations {station_x} and {station_y}")
            return None

        # if lines_to_voltage[line] < substations_to_voltage[station_x] and lines_to_voltage[line] < \
        #         substations_to_voltage[station_y]:
        #     print(f"Policy: Line {line} has a lower voltage than both substations {station_x} and {station_y}")
        #     return None

    # Policy 1: Power flow from a substation with higher voltage to a substation with lower voltage
    # Policy 1.1: Power flow direction between two substations does not change in different power lines
    if substations_to_voltage[station_x] > substations_to_voltage[station_y]:
        print(station_x, station_y, substations_to_voltage[station_x], substations_to_voltage[station_y])
        return station_x, station_y
    if substations_to_voltage[station_y] > substations_to_voltage[station_x]:
        print(station_y, station_x, substations_to_voltage[station_y], substations_to_voltage[station_x])
        return station_y, station_x

    return None


def policy_helper_voltage_allow_flow(station_x, station_y, substations_to_voltage, lines_to_voltage, lines):
    # If station_y have a higher voltage than station_x, then the flow is not allowed
    if substations_to_voltage[station_x] < substations_to_voltage[station_y]:
        return False

    # If for all power lines, station_x has a lower voltage than the power line, then the flow is not allowed
    for line in lines:
        if substations_to_voltage[station_x] >= lines_to_voltage[line] >= substations_to_voltage[station_y]:
            return True
    return False


def policy_helper_generator_source(station_x, station_y, substations_to_generators):
    # Policy 2: Power flow from a substation that is connected to a generator
    if station_x in substations_to_generators:
        return station_x, station_y
    if station_y in substations_to_generators:
        return station_y, station_x
    return None


def policy_helper_edge_station(station_x, station_y, stations_to_lines, substations_to_generators):
    # Policy 3: Power flow into a substation in the end of a power line without a generator
    if len(stations_to_lines[station_x]) == 1 and station_x not in substations_to_generators:
        return station_y, station_x
    if len(stations_to_lines[station_y]) == 1 and station_y not in substations_to_generators:
        return station_x, station_y
    if len(stations_to_lines[station_x]) == 1 and len(stations_to_lines[station_y]) == 1:
        print(f"Policy: Isolated substations pair {station_x} and {station_y}")
    return None


def policy_helper_power_line_same_direction(lines_to_station_pairs):
    direct_sync_dict = {}

    for line_id, station_pairs in lines_to_station_pairs.items():
        subG = nx.Graph()

        for (s1, s2) in station_pairs:
            subG.add_edge(s1, s2)

        if subG.number_of_edges() == 1:
            continue

        leaves = [n for (n, deg) in subG.degree() if deg == 1]

        if len(leaves) == 2:
            path_nodes = list(nx.dfs_preorder_nodes(subG, source=leaves[0]))
            if line_id not in direct_sync_dict:
                direct_sync_dict[line_id] = path_nodes
        else:
            print("A power line have an non-linear shape")

    return direct_sync_dict
