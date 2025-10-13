def search_power_lines_sync(station_pair, direct_sync_dict, lines):
    # Helpers for an outdated policy: power line with the same name should have the same direction

    sync_edges = []

    for line in lines:
        if line in direct_sync_dict:
            station_x, station_y = station_pair  # The power flow from station_x to station_y
            station_x_idx = direct_sync_dict[line].index(station_x)
            station_y_idx = direct_sync_dict[line].index(station_y)
            if station_x_idx > station_y_idx:
                lst = direct_sync_dict[line]
                rev_list = list(reversed(lst))
                sync_edges = list(zip(rev_list, rev_list[1:]))
            elif station_x_idx < station_y_idx:
                sync_edges = list(zip(direct_sync_dict[line], direct_sync_dict[line][1:]))

    # if sync_edges:
    #     print(sync_edges)

    return sync_edges
