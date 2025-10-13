def estimate_substation_demand_average_within_planning_area(planning_area_to_stations, planning_area_demand):
    substation_demand_dict = {}

    for planning_area, stations in planning_area_to_stations.items():
        if planning_area in planning_area_demand:
            total_demand = planning_area_demand[planning_area]  # Total Demand for the planning area
            num_stations = len(stations)  # The number of stations in the planning area

            if num_stations > 0:
                station_demand = total_demand / num_stations  # Average demand per station

                for station in stations:
                    substation_demand_dict[station] = station_demand
            else:
                print(f"Warning: Planning Area {planning_area} has demand but no stations!")

    return substation_demand_dict


def estimate_substation_demand_average_within_planning_area_urban_check(planning_area_to_stations, planning_area_demand,
                                                                        substations_in_city_areas,
                                                                        city_population_ratio):
    substation_demand_dict = {}

    for planning_area, stations in planning_area_to_stations.items():
        if planning_area in planning_area_demand:
            total_demand = planning_area_demand[planning_area]

            substation_in_city = [station for station in stations if station in substations_in_city_areas]
            substation_not_in_city = [station for station in stations if station not in substations_in_city_areas]

            if len(substation_in_city) > 0 and len(substation_not_in_city) > 0:
                total_demand_in_city = total_demand * city_population_ratio
                total_demand_not_in_city = total_demand * (1 - city_population_ratio)

                for station in substation_in_city:
                    substation_demand_dict[station] = total_demand_in_city / len(substation_in_city)
                for station in substation_not_in_city:
                    substation_demand_dict[station] = total_demand_not_in_city / len(substation_not_in_city)
            elif len(substation_in_city) > 0 and len(substation_not_in_city) == 0:
                for station in substation_in_city:
                    substation_demand_dict[station] = total_demand / len(substation_in_city)
            elif len(substation_in_city) == 0 and len(substation_not_in_city) > 0:
                for station in substation_not_in_city:
                    substation_demand_dict[station] = total_demand / len(substation_not_in_city)
            else:
                print(f"Warning: Planning Area {planning_area} has demand but no stations!")

    return substation_demand_dict
