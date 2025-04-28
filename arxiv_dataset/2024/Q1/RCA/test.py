"""
Copyright (C) 2024 Tom Mucke, Finn Seesemann
Ideas and Theory by Felix Engelhardt, Tom Mucke, Alexander Renneke, Finn Seesemann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import os
import sys
from random import seed

from gurobipy import GRB

from Classes import Water, SolveType, ExtendedJSONEncoder
from Solvers.run_gurobi import optimize
from data.DataHouse import DataHouse

import time
import logging

from tidal_gen.calculate_comp_val import calculate_obj_value


def test(seed_val: int, solve_type: SolveType, number_of_zones: int, water: Water, result_file,
         reduce_to_zones: int = None):
    if reduce_to_zones is None:
        reduce_to_zones = number_of_zones
    logging.info(
        f"Performing test with seed {seed_val} and {number_of_zones} zones and {solve_type.name} solver for {water.name}")
    seed(seed_val)
    dh = DataHouse(water)
    dh.create_all(max_zones=number_of_zones, seed=seed_val, consolidate=solve_type is not SolveType.GUROBI_BEST_TIDAL)

    logging.info("Starting solver")
    time_start = time.perf_counter()
    dh.cluster_zones(reduce_to_zones)
    logging.info("Reduced zones")
    match solve_type:
        case SolveType.GUROBI_MANY_ZONES:
            x = optimize(solve_type, dh.vessel_types, dh.stations, dh.zones_clustered, dh.incident_types, dh.shares)
        case SolveType.GUROBI_BETTER_TIDAL:
            x = optimize(solve_type, dh.vessel_types, dh.stations, dh.zones_clustered, dh.incident_types, dh.shares)
        case SolveType.GUROBI_BEST_TIDAL:
            x = optimize(solve_type, dh.vessel_types, dh.stations, dh.zones_clustered, dh.incident_types,
                         dh.unconsolidated_shares)
        case _:
            raise NotImplementedError("Unknown SolveType")
    time_end = time.perf_counter()
    runtime = time_end - time_start
    logging.info(f"Time: {runtime}")

    match solve_type:
        case SolveType.GUROBI_MANY_ZONES | SolveType.GUROBI_BETTER_TIDAL | SolveType.GUROBI_BEST_TIDAL:
            if x.status == GRB.OPTIMAL:
                result = x.ObjVal
            else:
                result = -1
        case _:
            raise NotImplementedError("Unknown SolveType")

    z = {"solver_type": solve_type, "seed": seed_val, "water": water, "number_of_zones": number_of_zones,
         "runtime": runtime, "solvetime": x.Runtime, "result": result}
    # create directory if not exists
    result_file = os.path.normpath(result_file)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    logging.info("Getting Assignment")
    if solve_type.name.lower().startswith("gurobi"):
        if x.status == GRB.OPTIMAL or x.status == GRB.TIME_LIMIT or x.Status == GRB.OPTIMAL or x.Status == GRB.TIME_LIMIT:
            stations_by_id = {station.identifier: station for station in dh.stations}
            vessels_by_id = {vessel.identifier: vessel for vessel in dh.vessel_types}
            assignment = {}
            for var in x.getVars():
                if var.VarName.startswith("x") and var.X > 0.5:
                    _, vessel_id, station_id = var.VarName.split("_")
                    assignment[stations_by_id[int(station_id)]] = vessels_by_id[int(vessel_id)]
        else:
            assignment = {}
    else:
        assignment = {}
        for station, vessel in x.get("x", []):
            assignment[station] = vessel
    logging.info("Calculating compare value")

    compare_val = calculate_obj_value(dh.vessel_types, dh.stations, dh.zones, dh.incident_types, assignment, water)
    z["compare_val"] = compare_val
    logging.info("Done calculating compare value, writing to json file")

    with open(result_file + '.json', "a") as f:
        json.dump(z, f, cls=ExtendedJSONEncoder)
        f.write(",\n")
    logging.info("Done writing to json file")

    logging.info("Writing gurobi solution to file")
    x.write(result_file + '_gurobi' + ".json")
    logging.info("Done writing gurobi solution to file")

    return runtime, x


if __name__ == '__main__':
    if __debug__:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    params = sys.argv
    if len(params) == 1:
        used_seed = int(input("Seed: "))
        number_of_zones = int(input("Number of zones: "))
        combined_number_of_zones = int(input("Combined number of zones: "))
        water = input(f"Water ({', '.join([e.name for e in Water])}): ")
        result_file = 'test/result/' + input("Result file: ")
        solve_type = input(f"Solve type ({', '.join([e.name for e in SolveType])}): ")
    else:
        used_seed = int(params[1])
        number_of_zones = int(params[2])
        combined_number_of_zones = int(params[3])
        water = params[4]
        result_file = params[5]
        solve_type = params[6]

    test(used_seed, SolveType[solve_type], number_of_zones, Water[water], result_file, combined_number_of_zones)
