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

import csv
import json
import logging
import os
import sys
from contextlib import ExitStack
from typing import Any, List

from Classes import GpsPoint, Water, ExtendedJSONEncoder, json_decoder_hook, Station, VesselType
from data.data_transformer import transform_stations, transform_vesseltypes


class Level:
    def __init__(self, identifier, longitude, latitude, water):
        self.identifier = int(identifier)
        self.position = GpsPoint(latitude=latitude, longitude=longitude)
        self.water = water
        if not isinstance(water, Water):
            self.water = Water(water)
        self.filepath = f"data/tidal_points/{self.water}/{self.identifier}.csv"
        self.current_level = None

    def __eq__(self, other):
        assert isinstance(other, Level)
        return self.identifier == other.identifier

    def __hash__(self):
        return self.identifier


def generate_intervals(water: Water, *, stations: List[Station] = None, vessels: List[VesselType] = None):
    if stations is None:
        stations = transform_stations()
    cutoff = 9.5721
    if water is Water.NORTH_SEA:
        stations = list(filter(lambda station: station.position.longitude < cutoff, stations))
    elif water is Water.BALTIC_SEA:
        stations = list(filter(lambda station: station.position.longitude >= cutoff, stations))

    if water is Water.ALL:
        with open("data/tidal_points/levels.json", "r") as f:
            f.readline()
            levels = [Level(l['number'], l['longitude'], l['latitude'], l['water']['shortname']) for l in
                      json.load(f) if (os.path.exists(
                    f"data/tidal_points/{Water.NORTH_SEA}/{l['number']}.csv") or os.path.exists(
                    f"data/tidal_points/{Water.BALTIC_SEA}/{l['number']}.csv"))]
    else:
        with open("data/tidal_points/levels.json", "r") as f:
            f.readline()
            levels = [Level(l['number'], l['longitude'], l['latitude'], l['water']['shortname']) for l in
                      json.load(f) if os.path.exists(f"data/tidal_points/{water}/{l['number']}.csv")]
    assert len(levels) > 0, "No levels found, please check if all data was downloaded"

    for station in stations:
        if station.position is None:
            stations.remove(station)
            continue
        # create ordered list of levels by distance to station
        levels_list = levels.copy()
        levels_list.sort(key=lambda level: level.position.distance_to(station.position))

        station.levels = levels_list

    if vessels is None:
        vessels = transform_vesseltypes()

    with ExitStack() as stack:
        files = {level: stack.enter_context(open(level.filepath, "r")) for level in levels}
        csv_readers = {level: csv.reader(files[level]) for level in levels}
        for level in levels:
            next(csv_readers[level])  # skip header

        go_on = True
        shares: dict[tuple, int] = dict()
        lines = 0
        while go_on:
            lines += 1
            sys.stdout.write(f"\r{lines} lines processed (Prognosed Progress: {lines / (31 * 24 * 60) * 100:.2f}%)")
            for level, line in ((level, next(csv_readers[level], None)) for level in levels):
                if line is None:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    logging.debug(f"Reached end of file for level {level.identifier}")
                    go_on = False
                    break
                if line[1] != "":
                    level.current_level = float(line[1])
                else:
                    level.current_level = None
            else:
                # all lines were read
                opened_station_vessels = list()
                for station in stations:
                    used_levels = set()
                    for level in station.levels:
                        if level.current_level is not None:
                            used_levels.add(level)
                            if len(used_levels) == 3:
                                break
                    if len(used_levels) < 1:
                        continue
                    dist = {level: station.position.distance_to(level.position) for level in used_levels}
                    total_distance = sum(dist.values())
                    water_level = sum(level.current_level * (total_distance - dist[level]) for level in used_levels)
                    water_level /= sum(total_distance - dist[level] for level in used_levels)
                    water_level -= station.depth * 100
                    opened = (station.identifier, tuple(sorted(vessel.identifier for vessel in vessels
                                                               if vessel.draught * 100 < water_level)))
                    if opened[1]:
                        opened_station_vessels.append(opened)
                if not opened_station_vessels:
                    logging.debug("No stations opened at line " + str(lines) + ", skipping")
                    raise Exception("No stations opened at line " + str(lines) + ", skipping")
                opened_station_vessels = tuple(sorted(opened_station_vessels, key=lambda x: x[0]))
                shares[opened_station_vessels] = shares.get(opened_station_vessels, 0) + 1
        logging.debug(f"Found {len(shares)} different shares")
        total = sum(shares.values())
        shares: dict[tuple, float] = {key: value / total for key, value in shares.items()}
        shares_dumpable = [{'key': key, 'value': value} for key, value in shares.items()]

    logging.debug("Now writing to file to increase speed for next time...")
    dir = "data/merged/" + str(water)
    dir = os.path.normpath(dir)
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "shares.json"), "w") as f:
        json.dump(shares_dumpable, f, indent=4, cls=ExtendedJSONEncoder)
    with open(os.path.join(dir, "stations.json"), "w") as f:
        json.dump(stations, f, indent=4, cls=ExtendedJSONEncoder)
    with open(os.path.join(dir, "vessels.json"), "w") as f:
        json.dump(vessels, f, indent=4, cls=ExtendedJSONEncoder)
    logging.debug("Done writing to file")
    return shares, stations, vessels


def get_connected_data(water: Water, *, stations: List[Station] = None, vessels: List[VesselType] = None) \
        -> tuple[dict[frozenset[tuple[Station, frozenset[VesselType]]], float], list[Station], list[VesselType]]:
    logging.debug("Entering get_connected_data")
    dir = "data/merged/" + str(water)
    dir = os.path.normpath(dir)

    if not (os.path.exists(os.path.join(dir, "shares.json")) and os.path.exists(
            os.path.join(dir, "stations.json")) and os.path.exists(
        os.path.join(dir, "vessels.json"))):
        logging.debug("No file found, generating data...")
        shares, stations, vessels = generate_intervals(water, stations=stations, vessels=vessels)
    else:
        logging.debug("Reading from file to increase speed...")
        with open(os.path.join(dir, "shares.json"), "r") as f:
            shares_dumpable = json.load(f, object_hook=json_decoder_hook)
        if stations is None:
            with open(os.path.join(dir, "stations.json"), "r") as f:
                stations = json.load(f, object_hook=json_decoder_hook)
        if vessels is None:
            with open(os.path.join(dir, "vessels.json"), "r") as f:
                vessels = json.load(f, object_hook=json_decoder_hook)
        shares = {share['key']: share['value'] for share in shares_dumpable}
        logging.debug("Done reading connected data from file")

    def replace_ids_with_objects(tup, v):
        result = set()
        for t in tup:
            station_id = t[0]
            vessel_ids = t[1]
            station = next((station for station in stations if station.identifier == station_id), None)
            assert station is not None, "Station not found"
            x = frozenset(vessel for vessel in v if vessel.identifier in vessel_ids)
            result.add((station, x))
        return frozenset(result)

    shares = {replace_ids_with_objects(key, vessels): value for key, value in shares.items()}
    logging.debug("Exiting get_connected_data")
    logging.debug(f"Found {len(shares)} different shares")

    return shares, stations, vessels


def consolidate_intervals(water: Water, del_shares: bool = True) -> tuple[dict[
    tuple[Station, VesselType], int | float | Any], list[Station], list[VesselType] | frozenset[VesselType]] | tuple[
                                                                        dict[frozenset[tuple[
                                                                            Station, frozenset[VesselType]]], float],
                                                                        dict[tuple[
                                                                            Station, VesselType], int | float | Any],
                                                                        list[Station], list[VesselType] | frozenset[
                                                                            VesselType]]:
    shares, stations, vessels = get_connected_data(water)
    logging.debug("Now consolidating intervals...")
    consolidated = dict()
    consolidated_by_station = dict()
    for key, value in shares.items():
        for station, vessels in key:
            for vessel in vessels:
                consolidated[(station, vessel)] = consolidated.get((station, vessel), 0) + value
                consolidated_by_station[station] = consolidated_by_station.get(station, 0) + value * vessel.amount

    number_of_vessels = sum(vessel.amount for vessel in vessels)
    consolidated_by_station = {key: max(min(1 - value / number_of_vessels, 1), 0) for key, value in
                               consolidated_by_station.items()}
    for station in stations:
        station.minimum_water_level = consolidated_by_station.get(station, 1)

    # eliminate rounding errors that cause values > 1 to prevent errors in the model
    consolidated = {key: 1 - min(value, 1) for key, value in consolidated.items()}
    logging.debug("Done consolidating intervals")
    if del_shares:
        del shares
        del consolidated_by_station
        import gc
        gc.collect()
        return consolidated, stations, vessels
    else:
        return shares, consolidated, stations, vessels
