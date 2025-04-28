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

import logging
import sys
from itertools import product

from Classes import VesselType, Station, Zone, Incident_Type, Water
from tidal_gen.generate_intervals import get_connected_data


def calculate_obj_value(vessels: list[VesselType], stations: list[Station], zones: list[Zone],
                        incidents: list[Incident_Type], assignment: dict[Station, VesselType], water: Water,
                        shares: dict[frozenset[tuple[Station, frozenset[VesselType]]], float] = None,
                        fullinfo: bool = False, printout: bool = True):
    val = 0
    if shares is None:
        shares, _, _ = get_connected_data(water, stations=stations, vessels=vessels)
    unfullfilled = 0
    unfilled_number = 0
    unfullfilled_weight = 0
    for (share_index, (share_obj, share)) in enumerate(shares.items()):
        for zone, incident in product(zones, incidents):
            share_obj: frozenset[tuple[Station, frozenset[VesselType]]]
            if incident.probability_by_zone.get(zone, 0) <= 0:
                continue
            curr_val = float('inf')
            for cur_station, cur_vessels in share_obj:
                cur_vessel = assignment.get(cur_station, None)
                if cur_vessel is None:
                    continue
                if cur_vessel not in incident.allowed_vessels:
                    continue
                if (cur_station, cur_vessel) not in zone.reachable_from_by:
                    continue
                dist = cur_station.position.distance_to(zone.position)
                curr_val = min(curr_val, dist / cur_vessel.speed)
            if curr_val == float('inf'):
                unfullfilled += share
                unfilled_number += 1
                unfullfilled_weight += incident.weight * incident.probability_by_zone.get(zone, 0)
            else:
                val += curr_val * share * incident.weight * incident.probability_by_zone.get(zone, 0)
        if printout:
            sys.stdout.write("\rCompare Progress {0:.2f}%, so far unfullfilled {1}"
                             .format(100.0 * share_index / len(shares), unfilled_number))
            sys.stdout.flush()
    if printout:
        print()
    logging.info(f"Unfullfilled: {unfullfilled} ({unfilled_number} incidents) with weight {unfullfilled_weight}")
    if fullinfo:
        return val, unfullfilled, unfilled_number, unfullfilled_weight
    else:
        if unfullfilled > 0:
            return float('inf')
        return val
