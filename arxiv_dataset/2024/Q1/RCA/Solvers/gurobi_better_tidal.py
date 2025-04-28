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

from gurobipy import GRB, Model, quicksum, tupledict
from Classes import *
from itertools import product, pairwise


def create_model(model: Model, /, *, vessels: list[VesselType], stations: list[Station], zones: list[Zone],
                 incidents: list[Incident_Type],
                 shares: dict[tuple[Station, VesselType], float], **_):
    # Decision variables
    x = tupledict()
    for vessel in vessels:
        for station in vessel.allowed_ports:
            x[vessel, station] = model.addVar(vtype=GRB.BINARY, name=f"x_{vessel.identifier}_{station.identifier}")

    waterlevels = set(shares.values()).union({0, 1})
    waterlevels = list(pairwise(sorted(waterlevels)))
    opened_by_waterlevel = {waterlevel: set() for waterlevel in waterlevels}
    for key, value in shares.items():
        for waterlevel in waterlevels:
            if value <= waterlevel[0]:
                opened_by_waterlevel[waterlevel].add(key)

    y = tupledict()
    for station, (waterlevelindex, waterlevel), vessel, zone, incident \
            in product(stations, enumerate(waterlevels), vessels, zones, incidents):
        if vessel in incident.allowed_vessels and station in vessel.allowed_ports \
                and (station, vessel) in opened_by_waterlevel[waterlevel] \
                and incident.probability_by_zone.get(zone, 0) > 0 and \
                (station, vessel) in zone.reachable_from_by:
            y[station.identifier, waterlevelindex, vessel.identifier, zone.identifier, incident.identifier] = \
                model.addVar(vtype=GRB.BINARY,
                             name=f"y_{station.identifier}_{waterlevelindex}_{vessel.identifier}"
                                  f"_{zone.identifier}_{incident.identifier}",
                             obj=incident.probability_by_zone.get(zone, 0)
                                 * (waterlevel[1] - waterlevel[0])
                                 * zone.position.distance_to(station.position) / vessel.speed
                                 * incident.weight)

    model.update()

    # Constraints
    model.addConstrs((quicksum(x[vessel, station] for station in vessel.allowed_ports) <= vessel.amount
                      for vessel in vessels), name="vessel_assignment_1")

    model.addConstrs((quicksum(x[vessel, station] for vessel in station.allowed_vessels) <= 1
                      for station in stations), name="vessel_assignment_2")

    for station in stations:
        for vessel in station.allowed_vessels:
            y_select = list(y.select(station.identifier, '*', vessel.identifier, '*', '*'))
            model.addConstr(quicksum(var for var in y_select) <= len(y_select) * x[vessel, station],
                            name=f"vessel_assigned_if_used_{station.identifier}_{vessel.identifier}")

    model.addConstrs((quicksum(var for var in y.select('*', waterlevel, '*', zone.identifier, incident.identifier))
                      == 1 for waterlevel in range(len(waterlevels)) for zone in zones for incident in incidents if
                      incident.probability_by_zone.get(zone, 0) > 0),
                     name="incident_covered")

    model.update()
    return model
