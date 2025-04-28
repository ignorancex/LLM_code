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


def create_model(model: Model, /, *,
                 vessels: list[VesselType], stations: list[Station], zones: list[Zone], incidents: list[Incident_Type],
                 shares: dict[frozenset[tuple[Station, frozenset[VesselType]]], float], **_):
    # Decision variables
    x = tupledict()
    for vessel in vessels:
        for station in vessel.allowed_ports:
            x[vessel, station] = model.addVar(vtype=GRB.BINARY, name=f"x_{vessel.identifier}_{station.identifier}")

    y = tupledict()
    for share_index, usabel_combs in enumerate(shares.keys()):
        share = shares[usabel_combs]
        for station, usable_vessels in usabel_combs:
            for incident in incidents:
                for vessel in filter(lambda vessel: station in vessel.allowed_ports and
                                                    vessel in incident.allowed_vessels, usable_vessels):
                    for zone in filter(lambda z: incident.probability_by_zone.get(z, 0) > 0
                                                 and (station, vessel) in z.reachable_from_by, zones):
                        y[station.identifier, share_index, vessel.identifier, zone.identifier, incident.identifier] = \
                            model.addVar(vtype=GRB.BINARY,
                                         name=f"y_{station.identifier}_{share_index}_{vessel.identifier}"
                                              f"_{zone.identifier}_{incident.identifier}",
                                         obj=incident.probability_by_zone.get(zone, 0)
                                             * zone.position.distance_to(station.position) / vessel.speed
                                             * incident.weight * share)

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
                      == 1 for waterlevel in range(len(shares.keys())) for zone in zones for incident in incidents if
                      incident.probability_by_zone.get(zone, 0) > 0),
                     name="incident_covered")

    model.update()
    return model
