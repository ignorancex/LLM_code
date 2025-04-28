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

from gurobipy import Env, GRB, Model

from Classes import VesselType, Station, Zone, Incident_Type, SolveType


def optimize(solver: SolveType, /, vessels: list[VesselType], stations: list[Station], zones: list[Zone],
             incidents: list[Incident_Type],
             shares: dict[tuple[Station, VesselType], float] | dict[
                 frozenset[tuple[Station, frozenset[VesselType]]], float]):
    params = {
        "OutputFlag": 0,
        "LogToConsole": 0,
        "Threads": 1,
    }
    with Env(params=params) as env, Model(solver.name.lower(), env=env):
        model = Model(solver.name.lower(), env=env)
        model.modelSense = GRB.MINIMIZE

        # Limit threads
        model.setParam('Threads', 1)
        model.params.TimeLimit = 6 * 60 * 60
        match solver:
            case SolveType.GUROBI_MANY_ZONES:
                from Solvers.gurobi_more_zones import create_model
            case SolveType.GUROBI_BETTER_TIDAL:
                from Solvers.gurobi_better_tidal import create_model
            case SolveType.GUROBI_BEST_TIDAL:
                from Solvers.gurobi_best_tidal import create_model
            case _:
                raise NotImplementedError("Unknown solver / not a gurobi solver")
        model = create_model(model, vessels=vessels, stations=stations, zones=zones, incidents=incidents, shares=shares)
        model.optimize()
        return model
