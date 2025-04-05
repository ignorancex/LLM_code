from gurobipy import *

from project_typing import *

EPS = 1e-6


def beck_fiala(pis: List[float], agents: List[Agent], direction: List[float]) -> Panel:
    assert len(pis) == len(agents) == len(direction)
    n = len(agents)
    assert n >= 1
    num_features = len(agents[0])
    k = round(sum(pis))

    bf_model = Model()
    bf_agent_vars = [bf_model.addVar(lb=0., ub=1.) for _ in agents]
    bf_model.addConstr(quicksum(bf_agent_vars) == k)
    bf_model.setObjective(quicksum(coeff * xi for coeff, xi in zip(direction, bf_agent_vars)), GRB.MAXIMIZE)

    constraint_num_active_variables: Dict[FeatureValue, int] = {}
    feature_value_sums: Dict[FeatureValue, float] = {}
    for pi, agent in zip(pis, agents):
        for feature_value in agent.items():
            if feature_value not in feature_value_sums:
                feature_value_sums[feature_value] = 0.
                constraint_num_active_variables[feature_value] = 0
            feature_value_sums[feature_value] += pi
            constraint_num_active_variables[feature_value] += 1

    feature_value_constraints: Dict[FeatureValue, Constr] = {}
    for (feature, value), fv_sum in feature_value_sums.items():
        constraint = quicksum(xi for agent, xi in zip(agents, bf_agent_vars) if agent[feature] == value) == fv_sum
        feature_value_constraints[(feature, value)] = bf_model.addConstr(constraint)

    determined_variables: Dict[int, bool] = {}

    while True:
        # Freeze additional variables
        bf_model.optimize()
        assert bf_model.status == GRB.OPTIMAL
        for i, agent in enumerate(agents):
            if i in determined_variables:
                continue
            lp_value = bf_agent_vars[i].X
            if lp_value < EPS:
                determined_variables[i] = False
                bf_model.addConstr(bf_agent_vars[i] == 0.)
                for feature_value in agent.items():
                    constraint_num_active_variables[feature_value] -= 1
            elif lp_value > 1 - EPS:
                determined_variables[i] = True
                bf_model.addConstr(bf_agent_vars[i] == 1.)
                for feature_value in agent.items():
                    constraint_num_active_variables[feature_value] -= 1

        if len(determined_variables) == n:
            panel = frozenset(i for i in range(n) if determined_variables[i])
            assert len(panel) == k
            return panel

        assert n - len(determined_variables) <= 1 + len(feature_value_constraints)
        assert len(feature_value_constraints) > 0

        constraints_to_delete = []
        for feature_value in feature_value_constraints:
            if constraint_num_active_variables[feature_value] == n - len(determined_variables):
                constraints_to_delete.append(feature_value)
            elif constraint_num_active_variables[feature_value] <= num_features:
                constraints_to_delete.append(feature_value)
        assert len(constraints_to_delete) > 0
        for feature_value in constraints_to_delete:
            bf_model.remove(feature_value_constraints[feature_value])
            del feature_value_constraints[feature_value]


def beck_fiala_column_gen(pis: List[float], agents: List[Agent]) -> List[Tuple[Panel, float]]:
    assert len(pis) == len(agents)

    setParam("OutputFlag", False)

    dual_model = Model()
    dual_agent_vars = [dual_model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in agents]
    dual_abs_vars = [dual_model.addVar(lb=0., ub=GRB.INFINITY) for _ in agents]
    dual_y_hat = dual_model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    dual_model.addConstr(quicksum(dual_abs_vars) == 1.)
    for yi, absi in zip(dual_agent_vars, dual_abs_vars):
        dual_model.addConstr(yi <= absi)
        dual_model.addConstr(-yi <= absi)
    dual_model.setObjective(quicksum(pi * yi for pi, yi in zip(pis, dual_agent_vars)) - dual_y_hat, GRB.MAXIMIZE)

    panels: Set[Panel] = set()

    direction = [1. for _ in agents]
    for _ in range(50):
        new_panel = beck_fiala(pis, agents, direction)
        panels.add(new_panel)
        dual_model.addConstr(dual_y_hat >= quicksum(dual_agent_vars[i] for i in new_panel))
        for i in new_panel:
            direction[i] *= .5
        sd = len(direction) / sum(direction)
        direction = [sd * dire for dire in direction]

    dual_model.optimize()
    assert dual_model.status == GRB.OPTIMAL
    while dual_model.objVal >= EPS:
        direction = [yi.x for yi in dual_agent_vars]
        new_panel = beck_fiala(pis, agents, direction)
        assert new_panel not in panels
        panels.add(new_panel)
        logging.info(f"{len(panels)} panels, gap: {dual_model.objVal} (target value: ≤{EPS})")
        dual_model.addConstr(dual_y_hat >= quicksum(dual_agent_vars[i] for i in new_panel))
        dual_model.optimize()
        assert dual_model.status == GRB.OPTIMAL

    panel_list = list(panels)
    primal_model = Model()
    primal_panel_vars = [primal_model.addVar(lb=0., ub=1.) for _ in panel_list]
    diff = primal_model.addVar(lb=0., ub=GRB.INFINITY)
    primal_model.addConstr(quicksum(primal_panel_vars) == 1.)
    for i, pi in enumerate(pis):
        agent_prob = quicksum(var for (panel, var) in zip(panel_list, primal_panel_vars) if i in panel)
        primal_model.addConstr(agent_prob >= pi - diff)
    primal_model.setObjective(diff, GRB.MINIMIZE)

    primal_model.optimize()
    assert primal_model.status == GRB.OPTIMAL

    probabilities = [var.x for var in primal_panel_vars]
    probabilities = [max(p, 0) for p in probabilities]
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]

    return list(zip(panel_list, probabilities))
