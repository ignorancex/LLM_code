def get_solvers(config):
    if config["dataset"] == "cifar10":
        from solvers.solver_cifar10 import Solver

        solver = Solver(config)
    elif config["dataset"] == "fashion":
        from solvers.solver_fashion import Solver

        solver = Solver(config)
    elif config["dataset"] == "coil20":
        from solvers.solver_coil20 import Solver

        solver = Solver(config)
    else:
        print("Unknown solver")
    return solver
