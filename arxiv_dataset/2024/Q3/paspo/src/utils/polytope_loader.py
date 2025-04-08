
def load_polytope(n_dim: int, storage_method: str, generation_method: str, polytope_generation_data: dict):
    if storage_method == "seed":
        seed = polytope_generation_data["seed"]
                
        if generation_method == "points":
            from envs.random_polytope_generator import generate_random_polytope
            vertices, A, b = generate_random_polytope(n_dim=n_dim, n_points=polytope_generation_data["n_points"], seed=seed)
            return A, b 
        elif generation_method == "rejection_sampling":            
            A, b = generate_random_polytope_rejection_sampling(n_dim=n_dim, number_of_constraints=polytope_generation_data["number_of_constraints"], seed=seed, bounds=tuple(polytope_generation_data["bounds"]))
            
            return A, b
        elif generation_method == "unconstrained":
            import numpy as np
            
            A = np.zeros((1, n_dim), dtype=np.float32)
            b = np.ones(1, dtype=np.float32)
            
            return A, b
        elif generation_method == "rejection_sampling_fin_env":
       
            A, b = generate_random_fin_env_polytope_rejection_sampling(n_dim=n_dim, number_of_constraints=polytope_generation_data["number_of_constraints"], seed=seed)
            
            return A, b
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")
    elif storage_method == "dictionary":
        raise NotImplementedError()
    elif storage_method == "file":
        import numpy as np
        
        path = polytope_generation_data["path"]
        data = np.load(path)
        return data["A"], data["b"]
    else:
        raise ValueError(f"Unknown storage method: {storage_method}")
        
def save_polytope(A, b, path):
    import numpy as np
    np.savez(path, A=A, b=b)
    
    
def generate_random_polytope_rejection_sampling(n_dim: int, number_of_constraints: int, seed: int, max_digits: int = 2, binary_constraints: bool = False, bounds=(-10.0, 10.0), min_number_of_affected_resources: int=2, max_tries: int = 100000):
    """ generates a random polytope using rejection sampling

    Args:
        n_dim (int): how many dimensions the polytope should have
        number_of_constraints (int): how many inequality constraints the polytope should have
        seed (int): seed
        max_digits (int, optional): how many digits the constraints should have. Defaults to 2.
        binary_constraints (bool, optional): Wheter to use 1 and 0 only in each constraint on the left side (A). Defaults to False.
        bounds (tuple, optional): Bounds for the constraints if they are not binary. Defaults to (-10, 10).
        min_number_of_affected_resources (int, optional): Min number of affected resources per constraint. Defaults to 2.
        max_tries (int, optional): Max tries to sample a polytope. Defaults to 100000.
    """
    import numpy as np
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed=seed)
    
    #init simplex constraints for lp solver
    A_eq = np.ones((1, n_dim), dtype=np.float32)
    b_eq = 1
    c = np.ones(n_dim)

    high = n_dim-1
    
    if n_dim == 3:
        high = 3
    
    for current_it in range(max_tries):
        number_of_affected_resource_per_constraint = rng.integers(low=min_number_of_affected_resources, high=high, size=number_of_constraints)
        
        A = np.zeros((number_of_constraints, n_dim), dtype=np.float32)
        
        for constraint in range(number_of_constraints):
            affected_resources = rng.choice(np.arange(0, n_dim), size=number_of_affected_resource_per_constraint[constraint], replace=False)
            
            if binary_constraints:
                #generate binary vector for the constraint randomly
                A[constraint, affected_resources] = 1
            else:
                #generate random values for the constraint randomly
                A[constraint, affected_resources] = rng.uniform(low=bounds[0], high=bounds[1], size=number_of_affected_resource_per_constraint[constraint]).astype(np.float32).round(max_digits)
        
        b = rng.uniform(low=0, high=1, size=number_of_constraints).astype(np.float32).round(max_digits)
        
        
        
        #check if the polytope has a feasible solution
        res = linprog(c , A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
        
        if res.success:
            print(f"Success! Found polytope after {current_it} iterations")
            print(A)
            print(b)
            print("********")
            return A, b
        
    raise Exception(f"Could not find a feasible polytope after {max_tries} iterations")

def generate_random_fin_env_polytope_rejection_sampling(n_dim, number_of_constraints, seed, max_tries: int = 100000):
    import numpy as np
    from scipy.optimize import linprog

    # init simplex constraints for lp solver
    A_eq = np.ones((1, n_dim), dtype=np.float32)
    b_eq = 1
    c = np.ones(n_dim)

    rounding_digits = 2
    list_asset_indices = list(range(0, n_dim))
    rng = np.random.default_rng(seed=seed)
    
    def sample_from_ranges(rng, list_of_lists):
        sampled_values = []
        for inner_list in list_of_lists:
            min_val = min(inner_list[1:])  # we are excluding the cash kpis that are set to zero
            max_val = max(inner_list[1:])
            sampled_value = rng.uniform(low=min_val, high=max_val)
            sampled_values.append(sampled_value)
        return sampled_values


    def convert_index_list_to_binary_list(original_list, index_list):
        return [1 if idx in index_list else 0 for idx, entry in enumerate(original_list)]

    from financial_markets_gym.envs.financial_markets_env import FinancialMarketsEnv
    dict_kpi = FinancialMarketsEnv.get_financial_kpi_data(num_dim=n_dim)
    # print(dict_kpi)
    # rejection_sampling_5_fin_env

    for current_it in range(max_tries):
        list_constraints_A = []
        list_constraints_b = []

        number_kpi_constraints = rng.integers(low=1, high=min(number_of_constraints, len(dict_kpi)))
        number_allocation_constraints = number_of_constraints - number_kpi_constraints
        
        for _ in range(number_allocation_constraints):

            tmp_number_impacted_resources = rng.integers(low=2, high=(
                        n_dim - 1))  # impact at least two assets, and max n_dim -1
            tmp_list_set = [1 if idx in list(
                rng.choice(list_asset_indices, size=tmp_number_impacted_resources, replace=False)) else 0 for idx, entry
                            in enumerate(list_asset_indices)]
            list_constraints_A.append(tmp_list_set)
            constraint_value = round(rng.uniform(low=0.0, high=1.0), rounding_digits)
            list_constraints_b.append(constraint_value)

        list_kpi = [dict_kpi[key] for key in list(dict_kpi.keys())]
        list_kpi_constraints_A = [list(entry) for entry in
                                      list(rng.choice(list_kpi, size=number_kpi_constraints, replace=False))]

        list_kpi_constraints_b = sample_from_ranges(rng, list_kpi_constraints_A)

        list_constraints_A.extend(list_kpi_constraints_A)
        list_constraints_b.extend(list_kpi_constraints_b)

        A = np.array(list_constraints_A)
        b = np.array(list_constraints_b)

            
        # check if the polytope has a feasible solution
        res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")

        if res.success:
            print(f"Success! Found polytope after {current_it} iterations")
            return A, b

if __name__ == "__main__":
    generate_random_polytope_rejection_sampling(n_dim=10, number_of_constraints=40, seed=1)
