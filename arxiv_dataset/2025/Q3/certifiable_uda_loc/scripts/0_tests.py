# This testing setup is a bit of a hack. Pytest was complaning in different ways across different systems when testing installation. 
# Future TODO: Fix. 
import certifiable_uda_loc.tests.test_constraints as test_constraints
import certifiable_uda_loc.tests.test_costs as test_costs
import certifiable_uda_loc.tests.test_gauss_newton_baseline as test_gauss_newton_baseline
import certifiable_uda_loc.tests.test_lost_in_the_woods as test_lost_in_the_woods
import certifiable_uda_loc.tests.test_max_mixture as test_max_mixture
import certifiable_uda_loc.tests.test_vech as test_vech



def main():
    print("----Test constraints---")
    test_constraints.test_constraints()
    print("----Test costs---")
    test_costs.test_rotation_frobenius()
    test_costs.test_known_landmark_cost()
    test_costs.test_known_pose_cost()
    print("----Test GN baseline---")
    test_gauss_newton_baseline.test_jacobians()
    test_gauss_newton_baseline.test_deadreckoning()
    print("----Test Lost in the Woods---")
    test_lost_in_the_woods.test_optimization()
    test_lost_in_the_woods.test_noise_properties()
    print("----Test Comparison with Max Mixture---")
    test_max_mixture.test_max_mixture_comparison()
    test_vech.test_vech()
    print("----All tests passed.---")
if __name__ == "__main__":
    main()

