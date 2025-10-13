#include "acados_kinematic_nonconvex_mpc_solver/acados_kinematic_nonconvex_mpc_solver.h"
#include "gtest/gtest.h"
namespace mpc_solvers
{
namespace kinematic_solvers
{
// Construct the acados solver and run all the functions.
// Solves where the car is at (0,0) and going to (1,0)
TEST(solverTests, basicCoverage)
{
  // Construction

  int horizon_length = 20;
  int num_segments = 3;
  int num_obstacles = 1;
  int num_rti_iterations = 5;
  AcadosNonconvexSolver solver(horizon_length, num_segments, num_obstacles, num_rti_iterations);
  // Setting params

  AcadosNonconvexSolver::tracking_costs costs = { .1, .1, .1, .1 };
  // auto car_dims = std::make_shared<AcadosNonconvexSolver::CarDimensions>();
  // car_dims->length = .7;
  // car_dims->width = .5;
  // car_dims->buffer_to_obstacle = .02;
  auto model_params = std::make_shared<crs_models::kinematic_model::kinematic_params>();
  model_params->lr = .038;
  model_params->lf = .052;
  model_params->a = 6;
  model_params->b = .01;
  model_params->tau = .6;
  model_params->car_length = .7;
  model_params->car_width = .5;
  model_params->min_dist_to_obstacle = .02;
  model_params->additional_buffer = .01;

  std::unique_ptr<crs_controls::Obstacle> obstacle = std::make_unique<crs_controls::Rhombus>(-3, -3, .2);
  auto obstacles = std::make_shared<crs_controls::ObstacleVector>();
  obstacles->push_back(std::move(obstacle));
  AcadosNonconvexSolver::kbm_params kbm_params = { costs, model_params, obstacles };
  AcadosNonconvexSolver::seg_params seg_params = { 0, 0, model_params, obstacles };
  solver.setParams(kbm_params, seg_params);
  solver.updateTarget(1, 0);

  // Setting the initial state
  AcadosNonconvexSolver::kbm_state kbm_state;
  kbm_state.x = 0;
  kbm_state.y = 0;
  kbm_state.yaw = 0;
  kbm_state.velocity = 0;
  kbm_state.torque = 0;
  kbm_state.steer = 0;

  // Setting the guesses
  AcadosNonconvexSolver::obstacle_plane plane;
  plane.xi[0] = 1;
  plane.xi[1] = 1;
  plane.mu = 1;

  AcadosNonconvexSolver::kbm_input kbm_input;
  kbm_input.dSteer = 0;
  kbm_input.dTorque = 0;
  kbm_input.planes = { plane };

  AcadosNonconvexSolver::seg_state seg_state;
  seg_state.x = 0;
  seg_state.y = 0;

  AcadosNonconvexSolver::seg_input seg_input;
  seg_input.dx = 0;
  seg_input.dy = 0;
  seg_input.planes = { plane };

  AcadosNonconvexSolver::solver_solution guesses(horizon_length, num_segments, num_obstacles);
  for (auto& kbm_state_guess : guesses.kbm_states)
  {
    kbm_state_guess = kbm_state;
  }
  guesses.ref = kbm_state;
  for (auto& kbm_input_guess : guesses.kbm_inputs)
  {
    kbm_input_guess = kbm_input;
  }
  for (auto& seg_state_guess : guesses.seg_states)
  {
    seg_state_guess = seg_state;
  }
  for (auto& seg_input_guess : guesses.seg_inputs)
  {
    seg_input_guess = seg_input;
  }
  guesses.seg_inputs[2].dx = 1;
  guesses.seg_states[3].x = 1;

  // solver.setInitialState(kbm_state);
  solver.setGuesses(guesses);

  int result = solver.solveFirstTime(kbm_state);
  EXPECT_EQ(result, 0);
  AcadosNonconvexSolver::solver_solution solution(horizon_length, num_segments, num_obstacles);

  solver.getSolution(solution);
  double SOLVER_TOL = 1e-4;
  EXPECT_NEAR(solution.seg_states[3].x, 1, SOLVER_TOL);
  EXPECT_NEAR(solution.seg_states[3].y, 0, SOLVER_TOL);

  solver.resetSolver();

  solver.setGuesses(guesses);
  solver.solve(kbm_state);
}

}  // namespace kinematic_solvers
}  // namespace mpc_solvers

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
