
#ifdef acados_kinematic_tracking_mpc_solver_FOUND
#include "mpc_controller/kinematic_controller/tracking_mpc_kinematic_controller.h"

#include "acados_kinematic_tracking_mpc_solver/acados_kinematic_tracking_mpc_solver.h"

namespace crs_controls
{
void KinematicTrackingMpcController::loadMpcSolver()
{
  if (config_.solver_type == "ACADOS")
  {
#ifdef acados_kinematic_tracking_mpc_solver_FOUND
    solver_ = std::make_shared<mpc_solvers::kinematic_tracking_solvers::AcadosKinematicTrackingMpcSolver>();
#else
    std::cout << "Requested ACADOS solver for kinematic MPC. Solver not found! Program will probably crash!"
              << std::endl;
    assert(true && "Requested ACADOS solver for kinematic MPC. Solver not found!");
#endif
  }
}

// Constructor, Creates a MPC controller based on the config and a track description
KinematicTrackingMpcController::KinematicTrackingMpcController(
    tracking_mpc_kinematic_config config, std::shared_ptr<crs_models::kinematic_model::DiscreteKinematicModel> model,
    std::shared_ptr<Trajectory> track)
  : MpcController(model, std::static_pointer_cast<Trajectory>(track))
{
  setConfig(config);
  loadMpcSolver();

  // Initialize last solutions with zeros
  last_solution.states_ = std::vector<double>(solver_->getStateDimension() * solver_->getHorizonLength(), 0.0);
  last_solution.inputs_ = std::vector<double>(solver_->getInputDimension() * solver_->getHorizonLength(), 0.0);

  // Initialize last references with zeros (only for visualization)
  last_solution.reference_ = std::vector<Eigen::Vector2d>(solver_->getHorizonLength(), Eigen::Vector2d::Zero());
}

std::optional<std::vector<std::vector<double>>> KinematicTrackingMpcController::getPlannedTrajectory() const
{
  std::vector<std::vector<double>> traj;
  for (int i = 0; i < solver_->getHorizonLength(); i++)
  {
    double x_pos = last_solution.states_[i * solver_->getStateDimension() + kinematic_tracking_vars::X];
    double y_pos = last_solution.states_[i * solver_->getStateDimension() + kinematic_tracking_vars::Y];
    double vel = last_solution.states_[i * solver_->getStateDimension() + kinematic_tracking_vars::V];
    // Append values to trajectory for visualization
    traj.push_back({
        x_pos,                                                                                   // Planned x
        y_pos,                                                                                   // Planned y
        vel,                                                                                     // Planned velocity
        last_solution.states_[i * solver_->getStateDimension() + kinematic_tracking_vars::YAW],  // Planned Yaw
    });
  }

  return traj;
}

void KinematicTrackingMpcController::initialize(crs_models::kinematic_model::kinematic_car_state state)
{
  auto model_dynamics = model_->getParams();
  int N_WARM_START = 10;

  mpc_solvers::kinematic_tracking_solvers::tracking_costs mpc_costs = { config_.Q1, config_.Q2, config_.R1,
                                                                        config_.R2 };

  // Initial inputs.
  last_input_.steer = 0;
  // Initial torque input.
  last_input_.torque = 0;

  double x_init[6];
  x_init[0] = state.pos_x;
  x_init[1] = state.pos_y;
  x_init[2] = state.yaw;
  x_init[3] = state.velocity;  // Should not be zero as otherwise the model gets NaN issues
  x_init[4] = last_input_.torque;
  x_init[5] = last_input_.steer;

  double u_init[2];
  u_init[0] = 0.0;
  u_init[1] = 0.0;

  solver_->setInitialState(x_init);

  // Setup parameter for initial solve
  for (int stage = 0; stage < solver_->getHorizonLength(); stage++)
  {
    // Set initial state
    for (int x_idx = 0; x_idx < solver_->getStateDimension(); x_idx++)
      last_solution.states_[stage * solver_->getStateDimension() + x_idx] = x_init[x_idx];

    // Set initial input (nothing to do, its zero)
    solver_->setStateInitialGuess(stage, x_init);
    solver_->setInputInitialGuess(stage, u_init);
  }

  // Run solver
  for (int run = 0; run < N_WARM_START; run++)
  {
    for (int stage = 0; stage < solver_->getHorizonLength(); stage++)
    {
      // Update tracking point
      mpc_solvers::kinematic_tracking_solvers::trajectory_track_point track_point;
      auto next_track_point = (*trajectory_)[counter_loop + stage];

      track_point.x = next_track_point(0);
      track_point.y = next_track_point(1);

      solver_->updateParams(stage,
                            model_dynamics,  // Model Dynamics
                            mpc_costs,       // Costs
                            track_point);    // Reference point

      // Use previous solution as initial guess this time
      solver_->setStateInitialGuess(stage, &last_solution.states_[stage * solver_->getStateDimension()]);
      solver_->setInputInitialGuess(stage, &last_solution.inputs_[stage * solver_->getInputDimension()]);
    }

    solver_->solve(&last_solution.states_[0], &last_solution.inputs_[0]);

    last_input_.torque = last_solution.states_[1 * solver_->getStateDimension() + kinematic_tracking_vars::TORQUE];
    last_input_.steer = last_solution.states_[1 * solver_->getStateDimension() + kinematic_tracking_vars::STEER];
  }
}

/**
 * @brief Executes MPC controller, returning a new input to apply
 *
 * @param state current measured state of the system
 * @param timestamp current timestamp, will be ignored
 * @return crs_models::kinematic_model::kinematic_car_input
 */
crs_models::kinematic_model::kinematic_car_input KinematicTrackingMpcController::getControlInput(
    crs_models::kinematic_model::kinematic_car_state state [[maybe_unused]], double timestamp [[maybe_unused]])
{
  return { .2, .2 };
  counter_loop++;

  mpc_solvers::kinematic_tracking_solvers::tracking_costs mpc_costs = { config_.Q1, config_.Q2, config_.R1,
                                                                        config_.R2 };

  if (!is_initialized)
  {
    initializing = true;
    initialize(state);
    initializing = false;
    is_initialized = true;
  }

  // initialize state to virtually advanced vehicle states and inputs
  // auto virtual_state = model_->applyModel(state, last_input_, config_.lag_compensation_time);

  double x0[] = {
    state.pos_x, state.pos_y, state.yaw, state.velocity, last_input_.torque, last_input_.steer,
  };

  solver_->setInitialState(x0);
  // Run solver
  for (int current_stage = 0; current_stage < solver_->getHorizonLength(); current_stage++)
  {
    // next stage points to current_stage + 1. If current_stage is at end of horizon, next stage directy points to
    // current stage i.e. current_stage = 2 -> next_stage = 3, current_stage = 29 -> next_stage = 29, assuming
    // horizon of 30
    int next_stage = current_stage + (current_stage != solver_->getHorizonLength() - 1);
    // Update tracking point based on predicted distance on track
    mpc_solvers::kinematic_tracking_solvers::trajectory_track_point track_point;
    auto next_track_point = (*trajectory_)[counter_loop + current_stage];

    track_point.x = next_track_point(0);
    track_point.y = next_track_point(1);

    // Update reference visualization
    last_solution.reference_[current_stage] = Eigen::Vector2d(track_point.x, track_point.y);

    solver_->updateParams(current_stage,
                          model_->getParams(),  // Model Dynamics
                          mpc_costs,            // Costs
                          track_point           // Tracking point
    );

    // Use previous solution as initial guess this time
    solver_->setStateInitialGuess(current_stage, &last_solution.states_[next_stage * solver_->getStateDimension()]);
    solver_->setInputInitialGuess(current_stage, &last_solution.inputs_[next_stage * solver_->getInputDimension()]);
  }
  solver_->solve(&last_solution.states_[0], &last_solution.inputs_[0]);

  last_input_.torque = last_solution.states_[1 * solver_->getStateDimension() + kinematic_tracking_vars::TORQUE];
  last_input_.steer = last_solution.states_[1 * solver_->getStateDimension() + kinematic_tracking_vars::STEER];

  return last_input_;
}

/**
 * @brief Returns the config
 *
 * @return mpc_config&
 */
tracking_mpc_kinematic_config& KinematicTrackingMpcController::getConfig()
{
  return config_;
}

void KinematicTrackingMpcController::setConfig(tracking_mpc_kinematic_config config)
{
  config_ = config;
}
}  // namespace crs_controls

#endif
