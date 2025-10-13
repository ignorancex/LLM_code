
#ifdef acados_kinematic_nonconvex_mpc_solver_FOUND
#include "mpc_controller/kinematic_controller/tracking_mpc_kinematic_nonconvex_controller.h"
#include <commons/geometry_utils.h>
#include "acados/utils/types.h"

#include "acados_kinematic_nonconvex_mpc_solver/acados_kinematic_nonconvex_mpc_solver.h"

namespace crs_controls
{
namespace kinematic_controllers
{
using crs_controls::Obstacle;
using crs_controls::ObstacleVector;
using crs_models::kinematic_model::DiscreteKinematicModel;
using geometry::Polygon;
using geometry::Vertex;
using mpc_solvers::kinematic_solvers::AcadosNonconvexSolver;
using mpc_solvers::kinematic_solvers::NonconvexSolver;

typedef NonconvexSolver::tracking_costs tracking_costs;
typedef NonconvexSolver::obstacle_plane obstacle_plane;
typedef NonconvexSolver::kbm_state kbm_state;
typedef NonconvexSolver::kbm_input kbm_input;
typedef NonconvexSolver::kbm_params kbm_params;
typedef NonconvexSolver::seg_state seg_state;
typedef NonconvexSolver::seg_input seg_input;
typedef NonconvexSolver::seg_params seg_params;
typedef NonconvexSolver::solver_solution solver_solution;

struct NonconvexController::Private
{
  std::unique_ptr<NonconvexSolver> solver_;

  // The MPC config (cost values)
  tracking_mpc_kinematic_nonconvex_config config_;

  bool is_initialized = false;

  std::shared_ptr<const ObstacleVector> obstacle_positions_;

  Eigen::Vector2d last_target_ = { std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };

  // Reference to last mpc solution
  solver_solution last_solution_;
  bool last_solve_succeeded_ = false;

  std::shared_ptr<DiscreteKinematicModel> model_;
  std::shared_ptr<DynamicPointTrajectory> trajectory_;
  double plan_id_;      // id of the plan we've received
  double tail_length_;  // the length of the remaining segments in the shortest path

  // Constructor, Creates a MPC controller based on the config and a track description
  Private(tracking_mpc_kinematic_nonconvex_config config, std::shared_ptr<DiscreteKinematicModel> model,
          std::shared_ptr<DynamicPointTrajectory> trajectory)
    : solver_(std::make_unique<AcadosNonconvexSolver>(config.solver_config))
    , config_(config)
    , last_solution_(config.solver_config)
    , model_(model)
    , trajectory_(trajectory)
  {
  }

  std::optional<std::vector<std::vector<double>>> getPlannedTrajectory() const
  {
    // Let N be the length of the KBM trajectory
    // The first N+1 elements will be of the form {x_pos, y_pos, vel, yaw}.
    // The final 3 correspond to the segments, and are just {x_pos, y_pos}

    if (!last_solve_succeeded_)
    {
      return std::nullopt;
    }

    std::vector<std::vector<double>> traj;

    for (const kbm_state& state : last_solution_.kbm_states)
    {
      traj.push_back({ state.x, state.y, state.velocity, state.yaw });
    }

    for (const seg_state& state : last_solution_.seg_states)
    {
      traj.push_back({ state.x, state.y, 0, 0 });
    }

    traj.push_back({ plan_id_, NAN, NAN, NAN });

    return traj;
  }

  static Eigen::Vector2d interpolate(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, double dist)
  {
    double line_dist = std::sqrt((p1 - p2).squaredNorm());
    double frac = dist / line_dist;
    if (frac > 1 + 1e-5)
    {
      std::cerr << "WARNING: provided segment initialization value is infeasible!" << std::endl;
    }
    frac = std::min(1.0, frac);
    Eigen::Vector2d ret;
    ret << p1[0] + (p2[0] - p1[0]) * frac, p1[1] + (p2[1] - p1[1]) * frac;
    return ret;
  }

  static void initialize_obstacle_variable(const Obstacle& obstacle, const kbm_state& car,
                                           const crs_models::kinematic_model::DiscreteKinematicModel& model,
                                           obstacle_plane& obst_var)
  {
    const crs_models::kinematic_model::kinematic_params params = model.getParams();
    double width = params.car_width;
    double length = params.car_length;

    // Generate the matrix corresponding to the vehicle
    Polygon<4> vehicle = geometry::create<4>({ { { -length / 2, width / 2 },
                                                 { length / 2, width / 2 },
                                                 { length / 2, -width / 2 },
                                                 { -length / 2, -width / 2 } } });
    vehicle = geometry::transform(vehicle, car.yaw, car.x, car.y);

    Polygon<4> obstacle_matrix = obstacle.getVertexMatrix();

    Eigen::Vector2d p_vehicle;
    Eigen::Vector2d p_obstacle;
    geometry::closest_points<4, 4>(vehicle, obstacle_matrix, p_vehicle, p_obstacle);

    p_vehicle = interpolate(p_obstacle, p_vehicle, params.min_dist_to_obstacle);

    Eigen::Vector2d s = p_obstacle - p_vehicle;
    s *= 2.;

    obst_var.xi[0] = -s[0];
    obst_var.xi[1] = -s[1];
    obst_var.mu = -s.dot(p_obstacle);
  }

  void initialize_segment_plane(const Obstacle& obstacle, const Vertex& p1, const Vertex& p2,
                                const DiscreteKinematicModel& model, obstacle_plane& obst_var)
  {
    if (config_.solver_config.euclidean_offset_cost)
    {
      // We do not care about segment obstacle constraints
      obst_var.xi[0] = 0;
      obst_var.xi[1] = 0;
      obst_var.mu = 0;
      return;
    }

    Polygon<4> obstacle_matrix = obstacle.getVertexMatrix();

    Eigen::Matrix<double, 2, 2> segment_matrix;
    segment_matrix << p1[0], p2[0], p1[1], p2[1];

    Eigen::Vector2d p_obstacle;
    Eigen::Vector2d p_segment;
    geometry::closest_points<2, 4>(segment_matrix, obstacle_matrix, p_segment, p_obstacle);

    double buffer = model.minDistCenterToObstacle();
    p_segment = interpolate(p_obstacle, p_segment, buffer);

    Eigen::Vector2d s = p_obstacle - p_segment;
    s *= 2.;

    obst_var.xi[0] = -s[0];
    obst_var.xi[1] = -s[1];
    obst_var.mu = -s.dot(p_obstacle);
  }

  std::tuple<std::vector<Vertex>, double, double> parse_plan()
  {
    // The field `trajectory_` stores
    //  1. n_segment+1 waypoints. The last waypoint is the "target". The waypoints are part of a
    //    shortest path sent from the planner.
    //  2. The "shortest path id"
    //  3. The length of the tail of the shortest path

    int num_waypoints = trajectory_->getTrajectoryLength() - 1;
    double path_id = (*trajectory_)[num_waypoints][0];
    double tail_length = (*trajectory_)[num_waypoints][1];

    // extract the waypoints
    std::vector<Vertex> waypoints;
    for (int i = 0; i < num_waypoints; ++i)
    {
      Vertex waypoint = (*trajectory_)[i];
      waypoints.push_back(waypoint);
    }
    return { waypoints, path_id, tail_length };
  }

  void prepare_segment_guesses(std::vector<Vertex>& waypoints, solver_solution& guesses)
  {
    if (waypoints.back() != last_target_)
    {
      // These waypoints are new
      last_target_ = waypoints.back();

      // Update the params
      solver_->updateTarget(last_target_[0], last_target_[1]);

      // If there are not enough waypoints, repeat the last ones
      while ((int)waypoints.size() < config_.solver_config.num_segments + 1)
      {
        waypoints.push_back(waypoints.back());
      }

      // Update the guesses
      for (int i = 0; i < (int)waypoints.size(); ++i)
      {
        guesses.seg_states[i] = { waypoints[i][0], waypoints[i][1] };
      }

      for (int i = 0; i < config_.solver_config.num_segments; ++i)
      {
        guesses.seg_inputs[i].dx = guesses.seg_states[i + 1].x - guesses.seg_states[i].x;
        guesses.seg_inputs[i].dy = guesses.seg_states[i + 1].y - guesses.seg_states[i].y;
        // label the endpoints
        Vertex p1 = { guesses.seg_states[i].x, guesses.seg_states[i].y };
        Vertex p2 = { guesses.seg_states[i + 1].x, guesses.seg_states[i + 1].y };

        for (size_t j = 0; j < obstacle_positions_->size(); ++j)
        {
          obstacle_plane& plane = guesses.seg_inputs[i].planes[j];
          initialize_segment_plane(*obstacle_positions_->at(j), p1, p2, *model_, plane);
        }
      }
    }
    else
    {
      // Base the guesses from the previous solution
      for (int i = 0; i < config_.solver_config.num_segments + 1; ++i)
      {
        guesses.seg_states[i] = last_solution_.seg_states[i];
      }
      for (int i = 0; i < config_.solver_config.num_segments; ++i)
      {
        guesses.seg_inputs[i] = last_solution_.seg_inputs[i];
      }
    }
  }

  void prepare_first_solve(crs_models::kinematic_model::kinematic_car_state& state, std::vector<Vertex>& waypoints,
                           solver_solution& guesses)
  {
    // Initialize the guesses
    kbm_state car_init;
    car_init.x = state.pos_x;
    car_init.y = state.pos_y;
    car_init.yaw = state.yaw;
    car_init.velocity = state.velocity;
    car_init.torque = 0.0;
    car_init.steer = 0.0;

    for (auto& car_guess : guesses.kbm_states)
    {
      car_guess = car_init;
    }
    guesses.ref = car_init;
    guesses.ref.velocity = 0;  // make sure we are at rest

    for (auto& car_input : guesses.kbm_inputs)
    {
      car_input.dSteer = 0.0;
      car_input.dTorque = 0.0;
      for (size_t j = 0; j < obstacle_positions_->size(); ++j)
      {
        initialize_obstacle_variable(*obstacle_positions_->at(j), car_init, *model_, car_input.planes[j]);
      }
    }

    prepare_segment_guesses(waypoints, guesses);
  }

  void prepare_solve(crs_models::kinematic_model::kinematic_car_state& state, std::vector<Vertex>& waypoints,
                     solver_solution& guesses)
  {
    kbm_state x0;
    x0.x = state.pos_x;
    x0.y = state.pos_y;
    x0.yaw = state.yaw;
    x0.velocity = state.velocity;
    x0.torque = last_solution_.kbm_states[1].torque;
    x0.steer = last_solution_.kbm_states[1].steer;

    guesses.kbm_states[0] = x0;

    const int horizon_length = config_.solver_config.horizon_length;

    for (int i = 1; i < horizon_length; ++i)
    {
      guesses.kbm_states[i] = last_solution_.kbm_states[i + 1];
    }
    guesses.kbm_states[horizon_length] = last_solution_.kbm_states[horizon_length];
    guesses.ref = last_solution_.ref;

    for (int i = 0; i < horizon_length - 1; ++i)
    {
      guesses.kbm_inputs[i] = last_solution_.kbm_inputs[i + 1];
    }
    // For the last input, we use the same obstacle planes, but 0 input
    guesses.kbm_inputs[horizon_length - 1] = last_solution_.kbm_inputs[horizon_length - 1];
    guesses.kbm_inputs[horizon_length - 1].dSteer = 0.0;
    guesses.kbm_inputs[horizon_length - 1].dTorque = 0.0;

    prepare_segment_guesses(waypoints, guesses);
  }

  void initialize(crs_models::kinematic_model::kinematic_car_state state [[maybe_unused]])
  {
    auto model_dynamics = std::make_shared<const crs_models::kinematic_model::kinematic_params>(model_->getParams());
    // int N_WARM_START = 10;

    kbm_params kbm_params = { { config_.Q1, config_.Q2, config_.R1, config_.R2 }, model_dynamics, obstacle_positions_ };

    seg_params seg_params = { 0, 0, model_dynamics, obstacle_positions_ };

    solver_->setParams(kbm_params, seg_params);
  }

  /**
   * @brief Executes MPC controller, returning a new input to apply
   *
   * @param state current measured state of the system
   * @param timestamp current timestamp, will be ignored
   * @return crs_models::kinematic_model::kinematic_car_input
   */
  crs_models::kinematic_model::kinematic_car_input
  getControlInput(crs_models::kinematic_model::kinematic_car_state state)
  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    bool first_solve = false;
    if (!is_initialized)
    {
      initialize(state);
      is_initialized = true;
      first_solve = true;
    }

    auto [waypoints, shortest_path_id, tail_length] = parse_plan();
    plan_id_ = shortest_path_id;
    tail_length_ = tail_length;
    if (shortest_path_id == -1)
    {
      // The planner is indicating that the goal is reached. Tell the car to stop.
      return { 0, 0 };
    }

    // initialize state to virtually advanced vehicle states and inputs
    crs_models::kinematic_model::kinematic_car_input last_input = { last_solution_.kbm_states[1].torque,
                                                                    last_solution_.kbm_states[1].steer };

    state = model_->applyModel(state, last_input, config_.lag_compensation_time);

    solver_solution guesses(config_.solver_config);

    if (first_solve)
    {
      prepare_first_solve(state, waypoints, guesses);
    }
    else
    {
      prepare_solve(state, waypoints, guesses);
    }

    // solver_->setInitialState(guesses.kbm_states[0]);
    solver_->setGuesses(guesses);

    // solver_->printDebugInfo();

    int status = solver_->solve(guesses.kbm_states[0]);

    if (status == ACADOS_MAXITER)
    {
      std::cerr << "WARNING: Max iterations exceeded" << std::endl;
    }
    else if (status != 0)
    {
      std::cout << "SOLVE FAILED with status " << status << std::endl;
      std::cout << "[NONCONVEX_COMPARISON ERROR]: Solver" << std::endl;
      last_solve_succeeded_ = false;
      return { last_solution_.kbm_states[1].torque, last_solution_.kbm_states[1].steer };
      // abort();
    }

    last_solve_succeeded_ = true;

    double cost = solver_->computeCost();
    double adjusted_cost = cost + tail_length_;
    std::cout << "[NONCONVEX_COMPARISON] Cost: " << adjusted_cost << std::endl;

    solver_->getSolution(last_solution_);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[NONCONVEX_COMPARISON] Controller: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    return { last_solution_.kbm_states[1].torque, last_solution_.kbm_states[1].steer };
  }

  bool isInitializing()
  {
    bool initialized = trajectory_->isInitialized() && trajectory_->getTrajectoryLength() > 0 && obstacle_positions_;
    return !initialized;
  }
};

/////// Function Stubs ///////
NonconvexController::NonconvexController(tracking_mpc_kinematic_nonconvex_config config,
                                         std::shared_ptr<crs_models::kinematic_model::DiscreteKinematicModel> model,
                                         std::shared_ptr<DynamicPointTrajectory> trajectory)
  : MpcController(model, trajectory)
{
  impl_ = new Private(config, model, trajectory);
}

std::optional<std::vector<std::vector<double>>> NonconvexController::getPlannedTrajectory() const
{
  return impl_->getPlannedTrajectory();
}

crs_models::kinematic_model::kinematic_car_input NonconvexController::getControlInput(
    crs_models::kinematic_model::kinematic_car_state state, double timestamp [[maybe_unused]])
{
  return impl_->getControlInput(state);
}

void NonconvexController::setObstacles(std::shared_ptr<const ObstacleVector> obstacles)
{
  impl_->obstacle_positions_ = obstacles;
}

bool NonconvexController::isInitializing()
{
  return impl_->isInitializing();
}

}  // namespace kinematic_controllers

}  // namespace crs_controls

#endif
