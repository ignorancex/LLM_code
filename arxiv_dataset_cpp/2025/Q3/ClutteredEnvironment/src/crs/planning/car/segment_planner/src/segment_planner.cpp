#include <commons/obstacle.h>
#include <segment_planner/roadmap.h>
#include <segment_planner/path_progressor.h>
#include <kinematic_model/kinematic_discrete.h>

#include "segment_planner/segment_planner.h"

using crs_controls::ObstacleVector;
using crs_models::kinematic_model::DiscreteKinematicModel;
using geometry::Vertex;

namespace crs_planning
{
// Private functions
struct SegmentPlanner::Private
{
  struct InitInfo
  {
    // We need to wait for the following info before we can initialize
    Vertex initial_start_;
    Vertex initial_target_;
    std::shared_ptr<const crs_controls::ObstacleVector> obstacle_positions_;

    bool received_initial_state_ = false;
    bool received_initial_target_ = false;
    bool setup_complete_ = false;
  };

  std::unique_ptr<DiscreteKinematicModel> model_;
  std::unique_ptr<const Config> config_;

  InitInfo init_info;

  std::unique_ptr<Roadmap> roadmap_;
  std::unique_ptr<PathProgressor> path_progressor_;
  Vertex current_coords_;
  Vertex current_target_;

  std::vector<Vertex> path_to_send_;
  bool send_new_path_ = false;

  double plan_id_;

  // If we have reached the goal, we use this indicator as the shortest path id
  static constexpr double GOAL_REACHED_INDICATOR = -1;

  /// Implementation

  Private(std::unique_ptr<DiscreteKinematicModel> model, std::unique_ptr<Config> config)
    : model_(std::move(model)), config_(std::move(config)), plan_id_(0)
  {
  }

  void setObstacles(std::shared_ptr<const crs_controls::ObstacleVector> obstacles)
  {
    init_info.obstacle_positions_ = obstacles;
    initialize_if_ready();
  }

  void setInitialState(const crs_models::kinematic_model::kinematic_car_state& state)
  {
    init_info.initial_start_ = { state.pos_x, state.pos_y };
    init_info.received_initial_state_ = true;
    initialize_if_ready();
  }

  void setTarget(const Vertex& target)
  {
    if (!init_info.received_initial_target_)
    {
      init_info.initial_target_ = target;
      init_info.received_initial_target_ = true;
      initialize_if_ready();
    }
    else
    {
      if (init_info.setup_complete_)
      {
        constructAndSendPath(current_coords_, target);
      }
    }
  }

  // Return the relevant parts of the trajectory:
  //  1. State
  //  2. Waypoints
  //  3. Shortest Path ID
  std::tuple<Vertex, std::vector<Vertex>, double>
  parseControllerPrediction(const std::vector<std::vector<double>>& prediction)
  {
    Vertex state_xy = { prediction[0][0], prediction[0][1] };

    // The last item contains the Shortest Path ID.
    // The previous N_SEGMENTS + 1 contain the waypoints

    int waypoints_idx_start = (int)prediction.size() - 1 - (config_->num_segments + 1);
    int waypoints_idx_end = (int)prediction.size() - 1;
    std::vector<Vertex> waypoints;
    for (int i = waypoints_idx_start; i < waypoints_idx_end; ++i)
    {
      waypoints.emplace_back(prediction[i][0], prediction[i][1]);
    }

    double plan_id = prediction.back()[0];

    return { state_xy, waypoints, plan_id };
  }

  // Return true if we are sufficiently close to the target
  bool targetReached(const Vertex& v)
  {
    double dx = v[0] - current_target_[0];
    double dy = v[1] - current_target_[1];
    double dist = std::hypot(dx, dy);
    return dist <= config_->goal_reached_buffer;
  }

  void handleTrajectory(const std::vector<std::vector<double>>& controller_prediction)
  {
    assert(init_info.setup_complete_);

    if (plan_id_ == GOAL_REACHED_INDICATOR)
    {
      return;  // the car has reached its goal. Nothing to do
    }

    auto [state_xy, waypoints, plan_id] = parseControllerPrediction(controller_prediction);

    if (plan_id != plan_id_)
    {
      return;  // This controller trajectory is out of date -- ignore it
    }

    if (targetReached(state_xy))
    {
      std::cout << "Target reached! Informing controller" << std::endl;
      // Indicate that we have reached the target
      plan_id_ = GOAL_REACHED_INDICATOR;
      path_to_send_ = {};
      send_new_path_ = true;
      return;
    }

    current_coords_ = waypoints[0];

    if (config_->euclidean_offset_cost)
    {
      // We never update the intermeidate target
      return;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto new_path = path_progressor_->update_path(waypoints);
    if (new_path)
    {
      send_new_path_ = true;
      path_to_send_ = new_path.value();
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[NONCONVEX_COMPARISON] Planner: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
  }

  std::optional<std::vector<cartesian_reference_point>> getPath()
  {
    if (!send_new_path_)
    {
      return std::nullopt;
    }

    send_new_path_ = false;

    if (plan_id_ == GOAL_REACHED_INDICATOR)
    {
      return { { { plan_id_, NAN } } };
    }

    ++plan_id_;

    std::cout << "New Path: ";

    std::vector<cartesian_reference_point> cartesian_path;
    for (const Vertex& pt : path_to_send_)
    {
      std::cout << "(" << pt[0] << ", " << pt[1] << "), ";
      cartesian_path.emplace_back(pt[0], pt[1]);
    }
    std::cout << std::endl;

    double tail_length = 0;
    if (!config_->euclidean_offset_cost)
    {
      tail_length = path_progressor_->compute_tail_length();
    }

    cartesian_path.emplace_back(plan_id_, tail_length);

    return cartesian_path;
  }

  // uses the helper functions
  void initialize_if_ready()
  {
    bool ready =
        init_info.received_initial_state_ && init_info.received_initial_target_ && init_info.obstacle_positions_;

    if (!ready)
    {
      return;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    double inflation_amount = model_->minDistCenterToObstacle();

    Roadmap::bounds bounds = { config_->bounds.x_min, config_->bounds.x_max, config_->bounds.y_min,
                               config_->bounds.y_max };
    roadmap_ = std::make_unique<Roadmap>(init_info.obstacle_positions_, inflation_amount, bounds);
    path_progressor_ =
        std::make_unique<PathProgressor>(config_->num_segments, inflation_amount, init_info.obstacle_positions_);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[NONCONVEX_COMPARISON] Roadmap: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    constructAndSendPath(init_info.initial_start_, init_info.initial_target_);

    init_info.setup_complete_ = true;
  }

  void constructAndSendPath(const Vertex& start, const Vertex& target)
  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (!roadmap_->pointIsSafe(target))
    {
      std::cerr << "Warning: target intersects with obstacle. Ignoring it" << std::endl;
      return;
    }

    if (!roadmap_->pointIsSafe(start))
    {
      std::cerr << "Warning: start intersects with obstacle. Proceeding anyways" << std::endl;
    }

    current_target_ = target;

    if (config_->euclidean_offset_cost)
    {
      path_to_send_ = { start, target };
    }
    else
    {
      std::vector<Vertex> shortest_path_vertices;
      roadmap_->shortestPath(start, target, shortest_path_vertices);
      path_to_send_ = path_progressor_->set_path(shortest_path_vertices);
    }
    send_new_path_ = true;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[NONCONVEX_COMPARISON] Dijkstra: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
  }

  std::vector<std::vector<geometry::Vertex>> getInflatedObstacles() const
  {
    return roadmap_->getPolygonVertices();
  }
};

///////// Segment Planner Function stubs

SegmentPlanner::SegmentPlanner(std::unique_ptr<crs_models::kinematic_model::DiscreteKinematicModel> model,
                               std::unique_ptr<Config> config)
  : impl_(std::make_unique<Private>(std::move(model), std::move(config)))
{
}

SegmentPlanner::~SegmentPlanner()
{
}

void SegmentPlanner::setObstacles(std::shared_ptr<const crs_controls::ObstacleVector> obstacles)
{
  impl_->setObstacles(obstacles);
}

void SegmentPlanner::setInitialState(const crs_models::kinematic_model::kinematic_car_state& state)
{
  impl_->setInitialState(state);
}

void SegmentPlanner::setTarget(const Vertex& target)
{
  impl_->setTarget(target);
}

bool SegmentPlanner::setupComplete()
{
  return impl_->init_info.setup_complete_;
}

void SegmentPlanner::getRoadMapVisualization(Roadmap::VisualizationInfo& vis_info) const
{
  if (!impl_->init_info.setup_complete_)
  {
    return;
  }
  impl_->roadmap_->getVisualizationInfo(vis_info);
}

void SegmentPlanner::getPlanVisualization(std::vector<Vertex>& plan) const
{
  impl_->path_progressor_->get_path(plan);
}

void SegmentPlanner::controllerTrajectoryCallback(const std::vector<std::vector<double>>& controller_trajectory)
{
  impl_->handleTrajectory(controller_trajectory);
}

std::optional<std::vector<cartesian_reference_point>> SegmentPlanner::getPath()
{
  return impl_->getPath();
}

std::vector<std::vector<geometry::Vertex>> SegmentPlanner::getInflatedObstacles() const
{
  return impl_->getInflatedObstacles();
}

}  // namespace crs_planning
