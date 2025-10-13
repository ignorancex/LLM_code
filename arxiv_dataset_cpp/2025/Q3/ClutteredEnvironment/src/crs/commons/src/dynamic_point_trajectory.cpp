#include "commons/dynamic_point_trajectory.h"
#include <iostream>
#include <vector>
#include <numeric>

namespace crs_controls
{

DynamicPointTrajectory::DynamicPointTrajectory() : Trajectory()
{
  initialized_ = false;
}

DynamicPointTrajectory::DynamicPointTrajectory(std::vector<double> x_coord, std::vector<double> y_coord)
  : Trajectory(x_coord, y_coord)
{
}

void DynamicPointTrajectory::resetTrajectory(std::vector<double> x_coord, std::vector<double> y_coord)
{
  initialized_ = true;

  assert((x_coord.size() == y_coord.size()) && "x_coord and y_coord differ in size");
  trajectory_coordinates_.clear();
  trajectory_coordinates_.reserve(x_coord.size());

  for (unsigned int i = 0; i < x_coord.size(); i++)
  {
    trajectory_coordinates_.push_back(Eigen::Vector2d(x_coord[i], y_coord[i]));
  }

  last_query_index_ = -1;
}

void DynamicPointTrajectory::resetTrajectory(std::vector<Eigen::Vector2d> pts)
{
  BaseTrajectory::resetTrajectory(pts);
}

void DynamicPointTrajectory::resetVorEdges(std::vector<double> x_edge, std::vector<double> y_edge)
{
  voronoi_edges_x_ = x_edge;
  voronoi_edges_y_ = y_edge;
}

double DynamicPointTrajectory::getLastRequestedTrackAngle() const
{
  return 0;
}

size_t DynamicPointTrajectory::getTrajectoryLength() const
{
  return trajectory_coordinates_.size();
}

}  // namespace crs_controls
