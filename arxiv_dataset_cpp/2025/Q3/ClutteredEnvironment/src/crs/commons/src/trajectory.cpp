#include "commons/trajectory.h"

namespace crs_controls
{

Eigen::Vector2d Trajectory::getClosestTrackPoint(const Eigen::Vector2d& query_point, int precision /* Default 16 */)
{
  return BaseTrajectory::getClosestTrajectoryPoint(query_point, precision);
}

int Trajectory::getClosestTrackPointIdx(const Eigen::Vector2d& query_point, int precision /* Default 16 */)
{
  return BaseTrajectory::getClosestTrajectoryPointIdx(query_point, precision);
}

const Eigen::Vector2d Trajectory::getLastRequestedTrackPoint() const
{
  return BaseTrajectory::getLastRequestedTrajectoryPoint();
}

std::vector<double> Trajectory::getVorEdgesX()
{
  return voronoi_edges_x_;
}

std::vector<double> Trajectory::getVorEdgesY()
{
  return voronoi_edges_y_;
}

}  // namespace crs_controls
