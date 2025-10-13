#include <segment_planner/path_progressor.h>
#include <CGAL/Simple_cartesian.h>
#include <commons/cgal_simple_cartesian_conversion.h>
#include <boost/range.hpp>
#include <boost/range/join.hpp>
#include <iterator>
#include <list>
#include <cmath>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;
typedef Kernel::Triangle_2 Triangle_2;
typedef Kernel::Line_2 Line_2;

namespace crs_planning
{
using crs_controls::Obstacle;
using crs_controls::ObstacleVector;
using geometry::Vertex;

struct PathProgressor::Private
{
  const int N_SEGMENTS;
  const double BUFFER_SQUARED;

  Private(int n_segments, double buffer, std::shared_ptr<const ObstacleVector> obstacles);
  std::vector<Vertex> set_path(const std::vector<Vertex>& shortest_path);
  std::optional<std::vector<Vertex>> update_path(const std::vector<Vertex>& prefix);
  double compute_tail_length(void);

  // uses the helper functions
  bool skip_point(const Point_2& a, const Point_2& b, const Point_2& c);
  bool segment_collides(Point_2 p1, Point_2 p2);

  std::vector<Point_2> shortest_path_;
  bool path_is_new_;
  std::vector<Point_2>::const_iterator one_past_curr_;
  std::list<Triangle_2> obstacle_triangles_;
  int last_path_size_;
};

PathProgressor::Private::Private(int n_segments, double buffer, std::shared_ptr<const ObstacleVector> obstacles)
  : N_SEGMENTS(n_segments), BUFFER_SQUARED(std::pow(buffer, 2))
{
  for (const std::unique_ptr<Obstacle>& obstacle : *obstacles)
  {
    std::vector<geometry::Triangle> obstacle_triangles;
    obstacle->triangulate(obstacle_triangles);

    for (const geometry::Triangle& triangle : obstacle_triangles)
    {
      Triangle_2 cgal_triangle = commons::to_cgal_triangle(triangle);
      obstacle_triangles_.push_back(cgal_triangle);
    }
  }
}

std::vector<Vertex> PathProgressor::Private::set_path(const std::vector<Vertex>& shortest_path_vertices)
{
  shortest_path_.clear();

  for (const Vertex& v : shortest_path_vertices)
  {
    shortest_path_.push_back(commons::to_cgal_point(v));
  }
  path_is_new_ = true;

  int distance = std::min(N_SEGMENTS + 1, (int)shortest_path_.size());
  one_past_curr_ = shortest_path_.begin() + distance;
  last_path_size_ = distance;
  return { shortest_path_vertices.begin(), shortest_path_vertices.begin() + distance };
}

std::optional<std::vector<Vertex>> PathProgressor::Private::update_path(const std::vector<Vertex>& prefix)
{
  // We are guaranteed that the last element in the prefix corresponds to curr_idx

  // Construct a range of all the relevant points

  std::vector<Point_2> all_points_head;
  for (const Vertex& vertex : prefix)
  {
    all_points_head.push_back(commons::to_cgal_point(vertex));
  }

  auto all_points_tail =
      boost::make_iterator_range<std::vector<Point_2>::const_iterator>(one_past_curr_, shortest_path_.end());

  auto all_points = boost::range::join(all_points_head, all_points_tail);

  // Iterate through all the new points until we build up a new path
  // The points in this path should not be collinear, and the segments should not
  // intersect with obstacles

  auto start = all_points.begin();
  auto mid = start + 1;
  std::vector<Vertex> new_path = { commons::from_cgal_point(*start) };

  while (mid != all_points.end() && (int)new_path.size() <= N_SEGMENTS)
  {
    bool add_mid = (mid + 1 == all_points.end()) || !skip_point(*start, *mid, *(mid + 1));
    if (add_mid)
    {
      new_path.push_back(commons::from_cgal_point(*mid));
      start = mid;
    }
    ++mid;
  }

  // compute how far into the tail we are
  int total_incremented = std::distance(all_points.begin(), mid);
  int position_in_tail = total_incremented - (int)all_points_head.size();
  assert(position_in_tail >= 0);
  bool send_path = path_is_new_             // the target just changed
                   || position_in_tail > 0  // we have incremented into the tail of the shortest path
                   || last_path_size_ != (int)new_path.size();  // we are at the end, and the path is further condensed

  if (!send_path)
  {
    return std::nullopt;
  }

  // the path changed
  path_is_new_ = false;
  one_past_curr_ += position_in_tail;  // the path has changed
  last_path_size_ = new_path.size();

  return new_path;
}

double PathProgressor::Private::compute_tail_length(void)
{
  auto curr = std::prev(one_past_curr_);
  assert(one_past_curr_ != shortest_path_.begin());

  // compute the length of the tail of the path
  double tail_length = 0;
  {
    for (; curr + 1 != shortest_path_.end(); ++curr)
    {
      auto next = curr + 1;
      double dx = next->x() - curr->x();
      double dy = next->y() - curr->y();
      tail_length += std::hypot(dx, dy);
    }
  }
  return tail_length;
}

bool PathProgressor::Private::skip_point(const Point_2& a, const Point_2& b, const Point_2& c)
{
  // Return true if point b is redundant, i.e.
  //  1. Points a, b, and c are "almost" collinear
  //  2. The segment {ac} doesn't intersect with any obstacles
  if (b == c)
  {
    return true;  // special case for numerical stability
  }

  // check for collinearity
  const double COLLINEAR_TOL = 3e-2;
  Line_2 line_ac(a, c);
  Point_2 b_on_ac = line_ac.projection(b);
  double d2 = CGAL::to_double(CGAL::squared_distance(b_on_ac, line_ac));
  if (d2 > COLLINEAR_TOL * COLLINEAR_TOL)
  {
    // B is far from segment AC, so these points are not collinear
    return false;
  }

  if (segment_collides(a, c))
  {
    return false;
  }

  // The segments are colinear and we are not too close to an obstacle.
  // Return true.
  return true;
}

bool PathProgressor::Private::segment_collides(Point_2 p1, Point_2 p2)
{
  // check for collisions
  Segment_2 seg(p1, p2);

  for (const Triangle_2& triangle : obstacle_triangles_)
  {
    if (CGAL::squared_distance(triangle, seg) < BUFFER_SQUARED - 1e-8)
    {
      return true;
    }
  }
  return false;
}

PathProgressor::PathProgressor(int n_segments, double buffer, std::shared_ptr<const ObstacleVector> obstacles)
  : impl_(std::make_unique<Private>(n_segments, buffer, obstacles))
{
}

PathProgressor::~PathProgressor()
{
}

std::vector<Vertex> PathProgressor::set_path(const std::vector<Vertex>& shortest_path)
{
  return impl_->set_path(shortest_path);
}

std::optional<std::vector<Vertex>> PathProgressor::update_path(const std::vector<Vertex>& prefix)
{
  return impl_->update_path(prefix);
}

double PathProgressor::compute_tail_length(void)
{
  return impl_->compute_tail_length();
}

void PathProgressor::get_path(std::vector<Vertex>& plan) const
{
  for (const auto& pt : impl_->shortest_path_)
  {
    plan.push_back(commons::from_cgal_point(pt));
  }
}

}  // end namespace crs_planning
