#include <segment_planner/polygon_tree.h>
#include <commons/geometry_utils.h>
#include <commons/cgal_simple_cartesian_conversion.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_2.h>
#include <CGAL/Polygon_2.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;
typedef Kernel::Vector_2 Vector_2;
typedef CGAL::Polygon_2<Kernel> Polygon_2;

// Define a primitive for the AABB tree where each segment refers to its polygon
struct Polygon_segment_primitive
{
  typedef std::pair<int, Segment_2> IdSegment;
  // Must define the types Point, Dataum, Id
  typedef Point_2 Point;
  typedef Segment_2 Datum;
  typedef IdSegment Id;

  typedef std::vector<Id>::const_iterator Iterator;

  Id id_;
  Polygon_segment_primitive(Iterator it) : id_(*it)
  {
  }

  // must define constructor and the functions datum(), reference_point(), and id()
  Datum datum() const
  {
    return id_.second;
  }
  Point reference_point() const
  {
    return id_.second.source();
  }
  Id id() const
  {
    return id_;
  }
};

typedef CGAL::AABB_traits_2<Kernel, Polygon_segment_primitive> AABB_polygon_traits;
typedef CGAL::AABB_tree<AABB_polygon_traits> AABB_tree;
typedef std::optional<AABB_tree::Intersection_and_primitive_id<Segment_2>::Type> Segment_intersection;

namespace crs_planning
{
using commons::from_cgal_point;
using commons::to_cgal_point;
using geometry::Polygon;
using geometry::Vertex;
typedef PolygonTree::PolygonId PolygonId;

struct PolygonTree::Private
{
  PolygonId insert(const Polygon<>& polygon)
  {
    // Construct a CGAL polygon from the current polygon
    std::vector<Point_2> points;
    for (int i = 0; i < polygon.cols(); ++i)
    {
      Vertex v = polygon.col(i);
      points.emplace_back(v[0], v[1]);
    }
    Polygon_2 cgal_polygon(points.begin(), points.end());

    PolygonId curr_id = (PolygonId)polygons_.size();
    std::vector<Polygon_segment_primitive::Id> segments;

    for (const Segment_2& polygon_edge : cgal_polygon.edges())
    {
      segments.emplace_back(curr_id, polygon_edge);
    }
    tree_.insert(segments.begin(), segments.end());
    polygons_.push_back(cgal_polygon);

    return curr_id;
  }

  // Given v1, a vertex on a polygon, checks if the point d leads inside the polygon.
  // Precondition: the two points should not correspond to the edge of a polygon
  bool segment_self_intersects(const VertexLookup& v1, const Point_2& d)
  {
    Polygon_2& polygon = polygons_[v1.polygon_idx];
    int n_sides = polygon.size();

    Point_2& a = polygon[(v1.point_idx - 1 + n_sides) % n_sides];
    Point_2& b = polygon[v1.point_idx];
    Point_2& c = polygon[(v1.point_idx + 1) % n_sides];

    // Check if d is in the angle formed by abc
    return CGAL::orientation(a, b, c) == CGAL::orientation(a, b, d) &&
           CGAL::orientation(c, b, a) == CGAL::orientation(c, b, d);
  }

  // Return the list of segments that intersect with the query
  std::list<AABB_tree::Primitive_id> visible_helper(const Point_2& p1, const Point_2& p2)
  {
    Segment_2 query = { p1, p2 };
    std::list<AABB_tree::Primitive_id> intersections;
    tree_.all_intersected_primitives(query, std::back_inserter(intersections));
    return intersections;
  }

  bool visible_polygon_edge(const VertexLookup& v1, const VertexLookup& v2)
  {
    assert(v1.polygon_idx == v2.polygon_idx);
    const Point_2& p1 = polygons_[v1.polygon_idx].vertex(v1.point_idx);
    const Point_2& p2 = polygons_[v2.polygon_idx].vertex(v2.point_idx);

    auto intersections = visible_helper(p1, p2);
    for (const auto& [polygon_idx, _] : intersections)
    {
      if (polygon_idx != v1.polygon_idx)
      {
        return false;
      }
    }
    return true;
  }

  bool visible(const VertexLookup& v1, const VertexLookup& v2)
  {
    const Point_2& p1 = polygons_[v1.polygon_idx].vertex(v1.point_idx);
    const Point_2& p2 = polygons_[v2.polygon_idx].vertex(v2.point_idx);

    if (segment_self_intersects(v1, p2) || segment_self_intersects(v2, p1))
    {
      return false;
    }

    // First check if the segment points directly into the two respective polygons

    auto intersections = visible_helper(p1, p2);
    for (const auto& [polygon_idx, _] : intersections)
    {
      if (polygon_idx != v1.polygon_idx && polygon_idx != v2.polygon_idx)
      {
        return false;
      }
    }
    return true;
  }

  bool visible(const VertexLookup& v1, const Vertex& v2)
  {
    const Point_2& p1 = polygons_[v1.polygon_idx].vertex(v1.point_idx);
    Point_2 p2 = to_cgal_point(v2);

    if (segment_self_intersects(v1, p2))
    {
      return false;
    }

    auto intersections = visible_helper(p1, p2);
    for (const auto& [polygon_idx, _] : intersections)
    {
      if (polygon_idx != v1.polygon_idx)
      {
        return false;
      }
    }
    return true;
  }

  bool visible(const Vertex& v1, const Vertex& v2)
  {
    Point_2 p1 = to_cgal_point(v1);
    Point_2 p2 = to_cgal_point(v2);

    auto intersections = visible_helper(p1, p2);
    return intersections.empty();
  }

  static Point_2 closest_point_on_segment(const Segment_2& segment, const Point_2& x)
  {
    const Point_2& p = segment.source();
    const Point_2& q = segment.target();

    Vector_2 pq = q - p;
    Vector_2 px = x - p;

    double t = px * pq / pq.squared_length();
    if (t <= 0)
    {
      return p;
    }
    else if (t >= 1)
    {
      return q;
    }
    else
    {
      return p + t * pq;
    }
  }

  // Return a list of polygons that the point is inside of.
  // Note that this is linear in the number of polygons. There is not a good way to use
  // the r-tree, since it only contains segments. If this is a bottleneck, we can create
  // a new rtree which contains the triangulated polygons and check efficiently
  std::vector<PolygonId> get_containing_polygons(const Vertex& v)
  {
    Point_2 pt = to_cgal_point(v);
    std::vector<PolygonId> containing_polygons;
    for (PolygonId i = 0; i < (int)polygons_.size(); ++i)
    {
      if (polygons_[i].has_on_bounded_side(pt))
      {
        containing_polygons.push_back(i);
      }
    }
    return containing_polygons;
  }

  bool point_is_safe(const Vertex& v)
  {
    std::vector<PolygonId> containing_polygons = get_containing_polygons(v);
    return containing_polygons.empty();
  }

  Vertex closest_point_outside_polygons(const Vertex& v)
  {
    std::vector<PolygonId> intersections = get_containing_polygons(v);
    if (intersections.empty())
    {
      return v;
    }

    // While it is technically possible that the point could be inside of multiple polygons,
    // this is very unlikely, and this case would be complicated to deal with.
    // We assume the point is only inside of one polygon.
    assert(intersections.size() == 1);
    const Polygon_2& polygon = polygons_[intersections[0]];

    // Find the closest point on the polygon border
    Point_2 p = to_cgal_point(v);
    Point_2 closest_pt;
    double min_dist_squared = std::numeric_limits<double>::max();
    for (const Segment_2& edge : polygon.edges())
    {
      Point_2 closest_on_edge = closest_point_on_segment(edge, p);
      double dist_to_edge_squared = CGAL::squared_distance(closest_on_edge, p);

      if (dist_to_edge_squared < min_dist_squared)
      {
        closest_pt = closest_on_edge;
        min_dist_squared = dist_to_edge_squared;
      }
    }
    // Extend the point a little past the border of the polygon
    const double extra_room = 1e-5;
    Vector_2 pq = closest_pt - p;
    Point_2 pt_outside_polygon = p + (1 + extra_room) * pq;

    return from_cgal_point(pt_outside_polygon);
  }

  std::vector<std::vector<geometry::Vertex>> getPolygonVertices() const
  {
    std::vector<std::vector<geometry::Vertex>> polygon_vertices;
    for (const auto& polygon : polygons_)
    {
      std::vector<geometry::Vertex> curr_polygon;
      for (const auto& vertex : polygon.vertices())
      {
        curr_polygon.push_back(from_cgal_point(vertex));
      }
      polygon_vertices.push_back(curr_polygon);
    }
    return polygon_vertices;
  }

  // Fields

  /// Vector of polygons that is inserted into the tree
  std::vector<Polygon_2> polygons_;

  /// The AABB tree with polygons
  AABB_tree tree_;
};

PolygonTree::PolygonTree()
{
  impl_ = std::make_unique<Private>();
}

PolygonTree::~PolygonTree()
{
}

PolygonId PolygonTree::insert(const Polygon<>& polygon)
{
  return impl_->insert(polygon);
}

bool PolygonTree::visible(const VertexLookup& v1, const VertexLookup& v2)
{
  return impl_->visible(v1, v2);
}

bool PolygonTree::visible_polygon_edge(const VertexLookup& v1, const VertexLookup& v2)
{
  return impl_->visible_polygon_edge(v1, v2);
}

bool PolygonTree::visible(const VertexLookup& v1, const Vertex& v2)
{
  return impl_->visible(v1, v2);
}

bool PolygonTree::visible(const Vertex& v1, const Vertex& v2)
{
  return impl_->visible(v1, v2);
}

Vertex PolygonTree::closest_point_outside_polygons(const Vertex& v)
{
  return impl_->closest_point_outside_polygons(v);
}

bool PolygonTree::point_is_safe(const Vertex& v)
{
  return impl_->point_is_safe(v);
}

std::vector<std::vector<geometry::Vertex>> PolygonTree::getPolygonVertices() const
{
  return impl_->getPolygonVertices();
}

}  // end namespace crs_planning
