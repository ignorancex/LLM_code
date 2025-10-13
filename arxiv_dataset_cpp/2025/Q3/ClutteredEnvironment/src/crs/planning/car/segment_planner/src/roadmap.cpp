#include <commons/obstacle.h>
#include <segment_planner/polygon_tree.h>
#include <commons/cgal_simple_cartesian_conversion.h>
#include <CGAL/Simple_cartesian.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <memory>

#include <segment_planner/roadmap.h>

/// CGAL definitions
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;

// typedef CGAL::Exact_predicates_inexact_constructions_kernel   Kernel;

/// Graph definitions

typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;

struct VertexProperty
{
  Point_2 point;
  int obstacle_idx;
  int point_idx;

  VertexProperty(const Point_2& point_in, int obstacle_idx_in, int point_idx_in)
    : point(point_in), obstacle_idx(obstacle_idx_in), point_idx(point_idx_in)
  {
  }

  // default constructor needed by boost graph library
  VertexProperty()
  {
  }
};

typedef boost::adjacency_list<boost::vecS,         // Vertex storage container (vector)
                              boost::vecS,         // Edge storage container (vector)
                              boost::undirectedS,  // Directed or undirected
                              VertexProperty,      // Vertex property
                              EdgeWeightProperty   // Edge weight property
                              >
    Graph;

typedef boost::graph_traits<Graph>::vertex_descriptor graph_vertex;
typedef boost::graph_traits<Graph>::edge_descriptor graph_edge;

/// Implementation

namespace crs_planning
{
using crs_controls::Obstacle;
using crs_controls::ObstacleVector;
using geometry::Polygon;
using geometry::Vertex;

struct Roadmap::Private
{
  //// Fields

  // Graph with obstacles: Vertices correspond to the inflated obstacles.
  // Two vertices are connected if they are members of the roadmap.
  Graph graph_;

  // Tree for fast intersection computation. Stores the inflated obstacles. Can be used
  // to efficiently check if a segment is blocked by an obstacle.
  PolygonTree tree_;

  // The bounds of our roadmap -- no vertices should be outside these bounds
  const bounds bounds_;

  const graph_vertex NO_VERTEX = -1;

  //// Initialization
  Private(std::shared_ptr<const ObstacleVector> obstacles, double inflation_amount, const bounds& bounds);

  // Uses the following helper functions:
  void add_vertices_to_graph(const std::vector<Polygon<>>& polygons,
                             std::vector<std::vector<graph_vertex>>& graph_vertices);

  void add_obstacle_sides_to_graph(const std::vector<std::vector<graph_vertex>>& graph_vertices);

  void construct_aabb_tree(const std::vector<Polygon<>>& inflated_polygons);

  void add_bitangent_edges_to_graph(const std::vector<std::vector<graph_vertex>>& graph_vertices);

  bool bitangent(const std::vector<std::vector<graph_vertex>>& vertices, const VertexProperty& prop_p,
                 const VertexProperty& prop_q) const;

  bool in_bounds(geometry::Vertex vertex);

  //// Shortest path
  void shortestPath(const Vertex& v1, const Vertex& v2, std::vector<Vertex>& path);

  // Uses the following helper functions
  std::pair<graph_vertex, graph_vertex> add_endpoints_to_graph(const Vertex& v1, const Vertex& v2);
  graph_vertex add_endpoint_to_graph(const Vertex& v);
  void remove_endpoints_from_graph(graph_vertex v1, graph_vertex v2);

  void compute_dijkstra(graph_vertex start, graph_vertex goal, std::vector<Vertex>& path);

  bool pointIsSafe(const Vertex& v);

  //// Visualization Info
  void getVisualizationInfo(VisualizationInfo& vis_info) const;
  std::vector<std::vector<geometry::Vertex>> getPolygonVertices() const;
};

static double distance(const Point_2& p1, const Point_2& p2)
{
  double d2 = CGAL::to_double(CGAL::squared_distance(p1, p2));
  return std::sqrt(d2);
}

static bool same_side(const Point_2& p, const Point_2& q, const Point_2& q1, const Point_2& q3)
{
  // check if q1 and q3 are on the same side of the ray pq.
  // assume general position. (ignore the posibility that they're collinear)
  return CGAL::left_turn(p, q, q1) == CGAL::left_turn(p, q, q3);
}

bool Roadmap::Private::bitangent(const std::vector<std::vector<graph_vertex>>& vertices, const VertexProperty& prop_p,
                                 const VertexProperty& prop_q) const
{
  // properties corresponding to this vertex
  const std::vector<graph_vertex>& obstacle_p = vertices[prop_p.obstacle_idx];
  const std::vector<graph_vertex>& obstacle_q = vertices[prop_q.obstacle_idx];

  int pi = prop_p.point_idx;
  int qi = prop_q.point_idx;
  int np = (int)obstacle_p.size();
  int nq = (int)obstacle_q.size();

  // Label the relevant points. We want to check if p2 and q2 are bitangent
  const Point_2& p1 = graph_[obstacle_p[(pi - 1 + np) % np]].point;
  const Point_2& p2 = graph_[obstacle_p[pi % np]].point;
  const Point_2& p3 = graph_[obstacle_p[(pi + 1) % np]].point;

  const Point_2& q1 = graph_[obstacle_q[(qi - 1 + nq) % nq]].point;
  const Point_2& q2 = graph_[obstacle_q[qi % nq]].point;
  const Point_2& q3 = graph_[obstacle_q[(qi + 1) % nq]].point;

  return same_side(p2, q2, q1, q3) && same_side(q2, p2, p1, p3);
}

// Initializing the roadmap consists of a few steps:
// 1. Inflate the obstacles, to allow for buffer room
// 2. Add the corners of the obstacles to a graph as vertices
// 3. Construct an AABB tree with all the obstacles
// 4. Add the sides of each obstacle to the graph as edges
// 5. Add mutually visible pairs of vertices to graph (checked efficiently with AABB tree)

Roadmap::Private::Private(std::shared_ptr<const ObstacleVector> obstacles, double inflation_amount,
                          const bounds& bounds)
  : bounds_(bounds)
{
  // 1. Inflate all the obstacles
  std::vector<Polygon<>> inflated_polygons;
  for (const std::unique_ptr<Obstacle>& obstacle : *obstacles)
  {
    inflated_polygons.push_back(obstacle->inflate(inflation_amount));
  }

  // 2. Add all the vertices to the boost graph. In addition, store the graph vertices
  // in a format with the same structure as the inflated polygons
  std::vector<std::vector<graph_vertex>> graph_vertices;
  add_vertices_to_graph(inflated_polygons, graph_vertices);

  // 3. Construct an AABB with the inflated obstacles
  for (const Polygon<>& polygon : inflated_polygons)
  {
    tree_.insert(polygon);
  }

  // 4. Add the sides of each obstacle to the graph as edges
  add_obstacle_sides_to_graph(graph_vertices);

  // 5. Add bitantent edges to the graph
  add_bitangent_edges_to_graph(graph_vertices);
}

bool Roadmap::Private::in_bounds(geometry::Vertex vertex)
{
  return bounds_.x_min <= vertex[0] && vertex[0] <= bounds_.x_max && bounds_.y_min <= vertex[1] &&
         vertex[1] <= bounds_.y_max;
}

void Roadmap::Private::add_vertices_to_graph(const std::vector<Polygon<>>& inflated_polygons,
                                             std::vector<std::vector<graph_vertex>>& graph_vertices)
{
  for (int i = 0; i < (int)inflated_polygons.size(); ++i)
  {
    Polygon<> inflated_polygon = inflated_polygons[i];
    std::vector<graph_vertex> obstacle_graph_vertices(inflated_polygon.cols(), NO_VERTEX);

    for (int j = 0; j < inflated_polygon.cols(); ++j)
    {
      if (!in_bounds(inflated_polygon.col(j)))
      {
        continue;
      }
      Point_2 point = commons::to_cgal_point(inflated_polygon.col(j));
      graph_vertex v = boost::add_vertex({ point, i, j }, graph_);
      obstacle_graph_vertices[j] = v;
    }

    graph_vertices.push_back(obstacle_graph_vertices);
  }
}

void Roadmap::Private::add_obstacle_sides_to_graph(const std::vector<std::vector<graph_vertex>>& graph_vertices)
{
  // Add all the edges of the polygons to the graph
  for (const std::vector<graph_vertex>& obstacle_vertices : graph_vertices)
  {
    int n_vertices = (int)obstacle_vertices.size();
    for (int i = 0; i < n_vertices; ++i)
    {
      graph_vertex v1 = obstacle_vertices[i];
      graph_vertex v2 = obstacle_vertices[(i + 1) % n_vertices];

      if (v1 == NO_VERTEX || v2 == NO_VERTEX)
      {
        continue;
      }

      const VertexProperty& prop1 = graph_[v1];
      const VertexProperty& prop2 = graph_[v2];
      PolygonTree::VertexLookup lookup1 = { prop1.obstacle_idx, prop1.point_idx };
      PolygonTree::VertexLookup lookup2 = { prop2.obstacle_idx, prop2.point_idx };
      if (!tree_.visible_polygon_edge(lookup1, lookup2))
      {
        continue;
      }

      double weight = distance(graph_[v1].point, graph_[v2].point);
      boost::add_edge(v1, v2, weight, graph_);
    }
  }
}

void Roadmap::Private::add_bitangent_edges_to_graph(const std::vector<std::vector<graph_vertex>>& graph_vertices)
{
  for (int i = 0; i < (int)graph_vertices.size(); ++i)
  {
    const std::vector<graph_vertex>& obstacle_vertices_1 = graph_vertices[i];

    for (int j = 0; j < i; ++j)
    {
      const std::vector<graph_vertex>& obstacle_vertices_2 = graph_vertices[j];

      // Iterate over all the pairs of points
      for (const graph_vertex& v1 : obstacle_vertices_1)
      {
        if (v1 == NO_VERTEX)
        {
          continue;
        }
        for (const graph_vertex& v2 : obstacle_vertices_2)
        {
          if (v2 == NO_VERTEX)
          {
            continue;
          }
          const VertexProperty& prop1 = graph_[v1];
          const VertexProperty& prop2 = graph_[v2];

          PolygonTree::VertexLookup lookup1 = { prop1.obstacle_idx, prop1.point_idx };
          PolygonTree::VertexLookup lookup2 = { prop2.obstacle_idx, prop2.point_idx };

          if (!tree_.visible(lookup1, lookup2))
          {
            continue;
          }

          if (!bitangent(graph_vertices, prop1, prop2))
          {
            continue;
          }

          double weight = distance(prop1.point, prop2.point);
          boost::add_edge(v1, v2, weight, graph_);
        }
      }
    }
  }
}

// This function works as follows
// 1. Add the endpoints to the graph, connecting edges to vertices that are visible
// 2. Call dijkstra's algorithm to find the shortest path
// 3. Remove the endpoints
void Roadmap::Private::shortestPath(const Vertex& v1, const Vertex& v2, std::vector<Vertex>& path)
{
  auto [graph_v1, graph_v2] = add_endpoints_to_graph(v1, v2);

  compute_dijkstra(graph_v1, graph_v2, path);

  remove_endpoints_from_graph(graph_v1, graph_v2);
}

std::pair<graph_vertex, graph_vertex> Roadmap::Private::add_endpoints_to_graph(const Vertex& v1, const Vertex& v2)
{
  Vertex safe_v1 = tree_.closest_point_outside_polygons(v1);
  Vertex safe_v2 = tree_.closest_point_outside_polygons(v2);

  graph_vertex graph_v1 = add_endpoint_to_graph(safe_v1);
  graph_vertex graph_v2 = add_endpoint_to_graph(safe_v2);

  if (tree_.visible(safe_v1, safe_v2))
  {
    double weight = distance(graph_[graph_v1].point, graph_[graph_v2].point);
    boost::add_edge(graph_v1, graph_v2, weight, graph_);
  }

  return { graph_v1, graph_v2 };
}

void Roadmap::Private::remove_endpoints_from_graph(graph_vertex v1, graph_vertex v2)
{
  boost::clear_vertex(v1, graph_);
  boost::clear_vertex(v2, graph_);
  boost::remove_vertex(v1, graph_);
  boost::remove_vertex(v2, graph_);
}

graph_vertex Roadmap::Private::add_endpoint_to_graph(const Vertex& end_vertex)
{
  Point_2 pt = commons::to_cgal_point(end_vertex);
  graph_vertex graph_end_vertex = boost::add_vertex({ pt, -1, -1 }, graph_);

  for (const graph_vertex& v : boost::make_iterator_range(boost::vertices(graph_)))
  {
    if (graph_[v].obstacle_idx == -1)
    {
      continue;
    }
    PolygonTree::VertexLookup lookup = { graph_[v].obstacle_idx, graph_[v].point_idx };
    if (!tree_.visible(lookup, end_vertex))
    {
      continue;
    }
    // possible optimization: only add if segment is tangent
    double weight = distance(graph_[v].point, graph_[graph_end_vertex].point);
    boost::add_edge(v, graph_end_vertex, weight, graph_);
  }
  return graph_end_vertex;
}

void Roadmap::Private::compute_dijkstra(graph_vertex v1, graph_vertex v2, std::vector<Vertex>& path)
{
  int n = boost::num_vertices(graph_);
  std::vector<double> dist_map(n);        // stores all the distances
  std::vector<graph_vertex> pred_map(n);  // stores the predecessor nodes

  boost::dijkstra_shortest_paths(
      graph_, v1,
      boost::distance_map(boost::make_iterator_property_map(dist_map.begin(), boost::get(boost::vertex_index, graph_)))
          .predecessor_map(
              boost::make_iterator_property_map(pred_map.begin(), boost::get(boost::vertex_index, graph_))));

  if (pred_map[v2] == v2)
  {
    // there was no shortest path
    std::cerr << "No shortest path found!" << std::endl;
    return;
  }

  graph_vertex curr = v2;
  path.push_back(commons::from_cgal_point(graph_[curr].point));
  while (curr != v1)
  {
    curr = pred_map[curr];
    path.push_back(commons::from_cgal_point(graph_[curr].point));
  }
  std::reverse(path.begin(), path.end());
}

bool Roadmap::Private::pointIsSafe(const geometry::Vertex& v)
{
  return tree_.point_is_safe(v);
}

void Roadmap::Private::getVisualizationInfo(VisualizationInfo& vis_info) const
{
  std::unordered_map<graph_vertex, int> vertex_lookup;
  int idx = 0;

  for (graph_vertex v : boost::make_iterator_range(boost::vertices(graph_)))
  {
    Vertex point = commons::from_cgal_point(graph_[v].point);
    vis_info.vertices.push_back(point);
    vertex_lookup[v] = idx;
    idx++;
  }

  for (graph_edge e : boost::make_iterator_range(boost::edges(graph_)))
  {
    graph_vertex v1 = boost::source(e, graph_);
    graph_vertex v2 = boost::target(e, graph_);
    vis_info.edges.emplace_back(vertex_lookup[v1], vertex_lookup[v2]);
  }
}

std::vector<std::vector<geometry::Vertex>> Roadmap::Private::getPolygonVertices() const
{
  return tree_.getPolygonVertices();
}

Roadmap::Roadmap(std::shared_ptr<const ObstacleVector> obstacles, double inflation_amount, bounds bounds)
  : impl_(std::make_unique<Private>(obstacles, inflation_amount, bounds))
{
}

Roadmap::~Roadmap()
{
}

void Roadmap::shortestPath(const Vertex& v1, const Vertex& v2, std::vector<Vertex>& path)
{
  return impl_->shortestPath(v1, v2, path);
}

bool Roadmap::pointIsSafe(const geometry::Vertex& v)
{
  return impl_->pointIsSafe(v);
}

void Roadmap::getVisualizationInfo(VisualizationInfo& vis_info) const
{
  return impl_->getVisualizationInfo(vis_info);
}

std::vector<std::vector<geometry::Vertex>> Roadmap::getPolygonVertices() const
{
  return impl_->getPolygonVertices();
}

}  // namespace crs_planning
