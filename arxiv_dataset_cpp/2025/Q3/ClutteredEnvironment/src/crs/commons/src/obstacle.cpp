#include <commons/geometry_utils.h>

#include <vector>
#include <Eigen/Core>

#include <commons/obstacle.h>

namespace crs_controls
{

using geometry::Polygon;
using geometry::Triangle;
using geometry::Vertex;

// Create an axis-aligned octagon. the x-coordinates are at {xlo, xlo + dx, xhi - dx, xhi}
// The y-coordinates follow a similar pattern.
static Polygon<> axisAlignedOctagon(double xlo, double xhi, double dx, double ylo, double yhi, double dy)
{
  // Start at the bottom left, go clockwise
  Polygon<8> octagon;
  octagon.col(0) = Vertex(xlo + dx, ylo);
  octagon.col(1) = Vertex(xlo, ylo + dy);
  octagon.col(2) = Vertex(xlo, yhi - dy);
  octagon.col(3) = Vertex(xlo + dx, yhi);
  octagon.col(4) = Vertex(xhi - dx, yhi);
  octagon.col(5) = Vertex(xhi, yhi - dy);
  octagon.col(6) = Vertex(xhi, ylo + dy);
  octagon.col(7) = Vertex(xhi - dx, ylo);

  return octagon;
}

template <int N_COLS>
static inline Polygon<N_COLS> toVertexMatrix(const std::vector<Vertex>& vertices)
{
  Polygon<N_COLS> matrix;
  for (int i = 0; i < matrix.cols(); ++i)
  {
    matrix.col(i) = vertices[i];
  }
  return matrix;
}

template <int N_COLS>
static inline void fromVertexMatrix(const Polygon<N_COLS> matrix, std::vector<Vertex>& vertices)
{
  for (int i = 0; i < matrix.cols(); ++i)
  {
    vertices.push_back(matrix.col(i));
  }
}

/////////////////////////////////// Rectangles ///////////////////////////////////

Rectangle::Rectangle(double x, double y, double width, double height) : width_(width), height_(height)
{
  lower_left_ << x, y;
  std::vector<Vertex> vertices = { { x, y }, { x, y + height }, { x + width, y + height }, { x + width, y } };
  vertex_matrix_ = toVertexMatrix<4>(vertices);
}

void Rectangle::getVertices(std::vector<Vertex>& vertices) const
{
  fromVertexMatrix(vertex_matrix_, vertices);
}

Polygon<> Rectangle::inflate(double amount) const
{
  double xlo = lower_left_[0] - amount;
  double xhi = lower_left_[0] + width_ + amount;
  double ylo = lower_left_[1] - amount;
  double yhi = lower_left_[1] + height_ + amount;
  double dx = amount * (std::sqrt(2) - 1);
  double dy = dx;

  return axisAlignedOctagon(xlo, xhi, dx, ylo, yhi, dy);
}

Polygon<Eigen::Dynamic> Rectangle::getVertexMatrix() const
{
  return vertex_matrix_;
}

double Rectangle::getWidth() const
{
  return width_;
}

double Rectangle::getHeight() const
{
  return height_;
}

geometry::Vertex Rectangle::getLowerLeft() const
{
  return lower_left_;
}

void Rectangle::triangulate(std::vector<Triangle>& triangles) const
{
  triangles.emplace_back(vertex_matrix_.col(0), vertex_matrix_.col(1), vertex_matrix_.col(2));
  triangles.emplace_back(vertex_matrix_.col(2), vertex_matrix_.col(3), vertex_matrix_.col(0));
}

/////////////////////////////////// Rhombuses ///////////////////////////////////

Rhombus::Rhombus(double x, double y, double theta_rad) : width_(WIDTH), height_(HEIGHT), theta_rad_(theta_rad)
{
  center_ << x, y;

  std::vector<Vertex> vertices = {
    { width_ / 2, 0 },
    { 0, height_ / 2 },
    { -width_ / 2, 0 },
    { 0, -height_ / 2 },
  };

  Polygon<4> base_matrix = toVertexMatrix<4>(vertices);
  vertex_matrix_ = geometry::transform(base_matrix, theta_rad, x, y);
}

void Rhombus::getVertices(std::vector<Vertex>& vertices) const
{
  fromVertexMatrix(vertex_matrix_, vertices);
}

Polygon<> Rhombus::inflate(double amount) const
{
  double w = width_ / 2;
  double h = height_ / 2;
  double l = std::hypot(w, h);
  double xdiff = amount / h * (l - w);
  double ydiff = amount / w * (l - h);

  double xlo = -w - amount;
  double xhi = +w + amount;
  double ylo = -h - amount;
  double yhi = +h + amount;
  double dx = xhi - xdiff;
  double dy = yhi - ydiff;
  Polygon<8> base_polygon = axisAlignedOctagon(xlo, xhi, dx, ylo, yhi, dy);

  return geometry::transform(base_polygon, theta_rad_, center_[0], center_[1]);
}

Polygon<Eigen::Dynamic> Rhombus::getVertexMatrix() const
{
  return vertex_matrix_;
}

void Rhombus::triangulate(std::vector<Triangle>& triangles) const
{
  triangles.emplace_back(vertex_matrix_.col(0), vertex_matrix_.col(1), vertex_matrix_.col(2));
  triangles.emplace_back(vertex_matrix_.col(2), vertex_matrix_.col(3), vertex_matrix_.col(0));
}

}  // namespace crs_controls
