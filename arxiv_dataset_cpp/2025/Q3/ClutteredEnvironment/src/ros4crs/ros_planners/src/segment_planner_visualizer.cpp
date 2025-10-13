#include <ros_planners/visualizers/segment_planner_visualizer.h>
#include <segment_planner/segment_planner.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "commons/geometry_utils.h"

using crs_planning::Roadmap;
using crs_planning::SegmentPlanner;
using geometry::Vertex;

SegmentPlannerVisualizer::SegmentPlannerVisualizer(ros::NodeHandle nh_private,
                                                   std::shared_ptr<SegmentPlanner> segment_planner)
  : planner_(segment_planner)
{
  // publish to the same topic as controller visualization
  roadmap_publisher_ = nh_private.advertise<visualization_msgs::Marker>("visualization/roadmap", 100);
  inflated_obstacles_publisher_ =
      nh_private.advertise<visualization_msgs::MarkerArray>("visualization/inflated_obstacles", 100);
  shortest_path_publisher_ = nh_private.advertise<visualization_msgs::Marker>("visualization/shortest_path", 100);

  // TODO: Find better method to see when rviz is ready
  while (inflated_obstacles_publisher_.getNumSubscribers() < 1)
  {
    if (!ros::ok())
    {
      abort();
    }
    sleep(1);
  }
}

void SegmentPlannerVisualizer::visualizeInflatedObstacles()
{
  // Create the marker for visualizing the obstacles.
  visualization_msgs::MarkerArray obstacle_markers;

  std::vector<std::vector<geometry::Vertex>> obstacles = planner_->getInflatedObstacles();

  int marker_id = 0;

  // Visualize the obstacles as a triangle list. Not sure if there's a more direct way
  for (const auto& obstacle : obstacles)
  {
    // Triangulate the polygon
    const int N_SIDES = 8;  // number of sizes in an inflated rhombus
    std::array<geometry::Vertex, N_SIDES> vertex_array;
    std::copy(obstacle.begin(), obstacle.end(), vertex_array.begin());
    geometry::Polygon<N_SIDES> polygon = geometry::create<N_SIDES>(vertex_array);
    std::vector<geometry::Triangle> triangles;
    geometry::triangulate(polygon, triangles);

    // Initialize the obstacle marker
    visualization_msgs::Marker obstacle_marker;
    obstacle_marker.header.frame_id = "crs_frame";
    obstacle_marker.header.stamp = ros::Time::now();
    obstacle_marker.id = marker_id++;
    obstacle_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    obstacle_marker.action = visualization_msgs::Marker::ADD;

    obstacle_marker.scale.x = 1;
    obstacle_marker.scale.y = 1;
    obstacle_marker.scale.z = 1;
    obstacle_marker.pose.orientation.w = 1;

    obstacle_marker.color.r = 0.3;
    obstacle_marker.color.g = 0.6;
    obstacle_marker.color.b = 0.6;
    obstacle_marker.color.a = 0.5;

    // Add the triangles to the marker
    for (const geometry::Triangle& triangle : triangles)
    {
      for (const geometry::Vertex& vertex : { triangle.v1, triangle.v2, triangle.v3 })
      {
        geometry_msgs::Point pt;
        pt.x = vertex[0];
        pt.y = vertex[1];
        pt.z = -0.001;  // slightly behind the actual obstacle
        obstacle_marker.points.push_back(pt);
        obstacle_marker.colors.push_back(obstacle_marker.color);
      }
    }

    obstacle_markers.markers.push_back(obstacle_marker);
  }

  inflated_obstacles_publisher_.publish(obstacle_markers);
}

void SegmentPlannerVisualizer::visualizeRoadmap()
{
  // Get the roadmap info
  Roadmap::VisualizationInfo vis_info;
  planner_->getRoadMapVisualization(vis_info);

  visualization_msgs::Marker roadmap;
  roadmap.header.frame_id = "crs_frame";
  roadmap.header.stamp = ros::Time::now();
  roadmap.ns = "roadmap";
  roadmap.id = 50;
  roadmap.action = visualization_msgs::Marker::ADD;
  roadmap.type = visualization_msgs::Marker::LINE_LIST;

  roadmap.scale.x = .02;
  roadmap.pose.orientation.w = 1.0;

  roadmap.color.r = .5;
  roadmap.color.g = .5;
  roadmap.color.b = .5;
  roadmap.color.a = .5;

  for (const auto& edge : vis_info.edges)
  {
    const Vertex& v1 = vis_info.vertices[edge.first];
    const Vertex& v2 = vis_info.vertices[edge.second];

    geometry_msgs::Point p1;
    p1.x = v1[0];
    p1.y = v1[1];
    p1.z = 0;
    roadmap.points.push_back(p1);

    geometry_msgs::Point p2;
    p2.x = v2[0];
    p2.y = v2[1];
    p2.z = 0;
    roadmap.points.push_back(p2);
  }

  roadmap_publisher_.publish(roadmap);
}

void SegmentPlannerVisualizer::visualizePath()
{
  std::vector<geometry::Vertex> plan;
  planner_->getPlanVisualization(plan);

  visualization_msgs::Marker shortest_path;
  shortest_path.header.frame_id = "crs_frame";
  shortest_path.header.stamp = ros::Time::now();
  shortest_path.ns = "shortest_path";
  shortest_path.id = 51;
  shortest_path.action = visualization_msgs::Marker::ADD;
  shortest_path.type = visualization_msgs::Marker::LINE_STRIP;

  shortest_path.scale.x = .03;
  shortest_path.pose.orientation.w = 1.0;

  shortest_path.color.r = 1.0;
  shortest_path.color.g = 0.0;
  shortest_path.color.b = 1.0;
  shortest_path.color.a = .5;

  for (const auto& pt : plan)
  {
    geometry_msgs::Point msg;
    msg.x = pt[0];
    msg.y = pt[1];
    msg.z = 0.0;
    shortest_path.points.push_back(msg);
  }

  shortest_path_publisher_.publish(shortest_path);
}

void SegmentPlannerVisualizer::visualize()
{
  if (!planner_->setupComplete())
  {
    // The planner is not ready. Nothing to show here
    return;
  }

  if (!roadmap_visualized_)
  {
    visualizeRoadmap();
    visualizeInflatedObstacles();
    roadmap_visualized_ = true;
  }

  visualizePath();
}
