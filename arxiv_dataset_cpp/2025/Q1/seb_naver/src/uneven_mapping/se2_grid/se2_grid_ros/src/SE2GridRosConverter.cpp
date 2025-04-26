#include "se2_grid_ros/SE2GridRosConverter.hpp"
#include "se2_grid_ros/SE2GridMsgHelpers.hpp"

// ROS
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// STL
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;
using namespace Eigen;

namespace se2_grid {

SE2GridRosConverter::SE2GridRosConverter()
{
}

SE2GridRosConverter::~SE2GridRosConverter()
{
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::SE2Grid& message, se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& /*layers*/)
{
  SE2Grid.initGeometry(Eigen::Array2d(message.info.length_x, message.info.length_y), 
                       message.info.pos_resolution, message.info.yaw_resolution,
                       Eigen::Vector2d(message.info.pose.position.x, message.info.pose.position.y));

  // Copy non-basic layers.
  unsigned int data_idx = 0;
  for (unsigned int i = 0u; i < message.layers.size(); ++i)
  {
    // TODO Could we use the data mapping (instead of copying) method here?
    std::vector<Eigen::MatrixXf> datas;
    int size_yaw = SE2Grid.getSizeYaw();
    if (message.has_so2[i])
    {
      for (int j=0; j<size_yaw; j++)
      {
        Eigen::MatrixXf d;
        if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
        {
          return false;
        }
        datas.emplace_back(d);
      }
      SE2Grid.add(message.layers[i], datas);
    }
    else
    {
      Eigen::MatrixXf d;
      if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
      {
        return false;
      }
      datas.emplace_back(d);
      SE2Grid.add(message.layers[i], datas);
    }
  }

  SE2Grid.setStartIndex(Eigen::Array2i(message.start_index_row, message.start_index_col));

  return true;
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::SE2Grid& message, se2_grid::SE2Grid& SE2Grid)
{
  return fromMessage(message, SE2Grid, std::vector<std::string>());
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, se2_grid_msgs::SE2Grid& message)
{
  toMessage(SE2Grid, SE2Grid.getLayers(), message);
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers,
                                    se2_grid_msgs::SE2Grid& message)
{
  message.info.header.stamp = ros::Time::now();
  message.info.header.frame_id = SE2Grid.getFrameId();
  message.info.pos_resolution = SE2Grid.getResolutionPos();
  message.info.yaw_resolution = SE2Grid.getResolutionYaw();
  message.info.length_x = SE2Grid.getLengthPos().x();
  message.info.length_y = SE2Grid.getLengthPos().y();
  message.info.length_yaw = SE2Grid.getLengthYaw();
  message.info.pose.position.x = SE2Grid.getPosition().x();
  message.info.pose.position.y = SE2Grid.getPosition().y();
  message.info.pose.position.z = 0.0;
  message.info.pose.orientation.x = 0.0;
  message.info.pose.orientation.y = 0.0;
  message.info.pose.orientation.z = 0.0;
  message.info.pose.orientation.w = 1.0;

  message.layers = layers;

  message.data.clear();
  message.has_so2.clear();
  for (size_t i=0; i<layers.size(); i++)
  {
    std_msgs::Float32MultiArray dataArray;
    if (SE2Grid.hasSO2(layers[i]))
      message.has_so2.emplace_back(true);
    else
      message.has_so2.emplace_back(false);
    
    std::vector<Eigen::MatrixXf> l = SE2Grid[layers[i]];
    for (size_t j=0; j<l.size(); j++)
    {
      matrixEigenCopyToMultiArrayMessage(l[j], dataArray);
      message.data.push_back(dataArray);
    }
  }

  message.start_index_row = SE2Grid.getStartIndex()(0);
  message.start_index_col = SE2Grid.getStartIndex()(1);
}

bool SE2GridRosConverter::saveToBag(const se2_grid::SE2Grid& SE2Grid, const std::string& pathToBag,
                                    const std::string& topic)
{
  se2_grid_msgs::SE2Grid message;
  toMessage(SE2Grid, message);
  ros::Time time = ros::Time::now();

  if (!time.isValid() || time.isZero()) {
    if (!ros::Time::isValid()) ros::Time::init();
    time = ros::Time::now();
  }

  rosbag::Bag bag;
  bag.open(pathToBag, rosbag::bagmode::Write);
  bag.write(topic, time, message);
  bag.close();
  return true;
}

bool SE2GridRosConverter::loadFromBag(const std::string& pathToBag, const std::string& topic,
                                      se2_grid::SE2Grid& SE2Grid)
{
  rosbag::Bag bag;
  bag.open(pathToBag, rosbag::bagmode::Read);
  rosbag::View view(bag, rosbag::TopicQuery(topic));

  bool isDataFound = false;
  for (const auto& messageInstance : view) {
    se2_grid_msgs::SE2Grid::ConstPtr message = messageInstance.instantiate<se2_grid_msgs::SE2Grid>();
    if (message != NULL) {
      fromMessage(*message, SE2Grid);
      isDataFound = true;
    } else {
      bag.close();
      ROS_WARN("Unable to load data from ROS bag.");
      return false;
    }
  }
  if (!isDataFound) ROS_WARN_STREAM("No data under the topic '" << topic << "' was found.");
  bag.close();
  return true;
}

} /* namespace */
