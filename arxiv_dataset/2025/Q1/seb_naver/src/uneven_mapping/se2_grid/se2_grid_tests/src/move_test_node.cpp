#include <ros/ros.h>
#include <se2_grid_ros/se2_grid_ros.hpp>

using namespace se2_grid;
using namespace ros;

int main(int argc, char** argv)
{
  init(argc, argv, "move_test");
  NodeHandle nh("~");
  Publisher publisher = nh.advertise<se2_grid_msgs::SE2Grid>("/se2_grid", 1, true);

  SE2Grid map({"elevation", "second", "third", "forth"}, {false, true, true, false});
  map.setrameId("world");
  map.initGeometry(Eigen::Array2d(30.0, 30.0), 1.0, 2.0, Eigen::Vector2d(0.0, 0.0));
  ROS_INFO("Created map with size %f x %f m x %f rad (%i x %i x %i cells).\n The center of the map is located at (%f, %f) in the %s frame.",
    map.getLengthPos().x(), map.getLengthPos().y(), map.getLengthYaw(), 
    map.getSizePos()(0), map.getSizePos()(1), map.getSizeYaw(), 
    map.getPosition().x(), map.getPosition().y(), map.getFrameId().c_str());
  int yaw_num = map.getSizeYaw();
  map["elevation"][0].setRandom();
  map["forth"][0].setRandom();
  for (int i=0; i<yaw_num; i++)
  {
    map["second"][i].setRandom();
    map["third"][i].setRandom();
  }
  
  ROS_INFO("Begin test.");
  while (nh.ok())
  {
    SE2Grid tempMap(map);
    Rate rate(10.0);
    ros::Time startTime = ros::Time::now();
    ros::Duration duration(0.0);

    while (duration <= ros::Duration(10.0))
    {
      ros::Time time = ros::Time::now();
      duration = time - startTime;

      const double t = duration.toSec();
      Eigen::Vector2d newPosition = 10.0 * Eigen::Vector2d(cos(t), sin(t));

      tempMap.move(newPosition);
      for (auto& l:tempMap.getLayers())
        for (auto& d:tempMap[l])
          for (int i=0; i<d.rows(); i++)
            for (int j=0; j<d.cols(); j++)
              if (isnan(d(i, j)))
                d(i, j) = 0.0;

      se2_grid_msgs::SE2Grid message;
      SE2GridRosConverter::toMessage(tempMap, message);
      publisher.publish(message);

      rate.sleep();
    }
  }

  return 0;
}
