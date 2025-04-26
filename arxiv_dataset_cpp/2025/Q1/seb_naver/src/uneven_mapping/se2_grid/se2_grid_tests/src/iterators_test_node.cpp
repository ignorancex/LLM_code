#include <ros/ros.h>
#include <se2_grid_ros/se2_grid_ros.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

using namespace se2_grid;
using namespace ros;

SE2Grid map({"elevation"}, {false});
ros::Publisher map_pub;
// Eigen::Vector2d off_set(0.0, 0.0);
Eigen::Vector2d off_set(-3.0, -3.0);

void publish()
{
    se2_grid_msgs::SE2Grid message;
    SE2GridRosConverter::toMessage(map, message);
    map_pub.publish(message);
    ROS_DEBUG("Grid map (timestamp %f) published.", message.info.header.stamp.toSec());
}

void demoSE2GridIterator()
{
  ROS_INFO("Running se2 grid iterator demo.");
  map["elevation"][0].setConstant(0.0);
  publish();

  Eigen::MatrixXf& data = map["elevation"][0];
  for (SE2GridIterator iterator(map); !iterator.isPastEnd(); ++iterator)
  {
    const int i = iterator.getLinearIndex();
    data(i) = 1.0;
    publish();
    ros::Duration duration(0.01);
    duration.sleep();
  }
  ros::Duration duration(1.0);
  duration.sleep();
}

void demoSubmapIterator()
{
    ROS_INFO("Running sub map iterator demo.");
    map["elevation"][0].setConstant(0.0);
    publish();

    Eigen::Array2i submapStartIndex(3, 5);
    Eigen::Array2i submapBufferSize(12, 7);

    for (SubmapIterator iterator(map, submapStartIndex, submapBufferSize);
        !iterator.isPastEnd(); ++iterator)
    {
        map.at("elevation", *iterator) = 1.0;
        publish();
        ros::Duration duration(0.02);
        duration.sleep();
    }

    ros::Duration duration(1.0);
    duration.sleep();
}

void demoCircleIterator()
{
    ROS_INFO("Running circle iterator demo.");
    map["elevation"][0].setConstant(0.0);
    publish();

    // Eigen::Vector2d center(5.0, 5.0);
    Eigen::Vector2d center(0.0, -1.5);
        center += off_set;
    double radius = 4.0;

    for (CircleIterator iterator(map, center, radius);
        !iterator.isPastEnd(); ++iterator)
    {
        map.at("elevation", *iterator) = 1.0;
        publish();
        ros::Duration duration(0.02);
        duration.sleep();
    }

    ros::Duration duration(1.0);
    duration.sleep();
}

void demoSpiralIterator()
{
    ROS_INFO("Running spiral iterator demo.");
    map["elevation"][0].setConstant(0.0);
    publish();

    Eigen::Vector2d center(0.0, -1.5);
        center += off_set;
    // double radius = 4.0;
    double radius = 7.0;

    for (SpiralIterator iterator(map, center, radius);
        !iterator.isPastEnd(); ++iterator)
    {
        map.at("elevation", *iterator) = 1.0;
        publish();
        ros::Duration duration(0.02);
        duration.sleep();
    }

    ros::Duration duration(1.0);
    duration.sleep();
}

void demoEllipseIterator()
{
    ROS_INFO("Running ellipse iterator demo.");
    map["elevation"][0].setConstant(0.0);
    publish();

    Eigen::Vector2d center(0.0, -1.5);
    center += off_set;
    Eigen::Array2d length(4.5, 9.0);

    for (EllipseIterator iterator(map, center, length, M_PI_4);
        !iterator.isPastEnd(); ++iterator)
    {
        map.at("elevation", *iterator) = 1.0;
        publish();
        ros::Duration duration(0.02);
        duration.sleep();
    }

    ros::Duration duration(1.0);
    duration.sleep();
}

int main(int argc, char** argv)
{
    init(argc, argv, "iterators_test");
    NodeHandle nh("~");

    map.setrameId("world");
    map.initGeometry(Eigen::Array2d(10.0, 10.0), 0.5, 2.0, Eigen::Vector2d(0.0, 0.0));
    ROS_INFO("Created map with size %f x %f m x %f rad (%i x %i x %i cells).\n The center of the map is located at (%f, %f) in the %s frame.",
    map.getLengthPos().x(), map.getLengthPos().y(), map.getLengthYaw(), 
    map.getSizePos()(0), map.getSizePos()(1), map.getSizeYaw(), 
    map.getPosition().x(), map.getPosition().y(), map.getFrameId().c_str());
    // map["elevation"][0].setConstant(0.0);
    map.move(off_set);
    ros::Time a = ros::Time::now();
    std::cout<<(ros::Time::now() - a).toSec()*1000<<"ms"<<std::endl;
    map["elevation"][0].setConstant(0.0);

    map_pub = nh.advertise<se2_grid_msgs::SE2Grid>("/se2_grid", 1, true);
    
    ros::Duration duration(2.0);
    duration.sleep();

    publish();
    duration.sleep();

    demoSE2GridIterator();
    demoSubmapIterator();
    demoCircleIterator();
    demoSpiralIterator();
    demoEllipseIterator();
    
    ros::requestShutdown();

    return 0;
}
