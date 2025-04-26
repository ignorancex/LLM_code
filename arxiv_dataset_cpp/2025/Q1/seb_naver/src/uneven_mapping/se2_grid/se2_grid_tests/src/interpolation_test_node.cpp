#include <ros/ros.h>
#include <se2_grid_ros/se2_grid_ros.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

using namespace se2_grid;
using namespace ros;

int yaw_size = 1;
bool has_so2 = true;

SE2Grid map({"elevation"}, {has_so2});
SE2Grid map_interp({"elevation_nearest", "elevation_linear"}, {has_so2, has_so2});

ros::Publisher map_pub, map_interp_pub;

// Eigen::Vector2d off_set(0.0, 0.0);
Eigen::Vector2d off_set(-3.0, -3.0);

void publishMap()
{
    se2_grid_msgs::SE2Grid message;
    SE2GridRosConverter::toMessage(map, message);
    map_pub.publish(message);
    ROS_DEBUG("Grid map (timestamp %f) published.", message.info.header.stamp.toSec());
}

void publishInterp()
{
    se2_grid_msgs::SE2Grid message;
    SE2GridRosConverter::toMessage(map_interp, message);
    map_interp_pub.publish(message);
    ROS_DEBUG("Grid map_interp (timestamp %f) published.", message.info.header.stamp.toSec());
}

void demoInterp()
{
    ROS_INFO("Running nearest interpolation demo.");

    publishMap();

    for (SE2GridIterator iterator(map_interp); !iterator.isPastEnd(); ++iterator)
    {
        Eigen::Array3i idx(*iterator);

        yaw_size = map_interp.getSizeYaw();
        for (int i=0; i<yaw_size; i++)
        {
            idx(2) = i;
            Eigen::Vector3d pos, grad;
            map_interp.index2Pos(idx, pos);
            map_interp.at("elevation_linear", idx) = map.getValue("elevation", pos, InterpolationMethods::INTER_LINEAR);
        }
    }
    publishInterp();
    ros::Duration duration(3.0);
    duration.sleep();
}

int main(int argc, char** argv)
{
    init(argc, argv, "iterators_test");
    NodeHandle nh("~");

    map.setrameId("world");
    map.initGeometry(Eigen::Array2d(10.0, 10.0), 1.0, 1.0, Eigen::Vector2d(0.0, 0.0));
    ROS_INFO("Created map with size %f x %f m x %f rad (%i x %i x %i cells).\n The center of the map is located at (%f, %f) in the %s frame.",
            map.getLengthPos().x(), map.getLengthPos().y(), map.getLengthYaw(), 
            map.getSizePos()(0), map.getSizePos()(1), map.getSizeYaw(), 
            map.getPosition().x(), map.getPosition().y(), map.getFrameId().c_str());
    map.move(off_set);
    ros::Time a = ros::Time::now();
    std::cout<<(ros::Time::now() - a).toSec()*1000<<"ms"<<std::endl;
    yaw_size = map.getSizeYaw();

    // map["elevation"][0].setConstant(1.0);
    // Eigen::MatrixXf p;
    // p.resize(map["elevation"][0].rows(), map["elevation"][0].cols());
    // p.setConstant(1.0);
    // for (int i=0; i<yaw_size; i++)
    // {
    //     if (i < yaw_size / 2)
    //         map["elevation"][i] = map["elevation"][0] + p * i;
    //     else
    //         map["elevation"][i] = map["elevation"][0] + p * (yaw_size - i);
    // }

    for (int i=0; i<yaw_size; i++)
        map["elevation"][i].setRandom();

    map_interp.setrameId("world");
    map_interp.initGeometry(Eigen::Array2d(10.0, 10.0), 0.2, 0.2, Eigen::Vector2d(0.0, 0.0));
    ROS_INFO("Created map with size %f x %f m x %f rad (%i x %i x %i cells).\n The center of the map is located at (%f, %f) in the %s frame.",
            map_interp.getLengthPos().x(), map_interp.getLengthPos().y(), map_interp.getLengthYaw(), 
            map_interp.getSizePos()(0), map_interp.getSizePos()(1), map_interp.getSizeYaw(), 
            map_interp.getPosition().x(), map_interp.getPosition().y(), map_interp.getFrameId().c_str());
    map_interp.move(off_set);

    map_pub = nh.advertise<se2_grid_msgs::SE2Grid>("/se2_grid", 1, true);
    map_interp_pub = nh.advertise<se2_grid_msgs::SE2Grid>("/se2_grid_interp", 1, true);

    demoInterp();
    
    ros::spin();

    return 0;
}
