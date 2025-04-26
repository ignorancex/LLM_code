#include "terrain_analyzer/TerrainAnalyzer.hpp"
#include <ros/ros.h>

using namespace terrain_analyzer;

int main( int argc, char * argv[] )
{ 
    ros::init(argc, argv, "terrain_analyzer_node");
    ros::NodeHandle nh("~");

    TerrainAnalyzer terrain_analyzer;
    terrain_analyzer.init(nh);

    ros::spin();

    return 0;
}
