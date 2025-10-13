// #include <ros_crs_utils/parameter_io.h>

// namespace parameter_io
// {
//     using crs_controls::Obstacle;
//     using crs_controls::Rectangle;
//     using crs_controls::Rhombus;
//     using crs_controls::ObstacleVector;

//     std::unique_ptr<const ObstacleVector> loadObstaclesFromParams(const ros::NodeHandle& nh) {
//         std::cout << "LOADING OBSTACLES FROM: " << nh.getNamespace() << std::endl;

//         ObstacleVector obstacles;

//         // No fancy parameter reading syntax for getting a list of objects
//         XmlRpc::XmlRpcValue rectangle_io;
//         if (nh.getParam("rectangles", rectangle_io))
//         {
//             ROS_ASSERT(rectangle_io.getType() == XmlRpc::XmlRpcValue::TypeArray);

//             for (int i = 0; i < rectangle_io.size(); ++i) {
//                 double xLowerLeft = static_cast<double>(rectangle_io[i]["xLowerLeft"]);
//                 double yLowerLeft = static_cast<double>(rectangle_io[i]["yLowerLeft"]);
//                 double width = static_cast<double>(rectangle_io[i]["width"]);
//                 double height = static_cast<double>(rectangle_io[i]["height"]);

//                 std::unique_ptr<Rectangle> rectangle = std::make_unique<Rectangle>(xLowerLeft, yLowerLeft, width,
//                 height); obstacles.push_back(std::move(rectangle));
//             }
//         }

//         XmlRpc::XmlRpcValue rhombus_io;
//         if (nh.getParam("rhombuses", rhombus_io))
//         {
//             ROS_ASSERT(rhombus_io.getType() == XmlRpc::XmlRpcValue::TypeArray);

//             for (int i = 0; i < rhombus_io.size(); ++i) {
//                 double xCenter = static_cast<double>(rhombus_io[i]["xCenter"]);
//                 double yCenter = static_cast<double>(rhombus_io[i]["yCenter"]);
//                 double width = static_cast<double>(rhombus_io[i]["width"]);
//                 double height = static_cast<double>(rhombus_io[i]["height"]);
//                 double theta_rad = static_cast<double>(rhombus_io[i]["theta_rad"]);

//                 std::unique_ptr<Rhombus> rhombus = std::make_unique<Rhombus>(xCenter, yCenter, width, height,
//                 theta_rad); obstacles.push_back(std::move(rhombus));
//             }
//         }

//         return std::make_unique<const ObstacleVector>(std::move(obstacles));
//     }
// }
