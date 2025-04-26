#include "se2_grid_ros/SE2GridMsgHelpers.hpp"

// Boost
#include <boost/assign.hpp>

namespace se2_grid {

int nDimensions()
{
  return 2;
}

std::map<StorageIndices, std::string> storageIndexNames = boost::assign::map_list_of
    (StorageIndices::Column,  "column_index")
    (StorageIndices::Row, "row_index");

} /* namespace */
