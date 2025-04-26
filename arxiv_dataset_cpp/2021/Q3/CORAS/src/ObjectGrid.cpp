#include "ObjectGrid.hpp"

#include <cmath>

#include <gsl/gsl_math.h>

#include "transformations.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

ObjectGrid::ObjectGrid(const double gridLength, const std::size_t gridBinNumber,
                       const double centralXCoordinate, const double centralYCoordinate, const double centralZCoordinate,
                       const std::vector<double> &xCoordinates, const std::vector<double> &yCoordinates, const std::vector<double> &zCoordinates,
                       const bool usePeriodicBoundaryConditions,
                       const std::function<bool(std::size_t index)> &condition)
    : GridLength(gridLength),
      GridBinNumber(gridBinNumber),
      GridBinWidth(GridLength / static_cast<double>(GridBinNumber)),
      XCoordinateShift(GridLength / 2.0 - centralXCoordinate),
      YCoordinateShift(GridLength / 2.0 - centralYCoordinate),
      ZCoordinateShift(GridLength / 2.0 - centralZCoordinate),
      XCoordinates(&xCoordinates),
      YCoordinates(&yCoordinates),
      ZCoordinates(&zCoordinates),
      UsePeriodicBoundaryConditions(usePeriodicBoundaryConditions),
      IndexGrid(gridBinNumber, std::vector<std::vector<std::vector<std::size_t>>>(gridBinNumber, std::vector<std::vector<std::size_t>>(gridBinNumber, std::vector<std::size_t>()))),
      ObjectNumber(0)

{
    initialize_grid(condition);
}

std::size_t ObjectGrid::object_number() const
{
    return ObjectNumber;
}

bool ObjectGrid::uses_periodic_boundary_conditions() const
{
    return UsePeriodicBoundaryConditions;
}

double ObjectGrid::x_coordinate(const std::size_t index) const
{
    return XCoordinates->operator[](index);
}

double ObjectGrid::y_coordinate(const std::size_t index) const
{
    return YCoordinates->operator[](index);
}

double ObjectGrid::z_coordinate(const std::size_t index) const
{
    return ZCoordinates->operator[](index);
}

double ObjectGrid::x_distance(const std::size_t index1, const std::size_t index2) const
{
    return coordinate_distance(x_coordinate(index1) - x_coordinate(index2));
}

double ObjectGrid::y_distance(const std::size_t index1, const std::size_t index2) const
{
    return coordinate_distance(y_coordinate(index1) - y_coordinate(index2));
}

double ObjectGrid::z_distance(const std::size_t index1, const std::size_t index2) const
{
    return coordinate_distance(z_coordinate(index1) - z_coordinate(index2));
}

double ObjectGrid::coordinate_distance(const double coordinateDifference) const
{
    if (UsePeriodicBoundaryConditions)
    {
        return coordinateDifference - GridLength * std::floor((coordinateDifference + GridLength / 2.0) / GridLength);
    }
    else
    {
        return coordinateDifference;
    }
}

double ObjectGrid::distance(const std::size_t index1, const std::size_t index2) const
{
    const double xDifference = x_coordinate(index1) - x_coordinate(index2);
    const double yDifference = y_coordinate(index1) - y_coordinate(index2);
    const double zDifference = z_coordinate(index1) - z_coordinate(index2);

    return distance(xDifference, yDifference, zDifference);
}

double ObjectGrid::distance(const double xDifference, const double yDifference, const double zDifference) const
{
    const double xDistance = coordinate_distance(xDifference);
    const double yDistance = coordinate_distance(yDifference);
    const double zDistance = coordinate_distance(zDifference);

    return std::sqrt(gsl_pow_2(xDistance) + gsl_pow_2(yDistance) + gsl_pow_2(zDistance));
}

void ObjectGrid::evaluate_for_all_objects(const std::function<void(std::size_t index)> &function) const
{
    for (std::size_t i_x = 0; i_x < GridBinNumber; ++i_x)
    {
        for (std::size_t i_y = 0; i_y < GridBinNumber; ++i_y)
        {
            for (std::size_t i_z = 0; i_z < GridBinNumber; ++i_z)
            {
                const std::vector<std::size_t> &gridCellObects = IndexGrid[i_x][i_y][i_z];

                for (std::size_t i_o = 0; i_o < gridCellObects.size(); ++i_o)
                {
                    function(gridCellObects[i_o]);
                }
            }
        }
    }
}

void ObjectGrid::evaluate_for_all_objects_within_distance(const double xCenter, const double yCenter, const double zCenter, const double maxDistance,
                                                          const std::function<void(double relativeX, double relativeY, double relativeZ, double distance, std::size_t index)> &function) const
{
    evaluate_for_all_objects_in_sub_grid(xCenter, yCenter, zCenter, maxDistance,
                                         [&](std::size_t index)
                                         {
                                             const double x = x_coordinate(index);
                                             const double y = y_coordinate(index);
                                             const double z = z_coordinate(index);

                                             const double relativeDistance = distance(x - xCenter, y - yCenter, z - zCenter);

                                             if (relativeDistance <= maxDistance)
                                             {
                                                 const double relativeX = coordinate_distance(x - xCenter);
                                                 const double relativeY = coordinate_distance(y - yCenter);
                                                 const double relativeZ = coordinate_distance(z - zCenter);

                                                 function(relativeX, relativeY, relativeZ, relativeDistance, index);
                                             }
                                         });
}

void ObjectGrid::evaluate_for_all_objects_within_distance(const std::size_t centralIndex, const double maxDistance,
                                                          const std::function<void(double relativeX, double relativeY, double relativeZ, double distance, std::size_t index)> &function) const
{
    const double xCenter = x_coordinate(centralIndex);
    const double yCenter = y_coordinate(centralIndex);
    const double zCenter = z_coordinate(centralIndex);

    evaluate_for_all_objects_within_distance(xCenter, yCenter, zCenter, maxDistance,
                                             function);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void ObjectGrid::initialize_grid(const std::function<bool(std::size_t index)> &condition)
{
    ObjectNumber = 0;

    for (std::size_t i_o = 0; i_o < XCoordinates->size(); ++i_o)
    {
        if (condition(i_o))
        {
            const double x = x_coordinate(i_o);
            const double y = y_coordinate(i_o);
            const double z = z_coordinate(i_o);

            std::size_t i_x, i_y, i_z;

            i_x = bin(shift_x_coordinate(x));
            i_y = bin(shift_y_coordinate(y));
            i_z = bin(shift_z_coordinate(z));

            IndexGrid[i_x][i_y][i_z].push_back(i_o);

            ObjectNumber++;
        }
    }
}

std::size_t ObjectGrid::bin(const double shiftedCoordinate) const
{
    if (UsePeriodicBoundaryConditions)
    {
        return static_cast<std::size_t>(std::floor((shiftedCoordinate - std::floor(shiftedCoordinate / GridLength) * GridLength) / GridBinWidth)) % GridBinNumber;
    }
    else
    {
        if (shiftedCoordinate == GridLength)
        {
            return GridBinNumber - 1;
        }
        else
        {
            return static_cast<std::size_t>(std::floor(shiftedCoordinate / GridBinWidth));
        }
    }
}

double ObjectGrid::shift_x_coordinate(const double x) const
{
    return x + XCoordinateShift;
}

double ObjectGrid::shift_y_coordinate(const double y) const
{
    return y + YCoordinateShift;
}

double ObjectGrid::shift_z_coordinate(const double z) const
{
    return z + ZCoordinateShift;
}

void ObjectGrid::evaluate_for_all_objects_in_sub_grid(double xCenter, double yCenter, double zCenter, const double minDistance,
                                                      const std::function<void(std::size_t index)> &function) const
{
    const double shiftedXCenter = shift_x_coordinate(xCenter);
    const double shiftedYCenter = shift_y_coordinate(yCenter);
    const double shiftedZCenter = shift_z_coordinate(zCenter);

    if (UsePeriodicBoundaryConditions)
    {
        const std::size_t i_x = bin(shiftedXCenter);
        const std::size_t i_y = bin(shiftedYCenter);
        const std::size_t i_z = bin(shiftedZCenter);

        const std::size_t gridBinDistance = bin(minDistance) + 1;

        const std::size_t j_xMin = (i_x + GridBinNumber - gridBinDistance) % GridBinNumber;
        const std::size_t j_yMin = (i_y + GridBinNumber - gridBinDistance) % GridBinNumber;
        const std::size_t j_zMin = (i_z + GridBinNumber - gridBinDistance) % GridBinNumber;

        const std::size_t j_xMax = (i_x + gridBinDistance + 1) % GridBinNumber;
        const std::size_t j_yMax = (i_y + gridBinDistance + 1) % GridBinNumber;
        const std::size_t j_zMax = (i_z + gridBinDistance + 1) % GridBinNumber;

        for (std::size_t j_x = j_xMin; j_x != j_xMax; j_x = (j_x + 1) % GridBinNumber)
        {
            for (std::size_t j_y = j_yMin; j_y != j_yMax; j_y = (j_y + 1) % GridBinNumber)
            {
                for (std::size_t j_z = j_zMin; j_z != j_zMax; j_z = (j_z + 1) % GridBinNumber)
                {
                    const std::vector<std::size_t> &gridCellObects = IndexGrid[j_x][j_y][j_z];

                    for (std::size_t j_o = 0; j_o < gridCellObects.size(); ++j_o)
                    {
                        function(gridCellObects[j_o]);
                    }
                }
            }
        }
    }
    else
    {
        const double shiftedXMin = std::max(shiftedXCenter - minDistance, 0.0);
        const double shiftedYMin = std::max(shiftedYCenter - minDistance, 0.0);
        const double shiftedZMin = std::max(shiftedZCenter - minDistance, 0.0);

        const double shiftedXMax = std::min(shiftedXCenter + minDistance, GridLength);
        const double shiftedYMax = std::min(shiftedYCenter + minDistance, GridLength);
        const double shiftedZMax = std::min(shiftedZCenter + minDistance, GridLength);

        if (shiftedXMin < shiftedXMax and
            shiftedYMin < shiftedYMax and
            shiftedZMin < shiftedZMax)
        {
            const std::size_t j_xMin = bin(shiftedXMin);
            const std::size_t j_yMin = bin(shiftedYMin);
            const std::size_t j_zMin = bin(shiftedZMin);

            const std::size_t j_xMax = bin(shiftedXMax);
            const std::size_t j_yMax = bin(shiftedYMax);
            const std::size_t j_zMax = bin(shiftedZMax);

            for (std::size_t j_x = j_xMin; j_x <= j_xMax; ++j_x)
            {
                for (std::size_t j_y = j_yMin; j_y <= j_yMax; ++j_y)
                {
                    for (std::size_t j_z = j_zMin; j_z <= j_zMax; ++j_z)
                    {
                        const std::vector<std::size_t> &gridCellObects = IndexGrid[j_x][j_y][j_z];

                        for (std::size_t j_o = 0; j_o < gridCellObects.size(); ++j_o)
                        {
                            function(gridCellObects[j_o]);
                        }
                    }
                }
            }
        }
    }
}