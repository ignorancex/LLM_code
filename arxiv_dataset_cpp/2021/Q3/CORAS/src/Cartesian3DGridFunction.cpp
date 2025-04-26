#include "Cartesian3DGridFunction.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

Cartesian3DGridFunction::Cartesian3DGridFunction(const double minXCoordinate, const double maxXCoordinate, std::size_t xBinNumber,
												 const double minYCoordinate, const double maxYCoordinate, std::size_t yBinNumber,
												 const double minZCoordinate, const double maxZCoordinate, std::size_t zBinNumber,
												 const double value)
	: MinXCoordinate(minXCoordinate),
	  MaxXCoordinate(maxXCoordinate),
	  MinYCoordinate(minYCoordinate),
	  MaxYCoordinate(maxYCoordinate),
	  MinZCoordinate(minZCoordinate),
	  MaxZCoordinate(maxZCoordinate),
	  XBinNumber(xBinNumber),
	  YBinNumber(yBinNumber),
	  ZBinNumber(zBinNumber),
	  XBinWidth((MaxXCoordinate - MinXCoordinate) / static_cast<double>(XBinNumber - 1)),
	  YBinWidth((MaxYCoordinate - MinYCoordinate) / static_cast<double>(YBinNumber - 1)),
	  ZBinWidth((MaxZCoordinate - MinZCoordinate) / static_cast<double>(ZBinNumber - 1)),
	  XCoordinates(XBinNumber),
	  YCoordinates(YBinNumber),
	  ZCoordinates(ZBinNumber),
	  FunctionValues(XBinNumber * YBinNumber * ZBinNumber, value)
{
	if ((MinXCoordinate >= MaxXCoordinate) or (MinYCoordinate >= MaxYCoordinate) or (MinZCoordinate >= MaxZCoordinate))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction Error: Minimal coordinate value not smaller than maximal coordinate value" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if ((XBinNumber < 2) or (YBinNumber < 2) or (ZBinNumber < 2))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction Error: Number of bins per axis is less than two" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_coordinates();
}

Cartesian3DGridFunction::Cartesian3DGridFunction(const double minXCoordinate, const double maxXCoordinate, std::size_t xBinNumber,
												 const double minYCoordinate, const double maxYCoordinate, std::size_t yBinNumber,
												 const double minZCoordinate, const double maxZCoordinate, std::size_t zBinNumber,
												 const std::function<double(double x, double y, double z)> &func)
	: Cartesian3DGridFunction(minXCoordinate, maxXCoordinate, xBinNumber,
							  minYCoordinate, maxYCoordinate, yBinNumber,
							  minZCoordinate, maxZCoordinate, zBinNumber)
{
	apply([&](double value, double x, double y, double z)
		  { return func(x, y, z); });
}

Cartesian3DGridFunction::Cartesian3DGridFunction(const std::string &fileName)
{
	load_object_from_file(fileName);
}

Cartesian3DGridFunction::Cartesian3DGridFunction(const Cartesian3DGridFunction &otherGridFunction, const std::function<double(double value)> &func)
	: Cartesian3DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian3DGridFunction::Cartesian3DGridFunction(const Cartesian3DGridFunction &otherGridFunction, const std::function<double(double value, double x, double y, double z)> &func)
	: Cartesian3DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian3DGridFunction::Cartesian3DGridFunction()
	: MinXCoordinate(0.0),
	  MaxXCoordinate(0.0),
	  MinYCoordinate(0.0),
	  MaxYCoordinate(0.0),
	  MinZCoordinate(0.0),
	  MaxZCoordinate(0.0),
	  XBinNumber(0),
	  YBinNumber(0),
	  ZBinNumber(0),
	  XBinWidth(0.0),
	  YBinWidth(0.0),
	  ZBinWidth(0.0),
	  XCoordinates(),
	  YCoordinates(),
	  ZCoordinates(),
	  FunctionValues()
{
}

double &Cartesian3DGridFunction::value(const std::size_t xBin, const std::size_t yBin, const std::size_t zBin)
{
	const std::size_t i_xyz = (xBin * YBinNumber + yBin) * ZBinNumber + zBin;

	return FunctionValues[i_xyz];
}

double Cartesian3DGridFunction::value(const std::size_t xBin, const std::size_t yBin, const std::size_t zBin) const
{
	const std::size_t i_xyz = (xBin * YBinNumber + yBin) * ZBinNumber + zBin;

	return FunctionValues[i_xyz];
}

std::vector<double> &Cartesian3DGridFunction::value_vector()
{
	return FunctionValues;
}

const std::vector<double> &Cartesian3DGridFunction::value_vector() const
{
	return FunctionValues;
}

double Cartesian3DGridFunction::interpolate(const double x, const double y, const double z) const
{
	if ((x < MinXCoordinate) or (x > MaxXCoordinate) or (y < MinYCoordinate) or (y > MaxYCoordinate) or (z < MinZCoordinate) or (z > MaxZCoordinate))
	{
		std::cout << "Cartesian3DGridFunction::interpolate Error: Coordinates out of range" << std::endl;
		std::cout << "xMin: " << MinXCoordinate << ", xMax: " << MaxXCoordinate << ", x: " << x << std::endl;
		std::cout << "yMin: " << MinYCoordinate << ", yMax: " << MaxYCoordinate << ", y: " << y << std::endl;
		std::cout << "zMin: " << MinZCoordinate << ", zMax: " << MaxZCoordinate << ", z: " << z << std::endl;

		exit(EXIT_FAILURE);
	}

	const std::size_t i_x_0 = static_cast<std::size_t>(std::floor((x - MinXCoordinate) / XBinWidth)); // index of nearest x-coordinate below point to evaluate
	const std::size_t i_y_0 = static_cast<std::size_t>(std::floor((y - MinYCoordinate) / YBinWidth)); // index of nearest y-coordinate below point to evaluate
	const std::size_t i_z_0 = static_cast<std::size_t>(std::floor((z - MinZCoordinate) / ZBinWidth)); // index of nearest z-coordinate below point to evaluate

	const std::size_t i_x_1 = (i_x_0 != XBinNumber - 1) ? i_x_0 + 1 : i_x_0; // index of nearest x-coordinate above point to evaluate
	const std::size_t i_y_1 = (i_y_0 != YBinNumber - 1) ? i_y_0 + 1 : i_y_0; // index of nearest y-coordinate above point to evaluate
	const std::size_t i_z_1 = (i_z_0 != ZBinNumber - 1) ? i_z_0 + 1 : i_z_0; // index of nearest z-coordinate above point to evaluate

	const double x_0 = x_coordinate(i_x_0);
	const double y_0 = y_coordinate(i_y_0);
	const double z_0 = z_coordinate(i_z_0);

	const double delta_x = (x - x_0) / XBinWidth;
	const double delta_y = (y - y_0) / YBinWidth;
	const double delta_z = (z - z_0) / ZBinWidth;

	const double f_000 = value(i_x_0, i_y_0, i_z_0);
	const double f_001 = value(i_x_0, i_y_0, i_z_1);
	const double f_010 = value(i_x_0, i_y_1, i_z_0);
	const double f_011 = value(i_x_0, i_y_1, i_z_1);
	const double f_100 = value(i_x_1, i_y_0, i_z_0);
	const double f_101 = value(i_x_1, i_y_0, i_z_1);
	const double f_110 = value(i_x_1, i_y_1, i_z_0);
	const double f_111 = value(i_x_1, i_y_1, i_z_1);

	const double f_x00 = (1.0 - delta_x) * f_000 + delta_x * f_100; // multilinear interpolation
	const double f_x01 = (1.0 - delta_x) * f_001 + delta_x * f_101;
	const double f_x10 = (1.0 - delta_x) * f_010 + delta_x * f_110;
	const double f_x11 = (1.0 - delta_x) * f_011 + delta_x * f_111;

	const double f_xy0 = (1.0 - delta_y) * f_x00 + delta_y * f_x10;
	const double f_xy1 = (1.0 - delta_y) * f_x01 + delta_y * f_x11;

	const double f_xyz = (1.0 - delta_z) * f_xy0 + delta_z * f_xy1;

	return f_xyz;
}

double Cartesian3DGridFunction::operator()(const double x, const double y, const double z) const
{
	return interpolate(x, y, z);
}

double Cartesian3DGridFunction::volume_element(const std::size_t xBin, const std::size_t yBin, const std::size_t zBin) const
{
	double volumeElement = XBinWidth * YBinWidth * ZBinWidth;

	if ((xBin == 0) or (xBin == XBinNumber - 1)) // end points of axes only contribute half the respective bin width and thus reduce the volume element accordingly
	{
		volumeElement /= 2.0;
	}

	if ((yBin == 0) or (yBin == YBinNumber - 1))
	{
		volumeElement /= 2.0;
	}

	if ((zBin == 0) or (zBin == ZBinNumber - 1))
	{
		volumeElement /= 2.0;
	}

	return volumeElement;
}

double Cartesian3DGridFunction::x_coordinate(std::size_t xBin) const
{
	return XCoordinates[xBin];
}

double Cartesian3DGridFunction::y_coordinate(std::size_t yBin) const
{
	return YCoordinates[yBin];
}

double Cartesian3DGridFunction::z_coordinate(std::size_t zBin) const
{
	return ZCoordinates[zBin];
}

std::vector<double> Cartesian3DGridFunction::x_coordinate_vector() const
{
	return XCoordinates;
}

std::vector<double> Cartesian3DGridFunction::y_coordinate_vector() const
{
	return YCoordinates;
}

std::vector<double> Cartesian3DGridFunction::z_coordinate_vector() const
{
	return ZCoordinates;
}

double Cartesian3DGridFunction::minimal_x_coordinate() const
{
	return MinXCoordinate;
}

double Cartesian3DGridFunction::maximal_x_coordinate() const
{
	return MaxXCoordinate;
}

double Cartesian3DGridFunction::minimal_y_coordinate() const
{
	return MinYCoordinate;
}

double Cartesian3DGridFunction::maximal_y_coordinate() const
{
	return MaxYCoordinate;
}

double Cartesian3DGridFunction::minimal_z_coordinate() const
{
	return MinZCoordinate;
}

double Cartesian3DGridFunction::maximal_z_coordinate() const
{
	return MaxZCoordinate;
}

double Cartesian3DGridFunction::x_bin_width() const
{
	return XBinWidth;
}

double Cartesian3DGridFunction::y_bin_width() const
{
	return YBinWidth;
}

double Cartesian3DGridFunction::z_bin_width() const
{
	return ZBinWidth;
}

std::size_t Cartesian3DGridFunction::x_bin_number() const
{
	return XBinNumber;
}

std::size_t Cartesian3DGridFunction::y_bin_number() const
{
	return YBinNumber;
}

std::size_t Cartesian3DGridFunction::z_bin_number() const
{
	return ZBinNumber;
}

void Cartesian3DGridFunction::apply(const std::function<double(double)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double currentValue = value(i_x, i_y, i_z);

				value(i_x, i_y, i_z) = func(currentValue);
			}
		}
	}
}

void Cartesian3DGridFunction::apply(const std::function<double(double value, double x, double y, double z)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double x = x_coordinate(i_x);
				const double y = y_coordinate(i_y);
				const double z = z_coordinate(i_z);

				const double currentValue = value(i_x, i_y, i_z);

				value(i_x, i_y, i_z) = func(currentValue, x, y, z);
			}
		}
	}
}

void Cartesian3DGridFunction::evaluate_for_all_grid_points(const std::function<void(std::size_t xBin, std::size_t yBin, std::size_t zBin)> &func, bool parallelized) const
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				func(i_x, i_y, i_z);
			}
		}
	}
}

double Cartesian3DGridFunction::average(const std::function<double(double value)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double val = value(i_x, i_y, i_z);
				const double volumeElement = volume_element(i_x, i_y, i_z);

				average += func(val) * volumeElement;
			}
		}
	}

	average /= (MaxXCoordinate - MinXCoordinate) * (MaxYCoordinate - MinYCoordinate) * (MaxZCoordinate - MinZCoordinate);

	return average;
}

double Cartesian3DGridFunction::average(const std::function<double(double value, double x, double y, double z)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		const double x = x_coordinate(i_x);

		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double y = y_coordinate(i_y);

			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double z = z_coordinate(i_z);

				const double val = value(i_x, i_y, i_z);
				const double volumeElement = volume_element(i_x, i_y, i_z);

				average += func(val, x, y, z) * volumeElement;
			}
		}
	}

	average /= (MaxXCoordinate - MinXCoordinate) * (MaxYCoordinate - MinYCoordinate) * (MaxZCoordinate - MinZCoordinate);

	return average;
}

void Cartesian3DGridFunction::write_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName);
	outputFile.setf(std::ios::scientific);

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		const double x = x_coordinate(i_x);

		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double y = y_coordinate(i_y);

			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double z = z_coordinate(i_z);

				const double val = value(i_x, i_y, i_z);

				outputFile << x << "\t"
						   << y << "\t"
						   << z << "\t"
						   << val
						   << std::endl;
			}
		}
	}

	outputFile.close();
}

void Cartesian3DGridFunction::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&MinXCoordinate, sizeof(double));
	outputFile.write((char *)&MaxXCoordinate, sizeof(double));
	outputFile.write((char *)&XBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&MinYCoordinate, sizeof(double));
	outputFile.write((char *)&MaxYCoordinate, sizeof(double));
	outputFile.write((char *)&YBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&MinZCoordinate, sizeof(double));
	outputFile.write((char *)&MaxZCoordinate, sizeof(double));
	outputFile.write((char *)&ZBinNumber, sizeof(std::size_t));

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				const double currentValue = value(i_x, i_y, i_z);

				outputFile.write((char *)&currentValue, sizeof(double));
			}
		}
	}

	outputFile.close();
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator+=(const Cartesian3DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction::operator+= Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
								 { value(xBin, yBin, zBin) += otherGridFunction.value(xBin, yBin, zBin); });

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator-=(const Cartesian3DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction::operator-= Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
								 { value(xBin, yBin, zBin) -= otherGridFunction.value(xBin, yBin, zBin); });

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator*=(const Cartesian3DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction::operator*= Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
								 { value(xBin, yBin, zBin) *= otherGridFunction.value(xBin, yBin, zBin); });

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator/=(const Cartesian3DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction::operator/= Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
								 { value(xBin, yBin, zBin) /= otherGridFunction.value(xBin, yBin, zBin); });

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator+=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
									 { value(xBin, yBin, zBin) += otherValue; });
	}

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator-=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
									 { value(xBin, yBin, zBin) -= otherValue; });
	}

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator*=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
									 { value(xBin, yBin, zBin) *= otherValue; });
	}

	return *this;
}

Cartesian3DGridFunction &Cartesian3DGridFunction::operator/=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
									 { value(xBin, yBin, zBin) /= otherValue; });
	}

	return *this;
}

bool Cartesian3DGridFunction::has_same_grid_as(const Cartesian3DGridFunction &otherGridFunction) const
{
	return (MinXCoordinate == otherGridFunction.MinXCoordinate and
			MaxXCoordinate == otherGridFunction.MaxXCoordinate and
			MinYCoordinate == otherGridFunction.MinYCoordinate and
			MaxYCoordinate == otherGridFunction.MaxYCoordinate and
			MinZCoordinate == otherGridFunction.MinZCoordinate and
			MaxZCoordinate == otherGridFunction.MaxZCoordinate and
			XBinNumber == otherGridFunction.XBinNumber and
			YBinNumber == otherGridFunction.YBinNumber and
			ZBinNumber == otherGridFunction.ZBinNumber);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void Cartesian3DGridFunction::initialize_coordinates()
{
	for (std::size_t i_x = 0; i_x < XBinNumber - 1; ++i_x)
	{
		XCoordinates[i_x] = MinXCoordinate + static_cast<double>(i_x) * XBinWidth;
	}

	XCoordinates[XBinNumber - 1] = MaxXCoordinate; // set maximal value explicitly to avoid floating point rounding error

	for (std::size_t i_y = 0; i_y < YBinNumber - 1; ++i_y)
	{
		YCoordinates[i_y] = MinYCoordinate + static_cast<double>(i_y) * YBinWidth;
	}

	YCoordinates[YBinNumber - 1] = MaxYCoordinate;

	for (std::size_t i_z = 0; i_z < ZBinNumber - 1; ++i_z)
	{
		ZCoordinates[i_z] = MinZCoordinate + static_cast<double>(i_z) * ZBinWidth;
	}

	ZCoordinates[ZBinNumber - 1] = MaxZCoordinate;
}

void Cartesian3DGridFunction::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " Cartesian3DGridFunction::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&MinXCoordinate, sizeof(double));
	inputFile.read((char *)&MaxXCoordinate, sizeof(double));
	inputFile.read((char *)&XBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&MinYCoordinate, sizeof(double));
	inputFile.read((char *)&MaxYCoordinate, sizeof(double));
	inputFile.read((char *)&YBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&MinZCoordinate, sizeof(double));
	inputFile.read((char *)&MaxZCoordinate, sizeof(double));
	inputFile.read((char *)&ZBinNumber, sizeof(std::size_t));

	XBinWidth = (MaxXCoordinate - MinXCoordinate) / static_cast<double>(XBinNumber - 1);
	YBinWidth = (MaxYCoordinate - MinYCoordinate) / static_cast<double>(YBinNumber - 1);
	ZBinWidth = (MaxZCoordinate - MinZCoordinate) / static_cast<double>(ZBinNumber - 1);
	XCoordinates = std::vector<double>(XBinNumber);
	YCoordinates = std::vector<double>(YBinNumber);
	ZCoordinates = std::vector<double>(ZBinNumber);
	FunctionValues = std::vector<double>(XBinNumber * YBinNumber * ZBinNumber);

	initialize_coordinates();

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			for (std::size_t i_z = 0; i_z < ZBinNumber; ++i_z)
			{
				inputFile.read((char *)&value(i_x, i_y, i_z), sizeof(double));
			}
		}
	}

	inputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

Cartesian3DGridFunction operator+(const Cartesian3DGridFunction &leftGridFunction, const Cartesian3DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator+ Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian3DGridFunction newCartesian3DGridFunction(leftGridFunction);

	newCartesian3DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
															{ newCartesian3DGridFunction.value(xBin, yBin, zBin) += rightGridFunction.value(xBin, yBin, zBin); });

	return newCartesian3DGridFunction;
}

Cartesian3DGridFunction operator-(const Cartesian3DGridFunction &leftGridFunction, const Cartesian3DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator- Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian3DGridFunction newCartesian3DGridFunction(leftGridFunction);

	newCartesian3DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
															{ newCartesian3DGridFunction.value(xBin, yBin, zBin) -= rightGridFunction.value(xBin, yBin, zBin); });

	return newCartesian3DGridFunction;
}

Cartesian3DGridFunction operator*(const Cartesian3DGridFunction &leftGridFunction, const Cartesian3DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator* Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian3DGridFunction newCartesian3DGridFunction(leftGridFunction);

	newCartesian3DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
															{ newCartesian3DGridFunction.value(xBin, yBin, zBin) *= rightGridFunction.value(xBin, yBin, zBin); });

	return newCartesian3DGridFunction;
}

Cartesian3DGridFunction operator/(const Cartesian3DGridFunction &leftGridFunction, const Cartesian3DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator/ Error: Coordinates of Cartesian 3D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian3DGridFunction newCartesian3DGridFunction(leftGridFunction);

	newCartesian3DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
															{ newCartesian3DGridFunction.value(xBin, yBin, zBin) /= rightGridFunction.value(xBin, yBin, zBin); });

	return newCartesian3DGridFunction;
}

Cartesian3DGridFunction operator+(const Cartesian3DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian3DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue + rightValue; });
	}
}

Cartesian3DGridFunction operator-(const Cartesian3DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian3DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue - rightValue; });
	}
}

Cartesian3DGridFunction operator*(const Cartesian3DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian3DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue * rightValue; });
	}
}

Cartesian3DGridFunction operator/(const Cartesian3DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian3DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue / rightValue; });
	}
}

Cartesian3DGridFunction operator+(const double leftValue, const Cartesian3DGridFunction &rightGridFunction)
{
	return rightGridFunction + leftValue;
}

Cartesian3DGridFunction operator-(const double leftValue, const Cartesian3DGridFunction &rightGridFunction)
{
	return Cartesian3DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue - rightValue; });
}

Cartesian3DGridFunction operator*(const double leftValue, const Cartesian3DGridFunction &rightGridFunction)
{
	return rightGridFunction * leftValue;
}

Cartesian3DGridFunction operator/(const double leftValue, const Cartesian3DGridFunction &rightGridFunction)
{
	return Cartesian3DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue / rightValue; });
}

void transform_cartesian_3D_grid_functions(const Cartesian3DGridFunction &input,
										   Cartesian3DGridFunction &output,
										   const std::function<void(std::size_t xBin, std::size_t yBin, std::size_t zBin,
																	double &output)> &trafo)
{
	const bool transformedInPlace = (&output == &input); // check if the transformation is in-place

	Cartesian3DGridFunction &fieldTemporary = transformedInPlace ? *(new Cartesian3DGridFunction(output)) // if it is (partially) in-place, work on a temporary copy of the output field
																 : output;

	if (not fieldTemporary.has_same_grid_as(input))
	{
		fieldTemporary = input;
	}

	fieldTemporary.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
												{
													double output;

													trafo(xBin, yBin, zBin,
														  output);

													fieldTemporary.value(xBin, yBin, zBin) = output;
												});

	if (transformedInPlace)
	{
		output = fieldTemporary;

		delete &fieldTemporary;
	}
}

void transform_cartesian_3D_grid_functions(const Cartesian3DGridFunction &input1, const Cartesian3DGridFunction &input2,
										   Cartesian3DGridFunction &output1, Cartesian3DGridFunction &output2,
										   const std::function<void(std::size_t xBin, std::size_t yBin, std::size_t zBin,
																	double &output1, double &output2)> &trafo)
{
	if (not input1.has_same_grid_as(input2))
	{
		std::cout << std::endl
				  << " transform_cartesian_3D_grid_functions Error: Coordinate grids of input Cartesian3DGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2);

	Cartesian3DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian3DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian3DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian3DGridFunction(output2))
																   : output2;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
										 {
											 double output1, output2;

											 trafo(xBin, yBin, zBin,
												   output1, output2);

											 fieldTemporary1.value(xBin, yBin, zBin) = output1;
											 fieldTemporary2.value(xBin, yBin, zBin) = output2;
										 });

	if (transformed1InPlace)
	{
		output1 = fieldTemporary1;

		delete &fieldTemporary1;
	}

	if (transformed2InPlace)
	{
		output2 = fieldTemporary2;

		delete &fieldTemporary2;
	}
}

void transform_cartesian_3D_grid_functions(const Cartesian3DGridFunction &input1, const Cartesian3DGridFunction &input2, const Cartesian3DGridFunction &input3,
										   Cartesian3DGridFunction &output1, Cartesian3DGridFunction &output2, Cartesian3DGridFunction &output3,
										   const std::function<void(std::size_t xBin, std::size_t yBin, std::size_t zBin,
																	double &output1, double &output2, double &output3)> &trafo)
{
	if (not input1.has_same_grid_as(input2) or
		not input1.has_same_grid_as(input3))
	{
		std::cout << std::endl
				  << " transform_cartesian_3D_grid_functions Error: Coordinate grids of input Cartesian3DGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1 or
									  &output3 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2 or
									  &output3 == &input2);

	const bool transformed3InPlace = (&output1 == &input3 or
									  &output2 == &input3 or
									  &output3 == &input3);

	Cartesian3DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian3DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian3DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian3DGridFunction(output2))
																   : output2;

	Cartesian3DGridFunction &fieldTemporary3 = transformed3InPlace ? *(new Cartesian3DGridFunction(output3))
																   : output3;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	if (not fieldTemporary3.has_same_grid_as(input1))
	{
		fieldTemporary3 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin, std::size_t zBin)
										 {
											 double output1, output2, output3;

											 trafo(xBin, yBin, zBin,
												   output1, output2, output3);

											 fieldTemporary1.value(xBin, yBin, zBin) = output1;
											 fieldTemporary2.value(xBin, yBin, zBin) = output2;
											 fieldTemporary3.value(xBin, yBin, zBin) = output3;
										 });

	if (transformed1InPlace)
	{
		output1 = fieldTemporary1;

		delete &fieldTemporary1;
	}

	if (transformed2InPlace)
	{
		output2 = fieldTemporary2;

		delete &fieldTemporary2;
	}

	if (transformed3InPlace)
	{
		output3 = fieldTemporary3;

		delete &fieldTemporary3;
	}
}