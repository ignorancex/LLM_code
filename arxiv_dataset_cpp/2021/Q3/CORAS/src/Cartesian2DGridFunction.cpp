#include "Cartesian2DGridFunction.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

Cartesian2DGridFunction::Cartesian2DGridFunction(const double minXCoordinate, const double maxXCoordinate, std::size_t xBinNumber,
												 const double minYCoordinate, const double maxYCoordinate, std::size_t yBinNumber,
												 const double value)
	: MinXCoordinate(minXCoordinate),
	  MaxXCoordinate(maxXCoordinate),
	  MinYCoordinate(minYCoordinate),
	  MaxYCoordinate(maxYCoordinate),
	  XBinNumber(xBinNumber),
	  YBinNumber(yBinNumber),
	  XBinWidth((MaxXCoordinate - MinXCoordinate) / static_cast<double>(XBinNumber - 1)),
	  YBinWidth((MaxYCoordinate - MinYCoordinate) / static_cast<double>(YBinNumber - 1)),
	  XCoordinates(xBinNumber),
	  YCoordinates(YBinNumber),
	  FunctionValues(XBinNumber * YBinNumber, value)
{
	if ((MinXCoordinate >= MaxXCoordinate) or (MinYCoordinate >= MaxYCoordinate))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction Error: Minimal coordinate value not smaller than maximal coordinate value" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if ((XBinNumber < 2) or (YBinNumber < 2))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction Error: Number of bins per axis is less than two" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_coordinates();
}

Cartesian2DGridFunction::Cartesian2DGridFunction(const double minXCoordinate, const double maxXCoordinate, std::size_t xBinNumber,
												 const double minYCoordinate, const double maxYCoordinate, std::size_t yBinNumber,
												 const std::function<double(double x, double y)> &func)
	: Cartesian2DGridFunction(minXCoordinate, maxXCoordinate, xBinNumber,
							  minYCoordinate, maxYCoordinate, yBinNumber)
{
	apply([&](double value, double x, double y)
		  { return func(x, y); });
}

Cartesian2DGridFunction::Cartesian2DGridFunction(const std::string &fileName)
{
	load_object_from_file(fileName);
}

Cartesian2DGridFunction::Cartesian2DGridFunction(const Cartesian2DGridFunction &otherGridFunction, const std::function<double(double value)> &func)
	: Cartesian2DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian2DGridFunction::Cartesian2DGridFunction(const Cartesian2DGridFunction &otherGridFunction, const std::function<double(double value, double x, double y)> &func)
	: Cartesian2DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian2DGridFunction::Cartesian2DGridFunction()
	: MinXCoordinate(0.0),
	  MaxXCoordinate(0.0),
	  MinYCoordinate(0.0),
	  MaxYCoordinate(0.0),
	  XBinNumber(0),
	  YBinNumber(0),
	  XBinWidth(0.0),
	  YBinWidth(0.0),
	  XCoordinates(),
	  YCoordinates(),
	  FunctionValues()
{
}

double &Cartesian2DGridFunction::value(const std::size_t xBin, const std::size_t yBin)
{
	const std::size_t i_xy = xBin * YBinNumber + yBin;

	return FunctionValues[i_xy];
}

double Cartesian2DGridFunction::value(const std::size_t xBin, const std::size_t yBin) const
{
	const std::size_t i_xy = xBin * YBinNumber + yBin;

	return FunctionValues[i_xy];
}

std::vector<double> &Cartesian2DGridFunction::value_vector()
{
	return FunctionValues;
}

const std::vector<double> &Cartesian2DGridFunction::value_vector() const
{
	return FunctionValues;
}

double Cartesian2DGridFunction::interpolate(const double x, const double y) const
{
	if ((x < MinXCoordinate) or (x > MaxXCoordinate) or (y < MinYCoordinate) or (y > MaxYCoordinate))
	{
		std::cout << "Cartesian2DGridFunction::interpolate Error: Coordinates out of range" << std::endl;
		std::cout << "xMin: " << MinXCoordinate << ", xMax: " << MaxXCoordinate << ", x: " << x << std::endl;
		std::cout << "yMin: " << MinYCoordinate << ", yMax: " << MaxYCoordinate << ", y: " << y << std::endl;

		exit(EXIT_FAILURE);
	}

	const std::size_t i_x_0 = static_cast<std::size_t>(std::floor((x - MinXCoordinate) / XBinWidth)); // index of nearest x-coordinate below point to evaluate
	const std::size_t i_y_0 = static_cast<std::size_t>(std::floor((y - MinYCoordinate) / YBinWidth)); // index of nearest y-coordinate below point to evaluate

	const std::size_t i_x_1 = (i_x_0 != XBinNumber - 1) ? i_x_0 + 1 : i_x_0; // index of nearest x-coordinate above point to evaluate
	const std::size_t i_y_1 = (i_y_0 != YBinNumber - 1) ? i_y_0 + 1 : i_y_0; // index of nearest y-coordinate above point to evaluate

	const double x_0 = x_coordinate(i_x_0);
	const double y_0 = y_coordinate(i_y_0);

	const double delta_x = (x - x_0) / XBinWidth;
	const double delta_y = (y - y_0) / YBinWidth;

	const double f_00 = value(i_x_0, i_y_0);
	const double f_01 = value(i_x_0, i_y_1);
	const double f_10 = value(i_x_1, i_y_0);
	const double f_11 = value(i_x_1, i_y_1);

	const double f_x0 = (1.0 - delta_x) * f_00 + delta_x * f_10; // multilinear interpolation
	const double f_x1 = (1.0 - delta_x) * f_01 + delta_x * f_11;

	const double f_xy = (1.0 - delta_y) * f_x0 + delta_y * f_x1;

	return f_xy;
}

double Cartesian2DGridFunction::operator()(const double x, const double y) const
{
	return interpolate(x, y);
}

double Cartesian2DGridFunction::volume_element(const std::size_t xBin, const std::size_t yBin) const
{
	double volumeElement = XBinWidth * YBinWidth;

	if ((xBin == 0) or (xBin == XBinNumber - 1)) // end points of axes only contribute half the respective bin width and thus reduce the volume element accordingly
	{
		volumeElement /= 2.0;
	}

	if ((yBin == 0) or (yBin == YBinNumber - 1))
	{
		volumeElement /= 2.0;
	}

	return volumeElement;
}

double Cartesian2DGridFunction::x_coordinate(std::size_t xBin) const
{
	return XCoordinates[xBin];
}

double Cartesian2DGridFunction::y_coordinate(std::size_t yBin) const
{
	return YCoordinates[yBin];
}

std::vector<double> Cartesian2DGridFunction::x_coordinate_vector() const
{
	return XCoordinates;
}

std::vector<double> Cartesian2DGridFunction::y_coordinate_vector() const
{
	return YCoordinates;
}

double Cartesian2DGridFunction::minimal_x_coordinate() const
{
	return MinXCoordinate;
}

double Cartesian2DGridFunction::maximal_x_coordinate() const
{
	return MaxXCoordinate;
}

double Cartesian2DGridFunction::minimal_y_coordinate() const
{
	return MinYCoordinate;
}

double Cartesian2DGridFunction::maximal_y_coordinate() const
{
	return MaxYCoordinate;
}

double Cartesian2DGridFunction::x_bin_width() const
{
	return XBinWidth;
}

double Cartesian2DGridFunction::y_bin_width() const
{
	return YBinWidth;
}

std::size_t Cartesian2DGridFunction::x_bin_number() const
{
	return XBinNumber;
}

std::size_t Cartesian2DGridFunction::y_bin_number() const
{
	return YBinNumber;
}

void Cartesian2DGridFunction::apply(const std::function<double(double)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double currentValue = value(i_x, i_y);

			value(i_x, i_y) = func(currentValue);
		}
	}
}

void Cartesian2DGridFunction::apply(const std::function<double(double value, double x, double y)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double x = x_coordinate(i_x);
			const double y = y_coordinate(i_y);

			const double currentValue = value(i_x, i_y);

			value(i_x, i_y) = func(currentValue, x, y);
		}
	}
}

void Cartesian2DGridFunction::evaluate_for_all_grid_points(const std::function<void(std::size_t xBin, std::size_t yBin)> &func, bool parallelized) const
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			func(i_x, i_y);
		}
	}
}

double Cartesian2DGridFunction::average(const std::function<double(double value)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double val = value(i_x, i_y);
			const double volumeElement = volume_element(i_x, i_y);

			average += func(val) * volumeElement;
		}
	}

	average /= (MaxXCoordinate - MinXCoordinate) * (MaxYCoordinate - MinYCoordinate);

	return average;
}

double Cartesian2DGridFunction::average(const std::function<double(double value, double x, double y)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		const double x = x_coordinate(i_x);

		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double y = y_coordinate(i_y);

			const double val = value(i_x, i_y);
			const double volumeElement = volume_element(i_x, i_y);

			average += func(val, x, y) * volumeElement;
		}
	}

	average /= (MaxXCoordinate - MinXCoordinate) * (MaxYCoordinate - MinYCoordinate);

	return average;
}

void Cartesian2DGridFunction::write_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName);
	outputFile.setf(std::ios::scientific);

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		const double x = x_coordinate(i_x);

		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double y = y_coordinate(i_y);

			const double val = value(i_x, i_y);

			outputFile << x << "\t"
					   << y << "\t"
					   << val
					   << std::endl;
		}
	}

	outputFile.close();
}

void Cartesian2DGridFunction::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&MinXCoordinate, sizeof(double));
	outputFile.write((char *)&MaxXCoordinate, sizeof(double));
	outputFile.write((char *)&XBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&MinYCoordinate, sizeof(double));
	outputFile.write((char *)&MaxYCoordinate, sizeof(double));
	outputFile.write((char *)&YBinNumber, sizeof(std::size_t));

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			const double currentValue = value(i_x, i_y);

			outputFile.write((char *)&currentValue, sizeof(double));
		}
	}

	outputFile.close();
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator+=(const Cartesian2DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction::operator+= Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
								 { value(xBin, yBin) += otherGridFunction.value(xBin, yBin); });

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator-=(const Cartesian2DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction::operator-= Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
								 { value(xBin, yBin) -= otherGridFunction.value(xBin, yBin); });

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator*=(const Cartesian2DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction::operator*= Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
								 { value(xBin, yBin) *= otherGridFunction.value(xBin, yBin); });

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator/=(const Cartesian2DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction::operator/= Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
								 { value(xBin, yBin) /= otherGridFunction.value(xBin, yBin); });

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator+=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
									 { value(xBin, yBin) += otherValue; });
	}

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator-=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
									 { value(xBin, yBin) -= otherValue; });
	}

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator*=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
									 { value(xBin, yBin) *= otherValue; });
	}

	return *this;
}

Cartesian2DGridFunction &Cartesian2DGridFunction::operator/=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
									 { value(xBin, yBin) /= otherValue; });
	}

	return *this;
}

bool Cartesian2DGridFunction::has_same_grid_as(const Cartesian2DGridFunction &otherGridFunction) const
{
	return (MinXCoordinate == otherGridFunction.MinXCoordinate and
			MaxXCoordinate == otherGridFunction.MaxXCoordinate and
			MinYCoordinate == otherGridFunction.MinYCoordinate and
			MaxYCoordinate == otherGridFunction.MaxYCoordinate and
			XBinNumber == otherGridFunction.XBinNumber and
			YBinNumber == otherGridFunction.YBinNumber);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void Cartesian2DGridFunction::initialize_coordinates()
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
}

void Cartesian2DGridFunction::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " Cartesian2DGridFunction::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&MinXCoordinate, sizeof(double));
	inputFile.read((char *)&MaxXCoordinate, sizeof(double));
	inputFile.read((char *)&XBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&MinYCoordinate, sizeof(double));
	inputFile.read((char *)&MaxYCoordinate, sizeof(double));
	inputFile.read((char *)&YBinNumber, sizeof(std::size_t));

	XBinWidth = (MaxXCoordinate - MinXCoordinate) / static_cast<double>(XBinNumber - 1);
	YBinWidth = (MaxYCoordinate - MinYCoordinate) / static_cast<double>(YBinNumber - 1);
	XCoordinates = std::vector<double>(XBinNumber);
	YCoordinates = std::vector<double>(YBinNumber);
	FunctionValues = std::vector<double>(XBinNumber * YBinNumber);

	initialize_coordinates();

	for (std::size_t i_x = 0; i_x < XBinNumber; ++i_x)
	{
		for (std::size_t i_y = 0; i_y < YBinNumber; ++i_y)
		{
			inputFile.read((char *)&value(i_x, i_y), sizeof(double));
		}
	}

	inputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

Cartesian2DGridFunction operator+(const Cartesian2DGridFunction &leftGridFunction, const Cartesian2DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator+ Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian2DGridFunction newCartesian2DGridFunction(leftGridFunction);

	newCartesian2DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
															{ newCartesian2DGridFunction.value(xBin, yBin) += rightGridFunction.value(xBin, yBin); });

	return newCartesian2DGridFunction;
}

Cartesian2DGridFunction operator-(const Cartesian2DGridFunction &leftGridFunction, const Cartesian2DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator- Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian2DGridFunction newCartesian2DGridFunction(leftGridFunction);

	newCartesian2DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
															{ newCartesian2DGridFunction.value(xBin, yBin) -= rightGridFunction.value(xBin, yBin); });

	return newCartesian2DGridFunction;
}

Cartesian2DGridFunction operator*(const Cartesian2DGridFunction &leftGridFunction, const Cartesian2DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator* Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian2DGridFunction newCartesian2DGridFunction(leftGridFunction);

	newCartesian2DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
															{ newCartesian2DGridFunction.value(xBin, yBin) *= rightGridFunction.value(xBin, yBin); });

	return newCartesian2DGridFunction;
}

Cartesian2DGridFunction operator/(const Cartesian2DGridFunction &leftGridFunction, const Cartesian2DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator/ Error: Coordinates of Cartesian 2D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian2DGridFunction newCartesian2DGridFunction(leftGridFunction);

	newCartesian2DGridFunction.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
															{ newCartesian2DGridFunction.value(xBin, yBin) /= rightGridFunction.value(xBin, yBin); });

	return newCartesian2DGridFunction;
}

Cartesian2DGridFunction operator+(const Cartesian2DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian2DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue + rightValue; });
	}
}

Cartesian2DGridFunction operator-(const Cartesian2DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian2DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue - rightValue; });
	}
}

Cartesian2DGridFunction operator*(const Cartesian2DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian2DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue * rightValue; });
	}
}

Cartesian2DGridFunction operator/(const Cartesian2DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian2DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue / rightValue; });
	}
}

Cartesian2DGridFunction operator+(const double leftValue, const Cartesian2DGridFunction &rightGridFunction)
{
	return rightGridFunction + leftValue;
}

Cartesian2DGridFunction operator-(const double leftValue, const Cartesian2DGridFunction &rightGridFunction)
{
	return Cartesian2DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue - rightValue; });
}

Cartesian2DGridFunction operator*(const double leftValue, const Cartesian2DGridFunction &rightGridFunction)
{
	return rightGridFunction * leftValue;
}

Cartesian2DGridFunction operator/(const double leftValue, const Cartesian2DGridFunction &rightGridFunction)
{
	return Cartesian2DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue / rightValue; });
}

void transform_cartesian_2D_grid_functions(const Cartesian2DGridFunction &input,
										   Cartesian2DGridFunction &output,
										   const std::function<void(std::size_t xBin, std::size_t yBin,
																	double &output)> &trafo)
{
	const bool transformedInPlace = (&output == &input); // check if the transformation is in-place

	Cartesian2DGridFunction &fieldTemporary = transformedInPlace ? *(new Cartesian2DGridFunction(output)) // if it is (partially) in-place, work on a temporary copy of the output field
																 : output;

	if (not fieldTemporary.has_same_grid_as(input))
	{
		fieldTemporary = input;
	}

	fieldTemporary.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
												{
													double output;

													trafo(xBin, yBin,
														  output);

													fieldTemporary.value(xBin, yBin) = output;
												});

	if (transformedInPlace)
	{
		output = fieldTemporary;

		delete &fieldTemporary;
	}
}

void transform_cartesian_2D_grid_functions(const Cartesian2DGridFunction &input1, const Cartesian2DGridFunction &input2,
										   Cartesian2DGridFunction &output1, Cartesian2DGridFunction &output2,
										   const std::function<void(std::size_t xBin, std::size_t yBin,
																	double &output1, double &output2)> &trafo)
{
	if (not input1.has_same_grid_as(input2))
	{
		std::cout << std::endl
				  << " transform_cartesian_2D_grid_functions Error: Coordinate grids of input Cartesian2DGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2);

	Cartesian2DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian2DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian2DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian2DGridFunction(output2))
																   : output2;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
										 {
											 double output1, output2;

											 trafo(xBin, yBin,
												   output1, output2);

											 fieldTemporary1.value(xBin, yBin) = output1;
											 fieldTemporary2.value(xBin, yBin) = output2;
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

void transform_cartesian_2D_grid_functions(const Cartesian2DGridFunction &input1, const Cartesian2DGridFunction &input2, const Cartesian2DGridFunction &input3,
										   Cartesian2DGridFunction &output1, Cartesian2DGridFunction &output2, Cartesian2DGridFunction &output3,
										   const std::function<void(std::size_t xBin, std::size_t yBin,
																	double &output1, double &output2, double &output3)> &trafo)
{
	if (not input1.has_same_grid_as(input2) or
		not input1.has_same_grid_as(input3))
	{
		std::cout << std::endl
				  << " transform_cartesian_2D_grid_functions Error: Coordinate grids of input Cartesian2DGridFunctions do not agree" << std::endl
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

	Cartesian2DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian2DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian2DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian2DGridFunction(output2))
																   : output2;

	Cartesian2DGridFunction &fieldTemporary3 = transformed3InPlace ? *(new Cartesian2DGridFunction(output3))
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

	output1.evaluate_for_all_grid_points([&](std::size_t xBin, std::size_t yBin)
										 {
											 double output1, output2, output3;

											 trafo(xBin, yBin,
												   output1, output2, output3);

											 fieldTemporary1.value(xBin, yBin) = output1;
											 fieldTemporary2.value(xBin, yBin) = output2;
											 fieldTemporary3.value(xBin, yBin) = output3;
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