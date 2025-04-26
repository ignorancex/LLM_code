#include "Cartesian1DGridFunction.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

Cartesian1DGridFunction::Cartesian1DGridFunction(const double minCoordinate, const double maxCoordinate, std::size_t binNumber,
												 const double value)
	: MinCoordinate(minCoordinate),
	  MaxCoordinate(maxCoordinate),
	  BinNumber(binNumber),
	  BinWidth((MaxCoordinate - MinCoordinate) / static_cast<double>(BinNumber - 1)),
	  Coordinates(binNumber),
	  FunctionValues(BinNumber, value)
{
	if (MinCoordinate >= MaxCoordinate)
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction Error: Minimal coordinate value not smaller than maximal coordinate value" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if (BinNumber < 2)
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction Error: Number of axis bins is less than two" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_coordinates();
}

Cartesian1DGridFunction::Cartesian1DGridFunction(const double minCoordinate, const double maxCoordinate, std::size_t binNumber,
												 const std::function<double(double x)> &func)
	: Cartesian1DGridFunction(minCoordinate, maxCoordinate, binNumber)
{
	apply([&](double value, double x)
		  { return func(x); });
}

Cartesian1DGridFunction::Cartesian1DGridFunction(const std::string &fileName)
{
	load_object_from_file(fileName);
}

Cartesian1DGridFunction::Cartesian1DGridFunction(const Cartesian1DGridFunction &otherGridFunction, const std::function<double(double value)> &func)
	: Cartesian1DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian1DGridFunction::Cartesian1DGridFunction(const Cartesian1DGridFunction &otherGridFunction, const std::function<double(double value, double x)> &func)
	: Cartesian1DGridFunction(otherGridFunction)
{
	apply(func);
}

Cartesian1DGridFunction::Cartesian1DGridFunction()
	: MinCoordinate(0.0),
	  MaxCoordinate(0.0),
	  BinNumber(0),
	  BinWidth(0.0),
	  Coordinates(),
	  FunctionValues()
{
}

double &Cartesian1DGridFunction::value(const std::size_t bin)
{
	return FunctionValues[bin];
}

double Cartesian1DGridFunction::value(const std::size_t bin) const
{
	return FunctionValues[bin];
}

std::vector<double> &Cartesian1DGridFunction::value_vector()
{
	return FunctionValues;
}

const std::vector<double> &Cartesian1DGridFunction::value_vector() const
{
	return FunctionValues;
}

double Cartesian1DGridFunction::interpolate(const double x) const
{
	if ((x < MinCoordinate) or (x > MaxCoordinate))
	{
		std::cout << "Cartesian1DGridFunction::interpolate Error: Coordinate out of range" << std::endl;

		exit(EXIT_FAILURE);
	}

	const std::size_t i_x_0 = static_cast<std::size_t>(std::floor((x - MinCoordinate) / BinWidth)); // index of nearest grid coordinate below point to evaluate

	const std::size_t i_x_1 = (i_x_0 != BinNumber - 1) ? i_x_0 + 1 : i_x_0; // index of nearest grid coordinate above point to evaluate

	const double x_0 = coordinate(i_x_0);

	const double delta_x = (x - x_0) / BinWidth;

	const double f_0 = value(i_x_0);
	const double f_1 = value(i_x_1);

	const double f_x = (1.0 - delta_x) * f_0 + delta_x * f_1; // linear interpolation

	return f_x;
}

double Cartesian1DGridFunction::operator()(const double x) const
{
	return interpolate(x);
}

double Cartesian1DGridFunction::line_element(const std::size_t bin) const
{
	double lineElement = BinWidth;

	if ((bin == 0) or (bin == BinNumber - 1)) // end points of axis only contribute half the bin width
	{
		lineElement /= 2.0;
	}

	return lineElement;
}

double Cartesian1DGridFunction::coordinate(const std::size_t bin) const
{
	return Coordinates[bin];
}

std::vector<double> Cartesian1DGridFunction::coordinate_vector() const
{
	return Coordinates;
}

double Cartesian1DGridFunction::minimal_coordinate() const
{
	return MinCoordinate;
}

double Cartesian1DGridFunction::maximal_coordinate() const
{
	return MaxCoordinate;
}

double Cartesian1DGridFunction::bin_width() const
{
	return BinWidth;
}

std::size_t Cartesian1DGridFunction::bin_number() const
{
	return BinNumber;
}

void Cartesian1DGridFunction::apply(const std::function<double(double)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) if (parallelized)
	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double currentValue = value(i_x);

		value(i_x) = func(currentValue);
	}
}

void Cartesian1DGridFunction::apply(const std::function<double(double value, double x)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) if (parallelized)
	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double x = coordinate(i_x);

		const double currentValue = value(i_x);

		value(i_x) = func(currentValue, x);
	}
}

void Cartesian1DGridFunction::evaluate_for_all_grid_points(const std::function<void(std::size_t bin)> &func, bool parallelized) const
{
#pragma omp parallel for schedule(static) if (parallelized)
	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		func(i_x);
	}
}

double Cartesian1DGridFunction::average(const std::function<double(double value)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double val = value(i_x);
		const double lineElement = line_element(i_x);

		average += func(val) * lineElement;
	}

	average /= (MaxCoordinate - MinCoordinate);

	return average;
}

double Cartesian1DGridFunction::average(const std::function<double(double value, double x)> &func) const
{
	double average = 0.0;

	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double x = coordinate(i_x);

		const double val = value(i_x);
		const double lineElement = line_element(i_x);

		average += func(val, x) * lineElement;
	}

	average /= (MaxCoordinate - MinCoordinate);

	return average;
}

void Cartesian1DGridFunction::write_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName);
	outputFile.setf(std::ios::scientific);

	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double x = coordinate(i_x);

		const double val = value(i_x);

		outputFile << x << "\t"
				   << val
				   << std::endl;
	}

	outputFile.close();
}

void Cartesian1DGridFunction::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&MinCoordinate, sizeof(double));
	outputFile.write((char *)&MaxCoordinate, sizeof(double));
	outputFile.write((char *)&BinNumber, sizeof(std::size_t));

	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		const double currentValue = value(i_x);

		outputFile.write((char *)&currentValue, sizeof(double));
	}

	outputFile.close();
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator+=(const Cartesian1DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction::operator+= Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t bin)
								 { value(bin) += otherGridFunction.value(bin); });

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator-=(const Cartesian1DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction::operator-= Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t bin)
								 { value(bin) -= otherGridFunction.value(bin); });

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator*=(const Cartesian1DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction::operator*= Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t bin)
								 { value(bin) *= otherGridFunction.value(bin); });

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator/=(const Cartesian1DGridFunction &otherGridFunction)
{
	if (not has_same_grid_as(otherGridFunction))
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction::operator/= Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t bin)
								 { value(bin) /= otherGridFunction.value(bin); });

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator+=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t bin)
									 { value(bin) += otherValue; });
	}

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator-=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t bin)
									 { value(bin) -= otherValue; });
	}

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator*=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t bin)
									 { value(bin) *= otherValue; });
	}

	return *this;
}

Cartesian1DGridFunction &Cartesian1DGridFunction::operator/=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t bin)
									 { value(bin) /= otherValue; });
	}

	return *this;
}

bool Cartesian1DGridFunction::has_same_grid_as(const Cartesian1DGridFunction &otherGridFunction) const
{
	return (MinCoordinate == otherGridFunction.MinCoordinate and
			MaxCoordinate == otherGridFunction.MaxCoordinate and
			BinNumber == otherGridFunction.BinNumber);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void Cartesian1DGridFunction::initialize_coordinates()
{
	for (std::size_t i_x = 0; i_x < BinNumber - 1; ++i_x)
	{
		Coordinates[i_x] = MinCoordinate + static_cast<double>(i_x) * BinWidth;
	}

	Coordinates[BinNumber - 1] = MaxCoordinate; // set maximal value explicitly to avoid floating point rounding error
}

void Cartesian1DGridFunction::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " Cartesian1DGridFunction::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&MinCoordinate, sizeof(double));
	inputFile.read((char *)&MaxCoordinate, sizeof(double));
	inputFile.read((char *)&BinNumber, sizeof(std::size_t));

	BinWidth = (MaxCoordinate - MinCoordinate) / static_cast<double>(BinNumber - 1);
	Coordinates = std::vector<double>(BinNumber);
	FunctionValues = std::vector<double>(BinNumber);

	initialize_coordinates();

	for (std::size_t i_x = 0; i_x < BinNumber; ++i_x)
	{
		inputFile.read((char *)&value(i_x), sizeof(double));
	}

	inputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

Cartesian1DGridFunction operator+(const Cartesian1DGridFunction &leftGridFunction, const Cartesian1DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator+ Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian1DGridFunction newCartesian1DGridFunction(leftGridFunction);

	newCartesian1DGridFunction.evaluate_for_all_grid_points([&](std::size_t bin)
															{ newCartesian1DGridFunction.value(bin) += rightGridFunction.value(bin); });

	return newCartesian1DGridFunction;
}

Cartesian1DGridFunction operator-(const Cartesian1DGridFunction &leftGridFunction, const Cartesian1DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator- Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian1DGridFunction newCartesian1DGridFunction(leftGridFunction);

	newCartesian1DGridFunction.evaluate_for_all_grid_points([&](std::size_t bin)
															{ newCartesian1DGridFunction.value(bin) -= rightGridFunction.value(bin); });

	return newCartesian1DGridFunction;
}

Cartesian1DGridFunction operator*(const Cartesian1DGridFunction &leftGridFunction, const Cartesian1DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator* Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian1DGridFunction newCartesian1DGridFunction(leftGridFunction);

	newCartesian1DGridFunction.evaluate_for_all_grid_points([&](std::size_t bin)
															{ newCartesian1DGridFunction.value(bin) *= rightGridFunction.value(bin); });

	return newCartesian1DGridFunction;
}

Cartesian1DGridFunction operator/(const Cartesian1DGridFunction &leftGridFunction, const Cartesian1DGridFunction &rightGridFunction)
{
	if (not leftGridFunction.has_same_grid_as(rightGridFunction))
	{
		std::cout << std::endl
				  << " operator/ Error: Coordinates of Cartesian 1D grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	Cartesian1DGridFunction newCartesian1DGridFunction(leftGridFunction);

	newCartesian1DGridFunction.evaluate_for_all_grid_points([&](std::size_t bin)
															{ newCartesian1DGridFunction.value(bin) /= rightGridFunction.value(bin); });

	return newCartesian1DGridFunction;
}

Cartesian1DGridFunction operator+(const Cartesian1DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian1DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue + rightValue; });
	}
}

Cartesian1DGridFunction operator-(const Cartesian1DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian1DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue - rightValue; });
	}
}

Cartesian1DGridFunction operator*(const Cartesian1DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian1DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue * rightValue; });
	}
}

Cartesian1DGridFunction operator/(const Cartesian1DGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return Cartesian1DGridFunction(leftGridFunction, [&](double leftValue)
									   { return leftValue / rightValue; });
	}
}

Cartesian1DGridFunction operator+(const double leftValue, const Cartesian1DGridFunction &rightGridFunction)
{
	return rightGridFunction + leftValue;
}

Cartesian1DGridFunction operator-(const double leftValue, const Cartesian1DGridFunction &rightGridFunction)
{
	return Cartesian1DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue - rightValue; });
}

Cartesian1DGridFunction operator*(const double leftValue, const Cartesian1DGridFunction &rightGridFunction)
{
	return rightGridFunction * leftValue;
}

Cartesian1DGridFunction operator/(const double leftValue, const Cartesian1DGridFunction &rightGridFunction)
{
	return Cartesian1DGridFunction(rightGridFunction, [&](double rightValue)
								   { return leftValue / rightValue; });
}

void transform_cartesian_1D_grid_functions(const Cartesian1DGridFunction &input,
										   Cartesian1DGridFunction &output,
										   const std::function<void(std::size_t bin,
																	double &output)> &trafo)
{
	const bool transformedInPlace = (&output == &input); // check if the transformation is in-place

	Cartesian1DGridFunction &fieldTemporary = transformedInPlace ? *(new Cartesian1DGridFunction(output)) // if it is (partially) in-place, work on a temporary copy of the output field
																 : output;

	if (not fieldTemporary.has_same_grid_as(input))
	{
		fieldTemporary = input;
	}

	fieldTemporary.evaluate_for_all_grid_points([&](std::size_t bin)
												{
													double output;

													trafo(bin,
														  output);

													fieldTemporary.value(bin) = output;
												});

	if (transformedInPlace)
	{
		output = fieldTemporary;

		delete &fieldTemporary;
	}
}

void transform_cartesian_1D_grid_functions(const Cartesian1DGridFunction &input1, const Cartesian1DGridFunction &input2,
										   Cartesian1DGridFunction &output1, Cartesian1DGridFunction &output2,
										   const std::function<void(std::size_t bin,
																	double &output1, double &output2)> &trafo)
{
	if (not input1.has_same_grid_as(input2))
	{
		std::cout << std::endl
				  << " transform_cartesian_1D_grid_functions Error: Coordinate grids of input Cartesian1DGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2);

	Cartesian1DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian1DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian1DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian1DGridFunction(output2))
																   : output2;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t bin)
										 {
											 double output1, output2;

											 trafo(bin,
												   output1, output2);

											 fieldTemporary1.value(bin) = output1;
											 fieldTemporary2.value(bin) = output2;
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

void transform_cartesian_1D_grid_functions(const Cartesian1DGridFunction &input1, const Cartesian1DGridFunction &input2, const Cartesian1DGridFunction &input3,
										   Cartesian1DGridFunction &output1, Cartesian1DGridFunction &output2, Cartesian1DGridFunction &output3,
										   const std::function<void(std::size_t bin,
																	double &output1, double &output2, double &output3)> &trafo)
{
	if (not input1.has_same_grid_as(input2) or
		not input1.has_same_grid_as(input3))
	{
		std::cout << std::endl
				  << " transform_cartesian_1D_grid_functions Error: Coordinate grids of input Cartesian1DGridFunctions do not agree" << std::endl
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

	Cartesian1DGridFunction &fieldTemporary1 = transformed1InPlace ? *(new Cartesian1DGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																   : output1;

	Cartesian1DGridFunction &fieldTemporary2 = transformed2InPlace ? *(new Cartesian1DGridFunction(output2))
																   : output2;

	Cartesian1DGridFunction &fieldTemporary3 = transformed3InPlace ? *(new Cartesian1DGridFunction(output3))
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

	output1.evaluate_for_all_grid_points([&](std::size_t bin)
										 {
											 double output1, output2, output3;

											 trafo(bin,
												   output1, output2, output3);

											 fieldTemporary1.value(bin) = output1;
											 fieldTemporary2.value(bin) = output2;
											 fieldTemporary3.value(bin) = output3;
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