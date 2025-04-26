#include "AngularGridFunction.hpp"

#include <cmath>
#include <iostream>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

AngularGridFunction::AngularGridFunction(const std::size_t thetaBinNumber, const std::size_t phiBinNumber,
										 const double value)
	: ThetaBinNumber(thetaBinNumber),
	  PhiBinNumber(phiBinNumber),
	  ThetaBinWidth(M_PI / static_cast<double>(ThetaBinNumber - 1)),
	  PhiBinWidth(2.0 * M_PI / static_cast<double>(PhiBinNumber)),
	  ThetaCoordinates(ThetaBinNumber),
	  PhiCoordinates(PhiBinNumber),
	  FunctionValues(ThetaBinNumber * PhiBinNumber, value)
{
	if ((ThetaBinNumber < 2) or (PhiBinNumber < 2))
	{
		std::cout << std::endl
				  << " AngularGridFunction Error: Number of bins per axis is less than two" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_coordinates();
}

AngularGridFunction::AngularGridFunction(const std::size_t thetaBinNumber, const std::size_t phiBinNumber,
										 const std::function<double(double theta, double phi)> &func)
	: AngularGridFunction(thetaBinNumber, phiBinNumber, 0.0)
{

	apply([&](double value, double theta, double phi)
		  { return func(theta, phi); });
}

AngularGridFunction::AngularGridFunction(const std::string &fileName)
{
	load_object_from_file(fileName);
}

AngularGridFunction::AngularGridFunction(const AngularGridFunction &otherAngularGridFunction, const std::function<double(double value)> &func)
	: AngularGridFunction(otherAngularGridFunction)
{
	apply(func);
}

AngularGridFunction::AngularGridFunction(const AngularGridFunction &otherAngularGridFunction, const std::function<double(double value, double theta, double phi)> &func)
	: AngularGridFunction(otherAngularGridFunction)
{
	apply(func);
}

AngularGridFunction::AngularGridFunction()
	: ThetaBinNumber(0),
	  PhiBinNumber(0),
	  ThetaBinWidth(0.0),
	  PhiBinWidth(0.0),
	  ThetaCoordinates(),
	  PhiCoordinates(),
	  FunctionValues()
{
}

double &AngularGridFunction::value(const std::size_t thetaBin, const std::size_t phiBin)
{
	const std::size_t i_tp = thetaBin * PhiBinNumber + phiBin;

	return FunctionValues[i_tp];
}

double AngularGridFunction::value(const std::size_t thetaBin, const std::size_t phiBin) const
{
	const std::size_t i_tp = thetaBin * PhiBinNumber + phiBin;

	return FunctionValues[i_tp];
}

std::vector<double> &AngularGridFunction::value_vector()
{
	return FunctionValues;
}

const std::vector<double> &AngularGridFunction::value_vector() const
{
	return FunctionValues;
}

double AngularGridFunction::interpolate(const double theta, const double phi) const
{
	if ((theta < 0.0) or (theta > M_PI))
	{
		std::cout << "AngularGridFunction::interpolate Error: Coordinates out of range" << std::endl;

		exit(EXIT_FAILURE);
	}

	const double phiMod2Pi = phi - 2.0 * M_PI * std::floor(phi / (2.0 * M_PI));

	const std::size_t i_t_0 = static_cast<std::size_t>(std::floor(theta / ThetaBinWidth));
	const std::size_t i_p_0 = static_cast<std::size_t>(std::floor(phiMod2Pi / PhiBinWidth));

	const std::size_t i_t_1 = (i_t_0 != ThetaBinNumber - 1) ? i_t_0 + 1 : i_t_0;
	const std::size_t i_p_1 = (i_p_0 + 1) % PhiBinNumber;

	const double theta_0 = theta_coordinate(i_t_0);
	const double phi_0 = phi_coordinate(i_p_0);

	const double delta_t = (theta - theta_0) / ThetaBinWidth;
	const double delta_p = (phiMod2Pi - phi_0) / PhiBinWidth;

	const double f_00 = value(i_t_0, i_p_0);
	const double f_01 = value(i_t_0, i_p_1);
	const double f_10 = value(i_t_1, i_p_0);
	const double f_11 = value(i_t_1, i_p_1);

	const double f_t0 = (1.0 - delta_t) * f_00 + delta_t * f_10;
	const double f_t1 = (1.0 - delta_t) * f_01 + delta_t * f_11;

	const double f_tp = (1.0 - delta_p) * f_t0 + delta_p * f_t1;

	return f_tp;
}

double AngularGridFunction::operator()(const double theta, const double phi) const
{
	return interpolate(theta, phi);
}

double AngularGridFunction::solid_angle_element(const std::size_t thetaBin) const
{
	double solidAngleElement;

	const double theta = theta_coordinate(thetaBin);

	if ((thetaBin == 0) or (thetaBin == ThetaBinNumber - 1))
	{
		solidAngleElement = (1.0 - std::cos(0.5 * ThetaBinWidth)) * PhiBinWidth;
	}
	else
	{
		solidAngleElement = 2.0 * std::sin(theta) * std::sin(0.5 * ThetaBinWidth) * PhiBinWidth;
	}

	return solidAngleElement;
}

double AngularGridFunction::theta_coordinate(const std::size_t thetaBin) const
{
	return ThetaCoordinates[thetaBin];
}

double AngularGridFunction::phi_coordinate(const std::size_t phiBin) const
{
	return PhiCoordinates[phiBin];
}

std::vector<double> AngularGridFunction::theta_coordinate_vector() const
{
	return ThetaCoordinates;
}

std::vector<double> AngularGridFunction::phi_coordinate_vector() const
{
	return PhiCoordinates;
}

double AngularGridFunction::theta_bin_width() const
{
	return ThetaBinWidth;
}

double AngularGridFunction::phi_bin_width() const
{
	return PhiBinWidth;
}

std::size_t AngularGridFunction::theta_bin_number() const
{
	return ThetaBinNumber;
}

std::size_t AngularGridFunction::phi_bin_number() const
{
	return PhiBinNumber;
}

void AngularGridFunction::apply(const std::function<double(double)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double currentValue = value(i_t, i_p);

			value(i_t, i_p) = func(currentValue);
		}
	}
}

void AngularGridFunction::apply(const std::function<double(double value, double theta, double phi)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double theta = theta_coordinate(i_t);
			const double phi = phi_coordinate(i_p);

			const double currentValue = value(i_t, i_p);

			value(i_t, i_p) = func(currentValue, theta, phi);
		}
	}
}

void AngularGridFunction::evaluate_for_all_grid_points(const std::function<void(std::size_t thetaBin, std::size_t phiBin)> &func, bool parallelized) const
{
#pragma omp parallel for schedule(static) collapse(2) if (parallelized)
	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			func(i_t, i_p);
		}
	}
}

double AngularGridFunction::average(const std::function<double(double value)> &func) const
{
	double average = 0.0;

	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		const double solidAngleElement = solid_angle_element(i_t);

		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double val = value(i_t, i_p);

			average += func(val) * solidAngleElement;
		}
	}

	average /= 4.0 * M_PI;

	return average;
}

double AngularGridFunction::average(const std::function<double(double value, double theta, double phi)> &func) const
{
	double average = 0.0;

	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		const double theta = theta_coordinate(i_t);

		const double solidAngleElement = solid_angle_element(i_t);

		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double phi = phi_coordinate(i_p);

			const double val = value(i_t, i_p);

			average += func(val, theta, phi) * solidAngleElement;
		}
	}

	average /= 4.0 * M_PI;

	return average;
}

void AngularGridFunction::write_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName);
	outputFile.setf(std::ios::scientific);

	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		const double theta = theta_coordinate(i_t);

		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double phi = phi_coordinate(i_p);

			const double currentValue = value(i_t, i_p);

			outputFile << theta << "\t"
					   << phi << "\t"
					   << currentValue
					   << std::endl;
		}
	}

	outputFile.close();
}

void AngularGridFunction::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&ThetaBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&PhiBinNumber, sizeof(std::size_t));

	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			const double currentValue = value(i_t, i_p);

			outputFile.write((char *)&currentValue, sizeof(double));
		}
	}

	outputFile.close();
}

AngularGridFunction &AngularGridFunction::operator+=(const AngularGridFunction &otherGridFunction)
{
	if ((otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " AngularGridFunction::operator+= Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
								 { value(thetaBin, phiBin) += otherGridFunction.value(thetaBin, phiBin); });

	return *this;
}

AngularGridFunction &AngularGridFunction::operator-=(const AngularGridFunction &otherGridFunction)
{
	if ((otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " AngularGridFunction::operator-= Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
								 { value(thetaBin, phiBin) -= otherGridFunction.value(thetaBin, phiBin); });

	return *this;
}

AngularGridFunction &AngularGridFunction::operator*=(const AngularGridFunction &otherGridFunction)
{
	if ((otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " AngularGridFunction::operator*= Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
								 { value(thetaBin, phiBin) *= otherGridFunction.value(thetaBin, phiBin); });

	return *this;
}

AngularGridFunction &AngularGridFunction::operator/=(const AngularGridFunction &otherGridFunction)
{
	if ((otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " AngularGridFunction::operator/= Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
								 { value(thetaBin, phiBin) /= otherGridFunction.value(thetaBin, phiBin); });

	return *this;
}

AngularGridFunction &AngularGridFunction::operator+=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
									 { value(thetaBin, phiBin) += otherValue; });
	}

	return *this;
}

AngularGridFunction &AngularGridFunction::operator-=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
									 { value(thetaBin, phiBin) -= otherValue; });
	}

	return *this;
}

AngularGridFunction &AngularGridFunction::operator*=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
									 { value(thetaBin, phiBin) *= otherValue; });
	}

	return *this;
}

AngularGridFunction &AngularGridFunction::operator/=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
									 { value(thetaBin, phiBin) /= otherValue; });
	}

	return *this;
}

bool AngularGridFunction::has_same_grid_as(const AngularGridFunction &otherGridFunction) const
{
	return (ThetaBinNumber == otherGridFunction.ThetaBinNumber and
			PhiBinNumber == otherGridFunction.PhiBinNumber);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void AngularGridFunction::initialize_coordinates()
{
	for (std::size_t i_t = 0; i_t < ThetaBinNumber - 1; ++i_t)
	{
		ThetaCoordinates[i_t] = static_cast<double>(i_t) * ThetaBinWidth;
	}

	ThetaCoordinates[ThetaBinNumber - 1] = M_PI;

	for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
	{
		PhiCoordinates[i_p] = static_cast<double>(i_p) * PhiBinWidth;
	}
}

void AngularGridFunction::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " AngularGridFunction::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&ThetaBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&PhiBinNumber, sizeof(std::size_t));

	ThetaBinWidth = M_PI / static_cast<double>(ThetaBinNumber - 1);
	PhiBinWidth = 2.0 * M_PI / static_cast<double>(PhiBinNumber);
	ThetaCoordinates = std::vector<double>(ThetaBinNumber);
	PhiCoordinates = std::vector<double>(PhiBinNumber);
	FunctionValues = std::vector<double>(ThetaBinNumber * PhiBinNumber);

	initialize_coordinates();

	for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
	{
		for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
		{
			inputFile.read((char *)&value(i_t, i_p), sizeof(double));
		}
	}

	inputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

AngularGridFunction operator+(const AngularGridFunction &leftGridFunction, const AngularGridFunction &rightGridFunction)
{
	if ((leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator+ Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	AngularGridFunction newAngularGridFunction(leftGridFunction);

	newAngularGridFunction.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
														{ newAngularGridFunction.value(thetaBin, phiBin) += rightGridFunction.value(thetaBin, phiBin); });

	return newAngularGridFunction;
}

AngularGridFunction operator-(const AngularGridFunction &leftGridFunction, const AngularGridFunction &rightGridFunction)
{
	if ((leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator- Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	AngularGridFunction newAngularGridFunction(leftGridFunction);

	newAngularGridFunction.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
														{ newAngularGridFunction.value(thetaBin, phiBin) -= rightGridFunction.value(thetaBin, phiBin); });

	return newAngularGridFunction;
}

AngularGridFunction operator*(const AngularGridFunction &leftGridFunction, const AngularGridFunction &rightGridFunction)
{
	if ((leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator* Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	AngularGridFunction newAngularGridFunction(leftGridFunction);

	newAngularGridFunction.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
														{ newAngularGridFunction.value(thetaBin, phiBin) *= rightGridFunction.value(thetaBin, phiBin); });

	return newAngularGridFunction;
}

AngularGridFunction operator/(const AngularGridFunction &leftGridFunction, const AngularGridFunction &rightGridFunction)
{
	if ((leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator/ Error: Coordinates of angular grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	AngularGridFunction newAngularGridFunction(leftGridFunction);

	newAngularGridFunction.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
														{ newAngularGridFunction.value(thetaBin, phiBin) /= rightGridFunction.value(thetaBin, phiBin); });

	return newAngularGridFunction;
}

AngularGridFunction operator+(const AngularGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return AngularGridFunction(leftGridFunction, [&](double leftValue)
								   { return leftValue + rightValue; });
	}
}

AngularGridFunction operator-(const AngularGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return AngularGridFunction(leftGridFunction, [&](double leftValue)
								   { return leftValue - rightValue; });
	}
}

AngularGridFunction operator*(const AngularGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return AngularGridFunction(leftGridFunction, [&](double leftValue)
								   { return leftValue * rightValue; });
	}
}

AngularGridFunction operator/(const AngularGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return AngularGridFunction(leftGridFunction, [&](double leftValue)
								   { return leftValue / rightValue; });
	}
}

AngularGridFunction operator+(const double leftValue, const AngularGridFunction &rightGridFunction)
{
	return rightGridFunction + leftValue;
}

AngularGridFunction operator-(const double leftValue, const AngularGridFunction &rightGridFunction)
{
	return AngularGridFunction(rightGridFunction, [&](double rightValue)
							   { return leftValue - rightValue; });
}

AngularGridFunction operator*(const double leftValue, const AngularGridFunction &rightGridFunction)
{
	return rightGridFunction * leftValue;
}

AngularGridFunction operator/(const double leftValue, const AngularGridFunction &rightGridFunction)
{
	return AngularGridFunction(rightGridFunction, [&](double rightValue)
							   { return leftValue / rightValue; });
}

void transform_angular_grid_functions(const AngularGridFunction &input,
									  AngularGridFunction &output,
									  const std::function<void(std::size_t thetaBin, std::size_t phiBin,
															   double &output)> &trafo)
{
	const bool transformedInPlace = (&output == &input); // check if the transformation is in-place

	AngularGridFunction &fieldTemporary = transformedInPlace ? *(new AngularGridFunction(output)) // if it is (partially) in-place, work on a temporary copy of the output field
															 : output;

	if (not fieldTemporary.has_same_grid_as(input))
	{
		fieldTemporary = input;
	}

	fieldTemporary.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
												{
													double output;

													trafo(thetaBin, phiBin,
														  output);

													fieldTemporary.value(thetaBin, phiBin) = output;
												});

	if (transformedInPlace)
	{
		output = fieldTemporary;

		delete &fieldTemporary;
	}
}

void transform_angular_grid_functions(const AngularGridFunction &input1, const AngularGridFunction &input2,
									  AngularGridFunction &output1, AngularGridFunction &output2,
									  const std::function<void(std::size_t thetaBin, std::size_t phiBin,
															   double &output1, double &output2)> &trafo)
{
	if (not input1.has_same_grid_as(input2))
	{
		std::cout << std::endl
				  << " transform_angular_grid_functions Error: Coordinate grids of input AngularGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2);

	AngularGridFunction &fieldTemporary1 = transformed1InPlace ? *(new AngularGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
															   : output1;

	AngularGridFunction &fieldTemporary2 = transformed2InPlace ? *(new AngularGridFunction(output2))
															   : output2;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
										 {
											 double output1, output2;

											 trafo(thetaBin, phiBin,
												   output1, output2);

											 fieldTemporary1.value(thetaBin, phiBin) = output1;
											 fieldTemporary2.value(thetaBin, phiBin) = output2;
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

void transform_angular_grid_functions(const AngularGridFunction &input1, const AngularGridFunction &input2, const AngularGridFunction &input3,
									  AngularGridFunction &output1, AngularGridFunction &output2, AngularGridFunction &output3,
									  const std::function<void(std::size_t thetaBin, std::size_t phiBin,
															   double &output1, double &output2, double &output3)> &trafo)
{
	if (not input1.has_same_grid_as(input2) or
		not input1.has_same_grid_as(input3))
	{
		std::cout << std::endl
				  << " transform_angular_grid_functions Error: Coordinate grids of input AngularGridFunctions do not agree" << std::endl
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

	AngularGridFunction &fieldTemporary1 = transformed1InPlace ? *(new AngularGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
															   : output1;

	AngularGridFunction &fieldTemporary2 = transformed2InPlace ? *(new AngularGridFunction(output2))
															   : output2;

	AngularGridFunction &fieldTemporary3 = transformed3InPlace ? *(new AngularGridFunction(output3))
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

	output1.evaluate_for_all_grid_points([&](std::size_t thetaBin, std::size_t phiBin)
										 {
											 double output1, output2, output3;

											 trafo(thetaBin, phiBin,
												   output1, output2, output3);

											 fieldTemporary1.value(thetaBin, phiBin) = output1;
											 fieldTemporary2.value(thetaBin, phiBin) = output2;
											 fieldTemporary3.value(thetaBin, phiBin) = output3;
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