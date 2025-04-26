#include "SphericalGridFunctionInterpolator.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

SphericalGridFunctionInterpolator::SphericalGridFunctionInterpolator(const double minParameter, const double maxParameter, const std::vector<SphericalGridFunction> &gridFunctions)
	: MinParameter(minParameter),
	  MaxParameter(maxParameter),
	  GridFunctions(gridFunctions),
	  BinNumber(GridFunctions.size()),
	  BinWidth((MaxParameter - MinParameter) / static_cast<double>(BinNumber - 1))
{
	if (MinParameter >= MaxParameter)
	{
		std::cout << std::endl
				  << " SphericalGridFunctionInterpolator Error: Minimal parameter value not smaller than maximal parameter value" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if (BinNumber < 2)
	{
		std::cout << std::endl
				  << " SphericalGridFunctionInterpolator Error: SphericalGridFunction vector contains less than two objects" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	ParameterValues.push_back(MinParameter);

	for (std::size_t i_p = 1; i_p < BinNumber - 1; ++i_p)
	{
		ParameterValues.push_back(MinParameter + static_cast<double>(i_p) * BinWidth);
	}

	ParameterValues.push_back(MaxParameter);
}

SphericalGridFunction SphericalGridFunctionInterpolator::operator()(const double parameter) const
{
	const std::size_t i_0 = static_cast<std::size_t>(std::floor((parameter - MinParameter) / BinWidth));
	const std::size_t i_1 = (i_0 != BinNumber - 1) ? i_0 + 1
												   : i_0;

	const double parameter_0 = ParameterValues[i_0];

	const SphericalGridFunction gridFunction_0 = GridFunctions[i_0];
	const SphericalGridFunction gridFunction_1 = GridFunctions[i_1];

	return gridFunction_0 * (BinWidth - parameter + parameter_0) / BinWidth + gridFunction_1 * (parameter - parameter_0) / BinWidth;
}

double SphericalGridFunctionInterpolator::operator()(const double parameter, const double radius, const double theta, const double phi) const
{
	const std::size_t i_0 = static_cast<std::size_t>(std::floor((parameter - MinParameter) / BinWidth));
	const std::size_t i_1 = (i_0 != BinNumber - 1) ? i_0 + 1
												   : i_0;

	const double parameter_0 = ParameterValues[i_0];

	const double value_0 = GridFunctions[i_0](radius, theta, phi);
	const double value_1 = GridFunctions[i_1](radius, theta, phi);

	return value_0 + (value_1 - value_0) * (parameter - parameter_0) / BinWidth;
}

SphericalGridFunction SphericalGridFunctionInterpolator::at(const std::size_t bin) const
{
	return GridFunctions[bin];
}

double SphericalGridFunctionInterpolator::at(const std::size_t bin, const double radius, const double theta, const double phi) const
{
	return GridFunctions[bin](radius, theta, phi);
}

double SphericalGridFunctionInterpolator::parameter(const std::size_t bin) const
{
	return ParameterValues[bin];
}

double SphericalGridFunctionInterpolator::min_parameter() const
{
	return MinParameter;
}

double SphericalGridFunctionInterpolator::max_parameter() const
{
	return MaxParameter;
}

double SphericalGridFunctionInterpolator::bin_width() const
{
	return BinWidth;
}

std::size_t SphericalGridFunctionInterpolator::bin_number() const
{
	return BinNumber;
}