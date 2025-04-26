#include "SplineInterpolator.hpp"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

SplineInterpolator::SplineInterpolator(const std::vector<double> &arguments, const std::vector<double> &functionValues, const gsl_interp_type *gslInterpolationType)
	: Arguments(arguments),
	  FunctionValues(functionValues),
	  InterpolationType(gslInterpolationType),
	  Accelerator(gsl_interp_accel_alloc()),
	  Interpolator(gsl_interp_alloc(gslInterpolationType, arguments.size()))
{
	if (Arguments.size() != FunctionValues.size())
	{
		std::cout << std::endl
				  << " SplineInterpolator Error: Numbers of arguments and function values are different" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	gsl_interp_init(Interpolator, Arguments.data(), FunctionValues.data(), Arguments.size());
}

double SplineInterpolator::evaluate(const double argument) const
{
	return gsl_interp_eval(Interpolator, Arguments.data(), FunctionValues.data(), argument, Accelerator);
}

double SplineInterpolator::operator()(const double argument) const
{
	return evaluate(argument);
}

double SplineInterpolator::derivative(const double argument) const
{
	return gsl_interp_eval_deriv(Interpolator, Arguments.data(), FunctionValues.data(), argument, Accelerator);
}

double SplineInterpolator::second_derivative(const double argument) const
{
	return gsl_interp_eval_deriv2(Interpolator, Arguments.data(), FunctionValues.data(), argument, Accelerator);
}

double SplineInterpolator::integral(const double lowerIntegrationLimit, const double upperIntegrationLimit) const
{
	return gsl_interp_eval_integ(Interpolator, Arguments.data(), FunctionValues.data(), lowerIntegrationLimit, upperIntegrationLimit, Accelerator);
}

SplineInterpolator::~SplineInterpolator()
{
	gsl_interp_accel_free(Accelerator);
	gsl_interp_free(Interpolator);
}

SplineInterpolator::SplineInterpolator(const SplineInterpolator &otherSplineInterpolator)
	: Arguments(otherSplineInterpolator.Arguments),
	  FunctionValues(otherSplineInterpolator.FunctionValues),
	  InterpolationType(otherSplineInterpolator.InterpolationType),
	  Accelerator(gsl_interp_accel_alloc()),
	  Interpolator(gsl_interp_alloc(otherSplineInterpolator.InterpolationType, otherSplineInterpolator.Arguments.size()))
{
	gsl_interp_init(Interpolator, Arguments.data(), FunctionValues.data(), Arguments.size());
}

SplineInterpolator::SplineInterpolator()
	: Arguments(),
	  FunctionValues(),
	  InterpolationType(),
	  Accelerator(),
	  Interpolator()
{
}

SplineInterpolator &SplineInterpolator::operator=(const SplineInterpolator &otherSplineInterpolator)
{
	Arguments = otherSplineInterpolator.Arguments;
	FunctionValues = otherSplineInterpolator.FunctionValues;
	InterpolationType = otherSplineInterpolator.InterpolationType;
	Accelerator = gsl_interp_accel_alloc();
	Interpolator = gsl_interp_alloc(otherSplineInterpolator.InterpolationType, otherSplineInterpolator.Arguments.size());

	gsl_interp_init(Interpolator, Arguments.data(), FunctionValues.data(), Arguments.size());

	return *this;
}