#include "NormalizedPowerSpectrum.hpp"

#include <cstddef>
#include <iostream>
#include <tr1/cmath>

#include <gsl/gsl_math.h>

#include "CQUADIntegrator.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

NormalizedPowerSpectrum::NormalizedPowerSpectrum(const std::vector<double> &wavenumbers, const std::vector<double> &powerSpectrumValues,
												 const double lengthRescaling, const double smoothingScale,
												 const bool useTophatFilter,
												 const double integrationAbsoluteError, const double integrationRelativeError,
												 const gsl_interp_type *gslInterpolationType)
	: SmoothingScale(smoothingScale),
	  Interpolator(),
	  Variance(),
	  MaxWavenumber()
{
	if (wavenumbers.size() != powerSpectrumValues.size())
	{
		std::cout << std::endl
				  << " NormalizedPowerSpectrum Error: Numbers of wavenumber and spectrum values are different" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	std::vector<double> rescaledWavenumbers = wavenumbers;
	std::vector<double> rescaledSpectrumValues = powerSpectrumValues;

	for (std::size_t i_k = 0; i_k < rescaledWavenumbers.size(); ++i_k)
	{
		rescaledWavenumbers[i_k] /= lengthRescaling;
		rescaledSpectrumValues[i_k] *= gsl_pow_3(lengthRescaling);
	}

	if (rescaledWavenumbers[0] > 0.0) // ensure that the power spectrum vanishes at zero wavenumber
	{
		rescaledWavenumbers.insert(rescaledWavenumbers.begin(), 0.0);
		rescaledSpectrumValues.insert(rescaledSpectrumValues.begin(), 0.0);
	}

	MaxWavenumber = rescaledWavenumbers.back();

	Interpolator = SplineInterpolator(rescaledWavenumbers, rescaledSpectrumValues, gslInterpolationType);

	CQUADIntegrator varianceIntegrator = useTophatFilter ? CQUADIntegrator([&](double k)
																		   { return tophat_filter_integrand(k); })
														 : CQUADIntegrator([&](double k)
																		   { return gaussian_filter_integrand(k); });

	varianceIntegrator.integrate(0.0, MaxWavenumber, integrationAbsoluteError, integrationRelativeError, Variance); // integrate up to maximal k provided as input
}

double NormalizedPowerSpectrum::evaluate(const double k) const
{
	return Interpolator(k) / Variance;
}

double NormalizedPowerSpectrum::operator()(const double k) const
{
	return evaluate(k);
}

double NormalizedPowerSpectrum::smoothing_scale() const
{
	return SmoothingScale;
}

double NormalizedPowerSpectrum::variance() const
{
	return Variance;
}

double NormalizedPowerSpectrum::variance(const std::function<double(double k)> &kernel,
										 const bool useTophatFilter,
										 const double integrationAbsoluteError, const double integrationRelativeError) const
{
	CQUADIntegrator varianceIntegrator = useTophatFilter ? CQUADIntegrator([&](double k)
																		   { return kernel(k) * tophat_filter_integrand(k); })
														 : CQUADIntegrator([&](double k)
																		   { return kernel(k) * gaussian_filter_integrand(k); });

	double variance;

	varianceIntegrator.integrate(0.0, MaxWavenumber, integrationAbsoluteError, integrationRelativeError, variance); // integrate up to maximal k provided as input

	return variance;
}

double NormalizedPowerSpectrum::tophat_filter_integrand(double k) const
{
	const double kR = k * SmoothingScale;
	const double P = Interpolator(k);
	const double W = 3.0 * std::tr1::sph_bessel(1, kR) / kR;

	return k * k * P * W * W / (2.0 * gsl_pow_2(M_PI));
}

double NormalizedPowerSpectrum::gaussian_filter_integrand(double k) const
{
	const double kR = k * SmoothingScale;
	const double P = Interpolator(k);
	const double W = std::exp(-gsl_pow_2(kR) / 2.0);

	return k * k * P * W * W / (2.0 * gsl_pow_2(M_PI));
}
