#include "SphericalFourierBesselDecomposition.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <tr1/cmath>
#include <vector>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>

#include "CQUADIntegrator.hpp"
#include "SphericalGridFunction.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

SphericalFourierBesselDecomposition::SphericalFourierBesselDecomposition(const double maxRadius, const double radialResolution, const std::size_t maxMultipole)
	: MaxRadius(maxRadius),
	  RadialResolution(radialResolution),
	  MaxMultipole(maxMultipole)
{
	initialize_SFB_decomposition();
}

SphericalFourierBesselDecomposition::SphericalFourierBesselDecomposition(const std::vector<double> &radialCoordinates, const std::vector<double> &thetaCoordinates, const std::vector<double> &phiCoordinates,
																		 const double maxRadius, const double radialResolution, const std::size_t maxMultipole,
																		 const std::function<double(double)> &weightingFunction, const std::function<double(double radius)> &isotropicContribution)
	: MaxRadius(maxRadius),
	  RadialResolution(radialResolution),
	  MaxMultipole(maxMultipole)
{
	if ((radialCoordinates.size() != thetaCoordinates.size()) || (radialCoordinates.size() != phiCoordinates.size()))
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition Error: Numbers of radial, theta and phi coordinates are different" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_SFB_decomposition();

	compute_spherical_fourier_bessel_decomposition_of_discrete_field(radialCoordinates, thetaCoordinates, phiCoordinates, weightingFunction, isotropicContribution);
}

SphericalFourierBesselDecomposition::SphericalFourierBesselDecomposition(const std::string &fileName)
{
	load_object_from_file(fileName);
}

SphericalFourierBesselDecomposition::SphericalFourierBesselDecomposition()
	: MaxRadius(0.0),
	  RadialResolution(0.0),
	  MaxMultipole(0),
	  MaxRadialWavenumber(0.0),
	  RadialModeNumber(1),
	  SFBCoefficients(),
	  RadialWavenumbers(),
	  RadialNormalizationConstants()
{
}

double SphericalFourierBesselDecomposition::radial_wavenumber(const std::size_t l, const std::size_t n) const
{
	const std::size_t i_ln = l * RadialModeNumber + n - 1;

	return RadialWavenumbers[i_ln];
}

double SphericalFourierBesselDecomposition::radial_normalization_constant(const std::size_t l, const std::size_t n) const
{
	const std::size_t i_ln = l * RadialModeNumber + n - 1;

	return RadialNormalizationConstants[i_ln];
}

std::complex<double> &SphericalFourierBesselDecomposition::SFB_coefficient(const std::size_t l, const std::size_t m, const std::size_t n)
{
	const std::size_t index = (l * (l + 1) / 2 + m) * RadialModeNumber + n - 1;

	return SFBCoefficients[index];
}

std::complex<double> SphericalFourierBesselDecomposition::SFB_coefficient(const std::size_t l, const std::intmax_t m, const std::size_t n) const
{
	if (l > MaxMultipole)
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition::SFB_coefficient Error: Angular mode number l larger than maximal multipole order specified" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if (static_cast<std::size_t>(std::abs(m)) > l)
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition::SFB_coefficient Error: Angular mode number m larger than angular mode number l" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if (n == 0)
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition::SFB_coefficient Error: Radial mode number n is not positive" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if (n > RadialModeNumber)
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition::SFB_coefficient Error: Radial mode number n is larger than maximal radial mode number specified" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const std::size_t index = (l * (l + 1) / 2 + std::abs(m)) * RadialModeNumber + n - 1;

	if (m >= 0)
	{
		return SFBCoefficients[index];
	}
	else
	{
		return std::conj(SFBCoefficients[index]) * static_cast<double>(1 - 2 * (std::abs(m) % 2));
	}
}

double SphericalFourierBesselDecomposition::field(const double radius, const double theta, const double phi, const double smoothingScale) const
{
	return compute_value_for_given_kernels(radius, theta, phi, smoothingScale,
										   radial_kernel_field, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

double SphericalFourierBesselDecomposition::radial_inverse_divergence(const double radius, const double theta, const double phi, const double smoothingScale) const
{
	return compute_value_for_given_kernels(radius, theta, phi, smoothingScale,
										   radial_kernel_radial_inverse_divergence, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

double SphericalFourierBesselDecomposition::theta_inverse_divergence(const double radius, const double theta, const double phi, const double smoothingScale) const
{
	return compute_value_for_given_kernels(radius, theta, phi, smoothingScale,
										   radial_kernel_angular_inverse_divergence, theta_kernel_theta_inverse_divergence, phi_kernel_scalar_radial_and_theta_component);
}

double SphericalFourierBesselDecomposition::phi_inverse_divergence(const double radius, const double theta, const double phi, const double smoothingScale) const
{
	return compute_value_for_given_kernels(radius, theta, phi, smoothingScale,
										   radial_kernel_angular_inverse_divergence, theta_kernel_phi_inverse_divergence, phi_kernel_phi_component);
}

double SphericalFourierBesselDecomposition::inverse_laplacian(const double radius, const double theta, const double phi, const double smoothingScale) const
{
	return compute_value_for_given_kernels(radius, theta, phi, smoothingScale,
										   radial_kernel_inverse_laplacian, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

SphericalGridFunction SphericalFourierBesselDecomposition::field_grid(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale) const
{
	return compute_grid_function_for_given_kernels(radialBinNumber, thetaBinNumber, phiBinNumber, smoothingScale,
												   radial_kernel_field, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

SphericalGridFunction SphericalFourierBesselDecomposition::radial_inverse_divergence_grid(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale) const
{
	return compute_grid_function_for_given_kernels(radialBinNumber, thetaBinNumber, phiBinNumber, smoothingScale,
												   radial_kernel_radial_inverse_divergence, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

SphericalGridFunction SphericalFourierBesselDecomposition::theta_inverse_divergence_grid(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale) const
{
	return compute_grid_function_for_given_kernels(radialBinNumber, thetaBinNumber, phiBinNumber, smoothingScale,
												   radial_kernel_angular_inverse_divergence, theta_kernel_theta_inverse_divergence, phi_kernel_scalar_radial_and_theta_component);
}

SphericalGridFunction SphericalFourierBesselDecomposition::phi_inverse_divergence_grid(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale) const
{
	return compute_grid_function_for_given_kernels(radialBinNumber, thetaBinNumber, phiBinNumber, smoothingScale,
												   radial_kernel_angular_inverse_divergence, theta_kernel_phi_inverse_divergence, phi_kernel_phi_component);
}

SphericalGridFunction SphericalFourierBesselDecomposition::inverse_laplacian_grid(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale) const
{
	return compute_grid_function_for_given_kernels(radialBinNumber, thetaBinNumber, phiBinNumber, smoothingScale,
												   radial_kernel_inverse_laplacian, theta_kernel_scalar_and_radial_component, phi_kernel_scalar_radial_and_theta_component);
}

double SphericalFourierBesselDecomposition::maximal_radius() const
{
	return MaxRadius;
}

double SphericalFourierBesselDecomposition::radial_resolution() const
{
	return RadialResolution;
}

std::size_t SphericalFourierBesselDecomposition::maximal_multipole() const
{
	return MaxMultipole;
}

double SphericalFourierBesselDecomposition::maximal_radial_wavenumber() const
{
	return MaxRadialWavenumber;
}

std::size_t SphericalFourierBesselDecomposition::radial_mode_number() const
{
	return RadialModeNumber;
}

void SphericalFourierBesselDecomposition::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&MaxRadius, sizeof(double));
	outputFile.write((char *)&RadialResolution, sizeof(double));
	outputFile.write((char *)&MaxMultipole, sizeof(std::size_t));

	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t m = 0; m <= l; ++m)
		{
			for (std::size_t n = 1; n <= RadialModeNumber; ++n)
			{
				const double currentCoefficientReal = SFB_coefficient(l, m, n).real();
				const double currentCoefficientImag = SFB_coefficient(l, m, n).imag();

				outputFile.write((char *)&currentCoefficientReal, sizeof(double));
				outputFile.write((char *)&currentCoefficientImag, sizeof(double));
			}
		}
	}

	outputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void SphericalFourierBesselDecomposition::initialize_SFB_decomposition()
{
	MaxRadialWavenumber = RadialResolution / MaxRadius;							// maximal wavenumber considered in the decomposition
	RadialModeNumber = static_cast<std::size_t>(RadialResolution / M_PI + 0.5); // maximal n needed to store all k_ln <= k_max is obtained by inverting the relation for l=0 since the monopole has the smallest radial wavenumbers

	SFBCoefficients = std::vector<std::complex<double>>(((MaxMultipole + 2) * (MaxMultipole + 1) * RadialModeNumber) / 2, std::complex<double>(0.0, 0.0)); // number of SFB coefficients with 0 <= l <= l_max, 0 <= m <= l and 1 <= n <= n_max
	RadialWavenumbers = std::vector<double>((MaxMultipole + 1) * RadialModeNumber);																		   // number of k_ln and C_ln with 0 <= l <= l_max and 1 <= n <= n_max
	RadialNormalizationConstants = std::vector<double>((MaxMultipole + 1) * RadialModeNumber);

	for (std::size_t l = 0; l <= MaxMultipole; ++l) // compute k_ln and C_ln for potential boundary conditions
	{
		for (std::size_t n = 1; n <= RadialModeNumber; ++n)
		{
			const double k_ln_R = (l == 0) ? (static_cast<double>(n) - 0.5) * M_PI : gsl_sf_bessel_zero_Jnu(static_cast<double>(l) - 0.5, n); // special case for l = 0 since the GSL implementation of the zeroes of Bessel functions J_nu only allows positive nu

			const double k_ln = k_ln_R / MaxRadius;

			const double C_ln = 2.0 / gsl_pow_3(MaxRadius) / gsl_pow_2(std::tr1::sph_bessel(l, k_ln_R));

			const std::size_t i_ln = l * RadialModeNumber + n - 1;

			RadialWavenumbers[i_ln] = k_ln;
			RadialNormalizationConstants[i_ln] = C_ln;
		}
	}
}

void SphericalFourierBesselDecomposition::compute_spherical_fourier_bessel_decomposition_of_discrete_field(const std::vector<double> &radialCoordinates, const std::vector<double> &thetaCoordinates, const std::vector<double> &phiCoordinates, const std::function<double(double radius)> &weightingFunction, const std::function<double(double radius)> &isotropicContribution)
{
	const std::size_t pointNumber = radialCoordinates.size();

	std::vector<double> radialKernelPointValues((MaxMultipole + 1) * RadialModeNumber * pointNumber, 0.0); // for all l, n and points, precompute the values of the purely radially dependent factors appearing in the SFB decomposition; this improves the performance by avoiding unnecessary recalculations

#pragma omp parallel for schedule(dynamic)
	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t n = 1; n <= RadialModeNumber; ++n)
		{
			const std::size_t i_ln = l * RadialModeNumber + n - 1;

			const double k_ln = RadialWavenumbers[i_ln];

			if (k_ln > MaxRadialWavenumber)
			{
				break;
			}

			for (std::size_t i_p = 0; i_p < pointNumber; ++i_p)
			{
				const double radius = radialCoordinates[i_p];

				const double w = weightingFunction(radius);

				const double j_ln = std::tr1::sph_bessel(l, k_ln * radius);

				const std::size_t i_lnd = i_ln * pointNumber + i_p;

				radialKernelPointValues[i_lnd] = j_ln * w;
			}
		}
	}

	std::vector<std::complex<double>> angularKernelPointValues((MaxMultipole + 1) * (MaxMultipole + 2) / 2 * pointNumber); // for all l, m and points, precompute the values of the purely angularly dependent factors appearing in the SFB decomposition; this improves the performance by avoiding unnecessary recalculations

#pragma omp parallel for schedule(dynamic)
	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t m = 0; m <= l; ++m)
		{
			for (std::size_t i_p = 0; i_p < pointNumber; ++i_p)
			{
				const double theta = thetaCoordinates[i_p];
				const double phi = phiCoordinates[i_p];

				const double YlmModulus = std::tr1::sph_legendre(l, m, theta);

				std::complex<double> Ylm = std::polar(YlmModulus, -static_cast<double>(m) * phi); // minus sign since the complex conjugate of the spherical harmonic is needed

				const std::size_t i_lmp = (l * (l + 1) / 2 + m) * pointNumber + i_p;

				angularKernelPointValues[i_lmp] = Ylm;
			}
		}
	}

#pragma omp parallel for schedule(dynamic)
	for (std::size_t l = 0; l <= MaxMultipole; ++l) // compute the SFB coefficients using the precomputed radial and angular contributions
	{
		for (std::size_t m = 0; m <= l; ++m)
		{
			for (std::size_t n = 1; n <= RadialModeNumber; ++n)
			{
				const std::size_t i_ln = l * RadialModeNumber + n - 1;

				const double k_ln = RadialWavenumbers[i_ln];

				if (k_ln > MaxRadialWavenumber)
				{
					break;
				}

				const std::size_t i_lmn = (l * (l + 1) / 2 + m) * RadialModeNumber + n - 1;

				for (std::size_t i_p = 0; i_p < pointNumber; ++i_p)
				{
					const std::size_t i_lnd = i_ln * pointNumber + i_p;
					const std::size_t i_lmd = (l * (l + 1) / 2 + m) * pointNumber + i_p;

					const double j_ln_w = radialKernelPointValues[i_lnd];
					const std::complex<double> Ylm = angularKernelPointValues[i_lmd];

					SFBCoefficients[i_lmn] += j_ln_w * Ylm;
				}

				if ((l == 0) && (m == 0)) // subtract isotropic contribution from monopole
				{
					auto integrand = [&](double r)
					{
						const double k_ln_r = k_ln * r;

						return 2.0 * std::sqrt(M_PI) * r * r * std::tr1::sph_bessel(0, k_ln_r) * isotropicContribution(r);
					};

					CQUADIntegrator integrator(integrand);

					double monopoleContribution = 0.0;

					integrator.integrate(0.0, MaxRadius,
										 0.0, 1e-5,
										 monopoleContribution);

					SFBCoefficients[i_lmn] -= monopoleContribution;
				}
			}
		}
	}
}

void SphericalFourierBesselDecomposition::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " SphericalFourierBesselDecomposition::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&MaxRadius, sizeof(double));
	inputFile.read((char *)&RadialResolution, sizeof(double));
	inputFile.read((char *)&MaxMultipole, sizeof(std::size_t));

	initialize_SFB_decomposition();

	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t m = 0; m <= l; ++m)
		{
			for (std::size_t n = 1; n <= RadialModeNumber; ++n)
			{
				double currentCoefficientReal, currentCoefficientImag;

				inputFile.read((char *)&currentCoefficientReal, sizeof(double));
				inputFile.read((char *)&currentCoefficientImag, sizeof(double));

				SFB_coefficient(l, m, n) = std::complex<double>(currentCoefficientReal, currentCoefficientImag);
			}
		}
	}

	inputFile.close();
}

double SphericalFourierBesselDecomposition::compute_value_for_given_kernels(const double radius, const double theta, const double phi, const double smoothingScale,
																			const std::function<double(std::size_t l, double k_ln, double radius)> &radialKernelFunction,
																			const std::function<double(std::size_t l, std::size_t m, double theta)> &thetaKernelFunction,
																			const std::function<std::complex<double>(std::size_t m, double phi)> &phiKernelFunction) const
{
	double value = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+ \
													 : value)
	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t n = 1; n <= RadialModeNumber; ++n)
		{
			const std::size_t i_ln = l * RadialModeNumber + n - 1;

			const double k_ln = RadialWavenumbers[i_ln];

			if (k_ln > MaxRadialWavenumber)
			{
				break;
			}

			const double C_ln = RadialNormalizationConstants[i_ln];

			const double windowFunction = std::exp(-gsl_pow_2(k_ln * smoothingScale) / 2.0);

			const double radialKernel = C_ln * windowFunction * radialKernelFunction(l, k_ln, radius);

			for (std::size_t m = 0; m <= l; ++m)
			{
				const std::complex<double> angularKernel = thetaKernelFunction(l, m, theta) * phiKernelFunction(m, phi);

				const std::size_t i_lmn = (l * (l + 1) / 2 + m) * RadialModeNumber + n - 1;

				const std::complex<double> radialContribution = radialKernel * SFBCoefficients[i_lmn];

				value += (m == 0 ? 1.0 : 2.0) * (std::real(radialContribution) * std::real(angularKernel) - std::imag(radialContribution) * std::imag(angularKernel)); // each term is the sum of the contributions from modes m and -m
			}
		}
	}

	return value;
}

SphericalGridFunction SphericalFourierBesselDecomposition::compute_grid_function_for_given_kernels(const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber, const double smoothingScale,
																								   const std::function<double(std::size_t l, double k_ln, double radius)> &radialKernelFunction,
																								   const std::function<double(std::size_t l, std::size_t m, double theta)> &thetaKernelFunction,
																								   const std::function<std::complex<double>(std::size_t m, double phi)> &phiKernelFunction) const
{
	SphericalGridFunction realSpaceField(MaxRadius, radialBinNumber, thetaBinNumber, phiBinNumber);

	std::vector<double> radialKernelGridValues((MaxMultipole + 1) * RadialModeNumber * radialBinNumber, 0.0); // for all l, n and radii, precompute the values of the purely radially dependent factors appearing in the SFB decomposition; this improves the performance by avoiding unnecessary recalculations

#pragma omp parallel for schedule(dynamic)
	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t n = 1; n <= RadialModeNumber; ++n)
		{
			const std::size_t i_ln = l * RadialModeNumber + n - 1;

			const double k_ln = RadialWavenumbers[i_ln];

			if (k_ln > MaxRadialWavenumber)
			{
				break;
			}

			const double C_ln = RadialNormalizationConstants[i_ln];

			const double windowFunction = std::exp(-gsl_pow_2(k_ln * smoothingScale) / 2.0);

			for (std::size_t i_r = 0; i_r < radialBinNumber; ++i_r)
			{
				const double radius = realSpaceField.radial_coordinate(i_r);

				const std::size_t i_lnr = i_ln * radialBinNumber + i_r;

				radialKernelGridValues[i_lnr] = C_ln * windowFunction * radialKernelFunction(l, k_ln, radius);
			}
		}
	}

	std::vector<std::complex<double>> angularKernelGridValues((MaxMultipole + 1) * (MaxMultipole + 2) / 2 * thetaBinNumber * phiBinNumber); // for all l, m and angles, precompute the values of the purely angularly dependent factors appearing in the SFB decomposition; this improves the performance by avoiding unnecessary recalculations

#pragma omp parallel for schedule(dynamic)
	for (std::size_t l = 0; l <= MaxMultipole; ++l)
	{
		for (std::size_t m = 0; m <= l; ++m)
		{
			for (std::size_t i_t = 0; i_t < thetaBinNumber; ++i_t)
			{
				const double theta = realSpaceField.theta_coordinate(i_t);

				const double thetaKernel = thetaKernelFunction(l, m, theta);

				for (std::size_t i_p = 0; i_p < phiBinNumber; ++i_p)
				{
					const double phi = realSpaceField.phi_coordinate(i_p);

					const std::size_t i_lmtp = ((l * (l + 1) / 2 + m) * thetaBinNumber + i_t) * phiBinNumber + i_p;

					angularKernelGridValues[i_lmtp] = thetaKernel * phiKernelFunction(m, phi);
				}
			}
		}
	}

#pragma omp parallel for schedule(dynamic)
	for (std::size_t i_r = 0; i_r < radialBinNumber; ++i_r) // compute the values at all grid points using the precomputed radial and angular contributions
	{
		for (std::size_t l = 0; l <= MaxMultipole; ++l)
		{
			for (std::size_t m = 0; m <= l; ++m)
			{
				std::complex<double> radialContribution(0.0, 0.0);

				for (std::size_t n = 1; n <= RadialModeNumber; ++n)
				{
					const std::size_t i_ln = l * RadialModeNumber + n - 1;

					const double k_ln = RadialWavenumbers[i_ln];

					if (k_ln > MaxRadialWavenumber)
					{
						break;
					}

					const std::size_t i_lnr = i_ln * radialBinNumber + i_r;
					const std::size_t i_lmn = (l * (l + 1) / 2 + m) * RadialModeNumber + n - 1;

					const double radialKernel = radialKernelGridValues[i_lnr];

					radialContribution += SFBCoefficients[i_lmn] * radialKernel;
				}

				const double multipliticity = (m == 0 ? 1.0 : 2.0);

				for (std::size_t i_t = 0; i_t < thetaBinNumber; ++i_t)
				{
					for (std::size_t i_p = 0; i_p < phiBinNumber; ++i_p)
					{
						const std::size_t i_lmtp = ((l * (l + 1) / 2 + m) * thetaBinNumber + i_t) * phiBinNumber + i_p;

						std::complex<double> angularKernel = angularKernelGridValues[i_lmtp];

						realSpaceField.value(i_r, i_t, i_p) += multipliticity * (std::real(radialContribution) * std::real(angularKernel) - std::imag(radialContribution) * std::imag(angularKernel)); // each term is the sum of the contributions from modes m and -m
					}
				}
			}
		}
	}

	return realSpaceField;
}

double SphericalFourierBesselDecomposition::radial_kernel_field(const std::size_t l, const double k_ln, const double radius)
{
	const double j_ln = std::tr1::sph_bessel(l, k_ln * radius);

	return j_ln;
}

double SphericalFourierBesselDecomposition::radial_kernel_radial_inverse_divergence(const std::size_t l, const double k_ln, const double radius)
{
	const double j_ln = std::tr1::sph_bessel(l, k_ln * radius);
	const double j_lPlus1n = std::tr1::sph_bessel(l + 1, k_ln * radius);
	const double j_ln_Deriv = (radius != 0.0) ? j_ln * static_cast<double>(l) / (k_ln * radius) - j_lPlus1n : (l == 1) ? 1.0 / 3.0
																													   : 0.0; // use relation between spherical Bessel functions and their derivatives

	return -j_ln_Deriv / k_ln;
}

double SphericalFourierBesselDecomposition::radial_kernel_angular_inverse_divergence(const std::size_t l, const double k_ln, const double radius)
{
	const double j_ln_over_r_kln = (radius != 0.0) ? std::tr1::sph_bessel(l, k_ln * radius) / radius / k_ln : (l == 1) ? 1.0 / 3.0
																													   : 0.0;

	return j_ln_over_r_kln / k_ln;
}

double SphericalFourierBesselDecomposition::radial_kernel_inverse_laplacian(const std::size_t l, const double k_ln, const double radius)
{
	const double j_ln = std::tr1::sph_bessel(l, k_ln * radius);

	return -j_ln / k_ln / k_ln;
}

double SphericalFourierBesselDecomposition::theta_kernel_scalar_and_radial_component(const std::size_t l, const std::size_t m, const double theta)
{
	const double YlmModulus = std::tr1::sph_legendre(l, m, theta);

	return YlmModulus;
}

double SphericalFourierBesselDecomposition::theta_kernel_theta_inverse_divergence(const std::size_t l, const std::size_t m, const double theta)
{
	const double YlmPlus1Modulus = (m != l) ? std::tr1::sph_legendre(l, m + 1, theta) : 0.0;
	const double YlmMinus1Modulus = (m != 0) ? std::tr1::sph_legendre(l, m - 1, theta) : ((l != 0) ? -std::tr1::sph_legendre(l, 1, theta) : 0.0);

	const double LlmPlus = std::sqrt(static_cast<double>((l - m) * (l + m + 1)));
	const double LlmMinus = std::sqrt(static_cast<double>((l + m) * (l - m + 1)));

	return (LlmMinus * YlmMinus1Modulus - LlmPlus * YlmPlus1Modulus) / 2.0;
}

double SphericalFourierBesselDecomposition::theta_kernel_phi_inverse_divergence(const std::size_t l, const std::size_t m, const double theta)
{
	const double cosTheta = std::cos(theta);
	const double sinTheta = std::sin(theta);

	const double YlmModulus = std::tr1::sph_legendre(l, m, theta);
	const double YlmPlus1Modulus = (m != l) ? std::tr1::sph_legendre(l, m + 1, theta) : 0.0;
	const double YlmMinus1Modulus = (m != 0) ? std::tr1::sph_legendre(l, m - 1, theta) : ((l != 0) ? -std::tr1::sph_legendre(l, 1, theta) : 0.0);

	const double LlmZ = static_cast<double>(m);
	const double LlmPlus = std::sqrt(static_cast<double>((l - m) * (l + m + 1)));
	const double LlmMinus = std::sqrt(static_cast<double>((l + m) * (l - m + 1)));

	return cosTheta * (LlmMinus * YlmMinus1Modulus + LlmPlus * YlmPlus1Modulus) / 2.0 - sinTheta * LlmZ * YlmModulus;
}

std::complex<double> SphericalFourierBesselDecomposition::phi_kernel_scalar_radial_and_theta_component(const std::size_t m, const double phi)
{
	const std::complex<double> YlmPhase = std::polar(1.0, static_cast<double>(m) * phi);

	return YlmPhase;
}

std::complex<double> SphericalFourierBesselDecomposition::phi_kernel_phi_component(const std::size_t m, const double phi)
{
	const std::complex<double> i_YlmPhase = std::polar(1.0, static_cast<double>(m) * phi + M_PI / 2.0); // express imaginary factor as phase shift

	return i_YlmPhase;
}