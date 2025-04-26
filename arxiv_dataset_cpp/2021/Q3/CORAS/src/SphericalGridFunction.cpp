#include "SphericalGridFunction.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Cartesian1DGridFunction.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

SphericalGridFunction::SphericalGridFunction(const double maxRadius, const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber,
											 const double value)
	: MaxRadius(maxRadius),
	  RadialBinNumber(radialBinNumber),
	  ThetaBinNumber(thetaBinNumber),
	  PhiBinNumber(phiBinNumber),
	  RadialBinWidth(MaxRadius / static_cast<double>(RadialBinNumber - 1)),
	  ThetaBinWidth(M_PI / static_cast<double>(ThetaBinNumber - 1)),
	  PhiBinWidth(2.0 * M_PI / static_cast<double>(PhiBinNumber)),
	  RadialCoordinates(RadialBinNumber),
	  ThetaCoordinates(ThetaBinNumber),
	  PhiCoordinates(PhiBinNumber),
	  FunctionValues(RadialBinNumber * ThetaBinNumber * PhiBinNumber, value)
{
	if (maxRadius <= 0)
	{
		std::cout << std::endl
				  << " SphericalGridFunction Error: Maximal radius is not positive" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	if ((RadialBinNumber < 2) or (ThetaBinNumber < 2) or (PhiBinNumber < 2))
	{
		std::cout << std::endl
				  << " SphericalGridFunction Error: Number of bins per axis is less than two" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	initialize_coordinates();
}

SphericalGridFunction::SphericalGridFunction(const double maxRadius, const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber,
											 const std::function<double(double radius, double theta, double phi)> &func)
	: SphericalGridFunction(maxRadius, radialBinNumber, thetaBinNumber, phiBinNumber, 0.0)
{
	apply([&](double value, double radius, double theta, double phi)
		  { return func(radius, theta, phi); });
}

SphericalGridFunction::SphericalGridFunction(const std::string &fileName)
{
	load_object_from_file(fileName);
}

SphericalGridFunction::SphericalGridFunction(const SphericalGridFunction &otherSphericalGridFunction, const std::function<double(double value)> &func)
	: SphericalGridFunction(otherSphericalGridFunction)
{
	apply(func);
}

SphericalGridFunction::SphericalGridFunction(const SphericalGridFunction &otherSphericalGridFunction, const std::function<double(double value, double radius, double theta, double phi)> &func)
	: SphericalGridFunction(otherSphericalGridFunction)
{
	apply(func);
}

SphericalGridFunction::SphericalGridFunction()
	: MaxRadius(0.0),
	  RadialBinNumber(0),
	  ThetaBinNumber(0),
	  PhiBinNumber(0),
	  RadialBinWidth(0.0),
	  ThetaBinWidth(0.0),
	  PhiBinWidth(0.0),
	  RadialCoordinates(),
	  ThetaCoordinates(),
	  PhiCoordinates(),
	  FunctionValues()
{
}

double &SphericalGridFunction::value(const std::size_t radialBin, const std::size_t thetaBin, const std::size_t phiBin)
{
	const std::size_t i_rtp = (radialBin * ThetaBinNumber + thetaBin) * PhiBinNumber + phiBin;

	return FunctionValues[i_rtp];
}

double SphericalGridFunction::value(const std::size_t radialBin, const std::size_t thetaBin, const std::size_t phiBin) const
{
	const std::size_t i_rtp = (radialBin * ThetaBinNumber + thetaBin) * PhiBinNumber + phiBin;

	return FunctionValues[i_rtp];
}

std::vector<double> &SphericalGridFunction::value_vector()
{
	return FunctionValues;
}

const std::vector<double> &SphericalGridFunction::value_vector() const
{
	return FunctionValues;
}

double SphericalGridFunction::interpolate(const double radius, const double theta, const double phi) const
{
	if ((radius < 0.0) or (radius > MaxRadius) or (theta < 0.0) or (theta > M_PI))
	{
		std::cout << "SphericalGridFunction::interpolate Error: Coordinates out of range: (" << radius << ", " << theta << ", " << phi << ")" << std::endl;

		exit(EXIT_FAILURE);
	}

	const double phiMod2Pi = phi - 2.0 * M_PI * std::floor(phi / (2.0 * M_PI));

	const std::size_t i_r_0 = static_cast<std::size_t>(std::floor(radius / RadialBinWidth));
	const std::size_t i_t_0 = static_cast<std::size_t>(std::floor(theta / ThetaBinWidth));
	const std::size_t i_p_0 = static_cast<std::size_t>(std::floor(phiMod2Pi / PhiBinWidth));

	const std::size_t i_r_1 = (i_r_0 != RadialBinNumber - 1) ? i_r_0 + 1 : i_r_0;
	const std::size_t i_t_1 = (i_t_0 != ThetaBinNumber - 1) ? i_t_0 + 1 : i_t_0;
	const std::size_t i_p_1 = (i_p_0 + 1) % PhiBinNumber;

	const double radius_0 = radial_coordinate(i_r_0);
	const double theta_0 = theta_coordinate(i_t_0);
	const double phi_0 = phi_coordinate(i_p_0);

	const double delta_r = (radius - radius_0) / RadialBinWidth;
	const double delta_t = (theta - theta_0) / ThetaBinWidth;
	const double delta_p = (phiMod2Pi - phi_0) / PhiBinWidth;

	const double f_000 = value(i_r_0, i_t_0, i_p_0);
	const double f_001 = value(i_r_0, i_t_0, i_p_1);
	const double f_010 = value(i_r_0, i_t_1, i_p_0);
	const double f_011 = value(i_r_0, i_t_1, i_p_1);
	const double f_100 = value(i_r_1, i_t_0, i_p_0);
	const double f_101 = value(i_r_1, i_t_0, i_p_1);
	const double f_110 = value(i_r_1, i_t_1, i_p_0);
	const double f_111 = value(i_r_1, i_t_1, i_p_1);

	const double f_r00 = (1.0 - delta_r) * f_000 + delta_r * f_100;
	const double f_r01 = (1.0 - delta_r) * f_001 + delta_r * f_101;
	const double f_r10 = (1.0 - delta_r) * f_010 + delta_r * f_110;
	const double f_r11 = (1.0 - delta_r) * f_011 + delta_r * f_111;

	const double f_rt0 = (1.0 - delta_t) * f_r00 + delta_t * f_r10;
	const double f_rt1 = (1.0 - delta_t) * f_r01 + delta_t * f_r11;

	const double f_rtp = (1.0 - delta_p) * f_rt0 + delta_p * f_rt1;

	return f_rtp;
}

double SphericalGridFunction::operator()(const double radius, const double theta, const double phi) const
{
	return interpolate(radius, theta, phi);
}

double SphericalGridFunction::volume_element(const std::size_t radialBin, const std::size_t thetaBin) const
{
	const double upperRadius = (radialBin < RadialBinNumber - 1) ? radial_coordinate(radialBin) + 0.5 * RadialBinWidth
																 : radial_coordinate(radialBin);
	const double lowerRadius = (radialBin > 0) ? radial_coordinate(radialBin) - 0.5 * RadialBinWidth
											   : radial_coordinate(radialBin);

	const double solidAngleElement = solid_angle_element(thetaBin);

	return (std::pow(upperRadius, 3.0) - std::pow(lowerRadius, 3.0)) / 3.0 * solidAngleElement;
}

double SphericalGridFunction::solid_angle_element(const std::size_t thetaBin) const
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

double SphericalGridFunction::radial_coordinate(const std::size_t radialBin) const
{
	return RadialCoordinates[radialBin];
}

double SphericalGridFunction::theta_coordinate(const std::size_t thetaBin) const
{
	return ThetaCoordinates[thetaBin];
}

double SphericalGridFunction::phi_coordinate(const std::size_t phiBin) const
{
	return PhiCoordinates[phiBin];
}

std::vector<double> SphericalGridFunction::radial_coordinate_vector() const
{
	return RadialCoordinates;
}

std::vector<double> SphericalGridFunction::theta_coordinate_vector() const
{
	return ThetaCoordinates;
}

std::vector<double> SphericalGridFunction::phi_coordinate_vector() const
{
	return PhiCoordinates;
}

double SphericalGridFunction::maximal_radius() const
{
	return MaxRadius;
}

double SphericalGridFunction::radial_bin_width() const
{
	return RadialBinWidth;
}

double SphericalGridFunction::theta_bin_width() const
{
	return ThetaBinWidth;
}

double SphericalGridFunction::phi_bin_width() const
{
	return PhiBinWidth;
}

std::size_t SphericalGridFunction::radial_bin_number() const
{
	return RadialBinNumber;
}

std::size_t SphericalGridFunction::theta_bin_number() const
{
	return ThetaBinNumber;
}

std::size_t SphericalGridFunction::phi_bin_number() const
{
	return PhiBinNumber;
}

void SphericalGridFunction::apply(const std::function<double(double)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double currentValue = value(i_r, i_t, i_p);

				value(i_r, i_t, i_p) = func(currentValue);
			}
		}
	}
}

void SphericalGridFunction::apply(const std::function<double(double value, double radius, double theta, double phi)> &func, bool parallelized)
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double radius = radial_coordinate(i_r);
				const double theta = theta_coordinate(i_t);
				const double phi = phi_coordinate(i_p);

				const double currentValue = value(i_r, i_t, i_p);

				value(i_r, i_t, i_p) = func(currentValue, radius, theta, phi);
			}
		}
	}
}

void SphericalGridFunction::evaluate_for_all_grid_points(const std::function<void(std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)> &func, bool parallelized) const
{
#pragma omp parallel for schedule(static) collapse(3) if (parallelized)
	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				func(i_r, i_t, i_p);
			}
		}
	}
}

double SphericalGridFunction::average(const std::function<double(double value)> &func) const
{
	double average = 0.0;

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double volumeElement = volume_element(i_r, i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double val = value(i_r, i_t, i_p);

				average += func(val) * volumeElement;
			}
		}
	}

	average /= 4.0 / 3.0 * M_PI * std::pow(MaxRadius, 3.0);

	return average;
}

Cartesian1DGridFunction SphericalGridFunction::averages(const std::function<double(double value)> &func) const
{
	Cartesian1DGridFunction averages(0.0, MaxRadius, RadialBinNumber);

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		const double radius = radial_coordinate(i_r);

		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double volumeElement = volume_element(i_r, i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double val = value(i_r, i_t, i_p);

				averages.value(i_r) += func(val) * volumeElement;
			}
		}

		if (i_r != RadialBinNumber - 1)
		{
			averages.value(i_r + 1) += averages.value(i_r);
			averages.value(i_r) /= 4.0 / 3.0 * M_PI * std::pow(radius + RadialBinWidth / 2.0, 3.0);
		}
		else
		{
			averages.value(i_r) /= 4.0 / 3.0 * M_PI * std::pow(radius, 3.0);
		}
	}

	return averages;
}

double SphericalGridFunction::average(const std::function<double(double value, double radius, double theta, double phi)> &func) const
{
	double average = 0.0;

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		const double radius = radial_coordinate(i_r);

		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double theta = theta_coordinate(i_t);

			const double volumeElement = volume_element(i_r, i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double phi = phi_coordinate(i_p);

				const double val = value(i_r, i_t, i_p);

				average += func(val, radius, theta, phi) * volumeElement;
			}
		}
	}

	average /= 4.0 / 3.0 * M_PI * std::pow(MaxRadius, 3.0);

	return average;
}

Cartesian1DGridFunction SphericalGridFunction::averages(const std::function<double(double value, double radius, double theta, double phi)> &func) const
{
	Cartesian1DGridFunction averages(0.0, MaxRadius, RadialBinNumber);

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		const double radius = radial_coordinate(i_r);

		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double theta = theta_coordinate(i_t);

			const double volumeElement = volume_element(i_r, i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double phi = phi_coordinate(i_p);

				const double val = value(i_r, i_t, i_p);

				averages.value(i_r) += func(val, radius, theta, phi) * volumeElement;
			}
		}

		if (i_r != RadialBinNumber - 1)
		{
			averages.value(i_r + 1) += averages.value(i_r);
			averages.value(i_r) /= 4.0 / 3.0 * M_PI * std::pow(radius + RadialBinWidth / 2.0, 3.0);
		}
		else
		{
			averages.value(i_r) /= 4.0 / 3.0 * M_PI * std::pow(radius, 3.0);
		}
	}

	return averages;
}

Cartesian1DGridFunction SphericalGridFunction::angular_average(const std::function<double(double value)> &func) const
{
	Cartesian1DGridFunction angularAverage(0.0, MaxRadius, RadialBinNumber);

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double solidAngleElement = solid_angle_element(i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double val = value(i_r, i_t, i_p);

				angularAverage.value(i_r) += func(val) * solidAngleElement / 4.0 / M_PI;
			}
		}
	}

	return angularAverage;
}

Cartesian1DGridFunction SphericalGridFunction::angular_average(const std::function<double(double value, double radius, double theta, double phi)> &func) const
{
	Cartesian1DGridFunction angularAverage(0.0, MaxRadius, RadialBinNumber);

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		const double radius = radial_coordinate(i_r);

		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double theta = theta_coordinate(i_r);

			const double solidAngleElement = solid_angle_element(i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double phi = phi_coordinate(i_r);

				const double val = value(i_r, i_t, i_p);

				angularAverage.value(i_r) += func(val, radius, theta, phi) * solidAngleElement / 4.0 / M_PI;
			}
		}
	}

	return angularAverage;
}

void SphericalGridFunction::write_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName);
	outputFile.setf(std::ios::scientific);

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		const double radius = radial_coordinate(i_r);

		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			const double theta = theta_coordinate(i_t);

			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double phi = phi_coordinate(i_p);

				const double currentValue = value(i_r, i_t, i_p);

				outputFile << radius << "\t"
						   << theta << "\t"
						   << phi << "\t"
						   << currentValue
						   << std::endl;
			}
		}
	}

	outputFile.close();
}

void SphericalGridFunction::save_object_to_file(const std::string &fileName) const
{
	std::ofstream outputFile(fileName, std::ios::binary);

	outputFile.write((char *)&MaxRadius, sizeof(double));
	outputFile.write((char *)&RadialBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&ThetaBinNumber, sizeof(std::size_t));
	outputFile.write((char *)&PhiBinNumber, sizeof(std::size_t));

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				const double currentValue = value(i_r, i_t, i_p);

				outputFile.write((char *)&currentValue, sizeof(double));
			}
		}
	}

	outputFile.close();
}

SphericalGridFunction &SphericalGridFunction::operator+=(const SphericalGridFunction &otherGridFunction)
{
	if ((otherGridFunction.MaxRadius != MaxRadius) or (otherGridFunction.RadialBinNumber != RadialBinNumber) or (otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " SphericalGridFunction::operator+= Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
								 { value(radialBin, thetaBin, phiBin) += otherGridFunction.value(radialBin, thetaBin, phiBin); });

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator-=(const SphericalGridFunction &otherGridFunction)
{
	if ((otherGridFunction.MaxRadius != MaxRadius) or (otherGridFunction.RadialBinNumber != RadialBinNumber) or (otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " SphericalGridFunction::operator-= Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
								 { value(radialBin, thetaBin, phiBin) -= otherGridFunction.value(radialBin, thetaBin, phiBin); });

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator*=(const SphericalGridFunction &otherGridFunction)
{
	if ((otherGridFunction.MaxRadius != MaxRadius) or (otherGridFunction.RadialBinNumber != RadialBinNumber) or (otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " SphericalGridFunction::operator*= Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
								 { value(radialBin, thetaBin, phiBin) *= otherGridFunction.value(radialBin, thetaBin, phiBin); });

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator/=(const SphericalGridFunction &otherGridFunction)
{
	if ((otherGridFunction.MaxRadius != MaxRadius) or (otherGridFunction.RadialBinNumber != RadialBinNumber) or (otherGridFunction.ThetaBinNumber != ThetaBinNumber) or (otherGridFunction.PhiBinNumber != PhiBinNumber))
	{
		std::cout << std::endl
				  << " SphericalGridFunction::operator/= Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
								 { value(radialBin, thetaBin, phiBin) /= otherGridFunction.value(radialBin, thetaBin, phiBin); });

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator+=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
									 { value(radialBin, thetaBin, phiBin) += otherValue; });
	}

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator-=(const double otherValue)
{
	if (otherValue != 0.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
									 { value(radialBin, thetaBin, phiBin) -= otherValue; });
	}

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator*=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
									 { value(radialBin, thetaBin, phiBin) *= otherValue; });
	}

	return *this;
}

SphericalGridFunction &SphericalGridFunction::operator/=(const double otherValue)
{
	if (otherValue != 1.0) // save computation time by avoiding identity operation
	{
		evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
									 { value(radialBin, thetaBin, phiBin) /= otherValue; });
	}

	return *this;
}

bool SphericalGridFunction::has_same_grid_as(const SphericalGridFunction &otherGridFunction) const
{
	return (MaxRadius == otherGridFunction.MaxRadius and
			RadialBinNumber == otherGridFunction.RadialBinNumber and
			ThetaBinNumber == otherGridFunction.ThetaBinNumber and
			PhiBinNumber == otherGridFunction.PhiBinNumber);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void SphericalGridFunction::initialize_coordinates()
{
	for (std::size_t i_r = 0; i_r < RadialBinNumber - 1; ++i_r)
	{
		RadialCoordinates[i_r] = static_cast<double>(i_r) * RadialBinWidth;
	}

	RadialCoordinates[RadialBinNumber - 1] = MaxRadius; // set maximal value explicitly to avoid floating point rounding error

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

void SphericalGridFunction::load_object_from_file(const std::string &fileName)
{
	std::ifstream inputFile(fileName, std::ios::binary);

	if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
	{
		std::cout << std::endl
				  << " SphericalGridFunction::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	inputFile.read((char *)&MaxRadius, sizeof(double));
	inputFile.read((char *)&RadialBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&ThetaBinNumber, sizeof(std::size_t));
	inputFile.read((char *)&PhiBinNumber, sizeof(std::size_t));

	RadialBinWidth = MaxRadius / static_cast<double>(RadialBinNumber - 1);
	ThetaBinWidth = M_PI / static_cast<double>(ThetaBinNumber - 1);
	PhiBinWidth = 2.0 * M_PI / static_cast<double>(PhiBinNumber);
	RadialCoordinates = std::vector<double>(RadialBinNumber);
	ThetaCoordinates = std::vector<double>(ThetaBinNumber);
	PhiCoordinates = std::vector<double>(PhiBinNumber);
	FunctionValues = std::vector<double>(RadialBinNumber * ThetaBinNumber * PhiBinNumber);

	initialize_coordinates();

	for (std::size_t i_r = 0; i_r < RadialBinNumber; ++i_r)
	{
		for (std::size_t i_t = 0; i_t < ThetaBinNumber; ++i_t)
		{
			for (std::size_t i_p = 0; i_p < PhiBinNumber; ++i_p)
			{
				inputFile.read((char *)&value(i_r, i_t, i_p), sizeof(double));
			}
		}
	}

	inputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

SphericalGridFunction operator+(const SphericalGridFunction &leftGridFunction, const SphericalGridFunction &rightGridFunction)
{
	if ((leftGridFunction.maximal_radius() != rightGridFunction.maximal_radius()) or (leftGridFunction.radial_bin_number() != rightGridFunction.radial_bin_number()) or (leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator+ Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	SphericalGridFunction newSphericalGridFunction(leftGridFunction);

	newSphericalGridFunction.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
														  { newSphericalGridFunction.value(radialBin, thetaBin, phiBin) += rightGridFunction.value(radialBin, thetaBin, phiBin); });

	return newSphericalGridFunction;
}

SphericalGridFunction operator-(const SphericalGridFunction &leftGridFunction, const SphericalGridFunction &rightGridFunction)
{
	if ((leftGridFunction.maximal_radius() != rightGridFunction.maximal_radius()) or (leftGridFunction.radial_bin_number() != rightGridFunction.radial_bin_number()) or (leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator- Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	SphericalGridFunction newSphericalGridFunction(leftGridFunction);

	newSphericalGridFunction.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
														  { newSphericalGridFunction.value(radialBin, thetaBin, phiBin) -= rightGridFunction.value(radialBin, thetaBin, phiBin); });

	return newSphericalGridFunction;
}

SphericalGridFunction operator*(const SphericalGridFunction &leftGridFunction, const SphericalGridFunction &rightGridFunction)
{
	if ((leftGridFunction.maximal_radius() != rightGridFunction.maximal_radius()) or (leftGridFunction.radial_bin_number() != rightGridFunction.radial_bin_number()) or (leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator* Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	SphericalGridFunction newSphericalGridFunction(leftGridFunction);

	newSphericalGridFunction.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
														  { newSphericalGridFunction.value(radialBin, thetaBin, phiBin) *= rightGridFunction.value(radialBin, thetaBin, phiBin); });

	return newSphericalGridFunction;
}

SphericalGridFunction operator/(const SphericalGridFunction &leftGridFunction, const SphericalGridFunction &rightGridFunction)
{
	if ((leftGridFunction.maximal_radius() != rightGridFunction.maximal_radius()) or (leftGridFunction.radial_bin_number() != rightGridFunction.radial_bin_number()) or (leftGridFunction.theta_bin_number() != rightGridFunction.theta_bin_number()) or (leftGridFunction.phi_bin_number() != rightGridFunction.phi_bin_number()))
	{
		std::cout << std::endl
				  << " operator/ Error: Coordinates of spherical grid functions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	SphericalGridFunction newSphericalGridFunction(leftGridFunction);

	newSphericalGridFunction.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
														  { newSphericalGridFunction.value(radialBin, thetaBin, phiBin) /= rightGridFunction.value(radialBin, thetaBin, phiBin); });

	return newSphericalGridFunction;
}

SphericalGridFunction operator+(const SphericalGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return SphericalGridFunction(leftGridFunction, [&](double leftValue)
									 { return leftValue + rightValue; });
	}
}

SphericalGridFunction operator-(const SphericalGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 0.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return SphericalGridFunction(leftGridFunction, [&](double leftValue)
									 { return leftValue - rightValue; });
	}
}

SphericalGridFunction operator*(const SphericalGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return SphericalGridFunction(leftGridFunction, [&](double leftValue)
									 { return leftValue * rightValue; });
	}
}

SphericalGridFunction operator/(const SphericalGridFunction &leftGridFunction, const double rightValue)
{
	if (rightValue == 1.0) // save computation time by avoiding identity operation
	{
		return leftGridFunction;
	}
	else
	{
		return SphericalGridFunction(leftGridFunction, [&](double leftValue)
									 { return leftValue / rightValue; });
	}
}

SphericalGridFunction operator+(const double leftValue, const SphericalGridFunction &rightGridFunction)
{
	return rightGridFunction + leftValue;
}

SphericalGridFunction operator-(const double leftValue, const SphericalGridFunction &rightGridFunction)
{
	return SphericalGridFunction(rightGridFunction, [&](double rightValue)
								 { return leftValue - rightValue; });
}

SphericalGridFunction operator*(const double leftValue, const SphericalGridFunction &rightGridFunction)
{
	return rightGridFunction * leftValue;
}

SphericalGridFunction operator/(const double leftValue, const SphericalGridFunction &rightGridFunction)
{
	return SphericalGridFunction(rightGridFunction, [&](double rightValue)
								 { return leftValue / rightValue; });
}

void transform_spherical_grid_functions(const SphericalGridFunction &input,
										SphericalGridFunction &output,
										const std::function<void(std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
																 double &output)> &trafo)
{
	const bool transformedInPlace = (&output == &input); // check if the transformation is in-place

	SphericalGridFunction &fieldTemporary = transformedInPlace ? *(new SphericalGridFunction(output)) // if it is (partially) in-place, work on a temporary copy of the output field
															   : output;

	if (not fieldTemporary.has_same_grid_as(input))
	{
		fieldTemporary = input;
	}

	fieldTemporary.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
												{
													double output;

													trafo(radialBin, thetaBin, phiBin,
														  output);

													fieldTemporary.value(radialBin, thetaBin, phiBin) = output;
												});

	if (transformedInPlace)
	{
		output = fieldTemporary;

		delete &fieldTemporary;
	}
}

void transform_spherical_grid_functions(const SphericalGridFunction &input1, const SphericalGridFunction &input2,
										SphericalGridFunction &output1, SphericalGridFunction &output2,
										const std::function<void(std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
																 double &output1, double &output2)> &trafo)
{
	if (not input1.has_same_grid_as(input2))
	{
		std::cout << std::endl
				  << " transform_spherical_grid_functions Error: Coordinate grids of input SphericalGridFunctions do not agree" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	const bool transformed1InPlace = (&output1 == &input1 or // check if the transformation is (partially) in-place
									  &output2 == &input1);

	const bool transformed2InPlace = (&output1 == &input2 or
									  &output2 == &input2);

	SphericalGridFunction &fieldTemporary1 = transformed1InPlace ? *(new SphericalGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																 : output1;

	SphericalGridFunction &fieldTemporary2 = transformed2InPlace ? *(new SphericalGridFunction(output2))
																 : output2;

	if (not fieldTemporary1.has_same_grid_as(input1))
	{
		fieldTemporary1 = input1;
	}

	if (not fieldTemporary2.has_same_grid_as(input1))
	{
		fieldTemporary2 = input1;
	}

	output1.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
										 {
											 double output1, output2;

											 trafo(radialBin, thetaBin, phiBin,
												   output1, output2);

											 fieldTemporary1.value(radialBin, thetaBin, phiBin) = output1;
											 fieldTemporary2.value(radialBin, thetaBin, phiBin) = output2;
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

void transform_spherical_grid_functions(const SphericalGridFunction &input1, const SphericalGridFunction &input2, const SphericalGridFunction &input3,
										SphericalGridFunction &output1, SphericalGridFunction &output2, SphericalGridFunction &output3,
										const std::function<void(std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
																 double &output1, double &output2, double &output3)> &trafo)
{
	if (not input1.has_same_grid_as(input2) or
		not input1.has_same_grid_as(input3))
	{
		std::cout << std::endl
				  << " transform_spherical_grid_functions Error: Coordinate grids of input SphericalGridFunctions do not agree" << std::endl
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

	SphericalGridFunction &fieldTemporary1 = transformed1InPlace ? *(new SphericalGridFunction(output1)) // if it is (partially) in-place, work on a temporary copy of the output fields
																 : output1;

	SphericalGridFunction &fieldTemporary2 = transformed2InPlace ? *(new SphericalGridFunction(output2))
																 : output2;

	SphericalGridFunction &fieldTemporary3 = transformed3InPlace ? *(new SphericalGridFunction(output3))
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

	output1.evaluate_for_all_grid_points([&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin)
										 {
											 double output1, output2, output3;

											 trafo(radialBin, thetaBin, phiBin,
												   output1, output2, output3);

											 fieldTemporary1.value(radialBin, thetaBin, phiBin) = output1;
											 fieldTemporary2.value(radialBin, thetaBin, phiBin) = output2;
											 fieldTemporary3.value(radialBin, thetaBin, phiBin) = output3;
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