#include "LogNormalPoissonRealization.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>

#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "Cartesian1DGridFunction.hpp"
#include "Cartesian3DGridFunction.hpp"
#include "cosmology.hpp"
#include "SphericalGridFunction.hpp"
#include "transformations.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

LogNormalPoissonRealization::LogNormalPoissonRealization(const double periodicBoundaryDistance, const std::size_t binNumber)
    : BoxLength(2.0 * periodicBoundaryDistance), // the box length is twice the distance from the center to the boundary
      BinNumber1D(4 * binNumber + 1),            // use an increased number of bins for the calculation of the log-density power spectrum to ensure sufficient resolution
      BinNumber3D(2 * binNumber + 1),            // number of bins per full box length
      BoxPointNumber(BinNumber3D * BinNumber3D * BinNumber3D),
      BinNumbers3D{BinNumber3D, BinNumber3D, BinNumber3D},
      FourierTrafoTypes1D{FFTW_RODFT11},                                                       // use discrete cosine transform of type IV (DST-IV) to perform the radial Fourier transform between power spectrum and correlation function because it is symmetric and automatically avoids evaluating the correlation function at zero distance
      RealBinWidth1D(BoxLength / static_cast<double>(BinNumber1D - 1) * 2.0 / std::sqrt(3.0)), // 1D bin width chosen such that the maximal absolute value of the 3D Fourier vector is reached
      FourierBinWidth1D(M_PI / RealBinWidth1D / static_cast<double>(BinNumber1D)),             // 1D bin width in Fourier space chosen to match the definition of the DST-IV in FFTW3 (see FFTW3 documentation for details)
      RealBinWidth3D(BoxLength / static_cast<double>(BinNumber3D - 1)),
      FourierBinWidth3D(2.0 * M_PI / RealBinWidth3D / static_cast<double>(BinNumber3D)), // 3D bin width in Fourier space chosen to match the definition of the discrete Fourier transform (DFT) in FFTW3 (see FFTW3 documentation for details)
      BackwardTrafoNormalization1D(FourierBinWidth1D / gsl_pow_2(2.0 * M_PI)),           // normalizations of 1D transforms correspond to the radial Fourier transform obtained by integrating out the angular parts of a standard 3D Fourier transform
      ForwardTrafoNormalization1D(RealBinWidth1D * 2.0 * M_PI),
      BackwardTrafoNormalization3D(gsl_pow_3(FourierBinWidth3D / (2.0 * M_PI))), // standard normalizations of 3D Fourier transform
      ForwardTrafoNormalization3D(gsl_pow_3(RealBinWidth3D)),
      Spectrum(fftw_alloc_real(BinNumber1D)),
      Density(fftw_alloc_complex(BoxPointNumber)),
      CurrentDensityRealization(fftw_alloc_complex(BoxPointNumber)),
      Velocity(fftw_alloc_complex(3 * BoxPointNumber)),
      BackwardTrafoPlan1D(fftw_plan_many_r2r(1, &BinNumber1D, 1, // FFT plan for transforming the power spectrum to the correlation function (see FFTW3 documentation for details)
                                             Spectrum, NULL,
                                             1, BinNumber1D, Spectrum, NULL,
                                             1, BinNumber1D,
                                             FourierTrafoTypes1D, FFTW_ESTIMATE)),
      ForwardTrafoPlan1D(fftw_plan_many_r2r(1, &BinNumber1D, 1, // FFT plan for simultaneously transforming the correlation function to the power spectrum (see FFTW3 documentation for details)
                                            Spectrum, NULL,
                                            1, BinNumber1D,
                                            Spectrum, NULL,
                                            1, BinNumber1D,
                                            FourierTrafoTypes1D, FFTW_ESTIMATE)),
      DensityBackwardTrafoPlan3D(fftw_plan_many_dft(3, BinNumbers3D, 1, // FFT plan for transforming the density field from Fourier to real space (see FFTW3 documentation for details)
                                                    Density, NULL,
                                                    1, BoxPointNumber,
                                                    Density, NULL,
                                                    1, BoxPointNumber,
                                                    FFTW_BACKWARD, FFTW_ESTIMATE)),
      DensityForwardTrafoPlan3D(fftw_plan_many_dft(3, BinNumbers3D, 1, // FFT plan for transforming the density field from real to Fourier space (see FFTW3 documentation for details)
                                                   Density, NULL,
                                                   1, BoxPointNumber,
                                                   Density, NULL,
                                                   1, BoxPointNumber,
                                                   FFTW_FORWARD, FFTW_ESTIMATE)),
      VelocityBackwardTrafoPlan3D(fftw_plan_many_dft(3, BinNumbers3D, 3, // FFT plan for simultaneously transforming all three components of the velocity field from Fourier to real space (see FFTW3 documentation for details)
                                                     Velocity, NULL,
                                                     1, BoxPointNumber,
                                                     Velocity, NULL,
                                                     1, BoxPointNumber,
                                                     FFTW_BACKWARD, FFTW_ESTIMATE)),
      CurrentMaxRadius(0.0),
      CurrentLogTrafoSmoothingScale(-1.0)
{
}

void LogNormalPoissonRealization::generate(const double maxRadius, const std::function<double(double)> &densityContrastPowerSpectrum, const double meanGalaxyDensity, const std::function<double(double)> &selectionFunction, const double logTrafoSmoothingScale,
                                           std::vector<double> &galaxyRadialCoordinatesRealization, std::vector<double> &galaxyThetaCoordinatesRealization, std::vector<double> &galaxyPhiCoordinatesRealization,
                                           const std::size_t randomNumberGeneratorSeed, const gsl_rng_type *randomNumberGeneratorType)
{
  CurrentMaxRadius = maxRadius; // set the current values of the maximal radius and the log-transform smoothing scale; these will be used in LogNormalPoissonRealization::get_fields
  CurrentLogTrafoSmoothingScale = logTrafoSmoothingScale;

  const double kMin = fourier_coordinate_1D(0);
  const double kMax = fourier_coordinate_1D(BinNumber1D - 1);

  Cartesian1DGridFunction logDensityPowerSpectrum(kMin, kMax, BinNumber1D, 0.0);

  for (int i_k = 0; i_k < BinNumber1D; ++i_k) // apply the Gaussian log-transform smoothing to the density contrast power spectra and transform them to real space
  {
    const double k = fourier_coordinate_1D(i_k);

    const double logTrafoWindowFunctionSquared = std::exp(-gsl_pow_2(k * logTrafoSmoothingScale));

    const double smoothedDensityContrastPowerSpectrum = densityContrastPowerSpectrum(k) * logTrafoWindowFunctionSquared;

    Spectrum[i_k] = smoothedDensityContrastPowerSpectrum * k * BackwardTrafoNormalization1D; // the multiplication with k appears because the 3D Fourier transform was turned into a radial Fourier transform by integrating out the independent angular parts
  }

  fftw_execute(BackwardTrafoPlan1D);

  for (int i_r = 0; i_r < BinNumber1D; ++i_r) // transform the resulting density contrast correlation functions to the correlation functions of the log-density, and transform these back into Fourier space
  {
    const double r = real_coordinate_1D(i_r);

    const double densityContrastcorrelationFunction = Spectrum[i_r] / r; // the division by r appears because the 3D Fourier transform was turned into a radial Fourier transform by integrating out the independent angular parts

    const double logDensityCorrelationFunction = std::log1p(densityContrastcorrelationFunction);

    Spectrum[i_r] = logDensityCorrelationFunction * r * ForwardTrafoNormalization1D; // the multiplication with r appears because the 3D Fourier transform was turned into a radial Fourier transform by integrating out the independent angular parts
  }

  fftw_execute(ForwardTrafoPlan1D);

  for (int i_k = 0; i_k < BinNumber1D; ++i_k) // write the resulting log-density power spectra into the respective grid functions
  {
    const double k = fourier_coordinate_1D(i_k);

    const double logDensityPowerSpectrumValue = Spectrum[i_k] / k; // the division by k appears because the 3D Fourier transform was turned into a radial Fourier transform by integrating out the independent angular parts

    if (logDensityPowerSpectrumValue >= 0) // if the log-transform yields negative values, the log-density power spectrum for that wavenumber is left at its initializing value of zero (see Xavier et al., MNRAS 459 (2016), p. 3693 (doi:10.1093/mnras/stw874) for details)
    {
      logDensityPowerSpectrum.value(i_k) = logDensityPowerSpectrumValue;
    }
  }

  double meanLogDensity = 0.0;

  galaxyRadialCoordinatesRealization.clear();
  galaxyThetaCoordinatesRealization.clear();
  galaxyPhiCoordinatesRealization.clear();

  gsl_rng *randomNumberGenerator = gsl_rng_alloc(randomNumberGeneratorType);
  gsl_rng_set(randomNumberGenerator, randomNumberGeneratorSeed);

  for (int i_x = 0; i_x < BinNumber3D; ++i_x)
  {
    const double kx = fourier_coordinate_3D(i_x);

    for (int i_y = 0; i_y < BinNumber3D; ++i_y)
    {
      const double ky = fourier_coordinate_3D(i_y);

      for (int i_z = 0; i_z < BinNumber3D; ++i_z)
      {
        const double kz = fourier_coordinate_3D(i_z);

        const double k = std::sqrt(kx * kx + ky * ky + kz * kz);

        meanLogDensity += -0.5 * ((k != 0.0) ? logDensityPowerSpectrum(k) : 0.0) * BackwardTrafoNormalization3D;

        const double fourierLogDensityVariance = (k != 0.0) ? logDensityPowerSpectrum(k) / BackwardTrafoNormalization3D
                                                            : 0.0;

        double fourierLogDensityRealizationRe = 0.0;
        double fourierLogDensityRealizationIm = 0.0;

        const std::size_t i_xyz = (i_x * BinNumber3D + i_y) * BinNumber3D + i_z;

        if ((i_x == 0) and (i_y == 0) and (i_z == 0))
        {
          const double standardNormalRandomNumberRe = gsl_ran_gaussian(randomNumberGenerator, 1.0);

          fourierLogDensityRealizationRe = standardNormalRandomNumberRe * std::sqrt(fourierLogDensityVariance) * BackwardTrafoNormalization3D;
        }
        else if ((i_x > BinNumber3D / 2) or ((i_x == 0) and (i_y > BinNumber3D / 2)) or ((i_x == 0) and (i_y == 0) and (i_z > BinNumber3D / 2)))
        {
          const std::size_t i_xyz_conjugated = (((BinNumber3D - i_x) % BinNumber3D) * BinNumber3D + ((BinNumber3D - i_y) % BinNumber3D)) * BinNumber3D + ((BinNumber3D - i_z) % BinNumber3D);

          fourierLogDensityRealizationRe = Density[i_xyz_conjugated][0];
          fourierLogDensityRealizationIm = -Density[i_xyz_conjugated][1];
        }
        else
        {
          const double standardNormalRandomNumberRe = gsl_ran_gaussian(randomNumberGenerator, 1.0);
          const double standardNormalRandomNumberIm = gsl_ran_gaussian(randomNumberGenerator, 1.0);

          fourierLogDensityRealizationRe = standardNormalRandomNumberRe * std::sqrt(fourierLogDensityVariance / 2.0) * BackwardTrafoNormalization3D;
          fourierLogDensityRealizationIm = standardNormalRandomNumberIm * std::sqrt(fourierLogDensityVariance / 2.0) * BackwardTrafoNormalization3D;
        }

        Density[i_xyz][0] = fourierLogDensityRealizationRe;
        Density[i_xyz][1] = fourierLogDensityRealizationIm;
      }
    }
  }

  fftw_execute(DensityBackwardTrafoPlan3D);

  for (int i_x = 0; i_x < BinNumber3D; ++i_x) // transform the resulting real-space log-densities to the corresponding density contrasts, obtain the mock galaxy positions via Poisson sampling, and transform the density contrasts back to Fourier space
  {
    const double x = real_coordinate_3D(i_x);

    for (int i_y = 0; i_y < BinNumber3D; ++i_y)
    {
      const double y = real_coordinate_3D(i_y);

      for (int i_z = 0; i_z < BinNumber3D; ++i_z)
      {
        const double z = real_coordinate_3D(i_z);

        const std::size_t i_xyz = (i_x * BinNumber3D + i_y) * BinNumber3D + i_z;

        const double logDensity = Density[i_xyz][0] + meanLogDensity; // add the previously computed log-density mean, to obtain correctly normalized zero-mean density contrasts

        const double densityContrast = std::expm1(logDensity);

        Density[i_xyz][0] = densityContrast * ForwardTrafoNormalization3D;
        Density[i_xyz][1] = 0.0;

        const double radius = std::sqrt(x * x + y * y + z * z);

        if (radius <= maxRadius) // for every Cartesian volume element within a sphere of radius maxRadius draw a number of mock galaxies from a Poisson distribution
        {
          const double meanGalaxyNumber = meanGalaxyDensity * (1.0 + densityContrast) * selectionFunction(radius) * gsl_pow_3(RealBinWidth3D); // the mean of the Poisson distribution is given by the expected number of galaxies per volume element given the total mean number density and the selection function

          std::size_t galaxyNumber = gsl_ran_poisson(randomNumberGenerator, meanGalaxyNumber);

          for (std::size_t i_g = 0; i_g < galaxyNumber; ++i_g) // randomly distribute the mock galaxies within the volume element and compute their spherical coordinates
          {
            const double galaxyXCoordinate = x + RealBinWidth3D * (gsl_rng_uniform(randomNumberGenerator) - 1.0 / 2.0);
            const double galaxyYCoordinate = y + RealBinWidth3D * (gsl_rng_uniform(randomNumberGenerator) - 1.0 / 2.0);
            const double galaxyZCoordinate = z + RealBinWidth3D * (gsl_rng_uniform(randomNumberGenerator) - 1.0 / 2.0);

            double galaxyRadialCoordinate, galaxyThetaCoordinate, galaxyPhiCoordinate;

            transform_cartesian_to_spherical_coordinates(galaxyXCoordinate, galaxyYCoordinate, galaxyZCoordinate,
                                                         galaxyRadialCoordinate, galaxyThetaCoordinate, galaxyPhiCoordinate);

            if (galaxyRadialCoordinate <= maxRadius)
            {
              galaxyRadialCoordinatesRealization.push_back(galaxyRadialCoordinate);
              galaxyThetaCoordinatesRealization.push_back(galaxyThetaCoordinate);
              galaxyPhiCoordinatesRealization.push_back(galaxyPhiCoordinate);
            }
          }
        }
      }
    }
  }

  gsl_rng_free(randomNumberGenerator);

  fftw_execute(DensityForwardTrafoPlan3D);

  for (int i_x = 0; i_x < BinNumber3D; ++i_x)
  {
    for (int i_y = 0; i_y < BinNumber3D; ++i_y)
    {
      for (int i_z = 0; i_z < BinNumber3D; ++i_z)
      {
        const std::size_t i_xyz = (i_x * BinNumber3D + i_y) * BinNumber3D + i_z;

        CurrentDensityRealization[i_xyz][0] = Density[i_xyz][0];
        CurrentDensityRealization[i_xyz][1] = Density[i_xyz][1];
      }
    }
  }
}

void LogNormalPoissonRealization::get_fields(const double normalizedGrowthRate, const double fieldSmoothingScale,
                                             const std::size_t radialBinNumber, const std::size_t thetaBinNumber, const std::size_t phiBinNumber,
                                             SphericalGridFunction &densityContrastRealization, SphericalGridFunction &radialVelocityRealization, SphericalGridFunction &thetaVelocityRealization, SphericalGridFunction &phiVelocityRealization) const
{
  if (CurrentLogTrafoSmoothingScale < 0.0)
  {
    std::cout << "LogNormalPoissonRealization::get_fields Error: Realization has not been generated yet" << std::endl;

    exit(EXIT_FAILURE);
  }

  if (fieldSmoothingScale < CurrentLogTrafoSmoothingScale)
  {
    std::cout << "LogNormalPoissonRealization::get_fields Error: Signal smoothing scale is smaller than the log-trafo smoothing used to generate the current realization" << std::endl;

    exit(EXIT_FAILURE);
  }

  const double residualSmoothingScale = std::sqrt(gsl_pow_2(fieldSmoothingScale) - gsl_pow_2(CurrentLogTrafoSmoothingScale)); // residual smoothing that has to be applied to the density and velocity fields after they have already been smoothed for the log-transform, to achieve the desired signal smoothing

  const double rMin = -BoxLength / 2.0;
  const double rMax = BoxLength / 2.0;

  Cartesian3DGridFunction cartesianDensityContrastRealization(rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, 0.0);
  Cartesian3DGridFunction cartesianXVelocityRealization(rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, 0.0);
  Cartesian3DGridFunction cartesianYVelocityRealization(rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, 0.0);
  Cartesian3DGridFunction cartesianZVelocityRealization(rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, rMin, rMax, BinNumber3D, 0.0);

  for (int i_x = 0; i_x < BinNumber3D; ++i_x) // apply the residual Gaussian smoothing needed to achieve the desired total signal smoothing
  {
    const double kx = fourier_coordinate_3D(i_x);

    for (int i_y = 0; i_y < BinNumber3D; ++i_y)
    {
      const double ky = fourier_coordinate_3D(i_y);

      for (int i_z = 0; i_z < BinNumber3D; ++i_z)
      {
        const double kz = fourier_coordinate_3D(i_z);

        const double kSquared = kx * kx + ky * ky + kz * kz;

        const std::size_t i_xyz = (i_x * BinNumber3D + i_y) * BinNumber3D + i_z;

        const double residualWindowFunction = std::exp(-kSquared * gsl_pow_2(residualSmoothingScale) / 2.0);

        Density[i_xyz][0] = CurrentDensityRealization[i_xyz][0] * residualWindowFunction * BackwardTrafoNormalization3D;
        Density[i_xyz][1] = CurrentDensityRealization[i_xyz][1] * residualWindowFunction * BackwardTrafoNormalization3D;

        if ((i_x == 0) and (i_y == 0) and (i_z == 0)) // corresponds to the zero-mode, which vanishes for the potential flow velocity field
        {
          Velocity[0][0] = 0.0;
          Velocity[0][1] = 0.0;
          Velocity[BoxPointNumber][0] = 0.0;
          Velocity[BoxPointNumber][1] = 0.0;
          Velocity[2 * BoxPointNumber][0] = 0.0;
          Velocity[2 * BoxPointNumber][1] = 0.0;
        }
        else
        {
          Velocity[i_xyz][0] = -normalizedGrowthRate * HUBBLE_NORMALIZATION * kx / kSquared * Density[i_xyz][1]; // v(k) = f sigma8 H delta_g(k) i k / |k|^2
          Velocity[i_xyz][1] = normalizedGrowthRate * HUBBLE_NORMALIZATION * kx / kSquared * Density[i_xyz][0];
          Velocity[i_xyz + BoxPointNumber][0] = -normalizedGrowthRate * HUBBLE_NORMALIZATION * ky / kSquared * Density[i_xyz][1];
          Velocity[i_xyz + BoxPointNumber][1] = normalizedGrowthRate * HUBBLE_NORMALIZATION * ky / kSquared * Density[i_xyz][0];
          Velocity[i_xyz + 2 * BoxPointNumber][0] = -normalizedGrowthRate * HUBBLE_NORMALIZATION * kz / kSquared * Density[i_xyz][1];
          Velocity[i_xyz + 2 * BoxPointNumber][1] = normalizedGrowthRate * HUBBLE_NORMALIZATION * kz / kSquared * Density[i_xyz][0];
        }
      }
    }
  }

  fftw_execute(DensityBackwardTrafoPlan3D);
  fftw_execute(VelocityBackwardTrafoPlan3D);

  for (int i_x = 0; i_x < BinNumber3D; ++i_x) // write real space density contrast and velocity fields into the respective Cartesian 3D grid functions
  {
    for (int i_y = 0; i_y < BinNumber3D; ++i_y)
    {
      for (int i_z = 0; i_z < BinNumber3D; ++i_z)
      {
        const std::size_t i_xyz = (i_x * BinNumber3D + i_y) * BinNumber3D + i_z;

        const double densityContrast = Density[i_xyz][0];

        const double xVelocity = Velocity[i_xyz][0];
        const double yVelocity = Velocity[i_xyz + BoxPointNumber][0];
        const double zVelocity = Velocity[i_xyz + 2 * BoxPointNumber][0];

        const std::size_t i_x_shifted = (i_x + BinNumber3D / 2) % BinNumber3D; // transform FFT bins, where (0,0,0) corresponds to the box center, to grid function bins, where (0,0,0) corresponds to a box corner
        const std::size_t i_y_shifted = (i_y + BinNumber3D / 2) % BinNumber3D;
        const std::size_t i_z_shifted = (i_z + BinNumber3D / 2) % BinNumber3D;

        cartesianDensityContrastRealization.value(i_x_shifted, i_y_shifted, i_z_shifted) = densityContrast;

        cartesianXVelocityRealization.value(i_x_shifted, i_y_shifted, i_z_shifted) = xVelocity;
        cartesianYVelocityRealization.value(i_x_shifted, i_y_shifted, i_z_shifted) = yVelocity;
        cartesianZVelocityRealization.value(i_x_shifted, i_y_shifted, i_z_shifted) = zVelocity;
      }
    }
  }

  densityContrastRealization = SphericalGridFunction(CurrentMaxRadius, radialBinNumber, thetaBinNumber, phiBinNumber, 0.0);

  radialVelocityRealization = SphericalGridFunction(CurrentMaxRadius, radialBinNumber, thetaBinNumber, phiBinNumber, 0.0);
  thetaVelocityRealization = SphericalGridFunction(CurrentMaxRadius, radialBinNumber, thetaBinNumber, phiBinNumber, 0.0);
  phiVelocityRealization = SphericalGridFunction(CurrentMaxRadius, radialBinNumber, thetaBinNumber, phiBinNumber, 0.0);

  for (std::size_t i_r = 0; i_r < radialBinNumber; ++i_r) // transform Cartesian velocity vector field components to spherical components, and map the density contrast and velocity fields to regular spherical grid functions of maximal radius maxRadius
  {
    const double radius = densityContrastRealization.radial_coordinate(i_r);

    for (std::size_t i_t = 0; i_t < thetaBinNumber; ++i_t)
    {
      const double theta = densityContrastRealization.theta_coordinate(i_t);

      for (std::size_t i_p = 0; i_p < phiBinNumber; ++i_p)
      {
        const double phi = densityContrastRealization.phi_coordinate(i_p);

        double x, y, z;

        transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                                     x, y, z);

        const double densityContrast = cartesianDensityContrastRealization(x, y, z);

        const double xVelocity = cartesianXVelocityRealization(x, y, z);
        const double yVelocity = cartesianYVelocityRealization(x, y, z);
        const double zVelocity = cartesianZVelocityRealization(x, y, z);

        double radialVelocity, thetaVelocity, phiVelocity;

        transform_cartesian_to_spherical_vector_field_value(theta, phi,
                                                            xVelocity, yVelocity, zVelocity,
                                                            radialVelocity, thetaVelocity, phiVelocity);

        densityContrastRealization.value(i_r, i_t, i_p) = densityContrast;
        radialVelocityRealization.value(i_r, i_t, i_p) = radialVelocity;
        thetaVelocityRealization.value(i_r, i_t, i_p) = thetaVelocity;
        phiVelocityRealization.value(i_r, i_t, i_p) = phiVelocity;
      }
    }
  }
}

LogNormalPoissonRealization::~LogNormalPoissonRealization()
{
  fftw_free(Spectrum);
  fftw_free(Density);
  fftw_free(CurrentDensityRealization);
  fftw_free(Velocity);

  fftw_destroy_plan(BackwardTrafoPlan1D);
  fftw_destroy_plan(ForwardTrafoPlan1D);
  fftw_destroy_plan(DensityBackwardTrafoPlan3D);
  fftw_destroy_plan(DensityForwardTrafoPlan3D);
  fftw_destroy_plan(VelocityBackwardTrafoPlan3D);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

double LogNormalPoissonRealization::real_coordinate_1D(const int bin) const
{
  return static_cast<double>(2 * bin + 1) / 2.0 * RealBinWidth1D; // discretization chosen to match the definition of the DST-IV in FFTW3 (see FFTW3 documentation for details)
}

double LogNormalPoissonRealization::fourier_coordinate_1D(const int bin) const
{
  return static_cast<double>(2 * bin + 1) / 2.0 * FourierBinWidth1D; // discretization chosen to match the definition of the DST-IV in FFTW3 (see FFTW3 documentation for details)
}

double LogNormalPoissonRealization::real_coordinate_3D(const int bin) const
{
  if (bin <= BinNumber3D / 2) // discretization chosen to match the definition of the DFT in FFTW3; the first half of the bins corresponds to positive, the second half to negative coordinates (see FFTW3 documentation for details)
  {
    return static_cast<double>(bin) * RealBinWidth3D;
  }
  else
  {
    return static_cast<double>(bin - BinNumber3D) * RealBinWidth3D;
  }
}

double LogNormalPoissonRealization::fourier_coordinate_3D(const int bin) const
{
  if (bin <= BinNumber3D / 2) // discretization chosen to match the definition of the DFT in FFTW3; the first half of the bins corresponds to positive, the second half to negative coordinates (see FFTW3 documentation for details)
  {
    return static_cast<double>(bin) * FourierBinWidth3D;
  }
  else
  {
    return static_cast<double>(bin - BinNumber3D) * FourierBinWidth3D;
  }
}