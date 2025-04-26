#include "survey.hpp"

#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>

#include "ObjectGrid.hpp"
#include "SphericalGridFunction.hpp"

void apply_partial_volume_limit(const std::vector<double> &inputRedshiftVelocities, const std::vector<double> &inputKCorrectionRedshiftVelocities, const std::vector<double> &inputThetaCoordinates, const std::vector<double> &inputPhiCoordinates, const std::vector<double> &inputApparentMagnitudes,
                                const ReferenceFrameChange referenceFrameChange, const double maxRadius, const double volumeLimitRadius, const double maxApparentMagnitude, const double omegaMatter,
                                const std::function<double(double redshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double kCorrectionRedshiftVelocity)> &kCorrection,
                                std::vector<double> &outputRedshiftVelocities, std::vector<double> &outputKCorrectionRedshiftVelocities, std::vector<double> &outputRadialCoordinates, std::vector<double> &outputThetaCoordinates, std::vector<double> &outputPhiCoordinates, std::vector<double> &outputApparentMagnitudes,
                                const bool excludeFaintGalaxies)
{
  outputRedshiftVelocities.clear();
  outputKCorrectionRedshiftVelocities.clear();
  outputRadialCoordinates.clear();
  outputThetaCoordinates.clear();
  outputPhiCoordinates.clear();
  outputApparentMagnitudes.clear();

  const double volumeLimitRedshiftVelocity = redshift_velocity_from_comoving_distance(volumeLimitRadius, omegaMatter);

  const double absoluteMagnitudeThreshold = absolute_magnitude(volumeLimitRedshiftVelocity, volumeLimitRedshiftVelocity, maxApparentMagnitude, omegaMatter, 1.0, // absolute magnitude threshold used for partial volume limit; set dimensionless Hubble constant h to 1 since its actual value does not matter
                                                               luminosityEvolutionCorrection, kCorrection);

  for (std::size_t i_input = 0; i_input < inputRedshiftVelocities.size(); ++i_input)
  {
    const double kCorrectionRedshiftVelocity = inputKCorrectionRedshiftVelocities[i_input];
    const double theta = inputThetaCoordinates[i_input];
    const double phi = inputPhiCoordinates[i_input];
    const double apparentMagnitude = inputApparentMagnitudes[i_input];

    const double redshiftVelocity = change_reference_frame(inputRedshiftVelocities[i_input], theta, phi, referenceFrameChange);
    const double radius = comoving_distance(redshiftVelocity, omegaMatter);

    const double absoluteMagnitude = absolute_magnitude(redshiftVelocity, kCorrectionRedshiftVelocity, apparentMagnitude, omegaMatter, 1.0,
                                                        luminosityEvolutionCorrection, kCorrection);

    if ((radius > 0.0) and (radius <= maxRadius) and (not excludeFaintGalaxies or apparentMagnitude <= maxApparentMagnitude) and (absoluteMagnitude <= absoluteMagnitudeThreshold)) // include all objects within the maximal radius that are bright enough and not blueshifted
    {
      outputRedshiftVelocities.push_back(redshiftVelocity);
      outputKCorrectionRedshiftVelocities.push_back(kCorrectionRedshiftVelocity);
      outputRadialCoordinates.push_back(radius);
      outputThetaCoordinates.push_back(theta);
      outputPhiCoordinates.push_back(phi);
      outputApparentMagnitudes.push_back(apparentMagnitude);
    }
  }
}

void prepare_2MRS_data(const std::vector<double> &inputRedshiftVelocities, const std::vector<double> &inputKCorrectionRedshiftVelocities, const std::vector<double> &inputLatitudes, const std::vector<double> &inputLongitudes, const std::vector<double> &inputApparentMagnitudes,
                       const ReferenceFrameChange referenceFrameChange, const double maxRadius, const double volumeLimitRadius, double maxApparentMagnitude, const double omegaMatter,
                       const std::function<double(double redshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double kCorrectionRedshiftVelocity)> &kCorrection,
                       std::vector<double> &outputRedshiftVelocities, std::vector<double> &outputKCorrectionRedshiftVelocities, std::vector<double> &outputRadialCoordinates, std::vector<double> &outputThetaCoordinates, std::vector<double> &outputPhiCoordinates, std::vector<double> &outputApparentMagnitudes,
                       const bool fillZOA, const bool excludeFaintGalaxies,
                       const std::size_t randomNumberGeneratorSeed, const gsl_rng_type *randomNumberGeneratorType)
{
  const std::size_t inputGalaxyNumber = inputRedshiftVelocities.size();

  std::vector<double> inputThetaCoordinates(inputGalaxyNumber);
  std::vector<double> inputPhiCoordinates(inputGalaxyNumber);

  for (std::size_t i_g = 0; i_g < inputGalaxyNumber; ++i_g)
  {
    transform_celestial_to_spherical_coordinates(inputLatitudes[i_g], inputLongitudes[i_g],
                                                 inputThetaCoordinates[i_g], inputPhiCoordinates[i_g]);
  }

  apply_partial_volume_limit(inputRedshiftVelocities, inputKCorrectionRedshiftVelocities, inputThetaCoordinates, inputPhiCoordinates, inputApparentMagnitudes,
                             referenceFrameChange, maxRadius, volumeLimitRadius, maxApparentMagnitude, omegaMatter,
                             luminosityEvolutionCorrection, kCorrection,
                             outputRedshiftVelocities, outputKCorrectionRedshiftVelocities, outputRadialCoordinates, outputThetaCoordinates, outputPhiCoordinates, outputApparentMagnitudes,
                             excludeFaintGalaxies);

  if (fillZOA)
  {
    const std::size_t radialBinNumber = static_cast<std::size_t>(std::round(maxRadius / 10.0)); // choose the radial bin number such that the radial bin width is as close to 10 Mpc/h as possible
    constexpr std::size_t phiBinNumber = 36;

    const double radialBinWidth = maxRadius / static_cast<double>(radialBinNumber);
    constexpr double thetaBinWidth = 10.0 / 180.0 * M_PI; // 10 degrees theta bin width
    constexpr double phiBinWidth = 10.0 / 180.0 * M_PI;   // 10 degrees phi bin width

    constexpr double zoaBulgeHalfThetaWidth = 8.0 / 180.0 * M_PI; // zone of avoidance bulge covers +- 8 degrees latitude
    constexpr double zoaDiskHalfThetaWidth = 5.0 / 180.0 * M_PI;  // zone of avoidance disk covers +- 5 degrees latitude

    constexpr std::size_t maxPhiBinFirstBulgeHalf = 2; // zone of avoidance bulge covers +- 30 degree longitude, corresponding to the first and last three phi bins
    constexpr std::size_t minPhiBinSecondBulgeHalf = 33;

    gsl_rng *randomNumberGenerator = gsl_rng_alloc(randomNumberGeneratorType);
    gsl_rng_set(randomNumberGenerator, randomNumberGeneratorSeed);

    for (std::size_t i_r = 0; i_r < radialBinNumber; ++i_r)
    {
      const double binRadiusMin = static_cast<double>(i_r) * radialBinWidth;
      const double binRadiusMax = (i_r < radialBinNumber - 1) ? binRadiusMin + radialBinWidth
                                                              : maxRadius;

      for (std::size_t i_p = 0; i_p < phiBinNumber; ++i_p)
      {
        const double binPhiMin = static_cast<double>(i_p) * phiBinWidth;
        const double binPhiMax = (i_p < phiBinNumber - 1) ? binPhiMin + phiBinWidth
                                                          : 2.0 * M_PI;

        std::vector<double> adjacentBinRedshiftVelocity, adjacentBinKCorrectionRedshiftVelocity, adjacentBinRadius, adjacentBinPhi, adjacentBinApparentMagnitude;

        std::size_t maskBinGalaxyNumber = 0;
        std::size_t adjacentBinGalaxyNumber = 0;

        const double thetaMaskCenter = M_PI / 2.0;                                                                                         // the mask is centered around 0 degrees latitude
        const double thetaMaskHalfWidth = ((i_p <= maxPhiBinFirstBulgeHalf) or (i_p >= minPhiBinSecondBulgeHalf)) ? zoaBulgeHalfThetaWidth // depending on phi, choose bulge or disk masking area
                                                                                                                  : zoaDiskHalfThetaWidth;

        for (std::size_t i_g = 0; i_g < outputRadialCoordinates.size(); ++i_g)
        {
          const double redshiftVelocity = outputRedshiftVelocities[i_g];
          const double kCorrectionRedshiftVelocity = outputKCorrectionRedshiftVelocities[i_g];
          const double radius = outputRadialCoordinates[i_g];
          const double theta = outputThetaCoordinates[i_g];
          const double phi = outputPhiCoordinates[i_g];
          const double apparentMagnitude = outputApparentMagnitudes[i_g];

          if ((radius >= binRadiusMin) and (radius < binRadiusMax) and (phi >= binPhiMin) and (phi < binPhiMax)) // only consider galaxies in the current radial and phi bin
          {
            if (std::fabs(theta - thetaMaskCenter) < thetaMaskHalfWidth) // count the galaxies in the mask region
            {
              ++maskBinGalaxyNumber;
            }
            else if (std::fabs(theta - thetaMaskCenter) < thetaMaskHalfWidth + thetaBinWidth) // count the galaxies adjacent to the mask region and remember their redshifts, radial and phi coordinates, and apparent magnitudes
            {
              adjacentBinRedshiftVelocity.push_back(redshiftVelocity);
              adjacentBinKCorrectionRedshiftVelocity.push_back(kCorrectionRedshiftVelocity);
              adjacentBinRadius.push_back(radius);
              adjacentBinPhi.push_back(phi);
              adjacentBinApparentMagnitude.push_back(apparentMagnitude);

              ++adjacentBinGalaxyNumber;
            }
          }
        }

        const double meanDemaskingGalaxyNumber = static_cast<double>(adjacentBinGalaxyNumber) * thetaMaskHalfWidth / thetaBinWidth; // mean number of galaxies expected to lie in the mask region

        std::size_t demaskingGalaxyNumber = gsl_ran_poisson(randomNumberGenerator, meanDemaskingGalaxyNumber); // draw number of galaxies to fill the mask region from Poisson distribution

        if (demaskingGalaxyNumber > maskBinGalaxyNumber) // subtract number of galaxies already lying in the mask region
        {
          demaskingGalaxyNumber -= maskBinGalaxyNumber;
        }
        else
        {
          demaskingGalaxyNumber = 0;
        }

        for (std::size_t i_d = 0; i_d < demaskingGalaxyNumber; ++i_d) // fill mask region with resulting number of mock galaxies whose, redshifts, radial and phi coordinates, and apparent magnitudes match random galaxies from the adjacent bins and whose theta coordinates are chosen randomly
        {
          std::size_t adjacentSampleGalaxy = gsl_rng_uniform_int(randomNumberGenerator, adjacentBinGalaxyNumber);

          const double mockRedshiftVelocity = adjacentBinRedshiftVelocity[adjacentSampleGalaxy];
          const double mockKCorrectionRedshiftVelocity = adjacentBinKCorrectionRedshiftVelocity[adjacentSampleGalaxy];
          const double mockRadius = adjacentBinRadius[adjacentSampleGalaxy];
          const double mockTheta = thetaMaskCenter + thetaMaskHalfWidth * (gsl_rng_uniform_pos(randomNumberGenerator) * 2.0 - 1.0);
          const double mockPhi = adjacentBinPhi[adjacentSampleGalaxy];
          const double mockApparentMagnitude = adjacentBinApparentMagnitude[adjacentSampleGalaxy];

          outputRedshiftVelocities.push_back(mockRedshiftVelocity);
          outputKCorrectionRedshiftVelocities.push_back(mockKCorrectionRedshiftVelocity);
          outputRadialCoordinates.push_back(mockRadius);
          outputThetaCoordinates.push_back(mockTheta);
          outputPhiCoordinates.push_back(mockPhi);
          outputApparentMagnitudes.push_back(mockApparentMagnitude);
        }
      }
    }

    gsl_rng_free(randomNumberGenerator);
  }
}

void prepare_distance_catalog_data(const std::vector<double> &inputRedshiftVelocities, const std::vector<double> &inputLatitudes, const std::vector<double> &inputLongitudes, const std::vector<double> &inputDistanceModuli, const std::vector<double> &inputDistanceModulusErrors,
                                   const ReferenceFrameChange inputToReferenceFrame, const double minRedshiftVelocity, const double maxRedshiftVelocity, const double omegaMatter,
                                   const SphericalGridFunction &fiducialRadialVelocity, const ReferenceFrameChange cmbToReferenceFrame, const double fiducialHubble, const double maxSigmaDeviation,
                                   std::vector<double> &outputRedshiftVelocities, std::vector<double> &outputRadialCoordinates, std::vector<double> &outputThetaCoordinates, std::vector<double> &outputPhiCoordinates, std::vector<double> &outputDistanceModuli, std::vector<double> &outputDistanceModulusErrors)
{
  outputRedshiftVelocities.clear();
  outputRadialCoordinates.clear();
  outputThetaCoordinates.clear();
  outputPhiCoordinates.clear();
  outputDistanceModuli.clear();
  outputDistanceModulusErrors.clear();

  for (std::size_t i_g = 0; i_g < inputRedshiftVelocities.size(); ++i_g)
  {
    const double latitude = inputLatitudes[i_g];
    const double longitude = inputLongitudes[i_g];
    const double distanceModulus = inputDistanceModuli[i_g];
    const double distanceModulusError = inputDistanceModulusErrors[i_g];

    double theta, phi;

    transform_celestial_to_spherical_coordinates(latitude, longitude,
                                                 theta, phi);

    const double redshiftVelocity = change_reference_frame(inputRedshiftVelocities[i_g], theta, phi, inputToReferenceFrame);

    if (redshiftVelocity >= minRedshiftVelocity and
        redshiftVelocity <= maxRedshiftVelocity)
    {
      const double radius = comoving_distance(redshiftVelocity, omegaMatter);

      const double distanceModulusAtObservedRedshift = distance_modulus(redshiftVelocity, omegaMatter, fiducialHubble);
      const double distanceModulusVelocityCorrection = distance_modulus_redshift_velocity_derivative(redshiftVelocity, omegaMatter) * change_reference_frame(fiducialRadialVelocity(radius, theta, phi), theta, phi, cmbToReferenceFrame);

      const double chiSquared = gsl_pow_2(distanceModulus - distanceModulusAtObservedRedshift + distanceModulusVelocityCorrection) / gsl_pow_2(distanceModulusError);

      if (chiSquared <= gsl_pow_2(maxSigmaDeviation))
      {
        outputRedshiftVelocities.push_back(redshiftVelocity);
        outputRadialCoordinates.push_back(radius);
        outputThetaCoordinates.push_back(theta);
        outputPhiCoordinates.push_back(phi);
        outputDistanceModuli.push_back(distanceModulus);
        outputDistanceModulusErrors.push_back(distanceModulusError);
      }
    }
  }
}

double estimate_mean_density(const double maxRadius, const std::vector<double> &observedGalaxyDistances, const std::function<double(double)> &selectionFunction)
{
  double expectedGalaxyNumber = 0.0;

  for (std::size_t i_g = 0; i_g < observedGalaxyDistances.size(); ++i_g)
  {
    expectedGalaxyNumber += 1.0 / selectionFunction(observedGalaxyDistances[i_g]); // expected number of galaxies at the position of the observed one
  }

  return expectedGalaxyNumber / 4.0 * 3.0 / M_PI / gsl_pow_3(maxRadius);
}

void estimate_selection_function(const std::vector<double> &redshiftVelocities, const std::vector<double> &kCorrectionRedshiftVelocities, const std::vector<double> &apparentMagnitudes, const double maxApparentMagnitude,
                                 const double maxRadius, const std::size_t radialBinNumber, const double omegaMatter,
                                 const std::function<double(double redshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double kCorrectionRedshiftVelocity)> &kCorrection,
                                 Cartesian1DGridFunction &selectionFunction, Cartesian1DGridFunction &selectionFunctionLogDerivative, Cartesian1DGridFunction &radialDistributionFunction)
{
  const std::size_t galaxyNumber = redshiftVelocities.size();

  selectionFunction = Cartesian1DGridFunction(0.0, maxRadius, radialBinNumber, 1.0); // compute the selection function, its logarithmic derivative and the radial distribution function on an equidistant 1D grid
  selectionFunctionLogDerivative = Cartesian1DGridFunction(0.0, maxRadius, radialBinNumber, 0.0);
  radialDistributionFunction = Cartesian1DGridFunction(0.0, maxRadius, radialBinNumber, 0.0);

  const double radialBinWidth = maxRadius / static_cast<double>(radialBinNumber - 1);

  double FT_prev = 0.0;

  for (std::size_t i_b = 1; i_b < radialBinNumber; ++i_b)
  {
    const double binRadius = selectionFunction.coordinate(i_b);
    const double nextBinRadius = binRadius + radialBinWidth;

    const double binRedshiftVelocity = redshift_velocity_from_comoving_distance(binRadius, omegaMatter);
    const double nextBinRedshiftVelocity = redshift_velocity_from_comoving_distance(nextBinRadius, omegaMatter);

    const double binAbsoluteMagnitudeThreshold = absolute_magnitude(binRedshiftVelocity, binRedshiftVelocity, maxApparentMagnitude, omegaMatter, 1.0, // set dimensionless Hubble constant h to 1 since it's value does not affect the F/T estimator
                                                                    luminosityEvolutionCorrection, kCorrection);
    const double nextBinAbsoluteMagnitudeThreshold = absolute_magnitude(nextBinRedshiftVelocity, nextBinRedshiftVelocity, maxApparentMagnitude, omegaMatter, 1.0,
                                                                        luminosityEvolutionCorrection, kCorrection);

    std::size_t F = 0;
    std::size_t T = 0;
    std::size_t dN = 0;

    for (std::size_t i_g = 0; i_g < galaxyNumber; ++i_g)
    {
      const double redshiftVelocity = redshiftVelocities[i_g];
      const double kCorrectionRedshiftVelocity = kCorrectionRedshiftVelocities[i_g];
      const double apparentMagnitude = apparentMagnitudes[i_g];

      const double radius = comoving_distance(redshiftVelocity, omegaMatter);
      const double absoluteMagnitude = absolute_magnitude(redshiftVelocity, kCorrectionRedshiftVelocity, apparentMagnitude, omegaMatter, 1.0,
                                                          luminosityEvolutionCorrection, kCorrection);

      if ((radius < binRadius) and (absoluteMagnitude < binAbsoluteMagnitudeThreshold)) // for every radial bin, T counts all galaxies within that radius bright enough to be detectable also at larger radii
      {
        ++T;

        if (absoluteMagnitude >= nextBinAbsoluteMagnitudeThreshold) // F counts the subset of these galaxies bright enough to still be detectable in the next radial bin but not beyond
        {
          ++F;
        }
      }

      if ((radius >= binRadius - radialBinWidth) and (radius < binRadius)) // for every radial bin, dN counts all galaxies within that bin
      {
        ++dN;
      }
    }

    const double FT = (T != 0) ? static_cast<double>(F) / static_cast<double>(T)
                               : FT_prev; // if there are no sufficiently bright galaxies in the bin, assume that F/T stays constant

    selectionFunctionLogDerivative.value(i_b) = (FT_prev != 0.0) ? -FT / radialBinWidth * binRadius // the logarithmic derivative of the selection function is estimated as -F/T * r/dr
                                                                 : 0.0;                             // unless the previous value of F/T was zero, implying that the selection function did not change from the previous bin (this case ensures that the derivative is zero up to and including the partial volume limit radius)
    selectionFunction.value(i_b) = selectionFunction.value(i_b - 1) * (1.0 - FT_prev);              // the selection function itself is then obtained by (binned) integration over the radius
    radialDistributionFunction.value(i_b) = static_cast<double>(dN) / radialBinWidth;               // the radial distribution function is estimated as dN/dr

    FT_prev = FT;
  }
}

void estimate_sigma_galaxy(const std::vector<double> &redshiftVelocities, const std::vector<double> &kCorrectionRedshiftVelocities, const std::vector<double> &thetaCoordinates, const std::vector<double> &phiCoordinates, const std::vector<double> &apparentMagnitudes,
                           const double minRadius, const double maxRadius, const std::size_t radialBinNumber, const double maxApparentMagnitude, const double omegaMatter, const double smoothingScale,
                           const std::function<double(double redshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double kCorrectionRedshiftVelocity)> &kCorrection,
                           Cartesian1DGridFunction &sigmaGalaxy, std::vector<std::size_t> &galaxyNumbers)
{
  sigmaGalaxy = Cartesian1DGridFunction(minRadius, maxRadius, radialBinNumber);
  galaxyNumbers = std::vector<std::size_t>(radialBinNumber);

  for (std::size_t i_r = 0; i_r < radialBinNumber; ++i_r)
  {
    const double volumeLimitRadius = sigmaGalaxy.coordinate(i_r);
    const double reconstructionRadius = volumeLimitRadius - smoothingScale;

    const std::size_t reconstructionRadialBinNumber = static_cast<std::size_t>(reconstructionRadius / smoothingScale * 4.0);
    const std::size_t reconstructionThetaBinNumber = reconstructionRadialBinNumber * 6;
    const std::size_t reconstructionPhiBinNumber = reconstructionRadialBinNumber * 3;

    std::vector<double> volumeLimitedRedshiftVelocities, volumeLimitedKCorrectionRedshiftVelocities, volumeLimitedRadialCoordinates, volumeLimitedThetaCoordinates, volumeLimitedPhiCoordinates, volumeLimitedApparentMagnitudes;

    apply_partial_volume_limit(redshiftVelocities, kCorrectionRedshiftVelocities, thetaCoordinates, phiCoordinates, apparentMagnitudes,
                               NO_REFERENCE_FRAME_CHANGE, volumeLimitRadius, volumeLimitRadius, maxApparentMagnitude, omegaMatter,
                               luminosityEvolutionCorrection, kCorrection,
                               volumeLimitedRedshiftVelocities, volumeLimitedKCorrectionRedshiftVelocities, volumeLimitedRadialCoordinates, volumeLimitedThetaCoordinates, volumeLimitedPhiCoordinates, volumeLimitedApparentMagnitudes);

    std::vector<double> xCoordinates, yCoordinates, zCoordinates;

    for (std::size_t i_g = 0; i_g < volumeLimitedRadialCoordinates.size(); ++i_g)
    {
      double x, y, z;

      transform_spherical_to_cartesian_coordinates(volumeLimitedRadialCoordinates[i_g], volumeLimitedThetaCoordinates[i_g], volumeLimitedPhiCoordinates[i_g],
                                                   x, y, z);

      xCoordinates.push_back(x);
      yCoordinates.push_back(y);
      zCoordinates.push_back(z);
    }

    const double galaxyGridLength = 2.0 * volumeLimitRadius;
    constexpr std::size_t galaxyGridBinNumber = 10;

    ObjectGrid galaxyGrid(galaxyGridLength, galaxyGridBinNumber,
                          0.0, 0.0, 0.0,
                          xCoordinates, yCoordinates, zCoordinates,
                          false);

    std::size_t galaxyNumber = galaxyGrid.object_number();

    const auto tophatSmoothingKernel = [&](double distance)
    {
      return (distance <= smoothingScale) ? 3.0 / 4.0 / M_PI / gsl_pow_3(smoothingScale)
                                          : 0.0;
    };

    SphericalGridFunction galaxyDensityContrastField(reconstructionRadius, reconstructionRadialBinNumber, reconstructionThetaBinNumber, reconstructionPhiBinNumber);

    galaxyDensityContrastField.apply(
        [&](double value, double radius, double theta, double phi)
        {
          double x, y, z;

          transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                                       x, y, z);

          double density = 0.0;

          galaxyGrid.evaluate_for_all_objects_within_distance(
              x, y, z, smoothingScale,
              [&](double relativeX, double relativeY, double relativeZ, double distance, std::size_t i_g)
              {
                density += tophatSmoothingKernel(distance);
              });

          return density;
        },
        true);

    const double meanGalaxyDensity = galaxyDensityContrastField.average();

    galaxyDensityContrastField = galaxyDensityContrastField / meanGalaxyDensity - 1.0;

    const double meanGalaxyNumberPerSmoothingVolume = static_cast<double>(galaxyNumber) * gsl_pow_3(smoothingScale / volumeLimitRadius); // for top-hat filter

    const double meanGalaxyDensityContrast = galaxyDensityContrastField.average();
    const double sigmaGalaxyDensityContrast = std::sqrt(galaxyDensityContrastField.average(gsl_pow_2) - gsl_pow_2(meanGalaxyDensityContrast) - 1.0 / meanGalaxyNumberPerSmoothingVolume);

    sigmaGalaxy.value(i_r) = sigmaGalaxyDensityContrast;
    galaxyNumbers[i_r] = galaxyNumber;
  }
}