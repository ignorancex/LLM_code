#include "ConstrainedRealizations.hpp"

#include <iostream>
#include <tr1/cmath>

#include <gsl/gsl_integration.h>

#include "cosmology.hpp"
#include "miscellaneous.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

ConstrainedRealizations::ConstrainedRealizations(const std::vector<double> &radialCoordinates, const std::vector<double> &thetaCoordinates, const std::vector<double> &phiCoordinates, const ReferenceFrame redshiftReferenceFrame,
                                                 const double maxRadius, const double radialResolution, const std::size_t maxMultipole, std::size_t radialBinNumber, std::size_t thetaBinNumber, std::size_t phiBinNumber,
                                                 const double fftPeriodicBoundaryDistance, const std::size_t fftBinNumber, const double logTransformSmoothingScale,
                                                 const double meanGalaxyDensity, const std::function<double(double)> &selectionFunction, const std::function<double(double)> &selectionFunctionLogDerivative, const std::function<double(double)> &sigmaGalaxy, const std::function<double(double)> &normalizedPowerSpectrum,
                                                 const std::string &directory, const std::string &identifier,
                                                 const bool usePrecomputedCouplingAndNoiseMatrices,
                                                 const gsl_rng_type *randomNumberGeneratorType)
    : MaxRadius(maxRadius),
      RadialResolution(radialResolution),
      MaxMultipole(maxMultipole),
      RadialBinNumber(radialBinNumber),
      ThetaBinNumber(thetaBinNumber),
      PhiBinNumber(phiBinNumber),
      FFTPeriodicBoundaryDistance(fftPeriodicBoundaryDistance),
      FFTBinNumber(fftBinNumber),
      LogTransformSmoothingScale(logTransformSmoothingScale),
      MeanGalaxyDensity(meanGalaxyDensity),
      SelectionFunction(selectionFunction),
      SigmaGalaxy(sigmaGalaxy),
      NormalizedPowerSpectrum(normalizedPowerSpectrum),
      RandomRealizations(FFTPeriodicBoundaryDistance, FFTBinNumber),
      SurveyRSDCorrection(),
      SurveyWienerFilter(),
      RandomWienerFilter(),
      RawSurveySFBD(),
      SurveySFBD(),
      RandomSFBD(),
      RandomNumberGeneratorType(randomNumberGeneratorType),
      RealizationNumber(0),
      RandomGalaxyRadialCoordinates(),
      RandomGalaxyThetaCoordinates(),
      RandomGalaxyPhiCoordinates(),
      NormalizedSurveyEstimateDensityContrast(),
      NormalizedSurveyEstimateRadialVelocity(),
      NormalizedSurveyEstimateThetaVelocity(),
      NormalizedSurveyEstimatePhiVelocity(),
      NormalizedRandomEstimateDensityContrast(),
      NormalizedRandomEstimateRadialVelocity(),
      NormalizedRandomEstimateThetaVelocity(),
      NormalizedRandomEstimatePhiVelocity(),
      NormalizedRandomSignalDensityContrast(),
      NormalizedRandomSignalRadialVelocity(),
      NormalizedRandomSignalThetaVelocity(),
      NormalizedRandomSignalPhiVelocity(),
      SurveyEstimateGrowthRate(-1.0),
      SurveyEstimateDensityContrastSmoothingScale(-1.0),
      SurveyEstimateRadialVelocitySmoothingScale(-1.0),
      SurveyEstimateThetaVelocitySmoothingScale(-1.0),
      SurveyEstimatePhiVelocitySmoothingScale(-1.0),
      RandomEstimateDensityContrastSmoothingScale(-1.0),
      RandomEstimateRadialVelocitySmoothingScale(-1.0),
      RandomEstimateThetaVelocitySmoothingScale(-1.0),
      RandomEstimatePhiVelocitySmoothingScale(-1.0),
      RandomSignalSmoothingScale(-1.0)
{
    if (fftPeriodicBoundaryDistance < maxRadius)
    {
        std::cout << std::endl
                  << " ConstrainedRealizations Error: Distance to the periodic boundary of the FFT box smaller than the maximal radius" << std::endl
                  << std::endl;

        exit(EXIT_FAILURE);
    }

    const std::string linearRSDCorrectionFileName = directory + "linear_RSD_correction" + identifier + ".bin";
    const std::string surveyWienerFilterFileName = directory + "survey_wiener_filter" + identifier + ".bin";
    const std::string randomWienerFilterFileName = directory + "random_wiener_filter" + identifier + ".bin";

    const bool linearRSDCorrectionFileExists = file_exists(linearRSDCorrectionFileName);
    const bool surveyWienerFilterFileExists = file_exists(surveyWienerFilterFileName);
    const bool randomWienerFilterFileExists = file_exists(randomWienerFilterFileName);

    const auto selectionTimesWeightingFunction = [&](double radius)
    {
        return 1.0 / SigmaGalaxy(radius);
    };

    const auto surveyNoiseWeightingFunction = [&](double radius)
    {
        return SelectionFunction(radius) * gsl_pow_2(SigmaGalaxy(radius));
    };

    const auto randomNoiseWeightingFunction = [&](double radius)
    {
        return SelectionFunction(radius);
    };

    if (usePrecomputedCouplingAndNoiseMatrices and linearRSDCorrectionFileExists)
    {
        SurveyRSDCorrection = LinearRSDCorrection(linearRSDCorrectionFileName);
    }
    else
    {
        SurveyRSDCorrection = LinearRSDCorrection(maxRadius, radialResolution, maxMultipole,
                                                  selectionFunctionLogDerivative, selectionTimesWeightingFunction, redshiftReferenceFrame);

        SurveyRSDCorrection.save_object_to_file(linearRSDCorrectionFileName);
    }

    if (usePrecomputedCouplingAndNoiseMatrices and surveyWienerFilterFileExists)
    {
        SurveyWienerFilter = WienerFilter(surveyWienerFilterFileName);
    }
    else
    {
        SurveyWienerFilter = WienerFilter(maxRadius, radialResolution, maxMultipole,
                                          surveyNoiseWeightingFunction);

        SurveyWienerFilter.save_object_to_file(surveyWienerFilterFileName);
    }

    if (usePrecomputedCouplingAndNoiseMatrices and randomWienerFilterFileExists)
    {
        RandomWienerFilter = WienerFilter(randomWienerFilterFileName);
    }
    else
    {
        RandomWienerFilter = WienerFilter(maxRadius, radialResolution, maxMultipole,
                                          randomNoiseWeightingFunction);

        RandomWienerFilter.save_object_to_file(randomWienerFilterFileName);
    }

    const auto surveyWeightingFunction = [&](double radius)
    {
        return 1.0 / MeanGalaxyDensity / SelectionFunction(radius) / SigmaGalaxy(radius);
    };

    RawSurveySFBD = SphericalFourierBesselDecomposition(radialCoordinates, thetaCoordinates, phiCoordinates,
                                                        MaxRadius, radialResolution, MaxMultipole,
                                                        surveyWeightingFunction, selectionTimesWeightingFunction);
}

void ConstrainedRealizations::generate(const std::size_t realizationNumber)
{
    RealizationNumber = realizationNumber;

    RandomRealizations.generate(MaxRadius, NormalizedPowerSpectrum, MeanGalaxyDensity, SelectionFunction, LogTransformSmoothingScale,
                                RandomGalaxyRadialCoordinates, RandomGalaxyThetaCoordinates, RandomGalaxyPhiCoordinates,
                                RealizationNumber, RandomNumberGeneratorType);

    const auto randomWeightingFunction = [&](double radius)
    {
        return 1.0 / MeanGalaxyDensity / SelectionFunction(radius);
    };

    RandomSFBD = SphericalFourierBesselDecomposition(RandomGalaxyRadialCoordinates, RandomGalaxyThetaCoordinates, RandomGalaxyPhiCoordinates,
                                                     MaxRadius, RadialResolution, MaxMultipole,
                                                     randomWeightingFunction, [](double)
                                                     { return 1.0; });

    RandomWienerFilter.apply_to(RandomSFBD, MeanGalaxyDensity, NormalizedPowerSpectrum, NormalizedPowerSpectrum);

    reset_random_fields();
}

std::vector<double> &ConstrainedRealizations::get_random_galaxy_radial_coordinates()
{
    return RandomGalaxyRadialCoordinates;
}

std::vector<double> &ConstrainedRealizations::get_random_galaxy_theta_coordinates()
{
    return RandomGalaxyThetaCoordinates;
}

std::vector<double> &ConstrainedRealizations::get_random_galaxy_phi_coordinates()
{
    return RandomGalaxyPhiCoordinates;
}

SphericalGridFunction ConstrainedRealizations::get_constrained_density_contrast(const double sigmaMatter, const double normalizedGrowthRate, const double smoothingScale)
{
    return get_survey_estimate_density_contrast(sigmaMatter, normalizedGrowthRate, smoothingScale) +
           get_noise_density_contrast(sigmaMatter, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_constrained_radial_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_survey_estimate_radial_velocity(normalizedGrowthRate, smoothingScale) +
           get_noise_radial_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_constrained_theta_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_survey_estimate_theta_velocity(normalizedGrowthRate, smoothingScale) +
           get_noise_theta_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_constrained_phi_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_survey_estimate_phi_velocity(normalizedGrowthRate, smoothingScale) +
           get_noise_phi_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_noise_density_contrast(const double sigmaMatter, const double smoothingScale)
{
    return get_random_signal_density_contrast(sigmaMatter, smoothingScale) -
           get_random_estimate_density_contrast(sigmaMatter, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_noise_radial_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_random_signal_radial_velocity(normalizedGrowthRate, smoothingScale) -
           get_random_estimate_radial_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_noise_theta_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_random_signal_theta_velocity(normalizedGrowthRate, smoothingScale) -
           get_random_estimate_theta_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_noise_phi_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    return get_random_signal_phi_velocity(normalizedGrowthRate, smoothingScale) -
           get_random_estimate_phi_velocity(normalizedGrowthRate, smoothingScale);
}

SphericalGridFunction ConstrainedRealizations::get_survey_estimate_density_contrast(const double sigmaMatter, const double normalizedGrowthRate, const double smoothingScale)
{
    update_survey_estimate_decomposition(normalizedGrowthRate);

    update_estimate_density_contrast(smoothingScale,
                                     NormalizedSurveyEstimateDensityContrast, SurveyEstimateDensityContrastSmoothingScale, SurveySFBD);

    return sigmaMatter * NormalizedSurveyEstimateDensityContrast;
}

SphericalGridFunction ConstrainedRealizations::get_survey_estimate_radial_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_survey_estimate_decomposition(normalizedGrowthRate);

    update_estimate_radial_velocity(smoothingScale,
                                    NormalizedSurveyEstimateRadialVelocity, SurveyEstimateRadialVelocitySmoothingScale, SurveySFBD);

    return normalizedGrowthRate * NormalizedSurveyEstimateRadialVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_survey_estimate_theta_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_survey_estimate_decomposition(normalizedGrowthRate);

    update_estimate_theta_velocity(smoothingScale,
                                   NormalizedSurveyEstimateThetaVelocity, SurveyEstimateThetaVelocitySmoothingScale, SurveySFBD);

    return normalizedGrowthRate * NormalizedSurveyEstimateThetaVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_survey_estimate_phi_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_survey_estimate_decomposition(normalizedGrowthRate);

    update_estimate_phi_velocity(smoothingScale,
                                 NormalizedSurveyEstimatePhiVelocity, SurveyEstimatePhiVelocitySmoothingScale, SurveySFBD);

    return normalizedGrowthRate * NormalizedSurveyEstimatePhiVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_estimate_density_contrast(const double sigmaMatter, const double smoothingScale)
{
    update_estimate_density_contrast(smoothingScale,
                                     NormalizedRandomEstimateDensityContrast, RandomEstimateDensityContrastSmoothingScale, RandomSFBD);

    return sigmaMatter * NormalizedRandomEstimateDensityContrast;
}

SphericalGridFunction ConstrainedRealizations::get_random_estimate_radial_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_estimate_radial_velocity(smoothingScale,
                                    NormalizedRandomEstimateRadialVelocity, RandomEstimateRadialVelocitySmoothingScale, RandomSFBD);

    return normalizedGrowthRate * NormalizedRandomEstimateRadialVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_estimate_theta_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_estimate_theta_velocity(smoothingScale,
                                   NormalizedRandomEstimateThetaVelocity, RandomEstimateThetaVelocitySmoothingScale, RandomSFBD);

    return normalizedGrowthRate * NormalizedRandomEstimateThetaVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_estimate_phi_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_estimate_phi_velocity(smoothingScale,
                                 NormalizedRandomEstimatePhiVelocity, RandomEstimatePhiVelocitySmoothingScale, RandomSFBD);

    return normalizedGrowthRate * NormalizedRandomEstimatePhiVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_signal_density_contrast(const double sigmaMatter, const double smoothingScale)
{
    update_random_signal_fields(smoothingScale);

    return sigmaMatter * NormalizedRandomSignalDensityContrast;
}

SphericalGridFunction ConstrainedRealizations::get_random_signal_radial_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_random_signal_fields(smoothingScale);

    return normalizedGrowthRate * NormalizedRandomSignalRadialVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_signal_theta_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_random_signal_fields(smoothingScale);

    return normalizedGrowthRate * NormalizedRandomSignalThetaVelocity;
}

SphericalGridFunction ConstrainedRealizations::get_random_signal_phi_velocity(const double normalizedGrowthRate, const double smoothingScale)
{
    update_random_signal_fields(smoothingScale);

    return normalizedGrowthRate * NormalizedRandomSignalPhiVelocity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void ConstrainedRealizations::reset_survey_fields()
{
    SurveyEstimateDensityContrastSmoothingScale = -1.0;
    SurveyEstimateRadialVelocitySmoothingScale = -1.0;
    SurveyEstimateThetaVelocitySmoothingScale = -1.0;
    SurveyEstimatePhiVelocitySmoothingScale = -1.0;
}

void ConstrainedRealizations::reset_random_fields()
{
    RandomEstimateDensityContrastSmoothingScale = -1.0;
    RandomEstimateRadialVelocitySmoothingScale = -1.0;
    RandomEstimateThetaVelocitySmoothingScale = -1.0;
    RandomEstimatePhiVelocitySmoothingScale = -1.0;
    RandomSignalSmoothingScale = -1.0;
}

void ConstrainedRealizations::update_estimate_density_contrast(const double smoothingScale,
                                                               SphericalGridFunction &normalizedDensityContrast, double &normalizedDensityContrastSmoothingScale,
                                                               const SphericalFourierBesselDecomposition &sfbd)
{
    if (smoothingScale != normalizedDensityContrastSmoothingScale)
    {
        normalizedDensityContrast = sfbd.field_grid(RadialBinNumber, ThetaBinNumber, PhiBinNumber, smoothingScale);

        normalizedDensityContrastSmoothingScale = smoothingScale;
    }
}

void ConstrainedRealizations::update_estimate_radial_velocity(const double smoothingScale,
                                                              SphericalGridFunction &normalizedRadialVelocity, double &normalizedRadialVelocitySmoothingScale,
                                                              const SphericalFourierBesselDecomposition &sfbd)
{
    if (smoothingScale != normalizedRadialVelocitySmoothingScale)
    {
        normalizedRadialVelocity = -HUBBLE_NORMALIZATION * sfbd.radial_inverse_divergence_grid(RadialBinNumber, ThetaBinNumber, PhiBinNumber, smoothingScale);

        normalizedRadialVelocitySmoothingScale = smoothingScale;
    }
}

void ConstrainedRealizations::update_estimate_theta_velocity(const double smoothingScale,
                                                             SphericalGridFunction &normalizedThetaVelocity, double &normalizedThetaVelocitySmoothingScale,
                                                             const SphericalFourierBesselDecomposition &sfbd)
{
    if (smoothingScale != normalizedThetaVelocitySmoothingScale)
    {
        normalizedThetaVelocity = -HUBBLE_NORMALIZATION * sfbd.theta_inverse_divergence_grid(RadialBinNumber, ThetaBinNumber, PhiBinNumber, smoothingScale);

        normalizedThetaVelocitySmoothingScale = smoothingScale;
    }
}

void ConstrainedRealizations::update_estimate_phi_velocity(const double smoothingScale,
                                                           SphericalGridFunction &normalizedPhiVelocity, double &normalizedPhiVelocitySmoothingScale,
                                                           const SphericalFourierBesselDecomposition &sfbd)
{
    if (smoothingScale != normalizedPhiVelocitySmoothingScale)
    {
        normalizedPhiVelocity = -HUBBLE_NORMALIZATION * sfbd.phi_inverse_divergence_grid(RadialBinNumber, ThetaBinNumber, PhiBinNumber, smoothingScale);

        normalizedPhiVelocitySmoothingScale = smoothingScale;
    }
}

void ConstrainedRealizations::update_survey_estimate_decomposition(const double normalizedGrowthRate)
{
    if (SurveyEstimateGrowthRate != normalizedGrowthRate)
    {
        SurveySFBD = RawSurveySFBD;

        SurveyRSDCorrection.transform_redshift_to_configuration_space(SurveySFBD, normalizedGrowthRate);

        double meanRealSpaceNormalizedDensityContrast = 0.0;

        for (std::size_t n = 1; n <= SurveySFBD.radial_mode_number(); ++n) // compute mean density contrast via sum over monopole coefficients
        {
            const double k_0n = SurveySFBD.radial_wavenumber(0, n);
            const double C_0n = SurveySFBD.radial_normalization_constant(0, n);
            const double delta_00n = (SurveySFBD.SFB_coefficient(0, 0, n)).real();

            meanRealSpaceNormalizedDensityContrast += C_0n * delta_00n * 3.0 / 2.0 / std::sqrt(M_PI) * std::tr1::sph_bessel(1, k_0n * MaxRadius) / k_0n / MaxRadius;
        }

        for (std::size_t n = 1; n <= SurveySFBD.radial_mode_number(); ++n) // subtract the SFB projection of this mean density contrast
        {
            const double k_0n = SurveySFBD.radial_wavenumber(0, n);
            const double monopoleCorrection = meanRealSpaceNormalizedDensityContrast * 2.0 * std::sqrt(M_PI) * std::tr1::sph_bessel(1, k_0n * MaxRadius) / k_0n * gsl_pow_2(MaxRadius);

            SurveySFBD.SFB_coefficient(0, 0, n) -= monopoleCorrection;
        }

        SurveyWienerFilter.apply_to(SurveySFBD, MeanGalaxyDensity, NormalizedPowerSpectrum, NormalizedPowerSpectrum);

        SurveyEstimateGrowthRate = normalizedGrowthRate;

        reset_survey_fields();
    }
}

void ConstrainedRealizations::update_random_signal_fields(const double smoothingScale)
{
    if (smoothingScale != RandomSignalSmoothingScale)
    {
        RandomRealizations.get_fields(1.0, smoothingScale,
                                      RadialBinNumber, ThetaBinNumber, PhiBinNumber,
                                      NormalizedRandomSignalDensityContrast, NormalizedRandomSignalRadialVelocity, NormalizedRandomSignalThetaVelocity, NormalizedRandomSignalPhiVelocity);

        RandomSignalSmoothingScale = smoothingScale;
    }
}