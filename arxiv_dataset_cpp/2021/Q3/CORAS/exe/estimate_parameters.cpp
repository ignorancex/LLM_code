#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <omp.h>

#include "../src/ConstrainedRealizations.hpp"
#include "../src/FileTable.hpp"
#include "../src/NormalizedPowerSpectrum.hpp"
#include "../src/SphericalGridFunction.hpp"
#include "../src/SphericalGridFunctionInterpolator.hpp"
#include "../src/survey.hpp"
#include "../src/transformations.hpp"
#include "../src/velocity_comparison.hpp"

#include "../configuration.hpp"

int main(int argc, char **argv)
{
    fftw_init_threads(); // initialize parallel execution of FFTs used in ConstrainedRealizations::generate
    fftw_plan_with_nthreads(omp_get_max_threads());

    const std::size_t cutNumber = PARAMETER_ESTIMATE_SMOOTHING_SCALES.size() * PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES.size();
    const std::size_t constrainedRealizationNumber = FINAL_CONSTRAINED_REALIZATION - INITIAL_CONSTRAINED_REALIZATION + 1;

    const std::string realizationRangeComment = "_CR" + std::to_string(INITIAL_CONSTRAINED_REALIZATION) + "-" + std::to_string(FINAL_CONSTRAINED_REALIZATION);

    const FileTable inputPowerSpectrumFileTable(FIDUCIAL_POWER_SPECTRUM_FILE_NAME, 2); // read the power spectrum from file

    const auto inputWavenumbers = inputPowerSpectrumFileTable.column<double>(0);
    const auto inputPowerSpectrumValues = inputPowerSpectrumFileTable.column<double>(1);

    NormalizedPowerSpectrum normalizedPowerSpectrum(inputWavenumbers, inputPowerSpectrumValues,
                                                    FIDUCIAL_HUBBLE, SIGMA_SCALE);

    const double fiducialNormalizedGrowthRate = FIDUCIAL_GROWTH_RATE * std::sqrt(normalizedPowerSpectrum.variance());

    const FileTable inputRedshiftCatalogFile(REDSHIFT_CATALOG_FILE_NAME, 10, '\t'); // read the redshift catalog properties from file

    const auto inputRedshiftCatalogLatitudes = inputRedshiftCatalogFile.column<double>(1);
    const auto inputRedshiftCatalogLongitudes = inputRedshiftCatalogFile.column<double>(2);
    const auto inputRedshiftCatalogRedshiftVelocities = REDSHIFT_CATALOG_COLLAPSE_FINGERS_OF_GOD ? inputRedshiftCatalogFile.column<double>(9)
                                                                                                 : inputRedshiftCatalogFile.column<double>(3);
    const auto inputRedshiftCatalogKCorrectionRedshiftVelocities = inputRedshiftCatalogFile.column<double>(3);
    const auto inputRedshiftCatalogApparentMagnitudes = inputRedshiftCatalogFile.column<double>(4);

    const FileTable inputDistanceCatalogFile(DISTANCE_CATALOG_GROUPS_FILE_NAME, 7, '\t'); // read the distance catalog properties from file

    const auto inputDistanceCatalogLatitudes = inputDistanceCatalogFile.column<double>(2);
    const auto inputDistanceCatalogLongitudes = inputDistanceCatalogFile.column<double>(3);
    const auto inputDistanceCatalogRedshiftVelocities = inputDistanceCatalogFile.column<double>(4);
    const auto inputDistanceCatalogDistanceModuli = inputDistanceCatalogFile.column<double>(5);
    const auto inputDistanceCatalogDistanceModulusErrors = inputDistanceCatalogFile.column<double>(6);

    for (std::size_t i_ref = 0; i_ref < RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES.size(); ++i_ref)
    {
        const ReferenceFrame redshiftReferenceFrame = RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES[i_ref];

        const ReferenceFrameChange redshiftCatalogInputToReferenceFrame = get_reference_frame_change(REDSHIFT_CATALOG_REDSHIFT_REFERENCE_FRAME, redshiftReferenceFrame);
        const ReferenceFrameChange distanceCatalogInputToReferenceFrame = get_reference_frame_change(DISTANCE_CATALOG_REDSHIFT_REFERENCE_FRAME, redshiftReferenceFrame);
        const ReferenceFrameChange cmbToReferenceFrame = get_reference_frame_change(CMB_FRAME, redshiftReferenceFrame);

        const std::string redshiftReferenceFrameName = get_reference_frame_name(redshiftReferenceFrame);
        const std::string redshiftReferenceFrameComment = "_z" + redshiftReferenceFrameName;

        const auto time1 = std::chrono::high_resolution_clock::now();
        std::cout << "Preparing " << redshiftReferenceFrameName << " reference frame..." << std::flush;

        std::vector<double> reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionRadialCoordinates, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentMagnitudes;

        prepare_2MRS_data(inputRedshiftCatalogRedshiftVelocities, inputRedshiftCatalogKCorrectionRedshiftVelocities, inputRedshiftCatalogLatitudes, inputRedshiftCatalogLongitudes, inputRedshiftCatalogApparentMagnitudes,
                          redshiftCatalogInputToReferenceFrame, RECONSTRUCTION_MAX_RADIUS, REDSHIFT_CATALOG_VOLUME_LIMIT_RADIUS, REDSHIFT_CATALOG_MAX_APPARENT_MAGNITUDE, FIDUCIAL_OMEGA_MATTER,
                          luminosity_evolution_correction_2MRS, k_correction_2MRS,
                          reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionRadialCoordinates, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentMagnitudes);

        Cartesian1DGridFunction sigmaGalaxyAboveVolumeLimit;
        std::vector<std::size_t> volumeLimitGalaxyNumbers;

        estimate_sigma_galaxy(reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentMagnitudes,
                              REDSHIFT_CATALOG_VOLUME_LIMIT_RADIUS, RECONSTRUCTION_MAX_RADIUS, SIGMA_GALAXY_RADIAL_BIN_NUMBER, REDSHIFT_CATALOG_MAX_APPARENT_MAGNITUDE, FIDUCIAL_OMEGA_MATTER, SIGMA_SCALE,
                              luminosity_evolution_correction_2MRS, k_correction_2MRS,
                              sigmaGalaxyAboveVolumeLimit, volumeLimitGalaxyNumbers);

        auto sigmaGalaxy = [&](double radius)
        {
            if (radius < sigmaGalaxyAboveVolumeLimit.coordinate(0))
            {
                return sigmaGalaxyAboveVolumeLimit.value(0);
            }
            else
            {
                return sigmaGalaxyAboveVolumeLimit(radius);
            }
        };

        Cartesian1DGridFunction selectionFunction, selectionFunctionLogDerivative, radialDistributionFunction;

        estimate_selection_function(reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionApparentMagnitudes, REDSHIFT_CATALOG_MAX_APPARENT_MAGNITUDE,
                                    RECONSTRUCTION_MAX_RADIUS, SELECTION_FUNCTION_BIN_NUMBER, FIDUCIAL_OMEGA_MATTER,
                                    luminosity_evolution_correction_2MRS, k_correction_2MRS,
                                    selectionFunction, selectionFunctionLogDerivative, radialDistributionFunction);

        const double meanGalaxyDensity = estimate_mean_density(RECONSTRUCTION_MAX_RADIUS, reconstructionRadialCoordinates, selectionFunction);

        const std::string precomputedRSDCorrectionAndWienerFilterComment = redshiftReferenceFrameComment + CONFIGURATION_COMMENT;

        ConstrainedRealizations constrainedRealizations(reconstructionRadialCoordinates, reconstructionThetaCoordinates, reconstructionPhiCoordinates, redshiftReferenceFrame,
                                                        RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_RESOLUTION, RECONSTRUCTION_MAX_MULTIPOLE, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER,
                                                        RECONSTRUCTION_FFT_PERIODIC_BOUNDARY_DISTANCE, RECONSTRUCTION_FFT_BIN_NUMBER, RECONSTRUCTION_LOG_TRAFO_SMOOTHING_SCALE,
                                                        meanGalaxyDensity, selectionFunction, selectionFunctionLogDerivative, sigmaGalaxy, normalizedPowerSpectrum,
                                                        DATA_DIRECTORY, precomputedRSDCorrectionAndWienerFilterComment,
                                                        RECONSTRUCTION_USE_PRECOMPUTED_RSD_CORRECTION_AND_WIENER_FILTER);

        const double normalizedGrowthRateBinWidth = (PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MAX - PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MIN) / static_cast<double>(PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_BIN_NUMBER - 1);

        std::vector<std::vector<SphericalGridFunction>> normalizedSurveyEstimateRadialVelocities(PARAMETER_ESTIMATE_SMOOTHING_SCALES.size(), std::vector<SphericalGridFunction>(PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_BIN_NUMBER));

        for (std::size_t i_growth = 0; i_growth < PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_BIN_NUMBER; ++i_growth)
        {
            const double normalizedGrowthRate = (i_growth == PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_BIN_NUMBER - 1) ? PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MAX
                                                                                                                       : PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MIN + static_cast<double>(i_growth) * normalizedGrowthRateBinWidth;

            for (std::size_t i_rSmooth = 0; i_rSmooth < PARAMETER_ESTIMATE_SMOOTHING_SCALES.size(); ++i_rSmooth)
            {
                const double smoothingScale = PARAMETER_ESTIMATE_SMOOTHING_SCALES[i_rSmooth];

                normalizedSurveyEstimateRadialVelocities[i_rSmooth][i_growth] = constrainedRealizations.get_survey_estimate_radial_velocity(normalizedGrowthRate, smoothingScale) / normalizedGrowthRate;
            }
        }

        SphericalGridFunction fiducialRadialVelocity = constrainedRealizations.get_survey_estimate_radial_velocity(fiducialNormalizedGrowthRate, FIDUCIAL_FIELD_SMOOTHING_SCALE);

        const double fiducialOriginXVelocity = fiducialRadialVelocity(0.0, M_PI_2, 0.0);
        const double fiducialOriginYVelocity = fiducialRadialVelocity(0.0, M_PI_2, M_PI_2);
        const double fiducialOriginZVelocity = fiducialRadialVelocity(0.0, 0.0, 0.0);

        double localGroupXVelocity, localGroupYVelocity, localGroupZVelocity;

        transform_spherical_to_cartesian_coordinates(LOCAL_GROUP_TO_CMB.relativeVelocityAmplitude, LOCAL_GROUP_TO_CMB.relativeVelocityTheta, LOCAL_GROUP_TO_CMB.relativeVelocityPhi,
                                                     localGroupXVelocity, localGroupYVelocity, localGroupZVelocity);

        const double fiducialExternalBulkXVelocity = localGroupXVelocity - fiducialOriginXVelocity;
        const double fiducialExternalBulkYVelocity = localGroupYVelocity - fiducialOriginYVelocity;
        const double fiducialExternalBulkZVelocity = localGroupZVelocity - fiducialOriginZVelocity;

        double fiducialExternalBulkVelocityAmplitude, fiducialExternalBulkVelocityTheta, fiducialExternalBulkVelocityPhi;

        transform_cartesian_to_spherical_coordinates(fiducialExternalBulkXVelocity, fiducialExternalBulkYVelocity, fiducialExternalBulkZVelocity,
                                                     fiducialExternalBulkVelocityAmplitude, fiducialExternalBulkVelocityTheta, fiducialExternalBulkVelocityPhi);

        const ReferenceFrameChange fiducialExternalBulkVelocityShift = {fiducialExternalBulkVelocityAmplitude, fiducialExternalBulkVelocityTheta, fiducialExternalBulkVelocityPhi};

        fiducialRadialVelocity.apply([&](double velocity, double radius, double theta, double phi)
                                     { return change_reference_frame(velocity, theta, phi, fiducialExternalBulkVelocityShift); });

        const auto time2 = std::chrono::high_resolution_clock::now();
        std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(time2 - time1).count() << "s)" << std::endl;

        std::vector<double> averageNormalizedGrowthRateEstimates(cutNumber, 0.0);
        std::vector<double> averageNormalizedGrowthRateShotVariances(cutNumber, 0.0);
        std::vector<double> averageNormalizedGrowthRateDistVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkXVelocityEstimates(cutNumber, 0.0);
        std::vector<double> averageExternalBulkXVelocityShotVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkXVelocityDistVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkYVelocityEstimates(cutNumber, 0.0);
        std::vector<double> averageExternalBulkYVelocityShotVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkYVelocityDistVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkZVelocityEstimates(cutNumber, 0.0);
        std::vector<double> averageExternalBulkZVelocityShotVariances(cutNumber, 0.0);
        std::vector<double> averageExternalBulkZVelocityDistVariances(cutNumber, 0.0);
        std::vector<double> averageHubbleEstimates(cutNumber, 0.0);
        std::vector<double> averageHubbleShotVariances(cutNumber, 0.0);
        std::vector<double> averageHubbleDistVariances(cutNumber, 0.0);
        std::vector<double> averageReducedChiSquares(cutNumber, 0.0);

        for (std::size_t i_CR = INITIAL_CONSTRAINED_REALIZATION; i_CR <= FINAL_CONSTRAINED_REALIZATION; ++i_CR)
        {
            const auto timeCR1 = std::chrono::high_resolution_clock::now();
            std::cout << "Analyzing CR " << i_CR << "..." << std::flush;

            constrainedRealizations.generate(i_CR);

            const std::string realizationComment = "_CR" + std::to_string(i_CR);
            const std::string individualComment = redshiftReferenceFrameComment + realizationComment + CONFIGURATION_COMMENT;

            const std::string individualParameterFileName = DATA_DIRECTORY + "parameters" + individualComment + ".dat";

            std::ofstream individualParameterFile = std::ofstream(individualParameterFileName);

            individualParameterFile.setf(std::ios::fixed);

            individualParameterFile << "# "
                                    << "rSmooth[Mpc/h]\t"
                                    << "czMin[km/s]\t"
                                    << "fSig8\t"
                                    << "fSig8_e\t"
                                    << "BExtX[km/s]\t"
                                    << "BExtX_e[km/s]\t"
                                    << "BExtY[km/s]\t"
                                    << "BExtY_e[km/s]\t"
                                    << "BExtZ[km/s]\t"
                                    << "BExtZ_e[km/s]\t"
                                    << "h\t"
                                    << "h_e\t"
                                    << "chiSq/dof"
                                    << std::endl;

            for (std::size_t i_rSmooth = 0; i_rSmooth < PARAMETER_ESTIMATE_SMOOTHING_SCALES.size(); ++i_rSmooth)
            {
                const double smoothingScale = PARAMETER_ESTIMATE_SMOOTHING_SCALES[i_rSmooth];

                SphericalGridFunction normalizedRandomNoiseRadialVelocity = constrainedRealizations.get_noise_radial_velocity(1.0, smoothingScale);

                SphericalGridFunctionInterpolator normalizedSurveyEstimateRadialVelocityInterpolator(PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MIN, PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MAX, normalizedSurveyEstimateRadialVelocities[i_rSmooth]);

                for (std::size_t i_czMin = 0; i_czMin < PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES.size(); ++i_czMin)
                {
                    const double minRedshiftVelocity = PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES[i_czMin];

                    const std::size_t i_cut = i_rSmooth * PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES.size() + i_czMin;

                    std::vector<double> comparisonRedshiftVelocities, comparisonRadialCoordinates, comparisonThetaCoordinates, comparisonPhiCoordinates, comparisonDistanceModuli, comparisonDistanceModulusErrors;

                    prepare_distance_catalog_data(inputDistanceCatalogRedshiftVelocities, inputDistanceCatalogLatitudes, inputDistanceCatalogLongitudes, inputDistanceCatalogDistanceModuli, inputDistanceCatalogDistanceModulusErrors,
                                                  distanceCatalogInputToReferenceFrame, minRedshiftVelocity, DISTANCE_CATALOG_MAX_REDSHIFT_VELOCITY, FIDUCIAL_OMEGA_MATTER,
                                                  fiducialRadialVelocity, cmbToReferenceFrame, FIDUCIAL_DISTANCE_CATALOG_HUBBLE, DISTANCE_CATALOG_OUTLIER_MAX_SIGMA,
                                                  comparisonRedshiftVelocities, comparisonRadialCoordinates, comparisonThetaCoordinates, comparisonPhiCoordinates, comparisonDistanceModuli, comparisonDistanceModulusErrors);

                    double normalizedGrowthRateEstimate, externalBulkXVelocityEstimate, externalBulkYVelocityEstimate, externalBulkZVelocityEstimate, reducedChiSquare, hubbleEstimate;

                    gsl_matrix *covarianceMatrix = gsl_matrix_alloc(5, 5);

                    estimate_parameters_via_radial_velocity_comparison(normalizedSurveyEstimateRadialVelocityInterpolator, normalizedRandomNoiseRadialVelocity, cmbToReferenceFrame,
                                                                       comparisonRedshiftVelocities, comparisonThetaCoordinates, comparisonPhiCoordinates, comparisonDistanceModuli, comparisonDistanceModulusErrors,
                                                                       FIDUCIAL_OMEGA_MATTER, fiducialNormalizedGrowthRate, PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MIN, PARAMETER_ESTIMATE_NORMALIZED_GROWTH_RATE_MAX, FIDUCIAL_DISTANCE_CATALOG_HUBBLE,
                                                                       normalizedGrowthRateEstimate, externalBulkXVelocityEstimate, externalBulkYVelocityEstimate, externalBulkZVelocityEstimate, hubbleEstimate,
                                                                       covarianceMatrix, reducedChiSquare);

                    const double normalizedGrowthRateError = std::sqrt(gsl_matrix_get(covarianceMatrix, 0, 0));
                    const double externalBulkXVelocityError = std::sqrt(gsl_matrix_get(covarianceMatrix, 1, 1));
                    const double externalBulkYVelocityError = std::sqrt(gsl_matrix_get(covarianceMatrix, 2, 2));
                    const double externalBulkZVelocityError = std::sqrt(gsl_matrix_get(covarianceMatrix, 3, 3));
                    const double hubbleError = std::sqrt(gsl_matrix_get(covarianceMatrix, 4, 4));

                    gsl_matrix_free(covarianceMatrix);

                    averageNormalizedGrowthRateEstimates[i_cut] += normalizedGrowthRateEstimate;
                    averageNormalizedGrowthRateShotVariances[i_cut] += gsl_pow_2(normalizedGrowthRateEstimate);
                    averageNormalizedGrowthRateDistVariances[i_cut] += gsl_pow_2(normalizedGrowthRateError);
                    averageExternalBulkXVelocityEstimates[i_cut] += externalBulkXVelocityEstimate;
                    averageExternalBulkXVelocityShotVariances[i_cut] += gsl_pow_2(externalBulkXVelocityEstimate);
                    averageExternalBulkXVelocityDistVariances[i_cut] += gsl_pow_2(externalBulkXVelocityError);
                    averageExternalBulkYVelocityEstimates[i_cut] += externalBulkYVelocityEstimate;
                    averageExternalBulkYVelocityShotVariances[i_cut] += gsl_pow_2(externalBulkYVelocityEstimate);
                    averageExternalBulkYVelocityDistVariances[i_cut] += gsl_pow_2(externalBulkYVelocityError);
                    averageExternalBulkZVelocityEstimates[i_cut] += externalBulkZVelocityEstimate;
                    averageExternalBulkZVelocityShotVariances[i_cut] += gsl_pow_2(externalBulkZVelocityEstimate);
                    averageExternalBulkZVelocityDistVariances[i_cut] += gsl_pow_2(externalBulkZVelocityError);
                    averageHubbleEstimates[i_cut] += hubbleEstimate;
                    averageHubbleShotVariances[i_cut] += gsl_pow_2(hubbleEstimate);
                    averageHubbleDistVariances[i_cut] += gsl_pow_2(hubbleError);
                    averageReducedChiSquares[i_cut] += reducedChiSquare;

                    individualParameterFile << std::setprecision(1)
                                            << smoothingScale << "\t"
                                            << minRedshiftVelocity << "\t"
                                            << std::setprecision(5)
                                            << normalizedGrowthRateEstimate << "\t"
                                            << normalizedGrowthRateError << "\t"
                                            << std::setprecision(2)
                                            << externalBulkXVelocityEstimate << "\t"
                                            << externalBulkXVelocityError << "\t"
                                            << externalBulkYVelocityEstimate << "\t"
                                            << externalBulkYVelocityError << "\t"
                                            << externalBulkZVelocityEstimate << "\t"
                                            << externalBulkZVelocityError << "\t"
                                            << std::setprecision(5)
                                            << hubbleEstimate << "\t"
                                            << hubbleError << "\t"
                                            << std::setprecision(3)
                                            << reducedChiSquare
                                            << std::endl;
                }
            }

            individualParameterFile.close();

            const auto timeCR2 = std::chrono::high_resolution_clock::now();
            std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(timeCR2 - timeCR1).count() << "s)" << std::endl;
        }

        const std::string averageComment = redshiftReferenceFrameComment + realizationRangeComment + CONFIGURATION_COMMENT;

        const std::string averageParameterFileName = DATA_DIRECTORY + "parameters" + averageComment + ".dat";

        std::ofstream averageParameterFile = std::ofstream(averageParameterFileName);

        averageParameterFile << "# "
                             << "rSmooth[Mpc/h]\t"
                             << "czMin[km/s]\t"
                             << "fSig8\t"
                             << "fSig8_eShot\t"
                             << "fSig8_eDist\t"
                             << "BExtX[km/s]\t"
                             << "BExtX_eShot[km/s]\t"
                             << "BExtX_eDist[km/s]\t"
                             << "BExtY[km/s]\t"
                             << "BExtY_eShot[km/s]\t"
                             << "BExtY_eDist[km/s]\t"
                             << "BExtZ[km/s]\t"
                             << "BExtZ_eShot[km/s]\t"
                             << "BExtZ_eDist[km/s]\t"
                             << "h\t"
                             << "h_eShot\t"
                             << "h_eDist\t"
                             << "chiSq/dof"
                             << std::endl;

        averageParameterFile.setf(std::ios::fixed);

        for (std::size_t i_rSmooth = 0; i_rSmooth < PARAMETER_ESTIMATE_SMOOTHING_SCALES.size(); ++i_rSmooth)
        {
            const double smoothingScale = PARAMETER_ESTIMATE_SMOOTHING_SCALES[i_rSmooth];

            for (std::size_t i_czMin = 0; i_czMin < PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES.size(); ++i_czMin)
            {
                const double minRedshiftVelocity = PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES[i_czMin];

                const std::size_t i_cut = i_rSmooth * PARAMETER_ESTIMATE_MIN_REDSHIFT_VELOCITIES.size() + i_czMin;

                const double averageNormalizedGrowthRateEstimate = averageNormalizedGrowthRateEstimates[i_cut] / static_cast<double>(constrainedRealizationNumber);
                const double averageNormalizedGrowthRateShotError = std::sqrt(averageNormalizedGrowthRateShotVariances[i_cut] / static_cast<double>(constrainedRealizationNumber) - gsl_pow_2(averageNormalizedGrowthRateEstimate));
                const double averageNormalizedGrowthRateDistError = std::sqrt(averageNormalizedGrowthRateDistVariances[i_cut] / static_cast<double>(constrainedRealizationNumber));
                const double averageExternalBulkXVelocityEstimate = averageExternalBulkXVelocityEstimates[i_cut] / static_cast<double>(constrainedRealizationNumber);
                const double averageExternalBulkXVelocityShotError = std::sqrt(averageExternalBulkXVelocityShotVariances[i_cut] / static_cast<double>(constrainedRealizationNumber) - gsl_pow_2(averageExternalBulkXVelocityEstimate));
                const double averageExternalBulkXVelocityDistError = std::sqrt(averageExternalBulkXVelocityDistVariances[i_cut] / static_cast<double>(constrainedRealizationNumber));
                const double averageExternalBulkYVelocityEstimate = averageExternalBulkYVelocityEstimates[i_cut] / static_cast<double>(constrainedRealizationNumber);
                const double averageExternalBulkYVelocityShotError = std::sqrt(averageExternalBulkYVelocityShotVariances[i_cut] / static_cast<double>(constrainedRealizationNumber) - gsl_pow_2(averageExternalBulkYVelocityEstimate));
                const double averageExternalBulkYVelocityDistError = std::sqrt(averageExternalBulkYVelocityDistVariances[i_cut] / static_cast<double>(constrainedRealizationNumber));
                const double averageExternalBulkZVelocityEstimate = averageExternalBulkZVelocityEstimates[i_cut] / static_cast<double>(constrainedRealizationNumber);
                const double averageExternalBulkZVelocityShotError = std::sqrt(averageExternalBulkZVelocityShotVariances[i_cut] / static_cast<double>(constrainedRealizationNumber) - gsl_pow_2(averageExternalBulkZVelocityEstimate));
                const double averageExternalBulkZVelocityDistError = std::sqrt(averageExternalBulkZVelocityDistVariances[i_cut] / static_cast<double>(constrainedRealizationNumber));
                const double averageHubbleEstimate = averageHubbleEstimates[i_cut] / static_cast<double>(constrainedRealizationNumber);
                const double averageHubbleShotError = std::sqrt(averageHubbleShotVariances[i_cut] / static_cast<double>(constrainedRealizationNumber) - gsl_pow_2(averageHubbleEstimate));
                const double averageHubbleDistError = std::sqrt(averageHubbleDistVariances[i_cut] / static_cast<double>(constrainedRealizationNumber));
                const double averageReducedChiSquare = averageReducedChiSquares[i_cut] / static_cast<double>(constrainedRealizationNumber);

                averageParameterFile << std::setprecision(1)
                                     << smoothingScale << "\t"
                                     << minRedshiftVelocity << "\t"
                                     << std::setprecision(5)
                                     << averageNormalizedGrowthRateEstimate << "\t"
                                     << averageNormalizedGrowthRateShotError << "\t"
                                     << averageNormalizedGrowthRateDistError << "\t"
                                     << std::setprecision(2)
                                     << averageExternalBulkXVelocityEstimate << "\t"
                                     << averageExternalBulkXVelocityShotError << "\t"
                                     << averageExternalBulkXVelocityDistError << "\t"
                                     << averageExternalBulkYVelocityEstimate << "\t"
                                     << averageExternalBulkYVelocityShotError << "\t"
                                     << averageExternalBulkYVelocityDistError << "\t"
                                     << averageExternalBulkZVelocityEstimate << "\t"
                                     << averageExternalBulkZVelocityShotError << "\t"
                                     << averageExternalBulkZVelocityDistError << "\t"
                                     << std::setprecision(5)
                                     << averageHubbleEstimate << "\t"
                                     << averageHubbleShotError << "\t"
                                     << averageHubbleDistError << "\t"
                                     << std::setprecision(3)
                                     << averageReducedChiSquare
                                     << std::endl;
            }
        }

        averageParameterFile.close();
    }

    fftw_cleanup_threads();

    return 0;
}