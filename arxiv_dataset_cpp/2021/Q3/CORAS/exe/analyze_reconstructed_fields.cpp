#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <omp.h>

#include "../src/Cartesian1DGridFunction.hpp"
#include "../src/Cartesian2DGridFunction.hpp"
#include "../src/ConstrainedRealizations.hpp"
#include "../src/FileTable.hpp"
#include "../src/NormalizedPowerSpectrum.hpp"
#include "../src/SphericalGridFunction.hpp"
#include "../src/survey.hpp"
#include "../src/transformations.hpp"

#include "../configuration.hpp"

double gaussian_volume_average(const SphericalGridFunction &field, double effectiveRadius)
{
    if (effectiveRadius == 0.0)
    {
        return field(0.0, 0.0, 0.0);
    }
    else
    {
        return field.average([&](double value, double radius, double theta, double phi)
                             { return value * std::pow(2.0 * M_PI * gsl_pow_2(effectiveRadius), -3.0 / 2.0) * std::exp(-gsl_pow_2(radius / effectiveRadius) / 2.0) * 4.0 / 3.0 * M_PI * gsl_pow_3(field.maximal_radius()); });
    }
}

int main(int argc, char **argv)
{
    fftw_init_threads(); // initialize parallel execution of FFTs used in ConstrainedRealizations::generate
    fftw_plan_with_nthreads(omp_get_max_threads());

    const FileTable inputPowerSpectrumFileTable(FIDUCIAL_POWER_SPECTRUM_FILE_NAME, 2); // read the power spectrum from file

    const auto inputWavenumbers = inputPowerSpectrumFileTable.column<double>(0);
    const auto inputPowerSpectrumValues = inputPowerSpectrumFileTable.column<double>(1);

    NormalizedPowerSpectrum normalizedPowerSpectrum(inputWavenumbers, inputPowerSpectrumValues,
                                                    FIDUCIAL_HUBBLE, SIGMA_SCALE);

    const FileTable inputRedshiftCatalogFile(REDSHIFT_CATALOG_FILE_NAME, 10, '\t'); // read the redshift catalog properties from file

    const auto inputRedshiftCatalogLatitudes = inputRedshiftCatalogFile.column<double>(1);
    const auto inputRedshiftCatalogLongitudes = inputRedshiftCatalogFile.column<double>(2);
    const auto inputRedshiftCatalogRedshiftVelocities = REDSHIFT_CATALOG_COLLAPSE_FINGERS_OF_GOD ? inputRedshiftCatalogFile.column<double>(9)
                                                                                                 : inputRedshiftCatalogFile.column<double>(3);
    const auto inputRedshiftCatalogKCorrectionRedshiftVelocities = inputRedshiftCatalogFile.column<double>(3);
    const auto inputRedshiftCatalogApparentMagnitudes = inputRedshiftCatalogFile.column<double>(4);

    for (std::size_t i_ref = 0; i_ref < RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES.size(); ++i_ref)
    {
        const ReferenceFrame redshiftReferenceFrame = RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES[i_ref];

        const ReferenceFrameChange redshiftCatalogInputToReferenceFrame = get_reference_frame_change(REDSHIFT_CATALOG_REDSHIFT_REFERENCE_FRAME, redshiftReferenceFrame);

        const std::string redshiftReferenceFrameName = get_reference_frame_name(redshiftReferenceFrame);
        const std::string redshiftReferenceFrameComment = "_z" + redshiftReferenceFrameName;
        const std::string realizationRangeComment = "_CR" + std::to_string(INITIAL_CONSTRAINED_REALIZATION) + "-" + std::to_string(FINAL_CONSTRAINED_REALIZATION);
        const std::string reconstructionComment = redshiftReferenceFrameComment + CONFIGURATION_COMMENT;
        const std::string averageComment = redshiftReferenceFrameComment + realizationRangeComment + CONFIGURATION_COMMENT;

        const std::string sigmaGalaxyFileName = DATA_DIRECTORY + "sigma_galaxy" + reconstructionComment + ".dat";
        const std::string superGalacticPlaneReconstructionFileName = DATA_DIRECTORY + "sgp_reconstruction" + reconstructionComment + ".dat";
        const std::string angularAveragesFileName = DATA_DIRECTORY + "angular_averages" + averageComment + ".dat";
        const std::string volumeAveragesTophatFileName = DATA_DIRECTORY + "volume_averages_tophat" + averageComment + ".dat";
        const std::string volumeAveragesGaussianFileName = DATA_DIRECTORY + "volume_averages_gaussian" + averageComment + ".dat";
        const std::string normalizedDensityContrastReconstructionErrorFileName = DATA_DIRECTORY + "normalized_density_contrast_reconstruction_error" + averageComment + ".bin";
        const std::string velocityComponentReconstructionErrorFileName = DATA_DIRECTORY + "velocity_component_reconstruction_error" + averageComment + ".bin";

        const auto time1 = std::chrono::high_resolution_clock::now();
        std::cout << "Preparing " << redshiftReferenceFrameName << " reference frame..." << std::flush;

        std::vector<double> reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionRadialCoordinates, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentKsMagnitudes;

        prepare_2MRS_data(inputRedshiftCatalogRedshiftVelocities, inputRedshiftCatalogKCorrectionRedshiftVelocities, inputRedshiftCatalogLatitudes, inputRedshiftCatalogLongitudes, inputRedshiftCatalogApparentMagnitudes,
                          redshiftCatalogInputToReferenceFrame, RECONSTRUCTION_MAX_RADIUS, REDSHIFT_CATALOG_VOLUME_LIMIT_RADIUS, REDSHIFT_CATALOG_MAX_APPARENT_MAGNITUDE, FIDUCIAL_OMEGA_MATTER,
                          luminosity_evolution_correction_2MRS, k_correction_2MRS,
                          reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionRadialCoordinates, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentKsMagnitudes);

        Cartesian1DGridFunction sigmaGalaxyAboveVolumeLimit;
        std::vector<std::size_t> volumeLimitGalaxyNumbers;

        estimate_sigma_galaxy(reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionThetaCoordinates, reconstructionPhiCoordinates, reconstructionApparentKsMagnitudes,
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

        std::ofstream sigmaGalaxyFile(sigmaGalaxyFileName);

        sigmaGalaxyFile.setf(std::ios::fixed);

        sigmaGalaxyFile << "# "
                        << "radius[Mpc/h]\t"
                        << "sig_gal\t"
                        << "N_gal_volLim"
                        << std::endl;

        for (std::size_t i_r = 0; i_r < sigmaGalaxyAboveVolumeLimit.bin_number(); ++i_r)
        {
            sigmaGalaxyFile << std::setprecision(2)
                            << sigmaGalaxyAboveVolumeLimit.coordinate(i_r) << "\t"
                            << std::setprecision(3)
                            << sigmaGalaxyAboveVolumeLimit.value(i_r) << "\t"
                            << volumeLimitGalaxyNumbers[i_r]
                            << std::endl;
        }

        sigmaGalaxyFile.close();

        Cartesian1DGridFunction selectionFunction, selectionFunctionLogDerivative, radialDistributionFunction;

        estimate_selection_function(reconstructionRedshiftVelocities, kCorrectionRedshiftVelocities, reconstructionApparentKsMagnitudes, REDSHIFT_CATALOG_MAX_APPARENT_MAGNITUDE,
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

        SphericalGridFunction reconstructedNormalizedDensityContrast = constrainedRealizations.get_survey_estimate_density_contrast(1.0, ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE); // these fields will be used for the SGP and angular average output
        SphericalGridFunction reconstructedRadialVelocity = constrainedRealizations.get_survey_estimate_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedThetaVelocity = constrainedRealizations.get_survey_estimate_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedPhiVelocity = constrainedRealizations.get_survey_estimate_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);

        SphericalGridFunction reconstructedNormalizedDensityContrastMinSmooth = constrainedRealizations.get_survey_estimate_density_contrast(1.0, ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE); // these fields will be used for the volume average output, allowing to resolve smaller radii
        SphericalGridFunction reconstructedRadialVelocityMinSmooth = constrainedRealizations.get_survey_estimate_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedThetaVelocityMinSmooth = constrainedRealizations.get_survey_estimate_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedPhiVelocityMinSmooth = constrainedRealizations.get_survey_estimate_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);

        SphericalGridFunction reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity,
            reconstructedXVelocityMinSmooth, reconstructedYVelocityMinSmooth, reconstructedZVelocityMinSmooth;

        transform_spherical_to_cartesian_vector_field(reconstructedRadialVelocity, reconstructedThetaVelocity, reconstructedPhiVelocity,
                                                      reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity);

        transform_spherical_to_cartesian_vector_field(reconstructedRadialVelocityMinSmooth, reconstructedThetaVelocityMinSmooth, reconstructedPhiVelocityMinSmooth,
                                                      reconstructedXVelocityMinSmooth, reconstructedYVelocityMinSmooth, reconstructedZVelocityMinSmooth);

        reconstructedXVelocity += ESTIMATED_EXTERNAL_BULK_X_VELOCITY;
        reconstructedYVelocity += ESTIMATED_EXTERNAL_BULK_Y_VELOCITY;
        reconstructedZVelocity += ESTIMATED_EXTERNAL_BULK_Z_VELOCITY;

        reconstructedXVelocityMinSmooth += ESTIMATED_EXTERNAL_BULK_X_VELOCITY;
        reconstructedYVelocityMinSmooth += ESTIMATED_EXTERNAL_BULK_Y_VELOCITY;
        reconstructedZVelocityMinSmooth += ESTIMATED_EXTERNAL_BULK_Z_VELOCITY;

        transform_cartesian_to_spherical_vector_field(reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity,
                                                      reconstructedRadialVelocity, reconstructedThetaVelocity, reconstructedPhiVelocity);

        transform_cartesian_to_spherical_vector_field(reconstructedXVelocityMinSmooth, reconstructedYVelocityMinSmooth, reconstructedZVelocityMinSmooth,
                                                      reconstructedRadialVelocityMinSmooth, reconstructedThetaVelocityMinSmooth, reconstructedPhiVelocityMinSmooth);

        SphericalGridFunction reconstructedNormalizedDensityContrastMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, reconstructedNormalizedDensityContrastMinSmooth); // these fields sample the minimal-smoothing fields at a higher resolution, to achieve a more accurate Gaussian volume average at small radii
        SphericalGridFunction reconstructedXVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, reconstructedXVelocityMinSmooth);
        SphericalGridFunction reconstructedYVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, reconstructedYVelocityMinSmooth);
        SphericalGridFunction reconstructedZVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, reconstructedZVelocityMinSmooth);

        const Cartesian1DGridFunction reconstructedNormalizedDensityContrastGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                                                   { return gaussian_volume_average(reconstructedNormalizedDensityContrastMinSmoothHighRes, effectiveRadius); });

        const Cartesian1DGridFunction reconstructedXVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                                   { return gaussian_volume_average(reconstructedXVelocityMinSmoothHighRes, effectiveRadius); });

        const Cartesian1DGridFunction reconstructedYVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                                   { return gaussian_volume_average(reconstructedYVelocityMinSmoothHighRes, effectiveRadius); });

        const Cartesian1DGridFunction reconstructedZVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                                   { return gaussian_volume_average(reconstructedZVelocityMinSmoothHighRes, effectiveRadius); });

        const Cartesian1DGridFunction reconstructedNormalizedDensityContrastAngularAverages = reconstructedNormalizedDensityContrast.angular_average();
        const Cartesian1DGridFunction reconstructedXVelocityAngularAverages = reconstructedXVelocity.angular_average();
        const Cartesian1DGridFunction reconstructedYVelocityAngularAverages = reconstructedYVelocity.angular_average();
        const Cartesian1DGridFunction reconstructedZVelocityAngularAverages = reconstructedZVelocity.angular_average();

        const Cartesian1DGridFunction reconstructedNormalizedDensityContrastTophatVolumeAverages = reconstructedNormalizedDensityContrastMinSmooth.averages();
        const Cartesian1DGridFunction reconstructedXVelocityTophatVolumeAverages = reconstructedXVelocityMinSmooth.averages();
        const Cartesian1DGridFunction reconstructedYVelocityTophatVolumeAverages = reconstructedYVelocityMinSmooth.averages();
        const Cartesian1DGridFunction reconstructedZVelocityTophatVolumeAverages = reconstructedZVelocityMinSmooth.averages();

        SphericalGridFunction reconstructedNormalizedDensityContrastSuperGalactic, reconstructedXVelocitySuperGalactic, reconstructedYVelocitySuperGalactic, reconstructedZVelocitySuperGalactic;

        rotate_scalar_field(reconstructedNormalizedDensityContrast,
                            reconstructedNormalizedDensityContrastSuperGalactic,
                            GalacticToSupergalactic);

        rotate_cartesian_vector_field(reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity,
                                      reconstructedXVelocitySuperGalactic, reconstructedYVelocitySuperGalactic, reconstructedZVelocitySuperGalactic,
                                      GalacticToSupergalactic);

        Cartesian2DGridFunction superGalacticPlaneSlice(-RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, SGP_BIN_NUMBER,
                                                        -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, SGP_BIN_NUMBER);

        std::ofstream superGalacticPlaneReconstructionFile(superGalacticPlaneReconstructionFileName);

        superGalacticPlaneReconstructionFile.setf(std::ios::fixed);

        superGalacticPlaneReconstructionFile << "# "
                                             << "SGX[Mpc/h]\t"
                                             << "SGY[Mpc/h]\t"
                                             << "deltaNorm\t"
                                             << "vX[km/s]\t"
                                             << "vY[km/s]"
                                             << std::endl;

        superGalacticPlaneSlice.evaluate_for_all_grid_points([&](std::size_t i_x, std::size_t i_y)
                                                             {
                                                                 const double sgx = superGalacticPlaneSlice.x_coordinate(i_x);
                                                                 const double sgy = superGalacticPlaneSlice.x_coordinate(i_y);

                                                                 double radius, theta, phi;

                                                                 transform_cartesian_to_spherical_coordinates(sgx, sgy, 0.0,
                                                                                                              radius, theta, phi);

                                                                 const double normalizedDensityContrast = (radius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedNormalizedDensityContrastSuperGalactic(radius, theta, phi) : 0.0;
                                                                 const double xVelocity = (radius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedXVelocitySuperGalactic(radius, theta, phi) : 0.0;
                                                                 const double yVelocity = (radius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedYVelocitySuperGalactic(radius, theta, phi) : 0.0;

                                                                 superGalacticPlaneReconstructionFile << std::setprecision(2)
                                                                                                      << sgx << "\t"
                                                                                                      << sgy << "\t"
                                                                                                      << std::setprecision(4)
                                                                                                      << normalizedDensityContrast << "\t"
                                                                                                      << std::setprecision(2)
                                                                                                      << xVelocity << "\t"
                                                                                                      << yVelocity
                                                                                                      << std::endl;
                                                             });

        superGalacticPlaneReconstructionFile.close();

        SphericalGridFunction noiseNormalizedDensityContrastVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);
        SphericalGridFunction noiseXVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);
        SphericalGridFunction noiseYVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);
        SphericalGridFunction noiseZVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);

        Cartesian1DGridFunction noiseNormalizedDensityContrastTophatVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseXVelocityTophatVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseYVelocityTophatVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseZVelocityTophatVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);

        Cartesian1DGridFunction noiseNormalizedDensityContrastGaussianVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseXVelocityGaussianVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseYVelocityGaussianVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);
        Cartesian1DGridFunction noiseZVelocityGaussianVolumeAverageVariances(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, 0.0);

        const auto time2 = std::chrono::high_resolution_clock::now();
        std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(time2 - time1).count() << "s)" << std::endl;

        for (std::size_t i_CR = INITIAL_CONSTRAINED_REALIZATION; i_CR <= FINAL_CONSTRAINED_REALIZATION; ++i_CR)
        {
            const auto timeCR1 = std::chrono::high_resolution_clock::now();
            std::cout << "Analyzing CR " << i_CR << "..." << std::flush;

            constrainedRealizations.generate(i_CR);

            SphericalGridFunction noiseNormalizedDensityContrast = constrainedRealizations.get_noise_density_contrast(1.0, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
            SphericalGridFunction noiseRadialVelocity = constrainedRealizations.get_noise_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
            SphericalGridFunction noiseThetaVelocity = constrainedRealizations.get_noise_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
            SphericalGridFunction noisePhiVelocity = constrainedRealizations.get_noise_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);

            SphericalGridFunction noiseNormalizedDensityContrastMinSmooth = constrainedRealizations.get_noise_density_contrast(1.0, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);
            SphericalGridFunction noiseRadialVelocityMinSmooth = constrainedRealizations.get_noise_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);
            SphericalGridFunction noiseThetaVelocityMinSmooth = constrainedRealizations.get_noise_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);
            SphericalGridFunction noisePhiVelocityMinSmooth = constrainedRealizations.get_noise_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_MIN_SMOOTHING_SCALE);

            SphericalGridFunction noiseXVelocity, noiseYVelocity, noiseZVelocity,
                noiseXVelocityMinSmooth, noiseYVelocityMinSmooth, noiseZVelocityMinSmooth;

            transform_spherical_to_cartesian_vector_field(noiseRadialVelocity, noiseThetaVelocity, noisePhiVelocity,
                                                          noiseXVelocity, noiseYVelocity, noiseZVelocity);

            transform_spherical_to_cartesian_vector_field(noiseRadialVelocityMinSmooth, noiseThetaVelocityMinSmooth, noisePhiVelocityMinSmooth,
                                                          noiseXVelocityMinSmooth, noiseYVelocityMinSmooth, noiseZVelocityMinSmooth);

            SphericalGridFunction noiseNormalizedDensityContrastMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, noiseNormalizedDensityContrastMinSmooth);
            SphericalGridFunction noiseXVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, noiseXVelocityMinSmooth);
            SphericalGridFunction noiseYVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, noiseYVelocityMinSmooth);
            SphericalGridFunction noiseZVelocityMinSmoothHighRes(RECONSTRUCTION_MAX_RADIUS, GAUSSIAN_VOLUME_AVERAGE_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, noiseZVelocityMinSmooth);

            if (std::find(SGP_NOISE_REALIZATIONS.begin(), SGP_NOISE_REALIZATIONS.end(), i_CR) != SGP_NOISE_REALIZATIONS.end())
            {
                const std::string realizationComment = "_CR" + std::to_string(i_CR);
                const std::string individualComment = redshiftReferenceFrameComment + realizationComment + CONFIGURATION_COMMENT;

                const std::string superGalacticPlaneNoiseFileName = DATA_DIRECTORY + "sgp_noise" + individualComment + ".dat";

                SphericalGridFunction noiseNormalizedDensityContrastSuperGalactic, noiseXVelocitySuperGalactic, noiseYVelocitySuperGalactic, noiseZVelocitySuperGalactic;

                rotate_scalar_field(noiseNormalizedDensityContrast,
                                    noiseNormalizedDensityContrastSuperGalactic,
                                    GalacticToSupergalactic);

                rotate_cartesian_vector_field(noiseXVelocity, noiseYVelocity, noiseZVelocity,
                                              noiseXVelocitySuperGalactic, noiseYVelocitySuperGalactic, noiseZVelocitySuperGalactic,
                                              GalacticToSupergalactic);

                std::ofstream superGalacticPlaneNoiseFile(superGalacticPlaneNoiseFileName);

                superGalacticPlaneNoiseFile.setf(std::ios::fixed);

                superGalacticPlaneNoiseFile << "# "
                                            << "SGX[Mpc/h]\t"
                                            << "SGY[Mpc/h]\t"
                                            << "deltaNorm\t"
                                            << "vX[km/s]\t"
                                            << "vY[km/s]"
                                            << std::endl;

                superGalacticPlaneSlice.evaluate_for_all_grid_points([&](std::size_t i_x, std::size_t i_y)
                                                                     {
                                                                         const double sgx = superGalacticPlaneSlice.x_coordinate(i_x);
                                                                         const double sgy = superGalacticPlaneSlice.x_coordinate(i_y);

                                                                         double radius, theta, phi;

                                                                         transform_cartesian_to_spherical_coordinates(sgx, sgy, 0.0,
                                                                                                                      radius, theta, phi);

                                                                         const double normalizedDensityContrast = (radius <= RECONSTRUCTION_MAX_RADIUS) ? noiseNormalizedDensityContrastSuperGalactic(radius, theta, phi) : 0.0;
                                                                         const double xVelocity = (radius <= RECONSTRUCTION_MAX_RADIUS) ? noiseXVelocitySuperGalactic(radius, theta, phi) : 0.0;
                                                                         const double yVelocity = (radius <= RECONSTRUCTION_MAX_RADIUS) ? noiseYVelocitySuperGalactic(radius, theta, phi) : 0.0;

                                                                         superGalacticPlaneNoiseFile << std::setprecision(2)
                                                                                                     << sgx << "\t"
                                                                                                     << sgy << "\t"
                                                                                                     << std::setprecision(4)
                                                                                                     << normalizedDensityContrast << "\t"
                                                                                                     << std::setprecision(2)
                                                                                                     << xVelocity << "\t"
                                                                                                     << yVelocity
                                                                                                     << std::endl;
                                                                     });

                superGalacticPlaneNoiseFile.close();
            }

            noiseNormalizedDensityContrastVariance += noiseNormalizedDensityContrast * noiseNormalizedDensityContrast;
            noiseXVelocityVariance += noiseXVelocity * noiseXVelocity;
            noiseYVelocityVariance += noiseYVelocity * noiseYVelocity;
            noiseZVelocityVariance += noiseZVelocity * noiseZVelocity;

            const Cartesian1DGridFunction noiseNormalizedDensityContrastTophatVolumeAverage = noiseNormalizedDensityContrastMinSmooth.averages();
            const Cartesian1DGridFunction noiseXVelocityTophatVolumeAverage = noiseXVelocityMinSmooth.averages();
            const Cartesian1DGridFunction noiseYVelocityTophatVolumeAverage = noiseYVelocityMinSmooth.averages();
            const Cartesian1DGridFunction noiseZVelocityTophatVolumeAverage = noiseZVelocityMinSmooth.averages();

            noiseNormalizedDensityContrastTophatVolumeAverageVariances += noiseNormalizedDensityContrastTophatVolumeAverage * noiseNormalizedDensityContrastTophatVolumeAverage;
            noiseXVelocityTophatVolumeAverageVariances += noiseXVelocityTophatVolumeAverage * noiseXVelocityTophatVolumeAverage;
            noiseYVelocityTophatVolumeAverageVariances += noiseYVelocityTophatVolumeAverage * noiseYVelocityTophatVolumeAverage;
            noiseZVelocityTophatVolumeAverageVariances += noiseZVelocityTophatVolumeAverage * noiseZVelocityTophatVolumeAverage;

            const Cartesian1DGridFunction noiseNormalizedDensityContrastGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                                               { return gaussian_volume_average(noiseNormalizedDensityContrastMinSmoothHighRes, effectiveRadius); });

            const Cartesian1DGridFunction noiseXVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                               { return gaussian_volume_average(noiseXVelocityMinSmoothHighRes, effectiveRadius); });

            const Cartesian1DGridFunction noiseYVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                               { return gaussian_volume_average(noiseYVelocityMinSmoothHighRes, effectiveRadius); });

            const Cartesian1DGridFunction noiseZVelocityGaussianVolumeAverages(0.0, RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, [&](double effectiveRadius)
                                                                               { return gaussian_volume_average(noiseZVelocityMinSmoothHighRes, effectiveRadius); });

            noiseNormalizedDensityContrastGaussianVolumeAverageVariances += noiseNormalizedDensityContrastGaussianVolumeAverages * noiseNormalizedDensityContrastGaussianVolumeAverages;
            noiseXVelocityGaussianVolumeAverageVariances += noiseXVelocityGaussianVolumeAverages * noiseXVelocityGaussianVolumeAverages;
            noiseYVelocityGaussianVolumeAverageVariances += noiseYVelocityGaussianVolumeAverages * noiseYVelocityGaussianVolumeAverages;
            noiseZVelocityGaussianVolumeAverageVariances += noiseZVelocityGaussianVolumeAverages * noiseZVelocityGaussianVolumeAverages;

            const auto timeCR2 = std::chrono::high_resolution_clock::now();
            std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(timeCR2 - timeCR1).count() << "s)" << std::endl;
        }

        const double realizationNumber = static_cast<double>(FINAL_CONSTRAINED_REALIZATION - INITIAL_CONSTRAINED_REALIZATION + 1);

        const Cartesian1DGridFunction noiseNormalizedDensityContrastAngularAverageVariances = noiseNormalizedDensityContrastVariance.angular_average() / realizationNumber;
        const Cartesian1DGridFunction noiseXVelocityAngularAverageVariances = noiseXVelocityVariance.angular_average() / realizationNumber;
        const Cartesian1DGridFunction noiseYVelocityAngularAverageVariances = noiseYVelocityVariance.angular_average() / realizationNumber;
        const Cartesian1DGridFunction noiseZVelocityAngularAverageVariances = noiseZVelocityVariance.angular_average() / realizationNumber;

        const Cartesian1DGridFunction normalizedDensityContrastReconstructionError(noiseNormalizedDensityContrastAngularAverageVariances, [](double variance)
                                                                                   { return std::sqrt(variance); });
        const Cartesian1DGridFunction velocityComponentReconstructionError((noiseXVelocityAngularAverageVariances + noiseYVelocityAngularAverageVariances + noiseZVelocityAngularAverageVariances) / 3.0, [](double variance)
                                                                           { return std::sqrt(variance); });

        normalizedDensityContrastReconstructionError.save_object_to_file(normalizedDensityContrastReconstructionErrorFileName);
        velocityComponentReconstructionError.save_object_to_file(velocityComponentReconstructionErrorFileName);

        noiseNormalizedDensityContrastTophatVolumeAverageVariances /= realizationNumber;
        noiseXVelocityTophatVolumeAverageVariances /= realizationNumber;
        noiseYVelocityTophatVolumeAverageVariances /= realizationNumber;
        noiseZVelocityTophatVolumeAverageVariances /= realizationNumber;

        noiseNormalizedDensityContrastGaussianVolumeAverageVariances /= realizationNumber;
        noiseXVelocityGaussianVolumeAverageVariances /= realizationNumber;
        noiseYVelocityGaussianVolumeAverageVariances /= realizationNumber;
        noiseZVelocityGaussianVolumeAverageVariances /= realizationNumber;

        std::ofstream angularAveragesFile(angularAveragesFileName);
        std::ofstream volumeAveragesTophatFile(volumeAveragesTophatFileName);
        std::ofstream volumeAveragesGaussianFile(volumeAveragesGaussianFileName);

        angularAveragesFile.setf(std::ios::fixed);
        volumeAveragesTophatFile.setf(std::ios::fixed);
        volumeAveragesGaussianFile.setf(std::ios::fixed);

        angularAveragesFile << "# "
                            << "radius[Mpc/h]\t"
                            << "deltaNorm\t"
                            << "deltaNorm_e\t"
                            << "vX[km/s]\t"
                            << "vX_e[km/s]\t"
                            << "vY[km/s]\t"
                            << "vY_e[km/s]\t"
                            << "vZ[km/s]\t"
                            << "vZ_e[km/s]"
                            << std::endl;

        volumeAveragesTophatFile << "# "
                                 << "radius[Mpc/h]\t"
                                 << "deltaNorm\t"
                                 << "deltaNorm_e\t"
                                 << "vX[km/s]\t"
                                 << "vX_e[km/s]\t"
                                 << "vY[km/s]\t"
                                 << "vY_e[km/s]\t"
                                 << "vZ[km/s]\t"
                                 << "vZ_e[km/s]"
                                 << std::endl;

        volumeAveragesGaussianFile << "# "
                                   << "radius[Mpc/h]\t"
                                   << "deltaNorm\t"
                                   << "deltaNorm_e\t"
                                   << "vX[km/s]\t"
                                   << "vX_e[km/s]\t"
                                   << "vY[km/s]\t"
                                   << "vY_e[km/s]\t"
                                   << "vZ[km/s]\t"
                                   << "vZ_e[km/s]"
                                   << std::endl;

        for (std::size_t i_r = 0; i_r < RECONSTRUCTION_RADIAL_BIN_NUMBER; ++i_r)
        {
            const double radius = reconstructedNormalizedDensityContrastAngularAverages.coordinate(i_r);

            angularAveragesFile << std::setprecision(2)
                                << radius << "\t"
                                << std::setprecision(4)
                                << reconstructedNormalizedDensityContrastAngularAverages.value(i_r) << "\t"
                                << std::sqrt(noiseNormalizedDensityContrastAngularAverageVariances.value(i_r)) << "\t"
                                << std::setprecision(2)
                                << reconstructedXVelocityAngularAverages.value(i_r) << "\t"
                                << std::sqrt(noiseXVelocityAngularAverageVariances.value(i_r)) << "\t"
                                << reconstructedYVelocityAngularAverages.value(i_r) << "\t"
                                << std::sqrt(noiseYVelocityAngularAverageVariances.value(i_r)) << "\t"
                                << reconstructedZVelocityAngularAverages.value(i_r) << "\t"
                                << std::sqrt(noiseZVelocityAngularAverageVariances.value(i_r))
                                << std::endl;

            volumeAveragesTophatFile << std::setprecision(2)
                                     << radius << "\t"
                                     << std::setprecision(4)
                                     << reconstructedNormalizedDensityContrastTophatVolumeAverages.value(i_r) << "\t"
                                     << std::sqrt(noiseNormalizedDensityContrastTophatVolumeAverageVariances.value(i_r)) << "\t"
                                     << std::setprecision(2)
                                     << reconstructedXVelocityTophatVolumeAverages.value(i_r) << "\t"
                                     << std::sqrt(noiseXVelocityTophatVolumeAverageVariances.value(i_r)) << "\t"
                                     << reconstructedYVelocityTophatVolumeAverages.value(i_r) << "\t"
                                     << std::sqrt(noiseYVelocityTophatVolumeAverageVariances.value(i_r)) << "\t"
                                     << reconstructedZVelocityTophatVolumeAverages.value(i_r) << "\t"
                                     << std::sqrt(noiseZVelocityTophatVolumeAverageVariances.value(i_r))
                                     << std::endl;

            volumeAveragesGaussianFile << std::setprecision(2)
                                       << radius << "\t"
                                       << std::setprecision(4)
                                       << reconstructedNormalizedDensityContrastGaussianVolumeAverages.value(i_r) << "\t"
                                       << std::sqrt(noiseNormalizedDensityContrastGaussianVolumeAverageVariances.value(i_r)) << "\t"
                                       << std::setprecision(2)
                                       << reconstructedXVelocityGaussianVolumeAverages.value(i_r) << "\t"
                                       << std::sqrt(noiseXVelocityGaussianVolumeAverageVariances.value(i_r)) << "\t"
                                       << reconstructedYVelocityGaussianVolumeAverages.value(i_r) << "\t"
                                       << std::sqrt(noiseYVelocityGaussianVolumeAverageVariances.value(i_r)) << "\t"
                                       << reconstructedZVelocityGaussianVolumeAverages.value(i_r) << "\t"
                                       << std::sqrt(noiseZVelocityGaussianVolumeAverageVariances.value(i_r))
                                       << std::endl;
        }

        angularAveragesFile.close();
        volumeAveragesTophatFile.close();
        volumeAveragesGaussianFile.close();
    }

    fftw_cleanup_threads();

    return 0;
}