#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <fftw3.h>
#include <omp.h>

#include "../src/Cartesian1DGridFunction.hpp"
#include "../src/ConstrainedRealizations.hpp"
#include "../src/FileTable.hpp"
#include "../src/miscellaneous.hpp"
#include "../src/NormalizedPowerSpectrum.hpp"
#include "../src/SphericalGridFunction.hpp"
#include "../src/survey.hpp"
#include "../src/transformations.hpp"

#include "../configuration.hpp"

int main(int argc, char **argv)
{
    fftw_init_threads(); // initialize parallel execution of the FFTs used to ConstrainedRealizations::generate
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

    const FileTable inputDistanceCatalogFile(DISTANCE_CATALOG_GALAXIES_FILE_NAME, 13, '\t'); // read the CF3 group member properties from file

    const auto inputDistanceCatalogGalaxyPGCNumbers = inputDistanceCatalogFile.column<std::size_t>(0);
    const auto inputDistanceCatalogGalaxyLatitudes = inputDistanceCatalogFile.column<double>(1);
    const auto inputDistanceCatalogGalaxyLongitudes = inputDistanceCatalogFile.column<double>(2);
    const auto inputDistanceCatalogGroupLatitudes = inputDistanceCatalogFile.column<double>(8);
    const auto inputDistanceCatalogGroupLongitudes = inputDistanceCatalogFile.column<double>(9);
    const auto inputDistanceCatalogGroupRedshiftVelocities = inputDistanceCatalogFile.column<double>(10);

    for (std::size_t i_ref = 0; i_ref < RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES.size(); ++i_ref)
    {
        const ReferenceFrame redshiftReferenceFrame = RECONSTRUCTION_REDSHIFT_REFERENCE_FRAMES[i_ref];

        const ReferenceFrameChange redshiftCatalogInputToReferenceFrame = get_reference_frame_change(REDSHIFT_CATALOG_REDSHIFT_REFERENCE_FRAME, redshiftReferenceFrame);
        const ReferenceFrameChange distanceCatalogInputToReferenceFrame = get_reference_frame_change(DISTANCE_CATALOG_REDSHIFT_REFERENCE_FRAME, redshiftReferenceFrame);

        const std::string redshiftReferenceFrameName = get_reference_frame_name(redshiftReferenceFrame);
        const std::string redshiftReferenceFrameComment = "_z" + redshiftReferenceFrameName;
        const std::string realizationRangeComment = "_CR" + std::to_string(INITIAL_CONSTRAINED_REALIZATION) + "-" + std::to_string(FINAL_CONSTRAINED_REALIZATION);
        const std::string reconstructionComment = redshiftReferenceFrameComment + realizationRangeComment + CONFIGURATION_COMMENT;

        const std::string velocityComponentReconstructionErrorFileName = DATA_DIRECTORY + "velocity_component_reconstruction_error" + reconstructionComment + ".bin";
        const std::string distanceCatalogVelocitiesFileName = DATA_DIRECTORY + "reconstructed_CF3_galaxy_and_group_velocities" + reconstructionComment + ".dat";

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

        SphericalGridFunction reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity;

        transform_spherical_to_cartesian_vector_field(reconstructedRadialVelocity, reconstructedThetaVelocity, reconstructedPhiVelocity,
                                                      reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity);

        reconstructedXVelocity += ESTIMATED_EXTERNAL_BULK_X_VELOCITY;
        reconstructedYVelocity += ESTIMATED_EXTERNAL_BULK_Y_VELOCITY;
        reconstructedZVelocity += ESTIMATED_EXTERNAL_BULK_Z_VELOCITY;

        transform_cartesian_to_spherical_vector_field(reconstructedXVelocity, reconstructedYVelocity, reconstructedZVelocity,
                                                      reconstructedRadialVelocity, reconstructedThetaVelocity, reconstructedPhiVelocity);

        const auto time2 = std::chrono::high_resolution_clock::now();
        std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(time2 - time1).count() << "s)" << std::endl;

        Cartesian1DGridFunction velocityComponentReconstructionError;

        if (ANALYSIS_USE_PRECOMPUTED_RECONSTRUCTION_ERRORS and file_exists(velocityComponentReconstructionErrorFileName)) // load precomputed velocity component variance data if it exists and should be used
        {
            velocityComponentReconstructionError = Cartesian1DGridFunction(velocityComponentReconstructionErrorFileName);
        }
        else // otherwise compute it by generating a set of CRs
        {
            SphericalGridFunction noiseXVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);
            SphericalGridFunction noiseYVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);
            SphericalGridFunction noiseZVelocityVariance(RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_RADIAL_BIN_NUMBER, RECONSTRUCTION_THETA_BIN_NUMBER, RECONSTRUCTION_PHI_BIN_NUMBER, 0.0);

            for (std::size_t i_CR = INITIAL_CONSTRAINED_REALIZATION; i_CR <= FINAL_CONSTRAINED_REALIZATION; ++i_CR)
            {
                const auto timeCR1 = std::chrono::high_resolution_clock::now();
                std::cout << "Analyzing CR " << i_CR << "..." << std::flush;

                constrainedRealizations.generate(i_CR);

                SphericalGridFunction noiseRadialVelocity = constrainedRealizations.get_noise_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
                SphericalGridFunction noiseThetaVelocity = constrainedRealizations.get_noise_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
                SphericalGridFunction noisePhiVelocity = constrainedRealizations.get_noise_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);

                SphericalGridFunction noiseXVelocity, noiseYVelocity, noiseZVelocity;

                transform_spherical_to_cartesian_vector_field(noiseRadialVelocity, noiseThetaVelocity, noisePhiVelocity,
                                                              noiseXVelocity, noiseYVelocity, noiseZVelocity);

                noiseXVelocityVariance += noiseXVelocity * noiseXVelocity;
                noiseYVelocityVariance += noiseYVelocity * noiseYVelocity;
                noiseZVelocityVariance += noiseZVelocity * noiseZVelocity;

                const auto timeCR2 = std::chrono::high_resolution_clock::now();
                std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(timeCR2 - timeCR1).count() << "s)" << std::endl;
            }

            const double realizationNumber = static_cast<double>(FINAL_CONSTRAINED_REALIZATION - INITIAL_CONSTRAINED_REALIZATION + 1);

            const Cartesian1DGridFunction noiseXVelocityAngularAverageVariances = noiseXVelocityVariance.angular_average() / realizationNumber;
            const Cartesian1DGridFunction noiseYVelocityAngularAverageVariances = noiseYVelocityVariance.angular_average() / realizationNumber;
            const Cartesian1DGridFunction noiseZVelocityAngularAverageVariances = noiseZVelocityVariance.angular_average() / realizationNumber;

            velocityComponentReconstructionError = (noiseXVelocityAngularAverageVariances + noiseYVelocityAngularAverageVariances + noiseZVelocityAngularAverageVariances) / 3.0;

            velocityComponentReconstructionError.save_object_to_file(velocityComponentReconstructionErrorFileName);
        }

        std::ofstream distanceCatalogVelocitiesFile(distanceCatalogVelocitiesFileName);

        distanceCatalogVelocitiesFile.setf(std::ios::fixed);

        distanceCatalogVelocitiesFile << "# "
                                      << "PGC\t"
                                      << "vR[km/s]\t"
                                      << "vX[km/s]\t"
                                      << "vY[km/s]\t"
                                      << "vZ[km/s]\t"
                                      << "vRGroup[km/s]\t"
                                      << "vXGroup[km/s]\t"
                                      << "vYGroup[km/s]\t"
                                      << "vZGroup[km/s]\t"
                                      << "v_e[km/s]\t"
                                      << "flag"
                                      << std::endl;

        for (std::size_t i_g = 0; i_g < inputDistanceCatalogGalaxyPGCNumbers.size(); ++i_g)
        {
            const double galaxyLatitude = inputDistanceCatalogGalaxyLatitudes[i_g];
            const double galaxyLongitude = inputDistanceCatalogGalaxyLongitudes[i_g];
            const double groupLatitude = inputDistanceCatalogGroupLatitudes[i_g];
            const double groupLongitude = inputDistanceCatalogGroupLongitudes[i_g];

            double galaxyTheta, galaxyPhi,
                groupTheta, groupPhi;

            transform_celestial_to_spherical_coordinates(galaxyLatitude, galaxyLongitude,
                                                         galaxyTheta, galaxyPhi);

            transform_celestial_to_spherical_coordinates(groupLatitude, groupLongitude,
                                                         groupTheta, groupPhi);

            const double groupRedshiftVelocity = change_reference_frame(inputDistanceCatalogGroupRedshiftVelocities[i_g], groupTheta, groupPhi, distanceCatalogInputToReferenceFrame);

            const double groupRadius = (groupRedshiftVelocity >= 0.0) ? comoving_distance(groupRedshiftVelocity, FIDUCIAL_OMEGA_MATTER)
                                                                      : 0.0;

            const double galaxyRadialVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedRadialVelocity(groupRadius, galaxyTheta, galaxyPhi)
                                                                                           : 0.0;
            const double galaxyXVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedXVelocity(groupRadius, galaxyTheta, galaxyPhi)
                                                                                      : 0.0;
            const double galaxyYVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedYVelocity(groupRadius, galaxyTheta, galaxyPhi)
                                                                                      : 0.0;
            const double galaxyZVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedZVelocity(groupRadius, galaxyTheta, galaxyPhi)
                                                                                      : 0.0;

            const double groupRadialVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedRadialVelocity(groupRadius, groupTheta, groupPhi)
                                                                                          : 0.0;
            const double groupXVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedXVelocity(groupRadius, groupTheta, groupPhi)
                                                                                     : 0.0;
            const double groupYVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedYVelocity(groupRadius, groupTheta, groupPhi)
                                                                                     : 0.0;
            const double groupZVelocity = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? reconstructedZVelocity(groupRadius, groupTheta, groupPhi)
                                                                                     : 0.0;

            const double velocityError = (groupRadius <= RECONSTRUCTION_MAX_RADIUS) ? velocityComponentReconstructionError(groupRadius)
                                                                                    : -1.0;

            int flag = 0;

            if (groupRedshiftVelocity < 0.0)
            {
                flag = -1;
            }
            else if (groupRadius > RECONSTRUCTION_MAX_RADIUS)
            {
                flag = 1;
            }

            distanceCatalogVelocitiesFile << std::setprecision(0)
                                          << inputDistanceCatalogGalaxyPGCNumbers[i_g] << "\t"
                                          << galaxyRadialVelocity << "\t"
                                          << galaxyXVelocity << "\t"
                                          << galaxyYVelocity << "\t"
                                          << galaxyZVelocity << "\t"
                                          << groupRadialVelocity << "\t"
                                          << groupXVelocity << "\t"
                                          << groupYVelocity << "\t"
                                          << groupZVelocity << "\t"
                                          << velocityError << "\t"
                                          << flag
                                          << std::endl;
        }

        distanceCatalogVelocitiesFile.close();
    }

    fftw_cleanup_threads();

    return 0;
}