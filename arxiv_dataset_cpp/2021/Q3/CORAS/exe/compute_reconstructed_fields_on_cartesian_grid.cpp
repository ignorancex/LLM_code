#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../src/Cartesian1DGridFunction.hpp"
#include "../src/Cartesian3DGridFunction.hpp"
#include "../src/ConstrainedRealizations.hpp"
#include "../src/FileTable.hpp"
#include "../src/NormalizedPowerSpectrum.hpp"
#include "../src/SphericalGridFunction.hpp"
#include "../src/survey.hpp"
#include "../src/transformations.hpp"

#include "../configuration.hpp"

double evaluate_in_enclosing_box(const SphericalGridFunction &field, double x, double y, double z)
{
    double radius, theta, phi;

    transform_cartesian_to_spherical_coordinates(x, y, z,
                                                 radius, theta, phi);

    return (radius <= RECONSTRUCTION_MAX_RADIUS) ? field(radius, theta, phi) : 0.0;
}

int main(int argc, char **argv)
{
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
        const std::string reconstructionComment = redshiftReferenceFrameComment + CONFIGURATION_COMMENT;

        const std::string cartesianGridDensityContrastFileName = DATA_DIRECTORY + "cartesian_grid_density" + reconstructionComment + ".dat";
        const std::string cartesianGridVelocityFileName = DATA_DIRECTORY + "cartesian_grid_velocity" + reconstructionComment + ".dat";

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
                                                        RECONSTRUCTION_FFT_PERIODIC_BOUNDARY_DISTANCE, 1, RECONSTRUCTION_LOG_TRAFO_SMOOTHING_SCALE, // set FFT bin number to 1, as no random realizations will be generated
                                                        meanGalaxyDensity, selectionFunction, selectionFunctionLogDerivative, sigmaGalaxy, normalizedPowerSpectrum,
                                                        DATA_DIRECTORY, precomputedRSDCorrectionAndWienerFilterComment,
                                                        RECONSTRUCTION_USE_PRECOMPUTED_RSD_CORRECTION_AND_WIENER_FILTER);

        const auto time2 = std::chrono::high_resolution_clock::now();
        std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(time2 - time1).count() << "s)" << std::endl;
        std::cout << "Computing reconstructed fields..." << std::flush;

        SphericalGridFunction reconstructedNormalizedDensityContrastSpherical = constrainedRealizations.get_survey_estimate_density_contrast(1.0, ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE); // these fields will be used for the SGP and angular average output
        SphericalGridFunction reconstructedRadialVelocitySpherical = constrainedRealizations.get_survey_estimate_radial_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedThetaVelocity = constrainedRealizations.get_survey_estimate_theta_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);
        SphericalGridFunction reconstructedPhiVelocity = constrainedRealizations.get_survey_estimate_phi_velocity(ESTIMATED_NORMALIZED_GROWTH_RATE, RECONSTRUCTED_FIELD_SMOOTHING_SCALE);

        SphericalGridFunction reconstructedXVelocitySpherical, reconstructedYVelocitySpherical, reconstructedZVelocitySpherical,
            reconstructedXVelocityMinSmooth, reconstructedYVelocityMinSmooth, reconstructedZVelocityMinSmooth;

        transform_spherical_to_cartesian_vector_field(reconstructedRadialVelocitySpherical, reconstructedThetaVelocity, reconstructedPhiVelocity,
                                                      reconstructedXVelocitySpherical, reconstructedYVelocitySpherical, reconstructedZVelocitySpherical);

        reconstructedXVelocitySpherical += ESTIMATED_EXTERNAL_BULK_X_VELOCITY;
        reconstructedYVelocitySpherical += ESTIMATED_EXTERNAL_BULK_Y_VELOCITY;
        reconstructedZVelocitySpherical += ESTIMATED_EXTERNAL_BULK_Z_VELOCITY;

        transform_cartesian_to_spherical_vector_field(reconstructedXVelocitySpherical, reconstructedYVelocitySpherical, reconstructedZVelocitySpherical,
                                                      reconstructedRadialVelocitySpherical, reconstructedThetaVelocity, reconstructedPhiVelocity);

        Cartesian3DGridFunction reconstructedNormalizedDensityContrastCartesian(-RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                                [&](double x, double y, double z)
                                                                                {
                                                                                    return evaluate_in_enclosing_box(reconstructedNormalizedDensityContrastSpherical, x, y, z);
                                                                                });

        Cartesian3DGridFunction reconstructedXVelocityCartesian(-RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                [&](double x, double y, double z)
                                                                {
                                                                    return evaluate_in_enclosing_box(reconstructedXVelocitySpherical, x, y, z);
                                                                });

        Cartesian3DGridFunction reconstructedYVelocityCartesian(-RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                [&](double x, double y, double z)
                                                                {
                                                                    return evaluate_in_enclosing_box(reconstructedYVelocitySpherical, x, y, z);
                                                                });

        Cartesian3DGridFunction reconstructedZVelocityCartesian(-RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                -RECONSTRUCTION_MAX_RADIUS, RECONSTRUCTION_MAX_RADIUS, CARTESIAN_GRID_BIN_NUMBER,
                                                                [&](double x, double y, double z)
                                                                {
                                                                    return evaluate_in_enclosing_box(reconstructedZVelocitySpherical, x, y, z);
                                                                });

        std::ofstream cartesianGridDensityContrastFile(cartesianGridDensityContrastFileName);
        std::ofstream cartesianGridVelocityFile(cartesianGridVelocityFileName);

        cartesianGridDensityContrastFile.setf(std::ios::fixed);
        cartesianGridVelocityFile.setf(std::ios::fixed);

        cartesianGridDensityContrastFile << "# "
                                         << "deltaNorm"
                                         << std::endl;

        cartesianGridVelocityFile << "# "
                                  << "vX[km/s]\t"
                                  << "vY[km/s]\t"
                                  << "vZ[km/s]"
                                  << std::endl;

        reconstructedNormalizedDensityContrastCartesian.evaluate_for_all_grid_points([&](std::size_t i_x, std::size_t i_y, std::size_t i_z)
                                                                                     {
                                                                                         const double normalizedDensityContrast = reconstructedNormalizedDensityContrastCartesian.value(i_x, i_y, i_z);
                                                                                         const double xVelocity = reconstructedXVelocityCartesian.value(i_x, i_y, i_z);
                                                                                         const double yVelocity = reconstructedYVelocityCartesian.value(i_x, i_y, i_z);
                                                                                         const double zVelocity = reconstructedZVelocityCartesian.value(i_x, i_y, i_z);

                                                                                         cartesianGridDensityContrastFile << std::setprecision(4) // to reduce the file size, do not write the coordinates to the file
                                                                                                                          << normalizedDensityContrast
                                                                                                                          << std::endl;

                                                                                         cartesianGridVelocityFile << std::setprecision(2)
                                                                                                                   << xVelocity << "\t"
                                                                                                                   << yVelocity << "\t"
                                                                                                                   << zVelocity
                                                                                                                   << std::endl;
                                                                                     });

        cartesianGridDensityContrastFile.close();
        cartesianGridVelocityFile.close();

        const auto time3 = std::chrono::high_resolution_clock::now();
        std::cout << " done (" << std::chrono::duration_cast<std::chrono::seconds>(time3 - time2).count() << "s)" << std::endl;
    }

    return 0;
}