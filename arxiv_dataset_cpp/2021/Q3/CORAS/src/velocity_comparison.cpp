#include "velocity_comparison.hpp"

#include <algorithm>
#include <cmath>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

#include "../src/cosmology.hpp"
#include "../src/miscellaneous.hpp"
#include "../src/ObjectGrid.hpp"

void estimate_parameters_via_radial_velocity_comparison(const std::function<double(double normalizedGrowthRate, double radius, double theta, double phi)> &normalizedWienerRadialVelocity, const SphericalGridFunction &normalizedNoiseRadialVelocity, const ReferenceFrameChange cmbToReferenceFrame,
                                                        const std::vector<double> &redshiftVelocities, const std::vector<double> &thetaCoordinates, const std::vector<double> &phiCoordinates, const std::vector<double> &distanceModuli, const std::vector<double> &distanceModulusErrors,
                                                        const double omegaMatter, const double normalizedGrowthRateGuess, const double normalizedGrowthRateMin, const double normalizedGrowthRateMax, const double hubbleGuess,
                                                        double &normalizedGrowthRateEstimate, double &externalBulkXVelocityEstimate, double &externalBulkYVelocityEstimate, double &externalBulkZVelocityEstimate, double &hubbleEstimate,
                                                        gsl_matrix *covarianceMatrix, double &reducedChiSquared)
{
    constexpr double normalizedGrowthRateDerivativeDelta = 0.001;

    const std::size_t velocityNumber = redshiftVelocities.size();

    const auto internal_radial_velocity = [&](double normalizedGrowthRate, double radius, double theta, double phi)
    {
        return normalizedGrowthRate * (normalizedWienerRadialVelocity(normalizedGrowthRate, radius, theta, phi) + normalizedNoiseRadialVelocity(radius, theta, phi));
    };

    const auto internal_radial_velocity_growth_rate_derivative = [&](double normalizedGrowthRate, double radius, double theta, double phi)
    {
        return normalizedWienerRadialVelocity(normalizedGrowthRateEstimate, radius, theta, phi) + normalizedGrowthRateEstimate * (normalizedWienerRadialVelocity(normalizedGrowthRate + normalizedGrowthRateDerivativeDelta, radius, theta, phi) - normalizedWienerRadialVelocity(normalizedGrowthRate - normalizedGrowthRateDerivativeDelta, radius, theta, phi)) / 2.0 / normalizedGrowthRateDerivativeDelta + normalizedNoiseRadialVelocity(radius, theta, phi);
    };

    const auto chi_squared = [&](const gsl_vector *minimizerArguments)
    {
        const double normalizedGrowthRate = std::atan(gsl_vector_get(minimizerArguments, 0)) / M_PI * (normalizedGrowthRateMax - normalizedGrowthRateMin - 2.0 * normalizedGrowthRateDerivativeDelta) + (normalizedGrowthRateMax + normalizedGrowthRateMin) / 2.0;
        const double externalBulkXVelocity = gsl_vector_get(minimizerArguments, 1);
        const double externalBulkYVelocity = gsl_vector_get(minimizerArguments, 2);
        const double externalBulkZVelocity = gsl_vector_get(minimizerArguments, 3);
        const double hubble = std::exp(gsl_vector_get(minimizerArguments, 4));

        double chiSquared = 0.0;

#pragma omp parallel for schedule(static) reduction(+ \
                                                    : chiSquared)
        for (std::size_t i_v = 0; i_v < velocityNumber; ++i_v)
        {
            const double theta = thetaCoordinates[i_v];
            const double phi = phiCoordinates[i_v];
            const double distanceModulus = distanceModuli[i_v];
            const double distanceModulusError = distanceModulusErrors[i_v];

            const double redshiftVelocity = redshiftVelocities[i_v];

            const double radius = comoving_distance(redshiftVelocity, omegaMatter);

            const double internalRadialVelocity = internal_radial_velocity(normalizedGrowthRate, radius, theta, phi);

            const double externalBulkRadialVelocity = std::sin(theta) * std::cos(phi) * externalBulkXVelocity + std::sin(theta) * std::sin(phi) * externalBulkYVelocity + std::cos(theta) * externalBulkZVelocity;

            const double distanceModulusAtObservedRedshift = distance_modulus(redshiftVelocity, omegaMatter, hubble);
            const double distanceModulusVelocityCorrection = distance_modulus_redshift_velocity_derivative(redshiftVelocity, omegaMatter) * change_reference_frame(internalRadialVelocity + externalBulkRadialVelocity, theta, phi, cmbToReferenceFrame);

            chiSquared += gsl_pow_2(distanceModulus - distanceModulusAtObservedRedshift + distanceModulusVelocityCorrection) / gsl_pow_2(distanceModulusError);
        }

        return chiSquared;
    };

    gsl_multimin_fminimizer *minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, 5);
    gsl_vector *minimizerStepSizes = gsl_vector_alloc(5);
    gsl_vector *minimizerArguments = gsl_vector_alloc(5);

    gsl_vector_set(minimizerArguments, 0, std::tan((normalizedGrowthRateGuess - (normalizedGrowthRateMax + normalizedGrowthRateMin) / 2.0) * M_PI / (normalizedGrowthRateMax - normalizedGrowthRateMin - 2.0 * normalizedGrowthRateDerivativeDelta))); // initial guesses
    gsl_vector_set(minimizerArguments, 1, 0.0);
    gsl_vector_set(minimizerArguments, 2, 0.0);
    gsl_vector_set(minimizerArguments, 3, 0.0);
    gsl_vector_set(minimizerArguments, 4, std::log(hubbleGuess));

    gsl_vector_set(minimizerStepSizes, 0, 0.1); // initial step sizes
    gsl_vector_set(minimizerStepSizes, 1, 50.0);
    gsl_vector_set(minimizerStepSizes, 2, 50.0);
    gsl_vector_set(minimizerStepSizes, 3, 50.0);
    gsl_vector_set(minimizerStepSizes, 4, 0.1);

    GSLMultiminFunctionWrapper Fp(5, chi_squared);
    gsl_multimin_function *minimizerFunction = static_cast<gsl_multimin_function *>(&Fp); // initialize method and iterate

    gsl_multimin_fminimizer_set(minimizer, minimizerFunction, minimizerArguments, minimizerStepSizes);

    int status = GSL_CONTINUE;
    std::size_t iteration = 0;

    while ((status == GSL_CONTINUE) and (iteration < 1000))
    {
        status = gsl_multimin_fminimizer_iterate(minimizer);

        if (status)
            break;

        const double size = gsl_multimin_fminimizer_size(minimizer);

        status = gsl_multimin_test_size(size, 1e-2);

        iteration++;
    }

    normalizedGrowthRateEstimate = std::atan(gsl_vector_get(gsl_multimin_fminimizer_x(minimizer), 0)) / M_PI * (normalizedGrowthRateMax - normalizedGrowthRateMin - 2.0 * normalizedGrowthRateDerivativeDelta) + (normalizedGrowthRateMax + normalizedGrowthRateMin) / 2.0;
    externalBulkXVelocityEstimate = gsl_vector_get(gsl_multimin_fminimizer_x(minimizer), 1);
    externalBulkYVelocityEstimate = gsl_vector_get(gsl_multimin_fminimizer_x(minimizer), 2);
    externalBulkZVelocityEstimate = gsl_vector_get(gsl_multimin_fminimizer_x(minimizer), 3);
    hubbleEstimate = std::exp(gsl_vector_get(gsl_multimin_fminimizer_x(minimizer), 4));
    reducedChiSquared = gsl_multimin_fminimizer_minimum(minimizer) / (static_cast<double>(velocityNumber) - 5.0);

    gsl_vector_free(minimizerArguments);
    gsl_vector_free(minimizerStepSizes);
    gsl_multimin_fminimizer_free(minimizer);

    double fisher00 = 0.0;
    double fisher10 = 0.0;
    double fisher20 = 0.0;
    double fisher30 = 0.0;
    double fisher40 = 0.0;
    double fisher11 = 0.0;
    double fisher21 = 0.0;
    double fisher31 = 0.0;
    double fisher41 = 0.0;
    double fisher22 = 0.0;
    double fisher32 = 0.0;
    double fisher42 = 0.0;
    double fisher33 = 0.0;
    double fisher43 = 0.0;
    double fisher44 = 0.0;

    for (std::size_t i_v = 0; i_v < velocityNumber; ++i_v)
    {
        const double redshiftVelocity = redshiftVelocities[i_v];
        const double theta = thetaCoordinates[i_v];
        const double phi = phiCoordinates[i_v];
        const double distanceModulusErrorSquared = gsl_pow_2(distanceModulusErrors[i_v]);

        const double radius = comoving_distance(redshiftVelocity, omegaMatter);
        const double distanceModulusVelocityCorrectionFactor = distance_modulus_redshift_velocity_derivative(redshiftVelocity, omegaMatter);

        const double growthRateDerivative = internal_radial_velocity_growth_rate_derivative(normalizedGrowthRateEstimate, radius, theta, phi) * distanceModulusVelocityCorrectionFactor;
        const double externalBulkXVelocityDerivative = std::sin(theta) * std::cos(phi) * distanceModulusVelocityCorrectionFactor;
        const double externalBulkYVelocityDerivative = std::sin(theta) * std::sin(phi) * distanceModulusVelocityCorrectionFactor;
        const double externalBulkZVelocityDerivative = std::cos(theta) * distanceModulusVelocityCorrectionFactor;
        const double hubbleDerivative = 5.0 / std::log(10.0) / hubbleEstimate;

        fisher00 += growthRateDerivative * growthRateDerivative / distanceModulusErrorSquared;
        fisher10 += externalBulkXVelocityDerivative * growthRateDerivative / distanceModulusErrorSquared;
        fisher20 += externalBulkYVelocityDerivative * growthRateDerivative / distanceModulusErrorSquared;
        fisher30 += externalBulkZVelocityDerivative * growthRateDerivative / distanceModulusErrorSquared;
        fisher40 += hubbleDerivative * growthRateDerivative / distanceModulusErrorSquared;
        fisher11 += externalBulkXVelocityDerivative * externalBulkXVelocityDerivative / distanceModulusErrorSquared;
        fisher21 += externalBulkYVelocityDerivative * externalBulkXVelocityDerivative / distanceModulusErrorSquared;
        fisher31 += externalBulkZVelocityDerivative * externalBulkXVelocityDerivative / distanceModulusErrorSquared;
        fisher41 += hubbleDerivative * externalBulkXVelocityDerivative / distanceModulusErrorSquared;
        fisher22 += externalBulkYVelocityDerivative * externalBulkYVelocityDerivative / distanceModulusErrorSquared;
        fisher32 += externalBulkZVelocityDerivative * externalBulkYVelocityDerivative / distanceModulusErrorSquared;
        fisher42 += hubbleDerivative * externalBulkYVelocityDerivative / distanceModulusErrorSquared;
        fisher33 += externalBulkZVelocityDerivative * externalBulkZVelocityDerivative / distanceModulusErrorSquared;
        fisher43 += hubbleDerivative * externalBulkZVelocityDerivative / distanceModulusErrorSquared;
        fisher44 += hubbleDerivative * hubbleDerivative / distanceModulusErrorSquared;
    }

    gsl_matrix_set(covarianceMatrix, 0, 0, fisher00);
    gsl_matrix_set(covarianceMatrix, 1, 0, fisher10);
    gsl_matrix_set(covarianceMatrix, 2, 0, fisher20);
    gsl_matrix_set(covarianceMatrix, 3, 0, fisher30);
    gsl_matrix_set(covarianceMatrix, 4, 0, fisher40);
    gsl_matrix_set(covarianceMatrix, 1, 1, fisher11);
    gsl_matrix_set(covarianceMatrix, 2, 1, fisher21);
    gsl_matrix_set(covarianceMatrix, 3, 1, fisher31);
    gsl_matrix_set(covarianceMatrix, 4, 1, fisher41);
    gsl_matrix_set(covarianceMatrix, 2, 2, fisher22);
    gsl_matrix_set(covarianceMatrix, 3, 2, fisher32);
    gsl_matrix_set(covarianceMatrix, 4, 2, fisher42);
    gsl_matrix_set(covarianceMatrix, 3, 3, fisher33);
    gsl_matrix_set(covarianceMatrix, 4, 3, fisher43);
    gsl_matrix_set(covarianceMatrix, 4, 4, fisher44);

    gsl_linalg_cholesky_decomp1(covarianceMatrix);
    gsl_linalg_cholesky_invert(covarianceMatrix);

    gsl_matrix_set(covarianceMatrix, 0, 1, gsl_matrix_get(covarianceMatrix, 1, 0));
    gsl_matrix_set(covarianceMatrix, 0, 2, gsl_matrix_get(covarianceMatrix, 2, 0));
    gsl_matrix_set(covarianceMatrix, 0, 3, gsl_matrix_get(covarianceMatrix, 3, 0));
    gsl_matrix_set(covarianceMatrix, 0, 4, gsl_matrix_get(covarianceMatrix, 4, 0));
    gsl_matrix_set(covarianceMatrix, 1, 2, gsl_matrix_get(covarianceMatrix, 2, 1));
    gsl_matrix_set(covarianceMatrix, 1, 3, gsl_matrix_get(covarianceMatrix, 3, 1));
    gsl_matrix_set(covarianceMatrix, 1, 4, gsl_matrix_get(covarianceMatrix, 4, 1));
    gsl_matrix_set(covarianceMatrix, 2, 3, gsl_matrix_get(covarianceMatrix, 3, 2));
    gsl_matrix_set(covarianceMatrix, 2, 4, gsl_matrix_get(covarianceMatrix, 4, 2));
    gsl_matrix_set(covarianceMatrix, 3, 4, gsl_matrix_get(covarianceMatrix, 4, 3));
}

void compute_radial_velocity_correlation_functions(const std::vector<double> &observedRedshiftVelocities, const std::vector<double> &observedThetaCoordinates, const std::vector<double> &observedPhiCoordinates, const std::vector<double> &observedDistanceModuli,
                                                   const SphericalGridFunction &reconstructedRadialVelocity,
                                                   const ReferenceFrameChange referenceToCMBFrame, const double omegaMatter, const double hubble,
                                                   const double maxDistance, const std::size_t distanceBinNumber,
                                                   Cartesian1DGridFunction &observedRadialVelocityCorrFunc, Cartesian1DGridFunction &reconstructedRadialVelocityCorrFunc, Cartesian1DGridFunction &residualRadialVelocityCorrFunc)
{
    observedRadialVelocityCorrFunc = Cartesian1DGridFunction(0.0, maxDistance, distanceBinNumber, 0.0);
    reconstructedRadialVelocityCorrFunc = observedRadialVelocityCorrFunc;
    residualRadialVelocityCorrFunc = observedRadialVelocityCorrFunc;

    Cartesian1DGridFunction radialVelocityCorrFuncNormalization = observedRadialVelocityCorrFunc;

    const double distanceBinWidth = observedRadialVelocityCorrFunc.coordinate(1);

    const double maxRedshiftVelocity = *std::max_element(observedRedshiftVelocities.begin(), observedRedshiftVelocities.end());

    const double gridLength = 2.0 * comoving_distance(maxRedshiftVelocity, omegaMatter);
    const std::size_t gridBinNumber = static_cast<std::size_t>(std::ceil(gridLength / maxDistance * 2.0));

    const std::size_t observationsNumber = observedRedshiftVelocities.size();

    std::vector<double> xCoordinates;
    std::vector<double> yCoordinates;
    std::vector<double> zCoordinates;
    std::vector<double> observedRadialVelocities;
    std::vector<double> reconstructedRadialVelocities;

    for (std::size_t i_g = 0; i_g < observationsNumber; ++i_g)
    {
        const double theta = observedThetaCoordinates[i_g];
        const double phi = observedPhiCoordinates[i_g];
        const double redshiftVelocity = observedRedshiftVelocities[i_g];
        const double distanceModulus = observedDistanceModuli[i_g];

        const double radius = comoving_distance(redshiftVelocity, omegaMatter);

        double x, y, z;

        transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                                     x, y, z);

        xCoordinates.push_back(x);
        yCoordinates.push_back(y);
        zCoordinates.push_back(z);

        const double distanceModulusAtObservedRedshift = distance_modulus(redshiftVelocity, omegaMatter, hubble);
        const double distanceModulusVelocityDerivative = distance_modulus_redshift_velocity_derivative(redshiftVelocity, omegaMatter);

        observedRadialVelocities.push_back(change_reference_frame((distanceModulusAtObservedRedshift - distanceModulus) / distanceModulusVelocityDerivative, theta, phi, referenceToCMBFrame));
        reconstructedRadialVelocities.push_back(reconstructedRadialVelocity(radius, theta, phi));
    }

    ObjectGrid galaxyGroupGrid(gridLength, gridBinNumber,
                               0.0, 0.0, 0.0,
                               xCoordinates, yCoordinates, zCoordinates,
                               false);

    galaxyGroupGrid.evaluate_for_all_objects(
        [&](std::size_t i_g1)
        {
            const double x1 = xCoordinates[i_g1];
            const double y1 = yCoordinates[i_g1];
            const double z1 = zCoordinates[i_g1];

            const double r1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);

            galaxyGroupGrid.evaluate_for_all_objects_within_distance(
                i_g1, maxDistance + distanceBinWidth,
                [&](double x12, double y12, double z12, double distance12, std::size_t i_g2)
                {
                    if (i_g2 < i_g1)
                    {
                        const double x2 = xCoordinates[i_g2];
                        const double y2 = yCoordinates[i_g2];
                        const double z2 = zCoordinates[i_g2];

                        const double r2 = std::sqrt(x2 * x2 + y2 * y2 + z2 * z2);

                        const double cos12 = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;

                        const std::size_t distanceBin = static_cast<std::size_t>(std::floor(distance12 / distanceBinWidth));

                        const double vObs1 = observedRadialVelocities[i_g1];
                        const double vObs2 = observedRadialVelocities[i_g2];
                        const double vRec1 = reconstructedRadialVelocities[i_g1];
                        const double vRec2 = reconstructedRadialVelocities[i_g2];
                        const double vRes1 = vObs1 - vRec1;
                        const double vRes2 = vObs2 - vRec2;

                        observedRadialVelocityCorrFunc.value(distanceBin) += cos12 * vObs1 * vObs2;
                        reconstructedRadialVelocityCorrFunc.value(distanceBin) += cos12 * vRec1 * vRec2;
                        residualRadialVelocityCorrFunc.value(distanceBin) += cos12 * vRes1 * vRes2;
                        radialVelocityCorrFuncNormalization.value(distanceBin) += gsl_pow_2(cos12);
                    }
                });
        });

    observedRadialVelocityCorrFunc /= radialVelocityCorrFuncNormalization;
    reconstructedRadialVelocityCorrFunc /= radialVelocityCorrFuncNormalization;
    residualRadialVelocityCorrFunc /= radialVelocityCorrFuncNormalization;
}

void compute_tensor_smoothed_radial_velocity_points(const std::vector<double> &observedRedshiftVelocities, const std::vector<double> &observedThetaCoordinates, const std::vector<double> &observedPhiCoordinates, const std::vector<double> &observedDistanceModuli, const std::vector<double> &observedDistanceModulusErrors,
                                                    const SphericalGridFunction &reconstructedRadialVelocity,
                                                    const ReferenceFrameChange referenceToCMBFrame, const double omegaMatter, const double hubble, const double minSmoothingScale, const double velocityCosmicSigma,
                                                    std::vector<double> &smoothObservedRadialVelocities, std::vector<double> &smoothReconstructedRadialVelocities, std::vector<double> &smoothObservedRadialVelocityErrors, std::vector<double> &adaptiveSmoothingScales,
                                                    const std::size_t minNeighbourNumber)
{
    const double searchDistance = 3.0 * minSmoothingScale;

    auto window_function = [&](double distance, double weight, double smoothingRadius)
    {
        return weight * std::exp(-gsl_pow_2(distance) / 2.0 / gsl_pow_2(smoothingRadius));
    };

    const double maxRedshiftVelocity = *std::max_element(observedRedshiftVelocities.begin(), observedRedshiftVelocities.end());

    const double gridLength = 2.0 * comoving_distance(maxRedshiftVelocity, omegaMatter);
    const std::size_t gridBinNumber = static_cast<std::size_t>(std::ceil(gridLength / searchDistance * 2.0));

    const std::size_t observationsNumber = observedRedshiftVelocities.size();

    std::vector<double> xCoordinates;
    std::vector<double> yCoordinates;
    std::vector<double> zCoordinates;
    std::vector<double> observedRadialVelocities;
    std::vector<double> reconstructedRadialVelocities;
    std::vector<double> observedRadialVelocityErrors;
    std::vector<double> windowFunctionWeights;

    std::vector<gsl_vector *> r_vector;
    std::vector<gsl_matrix *> Ainv_vector;

    for (std::size_t i_g = 0; i_g < observationsNumber; ++i_g)
    {
        const double theta = observedThetaCoordinates[i_g];
        const double phi = observedPhiCoordinates[i_g];
        const double redshiftVelocity = observedRedshiftVelocities[i_g];
        const double distanceModulus = observedDistanceModuli[i_g];
        const double distanceModulusError = observedDistanceModulusErrors[i_g];

        const double radius = comoving_distance(redshiftVelocity, omegaMatter);

        double x, y, z;

        transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                                     x, y, z);

        gsl_vector *r = gsl_vector_calloc(3);
        gsl_matrix *Ainv = gsl_matrix_calloc(3, 3);

        gsl_vector_set(r, 0, x / radius);
        gsl_vector_set(r, 1, y / radius);
        gsl_vector_set(r, 2, z / radius);

        r_vector.push_back(r);
        Ainv_vector.push_back(Ainv);

        xCoordinates.push_back(x);
        yCoordinates.push_back(y);
        zCoordinates.push_back(z);

        const double distanceModulusAtObservedRedshift = distance_modulus(redshiftVelocity, omegaMatter, hubble);
        const double distanceModulusVelocityDerivative = distance_modulus_redshift_velocity_derivative(redshiftVelocity, omegaMatter);

        observedRadialVelocities.push_back(change_reference_frame((distanceModulusAtObservedRedshift - distanceModulus) / distanceModulusVelocityDerivative, theta, phi, referenceToCMBFrame));
        reconstructedRadialVelocities.push_back(reconstructedRadialVelocity(radius, theta, phi));

        const double velocityError = distanceModulusError / distanceModulusVelocityDerivative;

        observedRadialVelocityErrors.push_back(velocityError);
        windowFunctionWeights.push_back(1.0 / (gsl_pow_2(velocityError) + gsl_pow_2(velocityCosmicSigma)));
    }

    ObjectGrid galaxyGroupGrid(gridLength, gridBinNumber,
                               0.0, 0.0, 0.0,
                               xCoordinates, yCoordinates, zCoordinates,
                               false);

    std::size_t consideredObservationsNumber = galaxyGroupGrid.object_number();

    adaptiveSmoothingScales = std::vector<double>(consideredObservationsNumber, minSmoothingScale);

    smoothObservedRadialVelocities = std::vector<double>(consideredObservationsNumber);
    smoothReconstructedRadialVelocities = std::vector<double>(consideredObservationsNumber);
    smoothObservedRadialVelocityErrors = std::vector<double>(consideredObservationsNumber);

    galaxyGroupGrid.evaluate_for_all_objects(
        [&](std::size_t i_g1)
        {
            gsl_vector *R_obs = gsl_vector_calloc(3);
            gsl_vector *R_rec = gsl_vector_calloc(3);
            gsl_vector *aux = gsl_vector_calloc(3);
            gsl_matrix *A = gsl_matrix_calloc(3, 3);

            std::size_t neighbourNumber = 0;

            galaxyGroupGrid.evaluate_for_all_objects_within_distance(
                i_g1, searchDistance,
                [&](double x12, double y12, double z12, double distance12, std::size_t i_g2)
                {
                    const double weight2 = windowFunctionWeights[i_g2];
                    const double W12 = window_function(distance12, weight2, minSmoothingScale);

                    gsl_blas_dsyr(CblasLower, W12, r_vector[i_g2], A);

                    gsl_blas_daxpy(W12 * observedRadialVelocities[i_g2], r_vector[i_g2], R_obs);
                    gsl_blas_daxpy(W12 * reconstructedRadialVelocities[i_g2], r_vector[i_g2], R_rec);

                    if (distance12 <= 2.0 * minSmoothingScale)
                    {
                        neighbourNumber++;
                    }
                });

            if (neighbourNumber < minNeighbourNumber)
            {
                gsl_vector_set_zero(R_obs);
                gsl_vector_set_zero(R_rec);
                gsl_matrix_set_zero(A);

                std::vector<double> neighbourDistances;

                galaxyGroupGrid.evaluate_for_all_objects_within_distance(
                    i_g1, gridLength / 2.0,
                    [&](double x12, double y12, double z12, double distance12, std::size_t i_g2)
                    {
                        neighbourDistances.push_back(distance12);
                    });

                double minNeighbourNumberDistance = 0.0;

                for (std::size_t i_n = 0; i_n < minNeighbourNumber; ++i_n)
                {
                    auto minDistance = std::min_element(neighbourDistances.begin(), neighbourDistances.end());
                    minNeighbourNumberDistance = *minDistance;
                    neighbourDistances.erase(minDistance);
                }

                const double adaptiveSmoothingScale = minNeighbourNumberDistance / 2.0;
                const double adaptiveSearchDistance = 3.0 * adaptiveSmoothingScale;

                adaptiveSmoothingScales[i_g1] = adaptiveSmoothingScale;

                galaxyGroupGrid.evaluate_for_all_objects_within_distance(
                    i_g1, adaptiveSearchDistance,
                    [&](double x12, double y12, double z12, double distance12, std::size_t i_g2)
                    {
                        const double weight2 = windowFunctionWeights[i_g2];
                        const double W12 = window_function(distance12, weight2, adaptiveSmoothingScale);

                        gsl_blas_dsyr(CblasLower, W12, r_vector[i_g2], A);

                        gsl_blas_daxpy(W12 * observedRadialVelocities[i_g2], r_vector[i_g2], R_obs);
                        gsl_blas_daxpy(W12 * reconstructedRadialVelocities[i_g2], r_vector[i_g2], R_rec);
                    });
            }

            gsl_linalg_cholesky_decomp1(A);
            gsl_linalg_cholesky_invert(A);

            gsl_matrix_memcpy(Ainv_vector[i_g1], A);

            gsl_blas_dsymv(CblasLower, 1.0, A, R_obs, 0.0, aux);
            gsl_blas_ddot(r_vector[i_g1], aux, &smoothObservedRadialVelocities[i_g1]);

            gsl_blas_dsymv(CblasLower, 1.0, A, R_rec, 0.0, aux);
            gsl_blas_ddot(r_vector[i_g1], aux, &smoothReconstructedRadialVelocities[i_g1]);

            gsl_vector_free(R_obs);
            gsl_vector_free(R_rec);
            gsl_vector_free(aux);
            gsl_matrix_free(A);
        });

    galaxyGroupGrid.evaluate_for_all_objects(
        [&](std::size_t i_g1)
        {
            gsl_vector *aux = gsl_vector_calloc(3);
            gsl_matrix *Ainv = Ainv_vector[i_g1];

            const double adaptiveSmoothingScale = adaptiveSmoothingScales[i_g1];
            const double adaptiveSearchDistance = 3.0 * adaptiveSmoothingScale;

            double smoothObservedRadialVelocityVariance = 0.0;

            galaxyGroupGrid.evaluate_for_all_objects_within_distance(
                i_g1, adaptiveSearchDistance,
                [&](double x12, double y12, double z12, double distance12, std::size_t i_g2)
                {
                    const double weight2 = windowFunctionWeights[i_g2];
                    const double W12 = window_function(distance12, weight2, adaptiveSmoothingScale);

                    double W12Tensor;

                    gsl_blas_dsymv(CblasLower, W12, Ainv, r_vector[i_g2], 0.0, aux);
                    gsl_blas_ddot(r_vector[i_g1], aux, &W12Tensor);

                    smoothObservedRadialVelocityVariance += gsl_pow_2(W12Tensor * observedRadialVelocityErrors[i_g2]) + gsl_pow_2(W12Tensor * (observedRadialVelocities[i_g2] - smoothObservedRadialVelocities[i_g2]));
                });

            smoothObservedRadialVelocityErrors[i_g1] = std::sqrt(smoothObservedRadialVelocityVariance);

            gsl_vector_free(aux);
            gsl_matrix_free(Ainv);
        });

    for (std::size_t i_g = 0; i_g < consideredObservationsNumber; ++i_g)
    {
        gsl_vector_free(r_vector[i_g]);
    }
}