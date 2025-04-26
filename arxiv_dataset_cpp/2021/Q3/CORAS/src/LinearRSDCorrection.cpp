#include "LinearRSDCorrection.hpp"

#include <fstream>
#include <iostream>
#include <tr1/cmath>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "CQUADIntegrator.hpp"
#include "miscellaneous.hpp"
#include "SphericalFourierBesselDecomposition.hpp"
#include "transformations.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

LinearRSDCorrection::LinearRSDCorrection(const double maxRadius, const double radialResolution, const std::size_t maxMultipole,
                                         const std::function<double(double)> &selectionFunctionLogDerivative, const std::function<double(double)> &selectionTimesWeightingFunction, const ReferenceFrame referenceFrame)
    : MaxRadius(maxRadius),
      RadialResolution(radialResolution),
      MaxMultipole(maxMultipole),
      MaxRadialWavenumber(SphericalFourierBesselDecomposition(maxRadius, radialResolution, maxMultipole).maximal_radial_wavenumber()),
      RadialModeNumber(SphericalFourierBesselDecomposition(maxRadius, radialResolution, maxMultipole).radial_mode_number()),
      ReducedCouplingMatrix((MaxMultipole + 1) * RadialModeNumber * RadialModeNumber, 0.0) // n_max^2 entries for each multipole order 0 <= l <= l_max
{
    if ((referenceFrame != CMB_FRAME) and (referenceFrame != LOCAL_GROUP_FRAME))
    {
        std::cout << std::endl
                  << " LinearRSDCorrection Error: Reference frame has to be either CMB_FRAME or LOCAL_GROUP_FRAME" << std::endl
                  << std::endl;

        exit(EXIT_FAILURE);
    }

    compute_coupling_matrix(selectionFunctionLogDerivative, selectionTimesWeightingFunction, referenceFrame);
}

LinearRSDCorrection::LinearRSDCorrection(const std::string &fileName)
{
    load_object_from_file(fileName);
}

LinearRSDCorrection::LinearRSDCorrection()
    : MaxRadius(0.0),
      RadialResolution(0.0),
      MaxMultipole(0),
      MaxRadialWavenumber(0.0),
      RadialModeNumber(1),
      ReducedCouplingMatrix()
{
}

void LinearRSDCorrection::transform_redshift_to_configuration_space(SphericalFourierBesselDecomposition &densityContrastSFBDecomposition, const double beta) const
{
    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
        gsl_vector *redshiftSpaceSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *redshiftSpaceSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *configurationSpaceSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *configurationSpaceSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);

        gsl_matrix *couplingMatrix = gsl_matrix_alloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix_set_identity(couplingMatrix); // ensures correct inversion for l-modes where the number of non-vanishing radial Fourier modes is below RadialModeNumber
        gsl_matrix *inverseCouplingMatrix = gsl_matrix_calloc(RadialModeNumber, RadialModeNumber);

#pragma omp parallel for schedule(static) collapse(2)
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1) // multiply precomputed reduced coupling matrix with beta and add the identity matrix to obtain the full coupling matrix
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = densityContrastSFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = densityContrastSFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    continue;
                }

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                const double couplingMatrixElement = beta * ReducedCouplingMatrix[i_ln1n2] + (n1 == n2 ? 1.0 : 0.0);

                gsl_matrix_set(couplingMatrix, n1 - 1, n2 - 1, couplingMatrixElement);
            }
        }

        invert_matrix_svd(couplingMatrix, inverseCouplingMatrix);

        for (std::size_t m = 0; m <= l; ++m) // multiply inverse coupling matrix with SFB coefficent vector
        {
            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                std::complex<double> redshiftSpaceDensityContrastMode = densityContrastSFBDecomposition.SFB_coefficient(l, m, n);

                gsl_vector_set(redshiftSpaceSFBCoefficientsRe, n - 1, std::real(redshiftSpaceDensityContrastMode));
                gsl_vector_set(redshiftSpaceSFBCoefficientsIm, n - 1, std::imag(redshiftSpaceDensityContrastMode));
            }

            gsl_blas_dgemv(CblasNoTrans, 1.0, inverseCouplingMatrix, redshiftSpaceSFBCoefficientsRe, 0.0, configurationSpaceSFBCoefficientsRe);
            gsl_blas_dgemv(CblasNoTrans, 1.0, inverseCouplingMatrix, redshiftSpaceSFBCoefficientsIm, 0.0, configurationSpaceSFBCoefficientsIm);

            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                const std::complex<double> configurationSpaceDensityContrastMode(gsl_vector_get(configurationSpaceSFBCoefficientsRe, n - 1), gsl_vector_get(configurationSpaceSFBCoefficientsIm, n - 1));

                densityContrastSFBDecomposition.SFB_coefficient(l, m, n) = configurationSpaceDensityContrastMode;
            }
        }

        gsl_matrix_free(couplingMatrix);
        gsl_matrix_free(inverseCouplingMatrix);

        gsl_vector_free(redshiftSpaceSFBCoefficientsRe);
        gsl_vector_free(redshiftSpaceSFBCoefficientsIm);
        gsl_vector_free(configurationSpaceSFBCoefficientsRe);
        gsl_vector_free(configurationSpaceSFBCoefficientsIm);
    }
}

void LinearRSDCorrection::transform_configuration_to_redshift_space(SphericalFourierBesselDecomposition &densityContrastSFBDecomposition, const double beta) const
{
    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
        gsl_vector *redshiftSpaceSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *redshiftSpaceSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *configurationSpaceSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *configurationSpaceSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);

        gsl_matrix *couplingMatrix = gsl_matrix_alloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix_set_identity(couplingMatrix); // ensures correct inversion for l-modes where the number of non-vanishing radial Fourier modes is below RadialModeNumber

#pragma omp parallel for schedule(static) collapse(2)
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1) // multiply the precomputed reduced coupling matrix with beta and add the identity matrix to obtain the full coupling matrix
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = densityContrastSFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = densityContrastSFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    continue;
                }

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                const double couplingMatrixElement = beta * ReducedCouplingMatrix[i_ln1n2] + (n1 == n2 ? 1.0 : 0.0);

                gsl_matrix_set(couplingMatrix, n1 - 1, n2 - 1, couplingMatrixElement);
            }
        }

        for (std::size_t m = 0; m <= l; ++m) // multiply the coupling matrix with the SFB coefficent vector
        {
            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                std::complex<double> configurationSpaceDensityContrastMode = densityContrastSFBDecomposition.SFB_coefficient(l, m, n);

                gsl_vector_set(configurationSpaceSFBCoefficientsRe, n - 1, std::real(configurationSpaceDensityContrastMode));
                gsl_vector_set(configurationSpaceSFBCoefficientsIm, n - 1, std::imag(configurationSpaceDensityContrastMode));
            }

            gsl_blas_dgemv(CblasNoTrans, 1.0, couplingMatrix, configurationSpaceSFBCoefficientsRe, 0.0, redshiftSpaceSFBCoefficientsRe);
            gsl_blas_dgemv(CblasNoTrans, 1.0, couplingMatrix, configurationSpaceSFBCoefficientsIm, 0.0, redshiftSpaceSFBCoefficientsIm);

            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                const std::complex<double> redshiftSpaceDensityContrastMode(gsl_vector_get(redshiftSpaceSFBCoefficientsRe, n - 1), gsl_vector_get(redshiftSpaceSFBCoefficientsIm, n - 1));

                densityContrastSFBDecomposition.SFB_coefficient(l, m, n) = redshiftSpaceDensityContrastMode;
            }
        }

        gsl_matrix_free(couplingMatrix);

        gsl_vector_free(redshiftSpaceSFBCoefficientsRe);
        gsl_vector_free(redshiftSpaceSFBCoefficientsIm);
        gsl_vector_free(configurationSpaceSFBCoefficientsRe);
        gsl_vector_free(configurationSpaceSFBCoefficientsIm);
    }
}

void LinearRSDCorrection::save_object_to_file(const std::string &fileName) const
{
    SphericalFourierBesselDecomposition SFBDecomposition(MaxRadius, RadialResolution, MaxMultipole); // zero-valued SFB decomposition used to access radial wavenumbers

    std::ofstream outputFile(fileName, std::ios::binary);

    outputFile.write((char *)&MaxRadius, sizeof(double));
    outputFile.write((char *)&RadialResolution, sizeof(double));
    outputFile.write((char *)&MaxMultipole, sizeof(std::size_t));

    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1)
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = SFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = SFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    break;
                }

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                outputFile.write((char *)&ReducedCouplingMatrix[i_ln1n2], sizeof(double));
            }
        }
    }

    outputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void LinearRSDCorrection::compute_coupling_matrix(const std::function<double(double)> &selectionFunctionLogDerivative, const std::function<double(double)> &selectionTimesWeightingFunction, const ReferenceFrame referenceFrame)
{
    SphericalFourierBesselDecomposition SFBDecomposition(MaxRadius, RadialResolution, MaxMultipole); // zero-valued SFB decomposition used to access radial wavenumbers and normalization constants

    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
#pragma omp parallel for schedule(static) collapse(2)
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1) // compute the reduced radial coupling matrix by integrating over the respective radial integral kernel
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = SFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = SFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    continue;
                }

                const double C_ln2 = SFBDecomposition.radial_normalization_constant(l, n2);

                std::function<double(double)> integrand;

                if (referenceFrame == CMB_FRAME)
                {
                    integrand = [&](double r)
                    {
                        return coupling_matrix_kernel(l, k_ln1, k_ln2, r, selectionFunctionLogDerivative, selectionTimesWeightingFunction, false);
                    };
                }
                else if (referenceFrame == LOCAL_GROUP_FRAME)
                {
                    integrand = [&](double r)
                    {
                        return coupling_matrix_kernel(l, k_ln1, k_ln2, r, selectionFunctionLogDerivative, selectionTimesWeightingFunction, true);
                    };
                }

                CQUADIntegrator integrator(integrand);

                double reducedCouplingMatrixElement = 0.0;

                integrator.integrate(0.0, MaxRadius,
                                     0.0, 1e-5,
                                     reducedCouplingMatrixElement);

                reducedCouplingMatrixElement *= -C_ln2;

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                ReducedCouplingMatrix[i_ln1n2] = reducedCouplingMatrixElement;
            }
        }
    }
}

void LinearRSDCorrection::load_object_from_file(const std::string &fileName)
{
    std::ifstream inputFile(fileName, std::ios::binary);

    if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
    {
        std::cout << std::endl
                  << " LinearRSDCorrection::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
                  << std::endl;

        exit(EXIT_FAILURE);
    }

    inputFile.read((char *)&MaxRadius, sizeof(double));
    inputFile.read((char *)&RadialResolution, sizeof(double));
    inputFile.read((char *)&MaxMultipole, sizeof(std::size_t));

    SphericalFourierBesselDecomposition SFBDecomposition(MaxRadius, RadialResolution, MaxMultipole); // zero-valued SFB decomposition used to access radial wavenumbers

    MaxRadialWavenumber = SFBDecomposition.maximal_radial_wavenumber();
    RadialModeNumber = SFBDecomposition.radial_mode_number();

    ReducedCouplingMatrix = std::vector<double>((MaxMultipole + 1) * RadialModeNumber * RadialModeNumber, 0.0); // n_max^2 entries for each multipole order 0 <= l <= l_max

    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1)
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = SFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = SFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    break;
                }

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                inputFile.read((char *)&ReducedCouplingMatrix[i_ln1n2], sizeof(double));
            }
        }
    }

    inputFile.close();
}

double LinearRSDCorrection::coupling_matrix_kernel(const std::size_t l, const double k_ln1, const double k_ln2, const double r, const std::function<double(double)> &selectionFunctionLogDerivative, const std::function<double(double)> &selectionTimesWeightingFunction, const bool subtractObserverVelocity) const
{
    if (r == 0.0)
    {
        return 0.0;
    }
    else
    {
        const double k_ln1_r = k_ln1 * r;
        const double k_ln2_r = k_ln2 * r;

        const double j_ln1 = std::tr1::sph_bessel(l, k_ln1_r);
        const double j_ln2 = std::tr1::sph_bessel(l, k_ln2_r);
        const double j_lPlus1n2 = std::tr1::sph_bessel(l + 1, k_ln2_r);
        const double j_ln2_Deriv = j_ln2 * static_cast<double>(l) / k_ln2_r - j_lPlus1n2;
        const double j_ln2_Deriv2 = j_ln2 * (static_cast<double>(l) * (static_cast<double>(l) - 1.0) / k_ln2_r / k_ln2_r - 1.0) + j_lPlus1n2 * 2.0 / k_ln2_r;

        const double observerVelocityDipoleContribution = (subtractObserverVelocity and l == 1) ? (1.0 - std::tr1::sph_bessel(0, k_ln2 * MaxRadius)) / k_ln2_r / 3.0
                                                                                                : 0.0;

        return r * r * selectionTimesWeightingFunction(r) * j_ln1 * (j_ln2_Deriv2 + (j_ln2_Deriv / k_ln2_r - observerVelocityDipoleContribution) * (2.0 + selectionFunctionLogDerivative(r)));
    }
}