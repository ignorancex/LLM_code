#include "WienerFilter.hpp"

#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <tr1/cmath>
#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "CQUADIntegrator.hpp"
#include "miscellaneous.hpp"
#include "SphericalFourierBesselDecomposition.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

WienerFilter::WienerFilter(const double maxRadius, const double radialResolution, const std::size_t maxMultipole,
                           const std::function<double(double)> &selectionFunction)
    : MaxRadius(maxRadius),
      RadialResolution(radialResolution),
      MaxMultipole(maxMultipole),
      MaxRadialWavenumber(SphericalFourierBesselDecomposition(maxRadius, radialResolution, maxMultipole).maximal_radial_wavenumber()),
      RadialModeNumber(SphericalFourierBesselDecomposition(maxRadius, radialResolution, maxMultipole).radial_mode_number()),
      ReducedNoiseMatrix((MaxMultipole + 1) * RadialModeNumber * RadialModeNumber, 0.0) // n_max^2 entries for each multipole order 0 <= l <= l_max

{
    compute_noise_matrix(selectionFunction);
}

WienerFilter::WienerFilter(const std::string &fileName)
{
    load_object_from_file(fileName);
}

WienerFilter::WienerFilter()
    : MaxRadius(0.0),
      RadialResolution(0.0),
      MaxMultipole(0),
      MaxRadialWavenumber(0.0),
      RadialModeNumber(1),
      ReducedNoiseMatrix()
{
}

void WienerFilter::apply_to(SphericalFourierBesselDecomposition &densityContrastSFBDecomposition, const double meanDensity, const std::function<double(double)> &dataDataPowerSpectrum, const std::function<double(double)> &signalDataPowerSpectrum) const
{
    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
        gsl_vector *dataSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *dataSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *wienerSFBCoefficientsRe = gsl_vector_calloc(RadialModeNumber);
        gsl_vector *wienerSFBCoefficientsIm = gsl_vector_calloc(RadialModeNumber);

        gsl_matrix *dataDataMatrix = gsl_matrix_alloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix_set_identity(dataDataMatrix); // ensures correct inversion for l-modes where the number of non-vanishing radial Fourier modes is below RadialModeNumber
        gsl_matrix *signalDataMatrix = gsl_matrix_alloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix_set_identity(signalDataMatrix); // ensures correct inversion for l-modes where the number of non-vanishing radial Fourier modes is below RadialModeNumber
        gsl_matrix *noiseMatrix = gsl_matrix_calloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix *dataDataPlusNoiseMatrix = gsl_matrix_calloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix *inverseDataDataPlusNoiseMatrix = gsl_matrix_calloc(RadialModeNumber, RadialModeNumber);
        gsl_matrix *wienerFilterMatrix = gsl_matrix_calloc(RadialModeNumber, RadialModeNumber);

#pragma omp parallel for schedule(static)
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1) // divide the precomputed reduced noise matrix by mean density to obtain the full noise matrix, and compute the diagonal data-data as well as signal-data matrices
        {
            const double k_ln1 = densityContrastSFBDecomposition.radial_wavenumber(l, n1);

            if (k_ln1 > MaxRadialWavenumber)
            {
                continue;
            }

            const double C_ln1 = densityContrastSFBDecomposition.radial_normalization_constant(l, n1);

            const double dataDataMatrixElement = dataDataPowerSpectrum(k_ln1) / C_ln1;
            const double signalDataMatrixElement = signalDataPowerSpectrum(k_ln1) / C_ln1;

            gsl_matrix_set(dataDataMatrix, n1 - 1, n1 - 1, dataDataMatrixElement);
            gsl_matrix_set(signalDataMatrix, n1 - 1, n1 - 1, signalDataMatrixElement);

            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln2 = densityContrastSFBDecomposition.radial_wavenumber(l, n2);

                if (k_ln2 > MaxRadialWavenumber)
                {
                    continue;
                }

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                const double noiseMatrixElement = ReducedNoiseMatrix[i_ln1n2] / meanDensity;

                gsl_matrix_set(noiseMatrix, n1 - 1, n2 - 1, noiseMatrixElement);
            }
        }

        gsl_matrix_memcpy(dataDataPlusNoiseMatrix, dataDataMatrix); // multiply the signal-data matrix with the inverse of the sum of data-data and noise matrix to obtain the Wiener filter matrix
        gsl_matrix_add(dataDataPlusNoiseMatrix, noiseMatrix);

        invert_matrix_svd(dataDataPlusNoiseMatrix, inverseDataDataPlusNoiseMatrix);

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, signalDataMatrix, inverseDataDataPlusNoiseMatrix, 0.0, wienerFilterMatrix);

        for (std::size_t m = 0; m <= l; ++m) // multiply the Wiener filter matrix with the SFB coefficent vector
        {
            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                std::complex<double> dataDensityContrastMode = densityContrastSFBDecomposition.SFB_coefficient(l, m, n);

                gsl_vector_set(dataSFBCoefficientsRe, n - 1, std::real(dataDensityContrastMode));
                gsl_vector_set(dataSFBCoefficientsIm, n - 1, std::imag(dataDensityContrastMode));
            }

            gsl_blas_dgemv(CblasNoTrans, 1.0, wienerFilterMatrix, dataSFBCoefficientsRe, 0.0, wienerSFBCoefficientsRe);
            gsl_blas_dgemv(CblasNoTrans, 1.0, wienerFilterMatrix, dataSFBCoefficientsIm, 0.0, wienerSFBCoefficientsIm);

            for (std::size_t n = 1; n <= RadialModeNumber; ++n)
            {
                const std::complex<double> wienerDensityContrastMode(gsl_vector_get(wienerSFBCoefficientsRe, n - 1), gsl_vector_get(wienerSFBCoefficientsIm, n - 1));

                densityContrastSFBDecomposition.SFB_coefficient(l, m, n) = wienerDensityContrastMode;
            }
        }

        gsl_matrix_free(dataDataMatrix);
        gsl_matrix_free(signalDataMatrix);
        gsl_matrix_free(noiseMatrix);
        gsl_matrix_free(dataDataPlusNoiseMatrix);
        gsl_matrix_free(inverseDataDataPlusNoiseMatrix);
        gsl_matrix_free(wienerFilterMatrix);

        gsl_vector_free(dataSFBCoefficientsRe);
        gsl_vector_free(dataSFBCoefficientsIm);
        gsl_vector_free(wienerSFBCoefficientsRe);
        gsl_vector_free(wienerSFBCoefficientsIm);
    }
}

void WienerFilter::save_object_to_file(const std::string &fileName) const
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

                outputFile.write((char *)&ReducedNoiseMatrix[i_ln1n2], sizeof(double));
            }
        }
    }

    outputFile.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void WienerFilter::compute_noise_matrix(const std::function<double(double)> &selectionFunction)
{
    SphericalFourierBesselDecomposition SFBDecomposition(MaxRadius, RadialResolution, MaxMultipole);

    for (std::size_t l = 0; l <= MaxMultipole; ++l)
    {
#pragma omp parallel for schedule(static) collapse(2)
        for (std::size_t n1 = 1; n1 <= RadialModeNumber; ++n1) // compute the reduced noise matrix by integrating over the respective radial integral kernel
        {
            for (std::size_t n2 = 1; n2 <= RadialModeNumber; ++n2)
            {
                const double k_ln1 = SFBDecomposition.radial_wavenumber(l, n1);
                const double k_ln2 = SFBDecomposition.radial_wavenumber(l, n2);

                if ((k_ln1 > MaxRadialWavenumber) or (k_ln2 > MaxRadialWavenumber))
                {
                    continue;
                }

                std::function<double(double)> integrand = [&](double r)
                {
                    return noise_matrix_kernel(l, k_ln1, k_ln2, r, selectionFunction);
                };

                CQUADIntegrator integrator(integrand);

                double reducedNoiseMatrixElement = 0.0;

                integrator.integrate(0.0, MaxRadius,
                                     0.0, 1e-5,
                                     reducedNoiseMatrixElement);

                std::size_t i_ln1n2 = (l * RadialModeNumber + n1 - 1) * RadialModeNumber + n2 - 1;

                ReducedNoiseMatrix[i_ln1n2] = reducedNoiseMatrixElement;
            }
        }
    }
}

void WienerFilter::load_object_from_file(const std::string &fileName)
{
    std::ifstream inputFile(fileName, std::ios::binary);

    if (inputFile.fail()) // if the file cannot be opened, send an error message and terminate the program
    {
        std::cout << std::endl
                  << " WienerFilter::load_object_from_file Error: File \"" << fileName << "\" cannot be opened" << std::endl
                  << std::endl;

        exit(EXIT_FAILURE);
    }

    inputFile.read((char *)&MaxRadius, sizeof(double));
    inputFile.read((char *)&RadialResolution, sizeof(double));
    inputFile.read((char *)&MaxMultipole, sizeof(std::size_t));

    SphericalFourierBesselDecomposition SFBDecomposition(MaxRadius, RadialResolution, MaxMultipole); // zero-valued SFB decomposition used to access radial wavenumbers

    MaxRadialWavenumber = SFBDecomposition.maximal_radial_wavenumber();
    RadialModeNumber = SFBDecomposition.radial_mode_number();

    ReducedNoiseMatrix = std::vector<double>((MaxMultipole + 1) * RadialModeNumber * RadialModeNumber, 0.0); // n_max^2 entries for each multipole order 0 <= l <= l_max

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

                inputFile.read((char *)&ReducedNoiseMatrix[i_ln1n2], sizeof(double));
            }
        }
    }

    inputFile.close();
}

double WienerFilter::noise_matrix_kernel(const std::size_t l, const double k_ln1, const double k_ln2, const double r, const std::function<double(double)> &selectionFunction) const
{
    const double k_ln1_r = k_ln1 * r;
    const double k_ln2_r = k_ln2 * r;

    const double j_ln1 = std::tr1::sph_bessel(static_cast<std::size_t>(l), k_ln1_r);
    const double j_ln2 = std::tr1::sph_bessel(static_cast<std::size_t>(l), k_ln2_r);

    return r * r * j_ln1 * j_ln2 / selectionFunction(r);
}