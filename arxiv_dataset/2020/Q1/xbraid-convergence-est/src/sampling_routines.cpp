#include "sampling_routines.hpp"

/**
 *  sample spatial eigenvalues \f$\delta_t \eta\f$ in the complex plane
 *  and compute the respective complex eigenvalues \f$\lambda\f$ of the \f$\Phi\f$ operator
 *  for a given Runge-Kutta method
*/
void sample_complex_plane(const int method,                     ///< Runge-Kutta method, see constants.hpp
                          double minreal,                       ///< minimum real value of \f$z = \delta_t \eta\f$
                          double maxreal,                       ///< maximum real value of \f$z = \delta_t \eta\f$
                          double minimag,                       ///< minimum complex value of \f$z = \delta_t \eta\f$
                          double maximag,                       ///< maximum complex part of \f$z = \delta_t \eta\f$
                          int numberOfRealSamples,              ///< number of points to sample real axis
                          int numberOfImagSamples,              ///< number of points to sample complex axis
                          arma::Col<arma::cx_double> *&z,       ///< on return, the sampled spatial eigenvalues \f$z = \delta_t \eta\f$; (note: we are passing a reference to a pointer)
                          arma::Col<arma::cx_double> *&lambda   ///< on return, the complex eigenvalues of the \f$\Phi\f$ operator
                          ){
    // sampling points
    z = new arma::Col<arma::cx_double>(numberOfRealSamples*numberOfImagSamples);
    (*z).fill({0.0,0.0});
    arma::Col<double>           realSamples = arma::linspace<arma::vec>(minreal,maxreal,numberOfRealSamples);
    arma::Col<double>           imagSamples = arma::linspace<arma::vec>(minimag,maximag,numberOfImagSamples);
    // sample complex plane
    int idx = 0;
    for(int realidx = 0; realidx < numberOfRealSamples; realidx++){
        for(int imagIdx = 0; imagIdx < numberOfImagSamples; imagIdx++){
            (*z)(idx++) = {realSamples(realidx),imagSamples(imagIdx)};
        }
    }
    // compute eigenvalues of \f$\Phi\f$ operator
    lambda = new arma::Col<arma::cx_double>(numberOfRealSamples*numberOfImagSamples);
    stability_function(method, z, lambda);
}