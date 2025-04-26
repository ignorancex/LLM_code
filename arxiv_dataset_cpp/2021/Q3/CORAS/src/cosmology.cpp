#include "cosmology.hpp"

#include <cmath>
#include <functional>

#include <gsl/gsl_math.h>

double expansion_function(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return std::sqrt(omegaMatter * gsl_pow_3(1.0 + z) + 1.0 - omegaMatter);
}

double proper_distance(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return HUBBLE_RADIUS * (z - (1.0 / 2.0 + 3.0 / 4.0 * omegaMatter) * z * z + (1.0 / 3.0 + 9.0 / 8.0 * omegaMatter * omegaMatter) * z * z * z);
}

double comoving_distance(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return HUBBLE_RADIUS * (z - 3.0 / 4.0 * omegaMatter * z * z + (-1.0 / 2.0 + 9.0 / 8.0 * omegaMatter) * omegaMatter * z * z * z);
}

double luminosity_distance(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return HUBBLE_RADIUS * (z + (1.0 - 3.0 / 4.0 * omegaMatter) * z * z + (-5.0 / 4.0 + 9.0 / 8.0 * omegaMatter) * omegaMatter * z * z * z);
}

double proper_distance_redshift_velocity_derivative(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return (1.0 - (1.0 + 3.0 / 2.0 * omegaMatter) * z + (1.0 + 27.0 / 8.0 * omegaMatter * omegaMatter) * z * z) / HUBBLE_NORMALIZATION;
}

double comoving_distance_redshift_velocity_derivative(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return (1.0 - 3.0 / 2.0 * omegaMatter * z + (-3.0 / 2.0 + 27.0 / 8.0 * omegaMatter) * omegaMatter * z * z) / HUBBLE_NORMALIZATION;
}

double luminosity_distance_redshift_velocity_derivative(const double redshiftVelocity, const double omegaMatter)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return (1.0 + (2.0 - 3.0 / 2.0 * omegaMatter) * z + (-15.0 / 4.0 + 27.0 / 8.0 * omegaMatter) * omegaMatter * z * z) / HUBBLE_NORMALIZATION;
}

double redshift_velocity_from_proper_distance(const double properDistance, const double omegaMatter)
{
  const double d = properDistance / HUBBLE_RADIUS;

  return SPEED_OF_LIGHT * (d + (1.0 / 2.0 + 3.0 / 4.0 * omegaMatter) * d * d + (1.0 / 6.0 + 3.0 / 2.0 * omegaMatter) * d * d * d);
}

double redshift_velocity_from_comoving_distance(const double comovingDistance, const double omegaMatter)
{
  const double d = comovingDistance / HUBBLE_RADIUS;

  return SPEED_OF_LIGHT * (d + 3.0 / 4.0 * omegaMatter * d * d + 1.0 / 2.0 * omegaMatter * d * d * d);
}

double redshift_velocity_from_luminosity_distance(const double luminosityDistance, const double omegaMatter)
{
  const double d = luminosityDistance / HUBBLE_RADIUS;

  return SPEED_OF_LIGHT * (d - (1.0 - 3.0 / 4.0 * omegaMatter) * d * d + (2.0 - 7.0 / 4.0 * omegaMatter) * d * d * d);
}

double distance_modulus(const double redshiftVelocity, const double omegaMatter, const double hubble)
{
  return 25.0 + 5.0 * std::log10(luminosity_distance(redshiftVelocity, omegaMatter) / hubble);
}

double distance_modulus_redshift_velocity_derivative(const double redshiftVelocity, const double omegaMatter)
{
  return 5.0 / std::log(10.0) * luminosity_distance_redshift_velocity_derivative(redshiftVelocity, omegaMatter) / luminosity_distance(redshiftVelocity, omegaMatter);
}

double k_correction_2MRS(const double redshiftVelocity)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return -2.1 * z; // Bell et al. ApJS 149 (2003) 289
}

double luminosity_evolution_correction_2MRS(const double redshiftVelocity)
{
  const double z = redshiftVelocity / SPEED_OF_LIGHT;

  return 0.8 * z; // Bell et al. ApJS 149 (2003) 289
}

double absolute_magnitude(const double cosmologicalRedshiftVelocity, const double observedRedshiftVelocity, const double apparentMagnitude, const double omegaMatter, const double hubble,
                          const std::function<double(double cosmologicalRedshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double observedRedshiftVelocity)> &kCorrection)
{
  return apparentMagnitude - distance_modulus(cosmologicalRedshiftVelocity, omegaMatter, hubble) - kCorrection(observedRedshiftVelocity) + luminosityEvolutionCorrection(cosmologicalRedshiftVelocity);
}

double apparent_magnitude(const double cosmologicalRedshiftVelocity, const double observedRedshiftVelocity, const double absoluteMagnitude, const double omegaMatter, const double hubble,
                          const std::function<double(double cosmologicalRedshiftVelocity)> &luminosityEvolutionCorrection, const std::function<double(double observedRedshiftVelocity)> &kCorrection)
{
  return absoluteMagnitude + distance_modulus(cosmologicalRedshiftVelocity, omegaMatter, hubble) + kCorrection(observedRedshiftVelocity) - luminosityEvolutionCorrection(cosmologicalRedshiftVelocity);
}