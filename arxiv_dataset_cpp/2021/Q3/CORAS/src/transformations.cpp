#include "transformations.hpp"

#include <cstddef>
#include <iostream>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

void cartesian_cross_product(const double x1Input, const double y1Input, const double z1Input,
                             const double x2Input, const double y2Input, const double z2Input,
                             double &xOutput, double &yOutput, double &zOutput)
{
  xOutput = y1Input * z2Input - z1Input * y2Input;
  yOutput = z1Input * x2Input - x1Input * z2Input;
  zOutput = x1Input * y2Input - y1Input * x2Input;
}

double cartesian_scalar_product(const double x1, const double y1, const double z1,
                                const double x2, const double y2, const double z2)
{
  return x1 * x2 + y1 * y2 + z1 * z2;
}

double cartesian_norm(const double x, const double y, const double z)
{
  return std::sqrt(x * x + y * y + z * z);
}

void spherical_cross_product(const double radius1Input, const double theta1Input, const double phi1Input,
                             const double radius2Input, const double theta2Input, const double phi2Input,
                             double &radiusOutput, double &thetaOutput, double &phiOutput)
{
  double x1Input, y1Input, z1Input,
      x2Input, y2Input, z2Input,
      xOutput, yOutput, zOutput;

  transform_spherical_to_cartesian_coordinates(radius1Input, theta1Input, phi1Input,
                                               x1Input, y1Input, z1Input);

  transform_spherical_to_cartesian_coordinates(radius2Input, theta2Input, phi2Input,
                                               x2Input, y2Input, z2Input);

  cartesian_cross_product(x1Input, y1Input, z1Input,
                          x2Input, y2Input, z2Input,
                          xOutput, yOutput, zOutput);

  transform_cartesian_to_spherical_coordinates(xOutput, yOutput, zOutput,
                                               radiusOutput, thetaOutput, phiOutput);
}

double spherical_scalar_product(const double radius1, const double theta1, const double phi1,
                                const double radius2, const double theta2, const double phi2)
{
  double x1, y1, z1,
      x2, y2, z2;

  transform_spherical_to_cartesian_coordinates(radius1, theta1, phi1,
                                               x1, y1, z1);

  transform_spherical_to_cartesian_coordinates(radius2, theta2, phi2,
                                               x2, y2, z2);

  return cartesian_scalar_product(x1, y1, z1,
                                  x2, y2, z2);
}

double spherical_norm(const double radius, const double theta, const double phi)
{
  double x, y, z;

  transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                               x, y, z);

  return cartesian_norm(x, y, z);
}

void celestial_cross_product(const double radius1Input, const double latitude1Input, const double longitude1Input,
                             const double radius2Input, const double latitude2Input, const double longitude2Input,
                             double &radiusOutput, double &latitudeOutput, double &longitudeOutput)
{
  double x1Input, y1Input, z1Input,
      x2Input, y2Input, z2Input,
      xOutput, yOutput, zOutput;

  transform_celestial_to_cartesian_coordinates(radius1Input, latitude1Input, longitude1Input,
                                               x1Input, y1Input, z1Input);

  transform_celestial_to_cartesian_coordinates(radius2Input, latitude2Input, longitude2Input,
                                               x2Input, y2Input, z2Input);

  cartesian_cross_product(x1Input, y1Input, z1Input,
                          x2Input, y2Input, z2Input,
                          xOutput, yOutput, zOutput);

  transform_cartesian_to_celestial_coordinates(xOutput, yOutput, zOutput,
                                               radiusOutput, latitudeOutput, longitudeOutput);
}

double celestial_scalar_product(const double radius1, const double latitude1, const double longitude1,
                                const double radius2, const double latitude2, const double longitude2)
{
  double x1, y1, z1,
      x2, y2, z2;

  transform_celestial_to_cartesian_coordinates(radius1, latitude1, longitude1,
                                               x1, y1, z1);

  transform_celestial_to_cartesian_coordinates(radius2, latitude2, longitude2,
                                               x2, y2, z2);

  return cartesian_scalar_product(x1, y1, z1,
                                  x2, y2, z2);
}

double celestial_norm(const double radius, const double latitude, const double longitude)
{
  double x, y, z;

  transform_celestial_to_cartesian_coordinates(radius, latitude, longitude,
                                               x, y, z);

  return cartesian_norm(x, y, z);
}

void transform_spherical_to_cartesian_coordinates(const double radius, const double theta, const double phi,
                                                  double &x, double &y, double &z)
{
  x = radius * std::sin(theta) * std::cos(phi);
  y = radius * std::sin(theta) * std::sin(phi);
  z = radius * std::cos(theta);
}

void transform_cartesian_to_spherical_coordinates(const double x, const double y, const double z,
                                                  double &radius, double &theta, double &phi)
{
  radius = std::sqrt(x * x + y * y + z * z);
  theta = radius > 0.0 ? std::acos(z / radius) : 0.0;
  phi = std::fmod(std::atan2(y, x) + 2.0 * M_PI, 2.0 * M_PI);
}

void transform_celestial_to_spherical_coordinates(const double latitude, const double longitude,
                                                  double &theta, double &phi)
{
  theta = (90.0 - latitude) * RadiansPerDegree;
  phi = longitude * RadiansPerDegree;
}

void transform_spherical_to_celestial_coordinates(const double theta, const double phi,
                                                  double &latitude, double &longitude)
{
  latitude = 90.0 - theta / RadiansPerDegree;
  longitude = phi / RadiansPerDegree;
}

void transform_celestial_to_cartesian_coordinates(const double radius, const double latitude, const double longitude,
                                                  double &x, double &y, double &z)
{
  double theta, phi;

  transform_celestial_to_spherical_coordinates(latitude, longitude,
                                               theta, phi);

  transform_spherical_to_cartesian_coordinates(radius, theta, phi,
                                               x, y, z);
}

void transform_cartesian_to_celestial_coordinates(const double x, const double y, const double z,
                                                  double &radius, double &latitude, double &longitude)
{
  double theta, phi;

  transform_cartesian_to_spherical_coordinates(x, y, z,
                                               radius, theta, phi);

  transform_spherical_to_celestial_coordinates(theta, phi,
                                               latitude, longitude);
}

void transform_spherical_to_cartesian_vector_field_value(const double theta, const double phi,
                                                         const double radialComponent, const double thetaComponent, const double phiComponent,
                                                         double &xComponent, double &yComponent, double &zComponent)
{
  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);
  const double sinPhi = std::sin(phi);
  const double cosPhi = std::cos(phi);

  xComponent = sinTheta * cosPhi * radialComponent + cosTheta * cosPhi * thetaComponent - sinPhi * phiComponent;
  yComponent = sinTheta * sinPhi * radialComponent + cosTheta * sinPhi * thetaComponent + cosPhi * phiComponent;
  zComponent = cosTheta * radialComponent - sinTheta * thetaComponent;
}

void transform_cartesian_to_spherical_vector_field_value(const double theta, const double phi,
                                                         const double xComponent, const double yComponent, const double zComponent,
                                                         double &radialComponent, double &thetaComponent, double &phiComponent)
{
  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);
  const double sinPhi = std::sin(phi);
  const double cosPhi = std::cos(phi);

  radialComponent = sinTheta * cosPhi * xComponent + sinTheta * sinPhi * yComponent + cosTheta * zComponent;
  thetaComponent = cosTheta * cosPhi * xComponent + cosTheta * sinPhi * yComponent - sinTheta * zComponent;
  phiComponent = -sinPhi * xComponent + cosPhi * yComponent;
}

void transform_spherical_to_cartesian_vector_field(const SphericalGridFunction &radialFieldComponents, const SphericalGridFunction &thetaFieldComponents, const SphericalGridFunction &phiFieldComponents,
                                                   SphericalGridFunction &xFieldComponents, SphericalGridFunction &yFieldComponents, SphericalGridFunction &zFieldComponents)
{
  transform_spherical_grid_functions(radialFieldComponents, thetaFieldComponents, phiFieldComponents,
                                     xFieldComponents, yFieldComponents, zFieldComponents,
                                     [&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
                                         double &xComponent, double &yComponent, double &zComponent)
                                     {
                                       const double theta = radialFieldComponents.theta_coordinate(thetaBin);
                                       const double phi = radialFieldComponents.phi_coordinate(phiBin);

                                       const double radialComponent = radialFieldComponents.value(radialBin, thetaBin, phiBin);
                                       const double thetaComponent = thetaFieldComponents.value(radialBin, thetaBin, phiBin);
                                       const double phiComponent = phiFieldComponents.value(radialBin, thetaBin, phiBin);

                                       transform_spherical_to_cartesian_vector_field_value(theta, phi,
                                                                                           radialComponent, thetaComponent, phiComponent,
                                                                                           xComponent, yComponent, zComponent);
                                     });
}

void transform_cartesian_to_spherical_vector_field(const SphericalGridFunction &xFieldComponents, const SphericalGridFunction &yFieldComponents, const SphericalGridFunction &zFieldComponents,
                                                   SphericalGridFunction &radialFieldComponents, SphericalGridFunction &thetaFieldComponents, SphericalGridFunction &phiFieldComponents)
{
  transform_spherical_grid_functions(xFieldComponents, yFieldComponents, zFieldComponents,
                                     radialFieldComponents, thetaFieldComponents, phiFieldComponents,
                                     [&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
                                         double &radialComponent, double &thetaComponent, double &phiComponent)
                                     {
                                       const double theta = xFieldComponents.theta_coordinate(thetaBin);
                                       const double phi = xFieldComponents.phi_coordinate(phiBin);

                                       const double xComponent = xFieldComponents.value(radialBin, thetaBin, phiBin);
                                       const double yComponent = yFieldComponents.value(radialBin, thetaBin, phiBin);
                                       const double zComponent = zFieldComponents.value(radialBin, thetaBin, phiBin);

                                       transform_cartesian_to_spherical_vector_field_value(theta, phi,
                                                                                           xComponent, yComponent, zComponent,
                                                                                           radialComponent, thetaComponent, phiComponent);
                                     });
}

void transform_spherical_to_cartesian_vector_field(const Cartesian3DGridFunction &radialFieldComponents, const Cartesian3DGridFunction &thetaFieldComponents, const Cartesian3DGridFunction &phiFieldComponents,
                                                   Cartesian3DGridFunction &xFieldComponents, Cartesian3DGridFunction &yFieldComponents, Cartesian3DGridFunction &zFieldComponents)
{
  transform_cartesian_3D_grid_functions(radialFieldComponents, thetaFieldComponents, phiFieldComponents,
                                        xFieldComponents, yFieldComponents, zFieldComponents,
                                        [&](std::size_t xBin, std::size_t yBin, std::size_t zBin,
                                            double &xComponent, double &yComponent, double &zComponent)
                                        {
                                          const double x = radialFieldComponents.x_coordinate(xBin);
                                          const double y = radialFieldComponents.y_coordinate(yBin);
                                          const double z = radialFieldComponents.z_coordinate(zBin);

                                          double radius, theta, phi;

                                          transform_cartesian_to_spherical_coordinates(x, y, z,
                                                                                       radius, theta, phi);

                                          const double radialComponent = radialFieldComponents.value(xBin, yBin, zBin);
                                          const double thetaComponent = thetaFieldComponents.value(xBin, yBin, zBin);
                                          const double phiComponent = phiFieldComponents.value(xBin, yBin, zBin);

                                          transform_spherical_to_cartesian_vector_field_value(theta, phi,
                                                                                              radialComponent, thetaComponent, phiComponent,
                                                                                              xComponent, yComponent, zComponent);
                                        });
}

void transform_cartesian_to_spherical_vector_field(const Cartesian3DGridFunction &xFieldComponents, const Cartesian3DGridFunction &yFieldComponents, const Cartesian3DGridFunction &zFieldComponents,
                                                   Cartesian3DGridFunction &radialFieldComponents, Cartesian3DGridFunction &thetaFieldComponents, Cartesian3DGridFunction &phiFieldComponents)
{
  transform_cartesian_3D_grid_functions(xFieldComponents, yFieldComponents, zFieldComponents,
                                        radialFieldComponents, thetaFieldComponents, phiFieldComponents,
                                        [&](std::size_t xBin, std::size_t yBin, std::size_t zBin,
                                            double &radialComponent, double &thetaComponent, double &phiComponent)
                                        {
                                          const double x = xFieldComponents.x_coordinate(xBin);
                                          const double y = xFieldComponents.y_coordinate(yBin);
                                          const double z = xFieldComponents.z_coordinate(zBin);

                                          double radius, theta, phi;

                                          transform_cartesian_to_spherical_coordinates(x, y, z,
                                                                                       radius, theta, phi);

                                          const double xComponent = xFieldComponents.value(xBin, yBin, zBin);
                                          const double yComponent = yFieldComponents.value(xBin, yBin, zBin);
                                          const double zComponent = zFieldComponents.value(xBin, yBin, zBin);

                                          transform_cartesian_to_spherical_vector_field_value(theta, phi,
                                                                                              xComponent, yComponent, zComponent,
                                                                                              radialComponent, thetaComponent, phiComponent);
                                        });
}

const CoordinateSystemRotation NoCoordinateSystemRotation = {(90.0 - 90.0) * RadiansPerDegree,
                                                             0.0 * RadiansPerDegree,
                                                             (90.0 - 0.0) * RadiansPerDegree,
                                                             0.0 * RadiansPerDegree};

const CoordinateSystemRotation GalacticToEquatorial = {(90.0 - 27.12835) * RadiansPerDegree, // NED Coordinate Calculator
                                                       122.93200 * RadiansPerDegree,
                                                       (90.0 - (-60.18846)) * RadiansPerDegree,
                                                       96.33724 * RadiansPerDegree};

const CoordinateSystemRotation EquatorialToGalactic = {(90.0 - 27.12835) * RadiansPerDegree, // NED Coordinate Calculator
                                                       192.85950 * RadiansPerDegree,
                                                       (90.0 - (-28.93616)) * RadiansPerDegree,
                                                       266.40507 * RadiansPerDegree};

const CoordinateSystemRotation SupergalacticToEquatorial = {(90.0 - 15.70886) * RadiansPerDegree, // NED Coordinate Calculator
                                                            26.45052 * RadiansPerDegree,
                                                            (90.0 - 13.23092) * RadiansPerDegree,
                                                            292.65879 * RadiansPerDegree};

const CoordinateSystemRotation EquatorialToSupergalactic = {(90.0 - 15.70894) * RadiansPerDegree, // NED Coordinate Calculator
                                                            283.75420 * RadiansPerDegree,
                                                            (90.0 - 59.52821) * RadiansPerDegree,
                                                            42.30998 * RadiansPerDegree};

const CoordinateSystemRotation SupergalacticToGalactic = {(90.0 - 6.32000) * RadiansPerDegree, // NED Coordinate Calculator
                                                          90.0 * RadiansPerDegree,
                                                          (90.0 - 42.31029) * RadiansPerDegree,
                                                          185.78611 * RadiansPerDegree};

const CoordinateSystemRotation GalacticToSupergalactic = {(90.0 - 6.32000) * RadiansPerDegree, // NED Coordinate Calculator
                                                          47.37000 * RadiansPerDegree,
                                                          (90.0 - 0.00000) * RadiansPerDegree,
                                                          137.37000 * RadiansPerDegree};

void rotate_cartesian_coordinates(const double xInput, const double yInput, const double zInput,
                                  double &xOutput, double &yOutput, double &zOutput,
                                  const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  const double thetaZ = coordinateSystemRotation.ThetaZ;
  const double phiZ = coordinateSystemRotation.PhiZ;
  const double thetaX = coordinateSystemRotation.ThetaX;
  const double phiX = coordinateSystemRotation.PhiX;

  const double cosThetaZ = std::cos(thetaZ);
  const double sinThetaZ = std::sin(thetaZ);
  const double cosPhiZ = std::cos(phiZ);
  const double sinPhiZ = std::sin(phiZ);
  const double cosThetaX = std::cos(thetaX);
  const double sinThetaX = std::sin(thetaX);
  const double cosPhiX = std::cos(phiX);
  const double sinPhiX = std::sin(phiX);
  const double sinPhiXMinusPhiZ = std::sin(phiX - phiZ);

  const double Rxx = cosPhiX * sinThetaX;
  const double Rxy = sinPhiX * sinThetaX;
  const double Rxz = cosThetaX;
  const double Ryx = sinPhiZ * sinThetaZ * cosThetaX - cosThetaZ * sinPhiX * sinThetaX;
  const double Ryy = cosThetaZ * cosPhiX * sinThetaX - cosPhiZ * sinThetaZ * cosThetaX;
  const double Ryz = sinThetaZ * sinThetaX * sinPhiXMinusPhiZ;
  const double Rzx = cosPhiZ * sinThetaZ;
  const double Rzy = sinPhiZ * sinThetaZ;
  const double Rzz = cosThetaZ;

  if (invert)
  {
    xOutput = Rxx * xInput + Ryx * yInput + Rzx * zInput;
    yOutput = Rxy * xInput + Ryy * yInput + Rzy * zInput;
    zOutput = Rxz * xInput + Ryz * yInput + Rzz * zInput;
  }
  else
  {
    xOutput = Rxx * xInput + Rxy * yInput + Rxz * zInput;
    yOutput = Ryx * xInput + Ryy * yInput + Ryz * zInput;
    zOutput = Rzx * xInput + Rzy * yInput + Rzz * zInput;
  }
}

void rotate_spherical_coordinates(const double thetaInput, const double phiInput,
                                  double &thetaOutput, double &phiOutput,
                                  const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  double xInput, yInput, zInput, xOutput, yOutput, zOutput, radiusOutput;

  constexpr double radiusInput = 1.0;

  transform_spherical_to_cartesian_coordinates(radiusInput, thetaInput, phiInput,
                                               xInput, yInput, zInput);

  rotate_cartesian_coordinates(xInput, yInput, zInput,
                               xOutput, yOutput, zOutput, coordinateSystemRotation, invert);

  transform_cartesian_to_spherical_coordinates(xOutput, yOutput, zOutput,
                                               radiusOutput, thetaOutput, phiOutput);
}

void rotate_celestial_coordinates(const double latitudeInput, const double longitudeInput,
                                  double &latitudeOutput, double &longitudeOutput,
                                  const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  double xInput, yInput, zInput, xOutput, yOutput, zOutput, radiusOutput;

  constexpr double radiusInput = 1.0;

  transform_celestial_to_cartesian_coordinates(radiusInput, latitudeInput, longitudeInput,
                                               xInput, yInput, zInput);

  rotate_cartesian_coordinates(xInput, yInput, zInput,
                               xOutput, yOutput, zOutput, coordinateSystemRotation, invert);

  transform_cartesian_to_celestial_coordinates(xOutput, yOutput, zOutput,
                                               radiusOutput, latitudeOutput, longitudeOutput);
}

void rotate_scalar_field(const SphericalGridFunction &fieldInput, SphericalGridFunction &fieldOutput,
                         const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  transform_spherical_grid_functions(fieldInput,
                                     fieldOutput,
                                     [&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
                                         double &output)
                                     {
                                       const double radius = fieldInput.radial_coordinate(radialBin);
                                       const double theta = fieldInput.theta_coordinate(thetaBin);
                                       const double phi = fieldInput.phi_coordinate(phiBin);

                                       double thetaOld, phiOld;

                                       rotate_spherical_coordinates(theta, phi,
                                                                    thetaOld, phiOld,
                                                                    coordinateSystemRotation, not invert); // invert coordinate change

                                       output = fieldInput(radius, thetaOld, phiOld);
                                     });
}

void rotate_cartesian_vector_field(const SphericalGridFunction &xFieldInput, const SphericalGridFunction &yFieldInput, const SphericalGridFunction &zFieldInput,
                                   SphericalGridFunction &xFieldOutput, SphericalGridFunction &yFieldOutput, SphericalGridFunction &zFieldOutput,
                                   const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  transform_spherical_grid_functions(xFieldInput, yFieldInput, zFieldInput,
                                     xFieldOutput, yFieldOutput, zFieldOutput,
                                     [&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
                                         double &xValue, double &yValue, double &zValue)
                                     {
                                       const double radius = xFieldInput.radial_coordinate(radialBin);
                                       const double theta = xFieldInput.theta_coordinate(thetaBin);
                                       const double phi = xFieldInput.phi_coordinate(phiBin);

                                       double thetaOld, phiOld;

                                       rotate_spherical_coordinates(theta, phi,
                                                                    thetaOld, phiOld,
                                                                    coordinateSystemRotation, not invert); // invert coordinate change

                                       const double xValueOld = xFieldInput(radius, thetaOld, phiOld);
                                       const double yValueOld = yFieldInput(radius, thetaOld, phiOld);
                                       const double zValueOld = zFieldInput(radius, thetaOld, phiOld);

                                       rotate_cartesian_coordinates(xValueOld, yValueOld, zValueOld,
                                                                    xValue, yValue, zValue,
                                                                    coordinateSystemRotation, invert);
                                     });
}

void rotate_spherical_vector_field(const SphericalGridFunction &radialFieldInput, const SphericalGridFunction &thetaFieldInput, const SphericalGridFunction &phiFieldInput,
                                   SphericalGridFunction &radialFieldOutput, SphericalGridFunction &thetaFieldOutput, SphericalGridFunction &phiFieldOutput,
                                   const CoordinateSystemRotation coordinateSystemRotation, const bool invert)
{
  transform_spherical_grid_functions(radialFieldInput, thetaFieldInput, phiFieldInput,
                                     radialFieldOutput, thetaFieldOutput, phiFieldOutput,
                                     [&](std::size_t radialBin, std::size_t thetaBin, std::size_t phiBin,
                                         double &radialValue, double &thetaValue, double &phiValue)
                                     {
                                       const double radius = radialFieldInput.radial_coordinate(radialBin);
                                       const double theta = radialFieldInput.theta_coordinate(thetaBin);
                                       const double phi = radialFieldInput.phi_coordinate(phiBin);

                                       double thetaOld, phiOld;

                                       rotate_spherical_coordinates(theta, phi,
                                                                    thetaOld, phiOld,
                                                                    coordinateSystemRotation, not invert); // invert coordinate change

                                       const double thetaValueOld = thetaFieldInput(radius, thetaOld, phiOld);
                                       const double phiValueOld = phiFieldInput(radius, thetaOld, phiOld);

                                       rotate_spherical_coordinates(thetaValueOld, phiValueOld,
                                                                    thetaValue, phiValue,
                                                                    coordinateSystemRotation, invert);
                                     });
}

CoordinateSystemRotation reference_vector_and_plane_coordinate_system_rotation(const double thetaReferenceInitial, const double phiReferenceInitial, const double thetaPlaneInitial, const double phiPlaneInitial,
                                                                               const double thetaReferenceFinal, const double phiReferenceFinal, const double thetaPlaneFinal, const double phiPlaneFinal)
{
  CoordinateSystemRotation rotateReferenceVector; // first use Rodrigues' formula to rotate the initial onto the final reference vector

  double xZ1, yZ1, zZ1,
      xX1, yX1, zX1;

  const double cosAngle1 = spherical_scalar_product(1.0, thetaReferenceInitial, phiReferenceInitial,
                                                    1.0, thetaReferenceFinal, phiReferenceFinal);

  if (cosAngle1 == -1.0) // if initial and final refernce vectors are antiparallel rotate by angle pi around any axis perpendicular to them
  {
    const double thetaAxis1 = thetaReferenceInitial + M_PI / 2.0; // this is always perpendicular to the initial reference vector
    const double phiAxis1 = phiReferenceInitial;

    double xAxis1, yAxis1, zAxis1;

    transform_spherical_to_cartesian_coordinates(1.0, thetaAxis1, phiAxis1,
                                                 xAxis1, yAxis1, zAxis1);

    xZ1 = 2.0 * xAxis1 * zAxis1; // apply inverse Rodrigues' formula to unit vector in z-direction
    yZ1 = 2.0 * yAxis1 * zAxis1;
    zZ1 = 2.0 * zAxis1 * zAxis1 - 1.0;

    xX1 = 2.0 * xAxis1 * xAxis1 - 1.0; // apply inverse Rodrigues' formula to unit vector in x-direction
    yX1 = 2.0 * yAxis1 * xAxis1;
    zX1 = 2.0 * zAxis1 * xAxis1;
  }
  else // otherwise rotate around the axis perpendicular to both initial and final reference vector by the angle between them
  {
    double radiusAxis1, thetaAxis1, phiAxis1,
        xAxis1, yAxis1, zAxis1;

    spherical_cross_product(1.0, thetaReferenceInitial, phiReferenceInitial,
                            1.0, thetaReferenceFinal, phiReferenceFinal,
                            radiusAxis1, thetaAxis1, phiAxis1);

    transform_spherical_to_cartesian_coordinates(radiusAxis1, thetaAxis1, phiAxis1,
                                                 xAxis1, yAxis1, zAxis1);

    xZ1 = xAxis1 * zAxis1 / (1.0 + cosAngle1) - yAxis1; // apply inverse Rodrigues' formula to unit vector in z-direction
    yZ1 = yAxis1 * zAxis1 / (1.0 + cosAngle1) + xAxis1;
    zZ1 = zAxis1 * zAxis1 / (1.0 + cosAngle1) + cosAngle1;

    xX1 = xAxis1 * xAxis1 / (1.0 + cosAngle1) + cosAngle1; // apply inverse Rodrigues' formula to unit vector in x-direction
    yX1 = yAxis1 * xAxis1 / (1.0 + cosAngle1) - zAxis1;
    zX1 = zAxis1 * xAxis1 / (1.0 + cosAngle1) + yAxis1;
  }

  double radiusZ1, thetaZ1, phiZ1,
      radiusX1, thetaX1, phiX1;

  transform_cartesian_to_spherical_coordinates(xZ1, yZ1, zZ1,
                                               radiusZ1, thetaZ1, phiZ1);

  transform_cartesian_to_spherical_coordinates(xX1, yX1, zX1,
                                               radiusX1, thetaX1, phiX1);

  rotateReferenceVector = {thetaZ1, phiZ1, thetaX1, phiX1};

  double thetaPlaneInitialTransformed, phiPlaneInitialTransformed;

  rotate_spherical_coordinates(thetaPlaneInitial, phiPlaneInitial,
                               thetaPlaneInitialTransformed, phiPlaneInitialTransformed,
                               rotateReferenceVector);

  CoordinateSystemRotation rotatePlane; // now rotate the transformed initial plane vector around the final reference vector until it lies in the plane spanned by the final reference and plane vectors

  double radiusNormalTransformed, thetaNormalTransformed, phiNormalTransformed, // compute the normal vectors of the transformed initial plane and the final plane
      radiusNormalFinal, thetaNormalFinal, phiNormalFinal;

  spherical_cross_product(1.0, thetaPlaneInitialTransformed, phiPlaneInitialTransformed,
                          1.0, thetaReferenceFinal, phiReferenceFinal,
                          radiusNormalTransformed, thetaNormalTransformed, phiNormalTransformed);

  spherical_cross_product(1.0, thetaPlaneFinal, phiPlaneFinal,
                          1.0, thetaReferenceFinal, phiReferenceFinal,
                          radiusNormalFinal, thetaNormalFinal, phiNormalFinal);

  if (radiusNormalTransformed == 0.0 or radiusNormalFinal == 0.0) // if the transformed initial or final plane vector is parallel to the final reference vector no additional rotation is needed
  {
    rotatePlane = {0.0, 0.0, M_PI / 2.0, 0.0}; // identity transformation
  }
  else // otherwise the rotation angle is given by the angle between the normal vectors of the transformed initial plane and the final plane
  {
    double radiusNormalTransformedCrossNormalFinal, thetaNormalTransformedCrossNormalFinal, phiNormalTransformedCrossNormalFinal;

    spherical_cross_product(radiusNormalTransformed, thetaNormalTransformed, phiNormalTransformed,
                            radiusNormalFinal, thetaNormalFinal, phiNormalFinal,
                            radiusNormalTransformedCrossNormalFinal, thetaNormalTransformedCrossNormalFinal, phiNormalTransformedCrossNormalFinal);

    const double angle2 = std::atan2(spherical_scalar_product(radiusNormalTransformedCrossNormalFinal, thetaNormalTransformedCrossNormalFinal, phiNormalTransformedCrossNormalFinal,
                                                              1.0, thetaReferenceFinal, phiReferenceFinal),
                                     spherical_scalar_product(radiusNormalTransformed, thetaNormalTransformed, phiNormalTransformed,
                                                              radiusNormalFinal, thetaNormalFinal, phiNormalFinal));

    const double cosAngle2 = std::cos(angle2);
    const double sinAngle2 = std::sin(angle2);

    double xAxis2, yAxis2, zAxis2;

    transform_spherical_to_cartesian_coordinates(1.0, thetaReferenceFinal, phiReferenceFinal,
                                                 xAxis2, yAxis2, zAxis2);

    const double xZ2 = xAxis2 * zAxis2 * (1.0 - cosAngle2) - yAxis2 * sinAngle2; // apply inverse Rodrigues' formula to unit vector in z-direction
    const double yZ2 = yAxis2 * zAxis2 * (1.0 - cosAngle2) + xAxis2 * sinAngle2;
    const double zZ2 = zAxis2 * zAxis2 * (1.0 - cosAngle2) + cosAngle2;

    const double xX2 = xAxis2 * xAxis2 * (1.0 - cosAngle2) + cosAngle2; // apply inverse Rodrigues' formula to unit vector in x-direction
    const double yX2 = yAxis2 * xAxis2 * (1.0 - cosAngle2) - zAxis2 * sinAngle2;
    const double zX2 = zAxis2 * xAxis2 * (1.0 - cosAngle2) + yAxis2 * sinAngle2;

    double radiusZ2, thetaZ2, phiZ2,
        radiusX2, thetaX2, phiX2;

    transform_cartesian_to_spherical_coordinates(xZ2, yZ2, zZ2,
                                                 radiusZ2, thetaZ2, phiZ2);

    transform_cartesian_to_spherical_coordinates(xX2, yX2, zX2,
                                                 radiusX2, thetaX2, phiX2);

    rotatePlane = {thetaZ2, phiZ2, thetaX2, phiX2};
  }

  double thetaZIntermediate, phiZIntermediate, thetaXIntermediate, phiXIntermediate, // perform inverse transforms to obtain the initial vectors that will be mapped to the final z- and x-axis, respectively, by the total transformation
      thetaZTotal, phiZTotal, thetaXTotal, phiXTotal;

  rotate_spherical_coordinates(0.0, 0.0,
                               thetaZIntermediate, phiZIntermediate,
                               rotatePlane, true);

  rotate_spherical_coordinates(thetaZIntermediate, phiZIntermediate,
                               thetaZTotal, phiZTotal,
                               rotateReferenceVector, true);

  rotate_spherical_coordinates(M_PI / 2.0, 0.0,
                               thetaXIntermediate, phiXIntermediate,
                               rotatePlane, true);

  rotate_spherical_coordinates(thetaXIntermediate, phiXIntermediate,
                               thetaXTotal, phiXTotal,
                               rotateReferenceVector, true);

  return CoordinateSystemRotation({thetaZTotal, phiZTotal, thetaXTotal, phiXTotal});
}

const ReferenceFrameChange NO_REFERENCE_FRAME_CHANGE = {0.0,
                                                        0.0,
                                                        0.0};

const ReferenceFrameChange HELIOCENTRIC_TO_LOCAL_GROUP = {299.0, // [Diaz et al., MNRAS 443 (2014) 1688]
                                                          (90.0 - (-5.9)) * RadiansPerDegree,
                                                          98.4 * RadiansPerDegree};

const ReferenceFrameChange LOCAL_GROUP_TO_HELIOCENTRIC = {-HELIOCENTRIC_TO_LOCAL_GROUP.relativeVelocityAmplitude,
                                                          HELIOCENTRIC_TO_LOCAL_GROUP.relativeVelocityTheta,
                                                          HELIOCENTRIC_TO_LOCAL_GROUP.relativeVelocityPhi};

const ReferenceFrameChange HELIOCENTRIC_TO_CMB = {369.82, // Planck 2018 [Aghanim et al., A&A 641 (2020) A1]
                                                  (90.0 - 48.253) * RadiansPerDegree,
                                                  264.021 * RadiansPerDegree};

const ReferenceFrameChange CMB_TO_HELIOCENTRIC = {-HELIOCENTRIC_TO_CMB.relativeVelocityAmplitude,
                                                  HELIOCENTRIC_TO_CMB.relativeVelocityTheta,
                                                  HELIOCENTRIC_TO_CMB.relativeVelocityPhi};

const ReferenceFrameChange LOCAL_GROUP_TO_CMB = {620.49406093409, // Combining Diaz and Planck (large number of digits for numerical consistency of changes between the three frames)
                                                 (90.0 - 29.617716070363) * RadiansPerDegree,
                                                 271.89078605012 * RadiansPerDegree};

const ReferenceFrameChange CMB_TO_LOCAL_GROUP = {-LOCAL_GROUP_TO_CMB.relativeVelocityAmplitude,
                                                 LOCAL_GROUP_TO_CMB.relativeVelocityTheta,
                                                 LOCAL_GROUP_TO_CMB.relativeVelocityPhi};

std::string get_reference_frame_name(const ReferenceFrame referenceFrame)
{
  switch (referenceFrame)
  {
  case HELIOCENTRIC_FRAME:
  {
    return "HC";
  }
  case LOCAL_GROUP_FRAME:
  {
    return "LG";
  }
  case CMB_FRAME:
  {
    return "CMB";
  }
  default:
  {
    std::cout << std::endl
              << " get_reference_frame_name Error: Invalid choice of reference frame" << std::endl
              << std::endl;

    exit(EXIT_FAILURE);
  }
  }
}

ReferenceFrameChange get_reference_frame_change(const ReferenceFrame inputFrame, const ReferenceFrame outputFrame)
{
  switch (inputFrame)
  {
  case HELIOCENTRIC_FRAME:
  {
    switch (outputFrame)
    {
    case HELIOCENTRIC_FRAME:
    {
      return NO_REFERENCE_FRAME_CHANGE;
    }
    case LOCAL_GROUP_FRAME:
    {
      return HELIOCENTRIC_TO_LOCAL_GROUP;
    }
    case CMB_FRAME:
    {
      return HELIOCENTRIC_TO_CMB;
    }
    }
  }
  case LOCAL_GROUP_FRAME:
  {
    switch (outputFrame)
    {
    case HELIOCENTRIC_FRAME:
    {
      return LOCAL_GROUP_TO_HELIOCENTRIC;
    }
    case LOCAL_GROUP_FRAME:
    {
      return NO_REFERENCE_FRAME_CHANGE;
    }
    case CMB_FRAME:
    {
      return LOCAL_GROUP_TO_CMB;
    }
    }
  }
  case CMB_FRAME:
  {
    switch (outputFrame)
    {
    case HELIOCENTRIC_FRAME:
    {
      return CMB_TO_HELIOCENTRIC;
    }
    case LOCAL_GROUP_FRAME:
    {
      return CMB_TO_LOCAL_GROUP;
    }
    case CMB_FRAME:
    {
      return NO_REFERENCE_FRAME_CHANGE;
    }
    }
  }
  }

  std::cout << std::endl // if no case matches, exit with error
            << " reference_frame_change Error: Invalid input and/or output frame" << std::endl
            << std::endl;

  exit(EXIT_FAILURE);
}

double change_reference_frame(const double velocity, const double theta, const double phi, const ReferenceFrameChange referenceFrameChange, const bool invert)
{
  const double relativeVelocity = invert ? -referenceFrameChange.relativeVelocityAmplitude
                                         : referenceFrameChange.relativeVelocityAmplitude;
  const double relativeVelocityTheta = referenceFrameChange.relativeVelocityTheta;
  const double relativeVelocityPhi = referenceFrameChange.relativeVelocityPhi;

  return velocity + relativeVelocity * (std::sin(theta) * std::sin(relativeVelocityTheta) * std::cos(phi - relativeVelocityPhi) + std::cos(theta) * std::cos(relativeVelocityTheta));
}