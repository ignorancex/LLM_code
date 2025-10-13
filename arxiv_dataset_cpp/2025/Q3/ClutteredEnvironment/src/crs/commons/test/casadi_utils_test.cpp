#include <Eigen/Core>
#include <gtest/gtest.h>
#include <casadi/casadi.hpp>
#include <vector>

#include "commons/casadi_utils.h"

// Test the CasadiFunction wrapper scalar case for y = x^2.

TEST(CasadiUtilsTest, testQuadraticFunction)
{
  casadi::SX x = casadi::SX::sym("x");
  casadi::SX y = x * x;

  commons::CasadiFunction f = casadi::Function("f", { x }, { y });

  std::vector<double> x_val{ 0.0, 1.0, -1.0, 2.0, -2.0 };      // x
  std::vector<double> y_val{ 0.0, 1.0, 1.0, 4.0, 4.0 };        // y = x*x
  std::vector<double> dy_dx_val{ 0.0, 2.0, -2.0, 4.0, -4.0 };  // dy/dx = 2*x

  for (size_t i = 0; i < x_val.size(); i++)
  {
    std::vector<const double*> arg = { &x_val[i] };

    // Check the function value
    Eigen::VectorXd y = f.evaluate(arg);
    EXPECT_TRUE(y.rows() == 1 && y.cols() == 1);
    EXPECT_NEAR(y(0), y_val[i], 1e-10);

    // Check the derivative
    Eigen::MatrixXd jacobian = f.evaluateJacobian(arg);
    EXPECT_TRUE(jacobian.rows() == 1 && jacobian.cols() == 1);
    EXPECT_NEAR(jacobian(0, 0), dy_dx_val[i], 1e-10);
  }
}

// Test the numerical Jacobian for z = x * y, where J = [y, x].
TEST(CasadiUtilsTest, testMultipleInputJacobian)
{
  casadi::SX x = casadi::SX::sym("x");
  casadi::SX y = casadi::SX::sym("y");
  casadi::SX z = x * y;

  commons::CasadiFunction f = casadi::Function("f", { x, y }, { z });

  std::vector<double> x_val{ 0.0, 1.0, -1.0, 2.0, -2.0 };
  std::vector<double> y_val{ 0.0, 1.0, -1.0, 2.0, -2.0 };
  std::vector<double> f_val{ 0.0, 1.0, 1.0, 4.0, 4.0 };

  for (size_t i = 0; i < x_val.size(); i++)
  {
    std::vector<const double*> arg = { &x_val[i], &y_val[i] };

    // Check function value
    Eigen::VectorXd z = f.evaluate(arg);
    EXPECT_TRUE(z.rows() == 1 && z.cols() == 1);
    EXPECT_NEAR(z(0), f_val[i], 1e-10);

    // Check the Jacobian
    double expected_derivative_x = y_val[i];
    double expected_derivative_y = x_val[i];

    Eigen::MatrixXd jacobian = f.evaluateJacobian(arg);
    EXPECT_TRUE(jacobian.rows() == 1 && jacobian.cols() == 2);
    EXPECT_EQ(jacobian(0, 0), expected_derivative_x);
    EXPECT_EQ(jacobian(0, 1), expected_derivative_y);
  }
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
