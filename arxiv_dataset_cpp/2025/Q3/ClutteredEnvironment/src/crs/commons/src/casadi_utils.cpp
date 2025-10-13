#include "commons/casadi_utils.h"

namespace commons
{
Eigen::VectorXd CasadiFunction::evaluate(const std::vector<const double*>& x) const
{
  casadi::DMVector inputs;
  for (const double* xi : x)
  {
    inputs.push_back(casadi::DM(*xi));
  }

  casadi::DMVector outputs;
  function_.call(inputs, outputs);

  return Eigen::Map<Eigen::VectorXd>(static_cast<std::vector<double>>(outputs[0]).data(), outputs[0].size1());
}

Eigen::MatrixXd CasadiFunction::evaluateJacobian(const std::vector<const double*>& x) const
{
  const size_t dim_f = function_.n_out();
  const size_t dim_x = function_.n_in();

  assert(dim_x == x.size() && "Number of input arguments does not match the function signature.");

  // The CasADi Jacobian function requires BOTH the input and output arguments to be provided. However,
  // in our case the Jacobian is not dependent on the output arguments (we have an explicit formulation).
  // Therefore, we pad the inputs to the Jacobian function with extra zeros to satisfy the function signature.
  // Our formulation is:  y = f(x)  =>  jac(x, y) = df/dx =>  jac(x, 0) = df/dx
  casadi::DMVector jacobian_inputs(dim_x + dim_f);
  for (size_t i = 0; i < dim_x + dim_f; i++)
  {
    if (i < dim_x)
      jacobian_inputs[i] = casadi::DM(*x[i]);
    else
      jacobian_inputs[i] = casadi::DM(0.0);
  }

  // Evaluate the Jacobian
  casadi::DMVector jacobian_outputs;
  jacobian_.call(jacobian_inputs, jacobian_outputs);

  // Convert DM matrix to regular Eigen matrix via a pure double vector
  auto jacobian_vector = std::vector<double>(jacobian_outputs.begin(), jacobian_outputs.end());

  Eigen::MatrixXd jacobian =
      Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(jacobian_vector.data(), dim_f, dim_x);
  return jacobian;
}
}  // namespace commons
