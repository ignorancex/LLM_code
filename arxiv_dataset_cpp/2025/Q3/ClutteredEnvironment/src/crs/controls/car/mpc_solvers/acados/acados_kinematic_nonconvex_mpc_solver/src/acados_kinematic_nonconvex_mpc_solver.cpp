#include "acados_kinematic_nonconvex_mpc_solver/acados_kinematic_nonconvex_mpc_solver.h"

#include <commons/geometry_utils.h>
#include <commons/obstacle.h>
#include "mpc_solvers/kinematic_nonconvex_mpc_solver.h"
#include "serialize_helpers.h"
#include "acados_solver_multiphase_ocp.h"
#include <chrono>

namespace mpc_solvers
{
namespace kinematic_solvers
{
using crs_controls::Obstacle;
using geometry::Polygon;
using geometry::Vertex;

enum phase_id
{
  KBM = 0,
  SEG = 1,
};

struct AcadosNonconvexSolver::Private
{
  //////////////////// INITIALIZATION ////////////////////
  // Horizon lengths
  const mpc_nonconvex_solver_config config_;

  Private(const mpc_nonconvex_solver_config& config) : config_{ config }
  {
    reset_solvers();
  }

  //////////////////// SETTING PARAMETERS ////////////////////

  void setParams(const kbm_params& kbm_params, const seg_params& seg_params)
  {
    // Note: this is very fragile. Must match the exact ordering in "generate_acados_nonconvex_solver_mpc.py"
    std::vector<double> kbm_array = {
      kbm_params.costs.Q1,
      kbm_params.costs.Q2,
      kbm_params.costs.R1,
      kbm_params.costs.R2,
      kbm_params.car_params->lr,
      kbm_params.car_params->lf,
      kbm_params.car_params->a,
      kbm_params.car_params->b,
      kbm_params.car_params->tau,
      kbm_params.car_params->car_width,
      kbm_params.car_params->car_length,
      kbm_params.car_params->min_dist_to_obstacle,
    };

    std::vector<double> seg_array = {
      seg_params.x_target,
      seg_params.y_target,
      seg_params.car_params->car_width,
      seg_params.car_params->car_length,
      seg_params.car_params->min_dist_to_obstacle,
      seg_params.car_params->additional_buffer,
    };

    std::vector<double> obstacle_array;
    for (const std::unique_ptr<Obstacle>& obstacle : *kbm_params.obstacles)
    {
      std::vector<Vertex> vertices;
      obstacle->getVertices(vertices);
      for (const Vertex& vertex : vertices)
      {
        obstacle_array.push_back(vertex[0]);
        obstacle_array.push_back(vertex[1]);
      }
    }

    kbm_array.insert(kbm_array.end(), obstacle_array.begin(), obstacle_array.end());
    seg_array.insert(seg_array.end(), obstacle_array.begin(), obstacle_array.end());

    assert((int)kbm_array.size() == getNumParams(phase_id::KBM));
    assert((int)seg_array.size() == getNumParams(phase_id::SEG));

    for (int stage = 0; stage < getHorizonLength(phase_id::KBM); stage++)
    {
      setKbmParams(stage, kbm_array.data());
    }

    for (int stage = getHorizonLength(phase_id::KBM) + 1; stage <= getTotalHorizonLength(); ++stage)
    {
      setSegParams(stage, seg_array.data());
    }
  }

  void updateTarget(double x, double y)
  {
    for (int i = getHorizonLength(phase_id::KBM) + 1; i <= getTotalHorizonLength(); ++i)
    {
      setParam(i, 0, x);
      setParam(i, 1, y);
    }
  }

  //////////////////// SETTING STATES AND INPUT ////////////////////

  /**
   * @brief Set the Initial State Constraint. The provided array must have the same length as the state dimension
   *
   * @param constraint
   */
  void setInitialState(const kbm_state& state)
  {
    static double buffer[1024];
    fill_array(state, buffer);

    setConstraint(0, "lbx", buffer);
    setConstraint(0, "ubx", buffer);
  }

  void setGuesses(const solver_solution& guess)
  {
    static const int BUFFER_SIZE = 1024;
    static double buffer[BUFFER_SIZE] = { 0 };

    fill_array(guess.kbm_states[0], buffer);

    for (size_t i = 0; i < guess.kbm_states.size(); ++i)
    {
      fill_array(guess.kbm_states[i], &buffer[0]);
      fill_array(guess.ref, &buffer[6]);
      setStateGuess(i, buffer);
    }

    for (size_t i = 0; i < guess.kbm_inputs.size(); ++i)
    {
      fill_array(guess.kbm_inputs[i], buffer);
      setInputGuess(i, buffer);
    }

    for (size_t i = 0; i < guess.seg_states.size(); ++i)
    {
      fill_array(guess.seg_states[i], buffer);
      setStateGuess(getHorizonLength(phase_id::KBM) + 1 + i, buffer);
    }

    for (size_t i = 0; i < guess.seg_inputs.size(); ++i)
    {
      fill_array(guess.seg_inputs[i], buffer);
      setInputGuess(getHorizonLength(phase_id::KBM) + 1 + i, buffer);
    }
  }

  //////////////////// SOLVING ////////////////////

  /**
   * @brief Solves the optimization problems and stores the solution in x and u.
   *
   * @param x State array or point with size N*StateDimenstion
   * @param u Input array or point with size N*Inputdimension
   * @return const int, return code. If no error occurred, return code is zero
   */
  int solve(const kbm_state& initial_state)
  {
    static double buffer[1024];
    fill_array(initial_state, buffer);

    setConstraint(0, "lbx", buffer);
    setConstraint(0, "ubx", buffer);

    // phase = 2;
    // ocp_nlp_solver_opts_set(acados_ocp_capsule_->nlp_config, acados_ocp_capsule_->nlp_opts, "rti_phase", &phase);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int result = multiphase_ocp_acados_solve(acados_ocp_capsule_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[NONCONVEX_COMPARISON] Solver: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    return result;
  }

  // Get the cost from the most recent solve. Assumes the solver was just invoked
  double computeCost(void)
  {
    ocp_nlp_eval_cost(acados_ocp_capsule_->nlp_solver, acados_ocp_capsule_->nlp_in, acados_ocp_capsule_->nlp_out);
    double cost = 0;
    ocp_nlp_get(acados_ocp_capsule_->nlp_config, acados_ocp_capsule_->nlp_solver, "cost_value", &cost);
    return cost;
  }

  //////////////////// READING THE SOLUTION ////////////////////

  void getSolution(solver_solution& solution) const
  {
    static const int BUFFER_SIZE = 1024;
    static double buffer[BUFFER_SIZE] = { 0 };

    for (size_t i = 0; i < solution.kbm_states.size(); ++i)
    {
      getStateSolution(i, buffer);
      read_array(solution.kbm_states[i], buffer);
    }

    solution.ref = solution.kbm_states.back();

    for (size_t i = 0; i < solution.kbm_inputs.size(); ++i)
    {
      getInputSolution(i, buffer);
      read_array(solution.kbm_inputs[i], buffer);
    }

    for (size_t i = 0; i < solution.seg_states.size(); ++i)
    {
      getStateSolution(getHorizonLength(phase_id::KBM) + 1 + i, buffer);
      read_array(solution.seg_states[i], buffer);
    }

    for (size_t i = 0; i < solution.seg_inputs.size(); ++i)
    {
      getInputSolution(getHorizonLength(phase_id::KBM) + 1 + i, buffer);
      read_array(solution.seg_inputs[i], buffer);
    }
  }

  //////////////////// DEBUGGING ////////////////////

  void printFieldHelp(const char* f) const
  {
    double scratch[1000];
    for (int i = 0; i < getTotalHorizonLength(); ++i)
    {
      std::cout << f << "_" << i << std::endl;
      int size = ocp_nlp_dims_get_from_attr(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), i, f);
      assert(size < 1000);
      ocp_nlp_out_get(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), i, f, scratch);
      for (int j = 0; j < size; ++j)
      {
        std::cout << scratch[j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  void printDebugInfo() const
  {
    printFieldHelp("x");
    printFieldHelp("u");
    printFieldHelp("lam");
  }

  //////////////////// COMMON HELPER FUNCTIONS ////////////////////

  /**
   * @brief Get the Horizon Length
   *
   * @return const int
   */
  int getHorizonLength(phase_id phase) const
  {
    switch (phase)
    {
      case phase_id::KBM:
        return config_.horizon_length;
      case phase_id::SEG:
        return config_.num_segments;
      default:
        std::cerr << "Unrecognized phase " << phase << std::endl;
        abort();
    }
    return -1;
  }

  int getTotalHorizonLength(void) const
  {
    return nlp_dims_->N;
  }

  int getStateDimension(phase_id phase) const
  {
    switch (phase)
    {
      case phase_id::KBM:
        return acados_ocp_capsule_->nlp_dims->nx[0];
      case phase_id::SEG:
        return acados_ocp_capsule_->nlp_dims->nx[getHorizonLength(phase_id::KBM) + 1];
      default:
        std::cerr << "Unrecognized phase " << phase << std::endl;
        abort();
    }
    return -1;
  }

  int getInputDimension(phase_id phase) const
  {
    switch (phase)
    {
      case phase_id::KBM:
        return acados_ocp_capsule_->nlp_dims->nu[0];
      case phase_id::SEG:
        return acados_ocp_capsule_->nlp_dims->nu[getHorizonLength(phase_id::KBM) + 1];
      default:
        std::cerr << "Unrecognized phase " << phase << std::endl;
        abort();
    }
    return -1;
  }

  int getNumParams(phase_id phase) const
  {
    switch (phase)
    {
      case phase_id::KBM:
        return acados_ocp_capsule_->nlp_dims->np[0];
      case phase_id::SEG:
        return acados_ocp_capsule_->nlp_dims->np[getHorizonLength(phase_id::KBM) + 1];
      default:
        std::cerr << "Unrecognized phase " << phase << std::endl;
        abort();
    }
    return -1;
  }

  //////////////////// ACADOS WRAPPERS ////////////////////
  // TODO: Much of this code is duplicated. It would be nice to
  // abstact into an acados wrapper package

  // Acados solver fields
  std::unique_ptr<multiphase_ocp_solver_capsule> acados_ocp_capsule_;
  std::unique_ptr<ocp_nlp_config> nlp_config_;
  std::unique_ptr<ocp_nlp_dims> nlp_dims_;
  std::unique_ptr<ocp_nlp_in> nlp_in_;
  std::unique_ptr<ocp_nlp_out> nlp_out_;
  std::unique_ptr<ocp_nlp_solver> nlp_solver_;

  void setKbmParams(int stage, double params[])
  {
    multiphase_ocp_acados_update_params(acados_ocp_capsule_.get(), stage, params, getNumParams(phase_id::KBM));
  }

  void setSegParams(int stage, double params[])
  {
    multiphase_ocp_acados_update_params(acados_ocp_capsule_.get(), stage, params, getNumParams(phase_id::SEG));
  }

  void setParam(int stage, int idx, double p)
  {
    multiphase_ocp_acados_update_params_sparse(acados_ocp_capsule_.get(), stage, &idx, &p, 1);
  }

  /**
   * @brief Sets an initial guess for the state at stage "stage" of the solver.
   * The provided array must have the same length as the state dimension
   *
   * @param constraint
   */
  void setStateGuess(int stage, double constraint[])
  {
    ocp_nlp_out_set(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), stage, "x", constraint);
  }

  /**
   * @brief Sets an initial guess for the input at stage "stage" of the solver.
   * The provided array must have the same length as the input dimension
   *
   * @param constraint
   */
  void setInputGuess(int stage, double constraint[])
  {
    ocp_nlp_out_set(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), stage, "u", constraint);
  }

  /**
   * @brief Sets an input bound constraint.
   *  This function just wraps the ocp_nlp_constraints_model_set call
   *
   * @param stage current stage of the mpc (0,....,horizon-1)
   * @param type type either "x" or "u"
   * @param constraint constraints must have same dimension as x or u (depending on type)
   */
  void setConstraint(int stage, std::string type, double constraint[])
  {
    ocp_nlp_constraints_model_set(nlp_config_.get(), nlp_dims_.get(), nlp_in_.get(), stage, type.c_str(), constraint);
  }

  void getStateSolution(int stage, double buffer[]) const
  {
    assert(stage <= getTotalHorizonLength());
    ocp_nlp_out_get(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), stage, "x", buffer);
  }

  void getInputSolution(int stage, double buffer[]) const
  {
    assert(stage < getTotalHorizonLength());
    ocp_nlp_out_get(nlp_config_.get(), nlp_dims_.get(), nlp_out_.get(), stage, "u", buffer);
  }

  void reset_solvers()
  {
    acados_ocp_capsule_.reset(multiphase_ocp_acados_create_capsule());
    multiphase_ocp_acados_create(acados_ocp_capsule_.get());

    nlp_config_.reset(multiphase_ocp_acados_get_nlp_config(acados_ocp_capsule_.get()));
    nlp_dims_.reset(multiphase_ocp_acados_get_nlp_dims(acados_ocp_capsule_.get()));
    nlp_in_.reset(multiphase_ocp_acados_get_nlp_in(acados_ocp_capsule_.get()));
    nlp_out_.reset(multiphase_ocp_acados_get_nlp_out(acados_ocp_capsule_.get()));
    nlp_solver_.reset(multiphase_ocp_acados_get_nlp_solver(acados_ocp_capsule_.get()));

    multiphase_ocp_acados_reset(acados_ocp_capsule_.get(), 1);
  }
};

//////////////////// STUBS ////////////////////

AcadosNonconvexSolver::AcadosNonconvexSolver(const mpc_nonconvex_solver_config& config)
{
  impl_ = std::make_unique<Private>(config);
}
AcadosNonconvexSolver::~AcadosNonconvexSolver()
{
}

void AcadosNonconvexSolver::setParams(const kbm_params& kbm_params, const seg_params& seg_params)
{
  return impl_->setParams(kbm_params, seg_params);
}
void AcadosNonconvexSolver::updateTarget(double x, double y)
{
  return impl_->updateTarget(x, y);
}
void AcadosNonconvexSolver::setInitialState(const kbm_state& state)
{
  return impl_->setInitialState(state);
}
void AcadosNonconvexSolver::setGuesses(const solver_solution& guess)
{
  return impl_->setGuesses(guess);
}
int AcadosNonconvexSolver::solve(const kbm_state& initial_state)
{
  return impl_->solve(initial_state);
}
double AcadosNonconvexSolver::computeCost(void)
{
  return impl_->computeCost();
}
void AcadosNonconvexSolver::getSolution(solver_solution& solution) const
{
  return impl_->getSolution(solution);
}
void AcadosNonconvexSolver::printDebugInfo() const
{
  return impl_->printDebugInfo();
}

}  // namespace kinematic_solvers

}  // namespace mpc_solvers
