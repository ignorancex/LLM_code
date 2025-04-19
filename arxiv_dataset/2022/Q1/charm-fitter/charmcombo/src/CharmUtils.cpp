#include <boost/algorithm/string.hpp>
#include <iostream>
#include <ostream>
#include <string>

#include <CharmUtils.h>

std::ostream& operator<<(std::ostream& os, const FSC& fsc) {
  switch (fsc) {
  case FSC::none:
    os << "none";
    break;
  case FSC::partial:
    os << "y x a_CP^d term only";
    break;
  case FSC::full:
    os << "full correction including dependence on delta_KK and delta_PP strong phases";
    break;
  default:
    std::cerr << "ERROR: You are trying to apply the '<<' operator to an unsupported FSC type" << std::endl;
    exit(1);
    break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const theory_config& th_cfg) {
  switch (th_cfg) {
  case theory_config::phenomenological:
    os << "phenomenological (x, y, q/p, phi_2)";
    break;
  case theory_config::theoretical:
    os << "theoretical (x12, y12, phiM, phiG)";
    break;
  case theory_config::superweak:
    os << "superweak (x12, y12, phiM, phiG)";
    break;
  case theory_config::d0_to_kpi:
    os << "D0 -> Kpi (yp, xp2, dyp, dxp2)";
    break;
  default:
    std::cerr << "ERROR: You are trying to apply the '<<' operator to an unsupported theory_cfg type" << std::endl;
    exit(1);
  }
  return os;
}

std::string CharmUtils::get_dy_expression(const theory_config th_cfg, const FSC fsc, const std::string fs) {
  std::string expression;
  switch (th_cfg) {
  case theory_config::phenomenological:
    expression = "0.5 * (- y*(qop+1 - 1/(qop+1))*cos(phi)"
                 "       + x*(qop+1 + 1/(qop+1))*sin(phi))";
    switch (fsc) {
    case FSC::none:
      break;
    case FSC::partial:
      expression += " + y * Acp_HH / 100";
      break;
    case FSC::full:
      expression += " + y * Acp_HH / 100 * (1 + x / y * cot_delta_HH)";
      break;
    }
    break;
  case theory_config::theoretical:
  case theory_config::superweak:
    expression = "-x12 * sin(phiM)";
    switch (fsc) {
    case FSC::none:
      break;
    case FSC::partial:
      expression += " + y12 * Acp_HH / 100";
      break;
    case FSC::full:
      expression += " + y12 * Acp_HH / 100 * (1 + x12 / y12 * cot_delta_HH)";
      break;
    }
    break;
  default:
    std::cerr << " ERROR [CharmUtils::get_dy_expression]: theory_config not supported." << std::endl;
    exit(1);
  }
  boost::replace_all(expression, "HH", fs);
  return "(" + expression + ")";
}
