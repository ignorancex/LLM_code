/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <map>
#include <vector>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_WS_NoCPV.h>
#include <ParametersCharmCombo.h>

namespace {
  // Map containing the expressions for the observables in the various parametrisations
  std::map<std::string, std::map<theory_config, std::string>> theory_expressions = {
      {"y'",
       {
           {theory_config::phenomenological, "y*cos(Delta_Kpi) + x*sin(Delta_Kpi)"},
           {theory_config::theoretical, "  y12 * cos(Delta_Kpi) * TMath::Sign(1.,cos(phiG)) "
                                        "+ x12 * sin(Delta_Kpi) * TMath::Sign(1.,cos(phiM))"},
           {theory_config::superweak, "  y12 * cos(Delta_Kpi)                            "
                                      "+ x12 * sin(Delta_Kpi) * TMath::Sign(1.,cos(phiM))"},
           {theory_config::d0_to_kpi, "yp"},
       }},
      {"x'2",
       {
           {theory_config::phenomenological, "pow(x*cos(Delta_Kpi) - y*sin(Delta_Kpi),2)"},
           {theory_config::theoretical, "pow(- y12*sin(Delta_Kpi) * TMath::Sign(1.,cos(phiG))"
                                        "    + x12*cos(Delta_Kpi) * TMath::Sign(1.,cos(phiM)), 2)"},
           {theory_config::superweak, "pow(- y12*sin(Delta_Kpi)                            "
                                      "    + x12*cos(Delta_Kpi) * TMath::Sign(1.,cos(phiM)), 2)"},
           {theory_config::d0_to_kpi, "xp2"},
       }},
  };
}  // namespace

PDF_WS_NoCPV::PDF_WS_NoCPV(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{3}, th_cfg{th_cfg} {
  name = measurement_id + "_WS_NoCPV";
  initParameters();
  initRelations();
  initObservables(measurement_id);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  build();
}

PDF_WS_NoCPV::~PDF_WS_NoCPV() {}

void PDF_WS_NoCPV::initParameters() {
  std::vector<std::string> param_names = {"R_Kpi"};
  if (th_cfg != theory_config::d0_to_kpi) param_names.emplace_back("Delta_Kpi");
  switch (th_cfg) {
  case theory_config::phenomenological:
    param_names.insert(param_names.end(), {"x", "y", "qop", "phi"});
    break;
  case theory_config::theoretical:
    param_names.emplace_back("phiG");
  case theory_config::superweak:
    param_names.insert(param_names.end(), {"x12", "y12", "phiM"});
    break;
  case theory_config::d0_to_kpi:
    param_names.insert(param_names.end(), {"yp", "xp2"});
    break;
  default:
    std::cout << "PDF_WS_NoCPV::initParameters : ERROR : theory_config " << th_cfg << " not supported." << std::endl;
    exit(1);
  }
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  for (const auto& par : param_names) parameters->add(*(p.get(par)));
}

void PDF_WS_NoCPV::initRelations() {
  theory = new RooArgList("theory");
  theory->add(*(Utils::makeTheoryVar("RD_th", "RD_th", "R_Kpi", parameters)));
  theory->add(*(Utils::makeTheoryVar("yp_th", "yp_th", theory_expressions["y'"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("xp2_th", "xp2_th", theory_expressions["x'2"][th_cfg], parameters)));
}

void PDF_WS_NoCPV::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("RD_obs", setName + "   #it{R_{K#pi}}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("yp_obs", setName + "   #it{y'}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("xp2_obs", setName + "   #it{x'}^{2}", 0., -1e4, 1e4)));
}

void PDF_WS_NoCPV::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("CDF")) {
    obsValSource = "https://inspirehep.net/literature/1254229";
    setObservable("RD_obs", 0.351);
    setObservable("yp_obs", 0.43);
    setObservable("xp2_obs", 0.8);
  } else if (c.EqualTo("BaBar")) {
    obsValSource = "https://inspirehep.net/literature/746245";
    setObservable("RD_obs", 0.303);
    setObservable("yp_obs", 0.97);
    setObservable("xp2_obs", -2.2);
  } else if (c.EqualTo("Belle")) {
    obsValSource = "https://inspirehep.net/literature/1277238";
    setObservable("RD_obs", 0.353);
    setObservable("yp_obs", 0.46);
    setObservable("xp2_obs", 0.9);
  } else {
    std::cout << "PDF_WS_NoCPV::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS_NoCPV::setUncertainties(TString c) {
  if (c.EqualTo("CDF")) {
    obsErrSource = "https://inspirehep.net/literature/1254229";
    StatErr[0] = 0.035;  // RD
    StatErr[1] = 0.43;   // y'
    StatErr[2] = 1.8;    // x'2
    SystErr[0] = 0;      // RD
    SystErr[1] = 0;      // y'
    SystErr[2] = 0;      // x'2
  } else if (c.EqualTo("BaBar")) {
    obsErrSource = "https://inspirehep.net/literature/746245";
    StatErr[0] = pow(pow(0.016, 2) + pow(0.010, 2), 0.5);  // RD
    StatErr[1] = pow(pow(0.44, 2) + pow(0.31, 2), 0.5);    // y'+
    StatErr[2] = pow(pow(3.0, 2) + pow(2.1, 2), 0.5);      // x'2+
    SystErr[0] = 0;                                        // RD
    SystErr[1] = 0;                                        // y'+
    SystErr[2] = 0;                                        // x'2+
  } else if (c.EqualTo("Belle")) {
    obsErrSource = "https://inspirehep.net/literature/1277238";
    StatErr[0] = 0.013;  // RD
    StatErr[1] = 0.34;   // y'
    StatErr[2] = 2.2;    // x'2
    SystErr[0] = 0;      // RD
    SystErr[1] = 0;      // y'
    SystErr[2] = 0;      // x'2
  } else {
    std::cout << "PDF_WS_NoCPV::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS_NoCPV::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("CDF")) {
    corSource = "https://inspirehep.net/literature/1254229";
    std::cout << "INFO [PDF_WS_NoCPV]: The correlation matrix of https://inspirehep.net/literature/1254229\n"
                 "                     (CDF) is not positive definite. It has been modified to avoid non convergence.\n"
              << std::endl;
    std::vector<double> dataStat = {
        // clang-format off
        1., -0.967, 0.900,  // RD
             1.,   -0.975,  // y'
                    1.      // x'2
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else if (c.EqualTo("BaBar")) {
    corSource = "https://inspirehep.net/literature/746245";
    std::vector<double> dataStat = {
        // clang-format off
        1., -0.87, 0.77,  // RD
             1.,  -0.94,  // y'
                   1.     // x'2
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else if (c.EqualTo("Belle")) {
    corSource = "https://inspirehep.net/literature/1277238";
    std::vector<double> dataStat = {
        // clang-format off
        1., -0.865, 0.737,  // RD
             1.,   -0.948,  // y'
                    1.      // x'2
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else {
    std::cout << "PDF_WS_NoCPV::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS_NoCPV::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
