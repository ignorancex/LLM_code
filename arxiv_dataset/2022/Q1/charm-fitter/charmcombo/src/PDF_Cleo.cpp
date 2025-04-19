/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_Cleo.h>
#include <ParametersCharmCombo.h>

PDF_Cleo::PDF_Cleo(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{5}, th_cfg{th_cfg} {
  name = "CLEO";
  initParameters();
  initRelations();
  initObservables(name);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_Cleo::~PDF_Cleo() {}

void PDF_Cleo::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("R_Kpi")));
  parameters->add(*(p.get("Delta_Kpi")));

  switch (th_cfg) {
  case theory_config::phenomenological:
    parameters->add(*(p.get("x")));
    parameters->add(*(p.get("y")));
    break;
  case theory_config::theoretical:
    parameters->add(*(p.get("phiG")));
  case theory_config::superweak:
    parameters->add(*(p.get("phiM")));
    parameters->add(*(p.get("x12")));
    parameters->add(*(p.get("y12")));
    break;
  default:
    std::cout << "PDF_Cleo::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_Cleo::initRelations() {
  theory = new RooArgList("theory");
  theory->add(*(Utils::makeTheoryVar("RD_th", "RD_th", "R_Kpi", parameters)));
  switch (th_cfg) {
  case theory_config::phenomenological:
    theory->add(*(Utils::makeTheoryVar("x2_th", "x2_th", "x*x", parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th", "y", parameters)));
    break;
  case theory_config::theoretical:
    theory->add(*(Utils::makeTheoryVar("x2_th", "x2_th",
                                       "0.5 * ("
                                       "      pow(x12,2) - pow(y12,2) "
                                       "    + pow(  pow(pow(x12,2) + pow(y12,2),2) "
                                       "          - pow(2 * x12 * y12 * sin(phiM - phiG),2), 0.5))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th",
                                       "pow(2, -0.5) * pow("
                                       "      pow(y12,2) - pow(x12,2) "
                                       "    + pow(  pow(pow(x12,2) + pow(y12,2),2) "
                                       "          - pow(2 * x12 * y12 * sin(phiM - phiG),2), 0.5), 0.5)",
                                       parameters)));
    break;
  case theory_config::superweak:
    theory->add(*(Utils::makeTheoryVar("x2_th", "x2_th",
                                       "0.5 * ("
                                       "      pow(x12,2) - pow(y12,2) "
                                       "    + pow(  pow(pow(x12,2) + pow(y12,2),2) "
                                       "          - pow(2 * x12 * y12 * sin(phiM),2), 0.5))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th",
                                       "pow(2, -0.5) * pow("
                                       "      pow(y12,2) - pow(x12,2) "
                                       "    + pow(  pow(pow(x12,2) + pow(y12,2),2) "
                                       "          - pow(2 * x12 * y12 * sin(phiM),2), 0.5), 0.5)",
                                       parameters)));
    break;
  default:
    std::cout << "PDF_Cleo::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
  theory->add(*(Utils::makeTheoryVar("cos_th", "cos_th", "cos(Delta_Kpi)", parameters)));
  theory->add(*(Utils::makeTheoryVar("sin_th", "sin_th", "-sin(Delta_Kpi)", parameters)));
}

void PDF_Cleo::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("RD_obs", setName + "   #it{R_{K#pi}}", 0., 0., 1e4)));
  observables->add(*(new RooRealVar("x2_obs", setName + "   #it{x}^{2}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("y_obs", setName + "   #it{y}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("cos_obs", setName + "   cos#Delta_{#it{K#pi}}", 0., -1., 1.)));
  observables->add(*(new RooRealVar("sin_obs", setName + "   #minussin#Delta_{#it{K#pi}}", 0., -1., 1.)));
}

void PDF_Cleo::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("Cleo-c")) {
    obsValSource = "https://inspirehep.net/literature/1189182";
    setObservable("RD_obs", 0.533);
    setObservable("x2_obs", 6.);
    setObservable("y_obs", 4.2);
    setObservable("cos_obs", 0.81);
    setObservable("sin_obs", -0.01);
  } else {
    std::cout << "PDF_Cleo::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Cleo::setUncertainties(TString c) {
  if (c.EqualTo("Cleo-c")) {
    obsErrSource = "https://inspirehep.net/literature/1189182";
    StatErr[0] = sqrt(pow(0.107, 2) + pow(0.045, 2));  // RD
    StatErr[1] = sqrt(pow(23., 2) + pow(11., 2));      // x2
    StatErr[2] = sqrt(pow(2., 2) + pow(1.0, 2));       // y
    StatErr[3] = sqrt(pow(0.20, 2) + pow(0.06, 2));    // cos
    StatErr[4] = sqrt(pow(0.41, 2) + pow(0.04, 2));    // sin
    SystErr[0] = 0;
    SystErr[1] = 0;
    SystErr[2] = 0;
    SystErr[3] = 0;
    SystErr[4] = 0;
  } else {
    std::cout << "PDF_Cleo::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Cleo::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("Cleo-c")) {
    corSource = "https://inspirehep.net/literature/1189182";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.,  0.,  -0.42,  0.01,  // RD
            1., -0.73, 0.39,  0.02,  // x2
                 1.,  -0.53, -0.03,  // y
                       1.,    0.04,  // cos
                              1.     // sin
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else {
    std::cout << "PDF_Cleo::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Cleo::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
