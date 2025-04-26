/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <vector>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_Kpipi0.h>
#include <ParametersCharmCombo.h>

PDF_Kpipi0::PDF_Kpipi0(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{2}, th_cfg{th_cfg} {
  name = measurement_id + "_Kpipi0";
  TString label;
  if (measurement_id.EqualTo("BaBar"))
    label = "BaBar #it{K}^{+}#pi^{#minus}#pi^{0}";
  else
    exit(1);
  initParameters();
  initRelations();
  initObservables(label);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_Kpipi0::~PDF_Kpipi0() {}

void PDF_Kpipi0::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("Delta_Kpipi0")));

  switch (th_cfg) {
  case theory_config::phenomenological:
    parameters->add(*(p.get("x")));
    parameters->add(*(p.get("y")));
    break;
  case theory_config::theoretical:
    parameters->add(*(p.get("phiG")));
  case theory_config::superweak:
    parameters->add(*(p.get("x12")));
    parameters->add(*(p.get("y12")));
    parameters->add(*(p.get("phiM")));
    break;
  default:
    std::cout << "PDF_Kpipi0::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_Kpipi0::initRelations() {
  theory = new RooArgList("theory");
  switch (th_cfg) {
  case theory_config::phenomenological:
    theory->add(*(Utils::makeTheoryVar("xpp_th", "xpp_th", "x*cos(Delta_Kpipi0) - y*sin(Delta_Kpipi0)", parameters)));
    theory->add(*(Utils::makeTheoryVar("ypp_th", "ypp_th", "y*cos(Delta_Kpipi0) + x*sin(Delta_Kpipi0)", parameters)));
    break;
  case theory_config::theoretical:
    theory->add(*(Utils::makeTheoryVar("xpp_th", "xpp_th",
                                       "  x12*cos(Delta_Kpipi0) * cos(phiM)"
                                       "- y12*sin(Delta_Kpipi0) * cos(phiG)",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("ypp_th", "ypp_th",
                                       "  y12 * cos(Delta_Kpipi0) * cos(phiG)"
                                       "+ x12 * sin(Delta_Kpipi0) * cos(phiM)",
                                       parameters)));
    break;
  case theory_config::superweak:
    theory->add(*(Utils::makeTheoryVar("xpp_th", "xpp_th",
                                       "  x12*cos(Delta_Kpipi0) * cos(phiM)"
                                       "- y12*sin(Delta_Kpipi0)            ",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("ypp_th", "ypp_th",
                                       "  y12 * cos(Delta_Kpipi0)            "
                                       "+ x12 * sin(Delta_Kpipi0) * cos(phiM)",
                                       parameters)));
    break;
  default:
    std::cout << "PDF_Kpipi0::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_Kpipi0::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("xpp_obs", setName + "   #it{x''}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("ypp_obs", setName + "   #it{y''}", 0., -1e4, 1e4)));
}

void PDF_Kpipi0::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("BaBar")) {
    obsValSource = "https://inspirehep.net/literature/791715";
    setObservable("xpp_obs", 2.61);
    setObservable("ypp_obs", -0.06);
  } else {
    std::cout << "PDF_Kpipi0::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kpipi0::setUncertainties(TString c) {
  if (c.EqualTo("BaBar")) {
    obsErrSource = "https://inspirehep.net/literature/791715";
    StatErr[0] = pow(pow(0.625, 2) + pow(0.39, 2), 0.5);  // x''
    StatErr[1] = pow(pow(0.595, 2) + pow(0.34, 2), 0.5);  // y''
  } else {
    std::cout << "PDF_Kpipi0::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kpipi0::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("BaBar")) {
    obsErrSource = "https://inspirehep.net/literature/791715";
    std::vector<double> dataStat = {1., -0.75, 1.};
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else {
    std::cout << "PDF_Kpipi0::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kpipi0::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
