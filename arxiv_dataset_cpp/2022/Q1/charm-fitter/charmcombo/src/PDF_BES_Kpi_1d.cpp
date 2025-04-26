/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <boost/algorithm/string.hpp>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_BES_Kpi_1d.h>
#include <ParametersCharmCombo.h>

PDF_BES_Kpi_1d::PDF_BES_Kpi_1d(const theory_config& th_cfg) : PDF_Abs{1}, th_cfg{th_cfg} {
  name = "BES";
  initParameters();
  initRelations();
  initObservables(name);
  setObservables("BES");
  setUncertainties("BES");
  setCorrelations("BES");
  buildCov();
  buildPdf();
}

PDF_BES_Kpi_1d::~PDF_BES_Kpi_1d() {}

void PDF_BES_Kpi_1d::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("Delta_Kpi")));
  parameters->add(*(p.get("R_Kpi")));

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
    std::cout << "PDF_BES_Kpi_1d::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi_1d::initRelations() {
  theory = new RooArgList("theory");
  std::string a_kpi_formula = "(2 * 10 * sqrt(R_Kpi) * cos(Delta_Kpi) + y) / (1 + R_Kpi/100)";
  switch (th_cfg) {
  case theory_config::phenomenological:
    break;
  case theory_config::theoretical:
    boost::replace_all(a_kpi_formula, "y", CharmUtils::y_to_theoretical);
    break;
  case theory_config::superweak:
    boost::replace_all(a_kpi_formula, "y", CharmUtils::y_to_superweak);
    break;
  default:
    std::cout << "PDF_BES_Kpi_1d::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
  theory->add(*(Utils::makeTheoryVar("A_kpi_th", "A_kpi_th", a_kpi_formula, parameters)));
}

void PDF_BES_Kpi_1d::initObservables(const TString& setName) {
  observables = new RooArgList("observables");
  observables->add(*(new RooRealVar("A_kpi_obs", setName + "   #it{A_{K#pi}^{CP}}", 0., -1e4, 1e4)));
}

void PDF_BES_Kpi_1d::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("BES")) {
    obsValSource = "http://inspirehep.net/record/1291279";
    setObservable("A_kpi_obs", 12.7);
  } else {
    std::cout << "PDF_BES_Kpi_1d::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi_1d::setUncertainties(TString c) {
  if (c.EqualTo("BES")) {
    obsErrSource = "http://inspirehep.net/record/1291279";
    StatErr[0] = 1.3;
    SystErr[0] = 0.7;
  } else {
    std::cout << "PDF_BES_Kpi_1d::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi_1d::setCorrelations(TString c) {
  resetCorrelations();
  corSource = "No correlations for one observable";
}

void PDF_BES_Kpi_1d::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
