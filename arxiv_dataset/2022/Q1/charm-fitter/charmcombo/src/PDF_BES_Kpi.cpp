/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: September 2022
 **/

#include <boost/algorithm/string.hpp>
#include <vector>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_BES_Kpi.h>
#include <ParametersCharmCombo.h>

PDF_BES_Kpi::PDF_BES_Kpi(const theory_config& th_cfg) : PDF_Abs{4}, th_cfg{th_cfg} {
  name = "charm-bes-kpi";
  initParameters();
  initRelations();
  initObservables();
  setObservables("3fb");
  setUncertainties("3fb");
  setCorrelations("3fb");
  buildCov();
  buildPdf();
}

PDF_BES_Kpi::~PDF_BES_Kpi() {}

void PDF_BES_Kpi::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("R_Kpi")));
  parameters->add(*(p.get("Delta_Kpi")));
  parameters->add(*(p.get("F_pipipi0")));
  switch (th_cfg) {
  case theory_config::phenomenological:
    parameters->add(*(p.get("y")));
    break;
  case theory_config::theoretical:
    parameters->add(*(p.get("phiG")));
    [[fallthrough]];
  case theory_config::superweak:
    parameters->add(*(p.get("x12")));
    parameters->add(*(p.get("y12")));
    parameters->add(*(p.get("phiM")));
    break;
  default:
    std::cout << "PDF_BES_Kpi::initParameters : ERROR : theory_config not supported." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi::initRelations() {
  theory = new RooArgList("theory");
  std::string a_kpi_formula = "(2 * 10 * sqrt(R_Kpi) * cos(Delta_Kpi) + y) / (1 + R_Kpi/100)";
  std::string a_kpi_pipipi0_formula =
      "F_pipipi0 * (2 * 10 * sqrt(R_Kpi) * cos(Delta_Kpi) + y)"
      "/ (1 + R_Kpi/100. + (1 - F_pipipi0) * (-2 * sqrt(R_Kpi/100.) * cos(Delta_Kpi) + y/100.))";
  switch (th_cfg) {
  case theory_config::phenomenological:
    break;
  case theory_config::theoretical:
    boost::replace_all(a_kpi_formula, "y", CharmUtils::y_to_theoretical);
    boost::replace_all(a_kpi_pipipi0_formula, "y", CharmUtils::y_to_theoretical);
    break;
  case theory_config::superweak:
    boost::replace_all(a_kpi_formula, "y", CharmUtils::y_to_superweak);
    boost::replace_all(a_kpi_pipipi0_formula, "y", CharmUtils::y_to_superweak);
    break;
  default:
    std::cout << "PDF_BES_Kpi_1d::initRelations : ERROR : theory_config not supported." << std::endl;
    exit(1);
  }
  using Utils::makeTheoryVar;
  theory->add(*(makeTheoryVar("A_kpi_th", "A_kpi_th", a_kpi_formula, parameters)));
  theory->add(*(makeTheoryVar("A_kpi_pipipi0_th", "A_kpi_pipipi0_th", a_kpi_pipipi0_formula, parameters)));
  theory->add(*(makeTheoryVar("rcos_th", "rcos_th", "-10 * sqrt(R_Kpi)*cos(Delta_Kpi)", parameters)));
  theory->add(*(makeTheoryVar("rsin_th", "rsin_th", " 10 * sqrt(R_Kpi)*sin(Delta_Kpi)", parameters)));
}

void PDF_BES_Kpi::initObservables() {
  observables = new RooArgList("observables");
  observables->add(*(new RooRealVar("A_kpi_obs", "#it{A_{K#pi}}", 0., -100., 100.)));
  observables->add(*(new RooRealVar("A_kpi_pipipi0_obs", "#it{A_{K#pi}^{#pi#pi#pi^{0}}}", 0., -150., 150)));
  observables->add(*(new RooRealVar("rcos_obs", "#minus#it{r_{D}^{K#pi}}cos#it{#Delta_{D}^{K#pi}}", 0., -20., 20.)));
  observables->add(*(new RooRealVar("rsin_obs", "#it{r_{D}^{K#pi}}sin#it{#Delta_{D}^{K#pi}}", 0., -20., 20.)));
}

void PDF_BES_Kpi::setObservables(TString c) {
  if (c.EqualTo("3fb")) {
    obsValSource = "https://arxiv.org/pdf/2208.09402v2.pdf";
    setObservable("A_kpi_obs", 13.2);
    setObservable("A_kpi_pipipi0_obs", 13.0);
    setObservable("rcos_obs", -5.62);
    setObservable("rsin_obs", -1.1);
  } else {
    std::cout << "PDF_BES_Kpi::setObservables() : ERROR : obs config " << c << " not found." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi::setUncertainties(TString c) {
  if (c.EqualTo("3fb")) {
    obsErrSource = "https://arxiv.org/pdf/2208.09402v2.pdf";
    StatErr[0] = 1.1;
    SystErr[0] = 0.7;
    StatErr[1] = 1.2;
    SystErr[1] = 0.8;
    // The stat. errs. for rcos and rsin include both the statistical and systematic components
    StatErr[2] = std::sqrt(std::pow(0.81, 2) + std::pow(0.50, 2) + std::pow(0.10, 2));
    SystErr[2] = 0.;
    StatErr[3] = std::sqrt(std::pow(1.2, 2) + std::pow(0.7, 2) + std::pow(0.3, 2));
    SystErr[3] = 0.;
  } else {
    std::cout << "PDF_BES_Kpi::setObservables() : ERROR : err config " << c << " not found." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("3fb")) {
    corSource = "https://arxiv.org/pdf/2208.09402v2.pdf";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.38, 0., 0.,    // A_Kpi
            1.,   0., 0.,    // A_Kpi^pipipi0
                  1., 0.02,  // rcos
                      1.,    // rsin
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
    std::vector<double> dataSyst = {
        // clang-format off
        1., 0.16, 0., 0.,  // A_Kpi
            1.,   0., 0.,  // A_Kpi^pipipi0
                  1., 0.,  // rcos
                      1.,  // rsin
        // clang-format on
    };
    corSystMatrix = Utils::buildCorMatrix(nObs, dataSyst);
  } else {
    std::cout << "PDF_BES_Kpi::setCorrelations() : ERROR : cor config " << c << " not found." << std::endl;
    exit(1);
  }
}

void PDF_BES_Kpi::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
