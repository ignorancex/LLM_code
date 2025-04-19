/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: June 2024
 **/

#include <map>
#include <string>
#include <vector>

#include <TMath.h>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_K3pi.h>
#include <ParametersCharmCombo.h>

namespace {
  std::map<std::string, std::map<theory_config, std::string>> theory_expressions = {
      {"c1",
       {
           {theory_config::phenomenological,
            "- k_K3pi * 0.5 * (      (qop+1) * (y*cos(Delta_K3pi - phi) + x*sin(Delta_K3pi - phi)) "
            "                  + 1 / (qop+1) * (y*cos(Delta_K3pi + phi) + x*sin(Delta_K3pi + phi)))"},
           {theory_config::theoretical,
            "-k_K3pi * (y12 * cos(Delta_K3pi) * cos(phiG) + x12 * sin(Delta_K3pi) * cos(phiM))"},
           {theory_config::superweak, "-k_K3pi * (y12 * cos(Delta_K3pi) + x12 * sin(Delta_K3pi) * cos(phiM))"},
       }},
      {"c2",
       {
           {theory_config::phenomenological, "(x * x + y * y) / 4"},
           {theory_config::theoretical, "(x12 * x12 + y12 * y12) / 4"},
           {theory_config::superweak, "(x12 * x12 + y12 * y12) / 4"},
       }},
  };
}

PDF_K3pi::PDF_K3pi(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{3}, th_cfg{th_cfg} {
  name = "K3pi_" + measurement_id;
  initParameters();
  initRelations();
  initObservables("LHCb R1");
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  build();
}

PDF_K3pi::~PDF_K3pi() {}

void PDF_K3pi::initParameters() {
  std::vector<std::string> param_names = {"r_K3pi", "k_K3pi", "Delta_K3pi"};
  if (th_cfg == theory_config::phenomenological)
    param_names.insert(param_names.end(), {"x", "y", "qop", "phi"});
  else
    param_names.insert(param_names.end(), {"x12", "y12", "phiM", "phiG"});
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  for (const auto& par : param_names) parameters->add(*(p.get(par)));
}

void PDF_K3pi::initRelations() {
  theory = new RooArgList("theory");  ///< the order of this list must match that of the COR matrix!
  theory->add(*(Utils::makeTheoryVar("r_K3pi_th", "r_K3pi_th", "r_K3pi", parameters)));
  theory->add(*(Utils::makeTheoryVar("c1_th", "c1_th", theory_expressions["c1"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("c2_th", "c2_th", theory_expressions["c2"][th_cfg], parameters)));
}

void PDF_K3pi::initObservables(const TString& label) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("r_K3pi_obs", label + "   #it{r_{K3#pi}}", 0, -1e4, 1e4)));
  observables->add(*(new RooRealVar("c1_obs", label + "   #it{#kappa_{K3#pi}y'}", 0, -1e4, 1e4)));
  observables->add(*(new RooRealVar("c2_obs", label + "   (#it{x}^{2}+#it{y}^{2})/4", 0, -1e4, 1e4)));
}

void PDF_K3pi::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("LHCb-run1")) {
    obsValSource = "https://arxiv.org/abs/1602.07224v2";
    setObservable("r_K3pi_obs", 5.67);
    setObservable("c1_obs", 0.03);
    setObservable("c2_obs", 0.48);
  } else {
    std::cout << "PDF_K3pi::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_K3pi::setUncertainties(TString c) {
  if (c.EqualTo("LHCb-run1")) {
    obsErrSource = "https://arxiv.org/abs/1602.07224v2";
    StatErr[0] = 0.12;
    StatErr[1] = 0.18;
    StatErr[2] = 0.18;
    SystErr[0] = 0;
    SystErr[1] = 0;
    SystErr[2] = 0;
  } else {
    std::cout << "PDF_K3pi::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_K3pi::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("LHCb-run1")) {
    corSource = "https://arxiv.org/abs/1602.07224v2";
    std::vector<double> corrs = {
        // clang-format off
        1., 0.91, 0.80,  // r
            1.,   0.94,  // c1
                  1.,    // c2
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, corrs);
  } else {
    std::cout << "PDF_K3pi::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_K3pi::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
