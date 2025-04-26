/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <map>
#include <string>
#include <vector>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_WS.h>
#include <ParametersCharmCombo.h>

namespace {
  // Map containing the expressions for the observables in the various parametrisations
  std::map<std::string, std::map<theory_config, std::string>> theory_expressions = {
      {"y'+",
       {
           {theory_config::phenomenological, "(qop+1)*(  y * cos(Delta_Kpi - phi)"
                                             "         + x * sin(Delta_Kpi - phi))"},
           {theory_config::theoretical, "  y12 * cos(Delta_Kpi + phiG)"
                                        "+ x12 * sin(Delta_Kpi + phiM)"},
           {theory_config::superweak, "  y12 * cos(Delta_Kpi)"
                                      "+ x12 * sin(Delta_Kpi + phiM)"},
           {theory_config::d0_to_kpi, "yp + dyp"},
       }},
      {"y'-",
       {
           {theory_config::phenomenological, "1/(qop+1)*(  y * cos(Delta_Kpi + phi)"
                                             "           + x * sin(Delta_Kpi + phi))"},
           {theory_config::theoretical, "  y12 * cos(Delta_Kpi-phiG)"
                                        "+ x12 * sin(Delta_Kpi-phiM)"},
           {theory_config::superweak, "(  y12 * cos(Delta_Kpi)"
                                      " + x12 * sin(Delta_Kpi-phiM))"},
           {theory_config::d0_to_kpi, "yp - dyp"},
       }},
      {"x'2+",
       {
           {theory_config::phenomenological, "pow((qop+1)*(  x * cos(Delta_Kpi - phi)"
                                             "             - y * sin(Delta_Kpi - phi)),2)"},
           {theory_config::theoretical, "pow(- y12 * sin(Delta_Kpi + phiG)"
                                        "    + x12 * cos(Delta_Kpi + phiM), 2)"},
           {theory_config::superweak, "pow(- y12 * sin(Delta_Kpi)"
                                      "    + x12 * cos(Delta_Kpi + phiM), 2)"},
           {theory_config::d0_to_kpi, "xp2 + dxp2"},
       }},
      {"x'2-",
       {
           {theory_config::phenomenological, "pow(1/(qop+1)*(  x * cos(Delta_Kpi + phi)"
                                             "               - y * sin(Delta_Kpi + phi)),2)"},
           {theory_config::theoretical, "pow(- y12 * sin(Delta_Kpi-phiG)"
                                        "    + x12 * cos(Delta_Kpi-phiM),2)"},
           {theory_config::superweak, "pow((- y12 * sin(Delta_Kpi)"
                                      "     + x12 * cos(Delta_Kpi-phiM)),2)"},
           {theory_config::d0_to_kpi, "xp2 - dxp2"},
       }},
      {"c",
       {
           {theory_config::phenomenological,
            "0.5 * (      (qop+1)*(  y*cos(Delta_Kpi - phi) + x*sin(Delta_Kpi - phi)) "
            "       + 1 / (qop+1)*(  y*cos(Delta_Kpi + phi) + x*sin(Delta_Kpi + phi)))"},
           {theory_config::theoretical, "y12 * cos(Delta_Kpi) * cos(phiG) + x12 * sin(Delta_Kpi) * cos(phiM)"},
           {theory_config::superweak, "y12 * cos(Delta_Kpi) + x12 * sin(Delta_Kpi) * cos(phiM)"},
           {theory_config::d0_to_kpi, "yp"},
       }},
      {"c'",
       {
           {theory_config::phenomenological, "0.125 * (pow(x, 2) + pow(y, 2)) * (pow(qop + 1, 2) + pow(qop + 1, -2))"},
           {theory_config::theoretical, "0.25 * (pow(x12, 2) + pow(y12, 2))"
                                        "+ 0.25 * R_Kpi / 100 * (pow(y12, 2) - pow(x12, 2))"},  // 2nd order corrections
           {theory_config::superweak, "0.25 * (pow(x12, 2) + pow(y12, 2))"
                                      "+ 0.25 * R_Kpi / 100 * (pow(y12, 2) - pow(x12, 2))"},  // 2nd order corrections
           {theory_config::d0_to_kpi, "(pow(yp, 2) + xp2) / 4"},
       }},
      {"dc",
       {
           {theory_config::phenomenological,
            "0.5 * (      (qop+1)*(  y*cos(Delta_Kpi - phi) + x*sin(Delta_Kpi - phi)) "
            "       - 1 / (qop+1)*(  y*cos(Delta_Kpi + phi) + x*sin(Delta_Kpi + phi)))"},
           {theory_config::theoretical, "  x12 * cos(Delta_Kpi) * sin(phiM)"
                                        "- y12 * sin(Delta_Kpi) * sin(phiG)"},
           {theory_config::superweak, "x12 * cos(Delta_Kpi) * sin(phiM)"},
           {theory_config::d0_to_kpi, "dyp"},
       }},
      {"dc'",
       {
           {theory_config::phenomenological, "1 / 8 * (pow(x, 2) + pow(y, 2)) * (pow(qop + 1, 2) - pow(qop + 1, -2))"},
           {theory_config::theoretical, "0.5 * x12 * y12 * sin(phiM - phiG)"},
           {theory_config::superweak, "0.5 * x12 * y12 * sin(phiM)"},
           {theory_config::d0_to_kpi, "(2 * yp * dyp + pow(dyp, 2) + dxp2) / 4"},
       }},
  };
}  // namespace

PDF_WS::PDF_WS(TString measurement_id, const theory_config& th_cfg, WS_parametrisation p)
    : PDF_Abs{measurement_id.EqualTo("LHCb_Prompt_Run12_appB") ? 9 : 6}, th_cfg{th_cfg}, ws_param{p} {
  TString label;
  if (measurement_id.EqualTo("BaBar"))
    label = "WS/RS BaBar CPV";
  else if (measurement_id.EqualTo("Belle"))
    label = "WS/RS Belle CPV";
  else if (measurement_id.EqualTo("LHCb_DT_Run1"))
    label = "WS/RS LHCb dt (Run 1)";
  else if (measurement_id.EqualTo("LHCb_Run1"))
    label = "WS/RS LHCb Run 1";
  else if (measurement_id.EqualTo("LHCb_Prompt_2011_2016"))
    label = "WS/RS LHCb prompt (15/16)";
  else if (measurement_id.EqualTo("LHCb_Prompt_Run12_sec9"))
    label = "WS/RS LHCb prompt (Run 1+2)";
  else if (measurement_id.EqualTo("LHCb_Prompt_Run12_appB"))
    label = "WS/RS LHCb prompt (Run 1+2)";
  else if (measurement_id.EqualTo("LHCb_DT_Run2"))
    label = "WS/RS LHCb dt (Run 2)";
  else if (measurement_id.EqualTo("LHCb_DT_Run12"))
    label = "WS/RS LHCb dt (Run 1+2)";
  else {
    std::cerr << "PDF_WS: Measurement ID " << measurement_id << " not supported\n";
    exit(1);
  }

  if (ws_param == WS_parametrisation::ccprime && !measurement_id.BeginsWith("LHCb_Prompt_Run12")) {
    std::cerr << "The c/c' parametrisation was introduced only with the LHCb Run 2 measurement\n";
    exit(1);
  }

  name = "WS_" + measurement_id;
  initParameters();
  initRelations();
  initObservables(label);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_WS::PDF_WS(TString val, TString err, const theory_config& th_cfg) : PDF_Abs{6}, th_cfg{th_cfg} {
  TString label;
  if (err.EqualTo("LHCb_Run12"))
    label = "WS/RS LHCb prompt (Run 1+2)";
  else
    exit(1);

  name = "WS_" + err;
  initParameters();
  initRelations();
  initObservables(label);
  setObservables(val);
  setUncertainties(err);
  setCorrelations(err);
  buildCov();
  buildPdf();
}

PDF_WS::~PDF_WS() {}

void PDF_WS::initParameters() {
  std::vector<std::string> param_names = {"R_Kpi"};
  if (th_cfg != theory_config::d0_to_kpi) param_names.emplace_back("Delta_Kpi");
  if (th_cfg != theory_config::superweak) param_names.emplace_back("Acp_KP");
  if (nObs == 9) param_names.emplace_back("Acp_KK");
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
    param_names.insert(param_names.end(), {"yp", "dyp", "xp2", "dxp2"});
    break;
  default:
    std::cout << "PDF_WS::initParameters : ERROR : theory_config " << th_cfg << " not supported.\n";
    exit(1);
  }
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  for (const auto& par : param_names) parameters->add(*(p.get(par)));
}

void PDF_WS::initRelations() {
  switch (ws_param) {
  case WS_parametrisation::raxy:
    initRelationsRAXY();
    break;
  case WS_parametrisation::rrxy:
    initRelationsRRXY();
    break;
  case WS_parametrisation::ccprime:
    initRelationsCCPrime();
    break;
  default:
    std::cout << "PDF_WS::initRelations : ERROR : ws_param not supported.\n";
    exit(1);
  }
}

void PDF_WS::initRelationsCCPrime() {
  theory = new RooArgList("theory");
  theory->add(*(Utils::makeTheoryVar("RD_th", "RD_th", "R_Kpi", parameters)));
  theory->add(*(Utils::makeTheoryVar("c_th", "c_th", theory_expressions["c"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("c'_th", "c'_th", theory_expressions["c'"][th_cfg], parameters)));
  theory->add(
      *(Utils::makeTheoryVar("AD_th", "AD_th", th_cfg != theory_config::superweak ? "Acp_KP" : "0", parameters)));
  theory->add(*(Utils::makeTheoryVar("dc_th", "dc_th", theory_expressions["dc"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("dc'_th", "dc'_th", theory_expressions["dc'"][th_cfg], parameters)));
  if (nObs == 9) {
    theory->add(*(Utils::makeTheoryVar(
        "ADt_th", "ADt_th", th_cfg != theory_config::superweak ? "Acp_KP - 2 * Acp_KK" : "- 2 * Acp_KK", parameters)));
    theory->add(*(Utils::makeTheoryVar("dc~_th", "dc~_th",
                                       theory_expressions["dc"][th_cfg] + " - 2 * sqrt(R_Kpi / 100) * (" +
                                           CharmUtils::get_dy_expression(th_cfg, FSC::none, "KK") + ")" +
                                           " - Acp_KK / 100 * (" + theory_expressions["c"][th_cfg] + ")",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("dc'~_th", "dc'~_th",
                                       theory_expressions["dc'"][th_cfg] + " - 2 * sqrt(R_Kpi / 100) * (" +
                                           theory_expressions["c"][th_cfg] + ") * (" +
                                           CharmUtils::get_dy_expression(th_cfg, FSC::none, "KK") + ")" +
                                           " - 2 * Acp_KK / 100 * (" + theory_expressions["c'"][th_cfg] + ")",
                                       parameters)));
  }
}

void PDF_WS::initRelationsRAXY() {
  theory = new RooArgList("theory");
  theory->add(*(Utils::makeTheoryVar("RD_th", "RD_th", "R_Kpi", parameters)));
  theory->add(*(Utils::makeTheoryVar("y'+_th", "y'+_th", theory_expressions["y'+"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("x'2+_th", "x'2+_th", theory_expressions["x'2+"][th_cfg], parameters)));
  theory->add(
      *(Utils::makeTheoryVar("AD_th", "AD_th", th_cfg != theory_config::superweak ? "Acp_KP" : "0", parameters)));
  theory->add(*(Utils::makeTheoryVar("y'-_th", "y'-_th", theory_expressions["y'-"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("x'2-_th", "x'2-_th", theory_expressions["x'2-"][th_cfg], parameters)));
}

void PDF_WS::initRelationsRRXY() {
  theory = new RooArgList("theory");
  std::string cpv_correction_p = th_cfg != theory_config::superweak ? " * (1 + Acp_KP / 100)" : "";
  std::string cpv_correction_m = th_cfg != theory_config::superweak ? " * (1 - Acp_KP / 100)" : "";
  theory->add(*(Utils::makeTheoryVar("RD_p_th", "RD_p_th", "R_Kpi" + cpv_correction_p, parameters)));
  theory->add(*(Utils::makeTheoryVar("y'+_th", "y'+_th", theory_expressions["y'+"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("x'2+_th", "x'2+_th", theory_expressions["x'2+"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("RD_m_th", "RD_m_th", "R_Kpi" + cpv_correction_m, parameters)));
  theory->add(*(Utils::makeTheoryVar("y'-_th", "y'-_th", theory_expressions["y'-"][th_cfg], parameters)));
  theory->add(*(Utils::makeTheoryVar("x'2-_th", "x'2-_th", theory_expressions["x'2-"][th_cfg], parameters)));
}

void PDF_WS::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  // the order of this list must match that of the COR matrix!
  switch (ws_param) {
  case WS_parametrisation::raxy:
    observables->add(*(new RooRealVar("RD_obs", setName + "   #it{R_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("y'+_obs", setName + "   #it{y'^{+}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("x'2+_obs", setName + "   #it{x'^{+2}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("AD_obs", setName + "   #it{A_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("y'-_obs", setName + "   #it{y'}^{#minus}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("x'2-_obs", setName + "   #it{x'}^{#minus2}", 0., -1e4, 1e4)));
    break;
  case WS_parametrisation::rrxy:
    observables->add(*(new RooRealVar("RD_p_obs", setName + "   #it{R_{K#pi}^{+}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("y'+_obs", setName + "   #it{y'^{+}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("x'2+_obs", setName + "   #it{x'^{+2}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("RD_m_obs", setName + "   #it{R_{K#pi}^{#minus}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("y'-_obs", setName + "   #it{y'}^{#minus}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("x'2-_obs", setName + "   #it{x'}^{#minus2}", 0., -1e4, 1e4)));
    break;
  case WS_parametrisation::ccprime:
    observables->add(*(new RooRealVar("RD_obs", setName + "   #it{R_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("c_obs", setName + "   #it{c_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("c'_obs", setName + "   #it{c'_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("AD_obs", setName + "   #it{A_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("dc_obs", setName + "   #it{#Deltac_{K#pi}}", 0., -1e4, 1e4)));
    observables->add(*(new RooRealVar("dc'_obs", setName + "   #it{#Deltac'_{K#pi}}", 0., -1e4, 1e4)));
    if (nObs == 9) {
      observables->add(*(new RooRealVar("ADt_obs", setName + "   #it{#tilde{A}_{K#pi}}", 0., -1e4, 1e4)));
      observables->add(*(new RooRealVar("dc~_obs", setName + "   #it{#Delta#tilde{c}_{K#pi}}", 0., -1e4, 1e4)));
      observables->add(*(new RooRealVar("dc'~_obs", setName + "   #it{#Delta#tilde{c}'_{K#pi}}", 0., -1e4, 1e4)));
    }
    break;
  default:
    std::cout << "PDF_WS::initRelations : ERROR : ws_param not supported.\n";
    exit(1);
  }
}

void PDF_WS::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("BaBar")) {
    obsValSource = "https://inspirehep.net/literature/746245";
    // setObservable("RD_obs", 0.303); TODO
    setObservable("RD_p_obs", 0.297);
    setObservable("y'+_obs", 0.98);
    setObservable("x'2+_obs", -2.4);
    // setObservable("AD_obs", -2.1); TODO
    setObservable("RD_m_obs", 0.309);
    setObservable("y'-_obs", 0.96);
    setObservable("x'2-_obs", -2.0);
  } else if (c.EqualTo("Belle")) {
    obsValSource = "http://belle.kek.jp/belle/theses/doctor/lmzhang06/phd-mix-400.ps.gz";
    setObservable("RD_p_obs", 0.373);
    setObservable("y'+_obs", -0.12);
    setObservable("x'2+_obs", 3.2);
    setObservable("RD_m_obs", 0.356);
    setObservable("y'-_obs", 0.2);
    setObservable("x'2-_obs", 0.6);
  } else if (c.EqualTo("LHCb_DT_Run1")) {
    obsValSource = "https://inspirehep.net/literature/1499047";
    setObservable("RD_p_obs", 0.338);
    setObservable("y'+_obs", 0.581);
    setObservable("x'2+_obs", -0.19);
    setObservable("RD_m_obs", 0.360);
    setObservable("y'-_obs", 0.332);
    setObservable("x'2-_obs", 0.79);
  } else if (c.EqualTo("LHCb_Run1")) {
    obsValSource = "https://inspirehep.net/literature/1499047";
    setObservable("RD_p_obs", 0.3474);
    setObservable("y'+_obs", 0.597);
    setObservable("x'2+_obs", 0.11);
    setObservable("RD_m_obs", 0.3591);
    setObservable("y'-_obs", 0.450);
    setObservable("x'2-_obs", 0.61);
  } else if (c.EqualTo("LHCb_Prompt_2011_2016")) {
    obsValSource = "https://inspirehep.net/literature/1642234";
    setObservable("RD_p_obs", 0.3454);
    setObservable("y'+_obs", 0.501);
    setObservable("x'2+_obs", 0.61);
    setObservable("RD_m_obs", 0.3454);
    setObservable("y'-_obs", 0.554);
    setObservable("x'2-_obs", 0.16);
  } else if (c.EqualTo("LHCb_Prompt_Run12_sec9")) {
    obsValSource = "https://indico.cern.ch/event/1355805/";
    setObservable("RD_obs", 0.3427);
    setObservable("c_obs", 0.528);
    setObservable("c'_obs", 0.120);
    setObservable("AD_obs", -0.66);
    setObservable("dc_obs", 0.020);
    setObservable("dc'_obs", -0.007);
  } else if (c.EqualTo("LHCb_Prompt_Run12_appB")) {
    obsValSource = "https://indico.cern.ch/event/1355805/";
    setObservable("RD_obs", 0.3427);
    setObservable("c_obs", 0.528);
    setObservable("c'_obs", 0.120);
    setObservable("AD_obs", -0.9);
    setObservable("dc_obs", -0.01);
    setObservable("dc'_obs", 0.046);
    setObservable("ADt_obs", -0.82);
    setObservable("dc~_obs", 0.032);
    setObservable("dc'~_obs", -0.020);
  } else if (c.EqualTo("LHCb_DT_Run2")) {
    obsValSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    setObservable("RD_p_obs", 0.355);
    setObservable("y'+_obs", 0.356);
    setObservable("x'2+_obs", 1.086);
    setObservable("RD_m_obs", 0.339);
    setObservable("y'-_obs", 0.811);
    setObservable("x'2-_obs", -1.129);
  } else if (c.EqualTo("LHCb_DT_Run12")) {
    obsValSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    setObservable("RD_p_obs", 0.350);
    setObservable("y'+_obs", 0.414);
    setObservable("x'2+_obs", 0.784);
    setObservable("RD_m_obs", 0.344);
    setObservable("y'-_obs", 0.681);
    setObservable("x'2-_obs", -0.486);
  } else {
    std::cout << "PDF_WS::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS::setUncertainties(TString c) {
  if (c.EqualTo("BaBar")) {
    obsErrSource = "https://inspirehep.net/literature/746245";
    // StatErr[0] = 0.0189; // RD TODO
    StatErr[0] = 0.0267;  // RD+
    StatErr[1] = 0.78;    // y'+
    StatErr[2] = 5.2;     // x'2+
    // StatErr[3] = 5.4;    // AD TODO
    StatErr[3] = 0.0267;  // RD-
    StatErr[4] = 0.75;    // y'-
    StatErr[5] = 5.0;     // x'2-
    SystErr[0] = 0;       // RD+ (or RD)
    SystErr[1] = 0;       // y'+
    SystErr[2] = 0;       // x'2+
    SystErr[3] = 0;       // RD- (or AD)
    SystErr[4] = 0;       // y'-
    SystErr[5] = 0;       // x'2-
  } else if (c.EqualTo("Belle")) {
    obsErrSource = "http://belle.kek.jp/belle/theses/doctor/lmzhang06/phd-mix-400.ps.gz";
    StatErr[0] = 0.024;  // RD+
    StatErr[1] = 0.57;   // y'+
    StatErr[2] = 3.1;    // x'2+
    StatErr[3] = 0.024;  // RD-
    StatErr[4] = 0.54;   // y'-
    StatErr[5] = 2.9;    // x'2-
    SystErr[0] = 0;      // RD+
    SystErr[1] = 0;      // y'+
    SystErr[2] = 0;      // x'2+
    SystErr[3] = 0;      // RD-
    SystErr[4] = 0;      // y'-
    SystErr[5] = 0;      // x'2-
  } else if (c.EqualTo("LHCb_DT_Run1")) {
    obsErrSource = "https://inspirehep.net/literature/1499047";
    StatErr[0] = pow(pow(0.15, 2) + pow(0.06, 2), 0.5);    // RD+
    StatErr[1] = pow(pow(0.525, 2) + pow(0.032, 2), 0.5);  // y'+
    StatErr[2] = pow(pow(4.46, 2) + pow(0.31, 2), 0.5);    // x'2+
    StatErr[3] = pow(pow(0.15, 2) + pow(0.07, 2), 0.5);    // RD-
    StatErr[4] = pow(pow(0.521, 2) + pow(0.040, 2), 0.5);  // y'-
    StatErr[5] = pow(pow(4.31, 2) + pow(0.38, 2), 0.5);    // x'2-
    SystErr[0] = 0;                                        // RD+
    SystErr[1] = 0;                                        // y'+
    SystErr[2] = 0;                                        // x'2+
    SystErr[3] = 0;                                        // RD-
    SystErr[4] = 0;                                        // y'-
    SystErr[5] = 0;                                        // x'2-
  } else if (c.EqualTo("LHCb_Run1")) {
    obsErrSource = "https://inspirehep.net/literature/1499047";
    StatErr[0] = 0.081;  // RD+
    StatErr[1] = 0.125;  // y'+
    StatErr[2] = 0.65;   // x'2+
    StatErr[3] = 0.081;  // RD-
    StatErr[4] = 0.121;  // y'-
    StatErr[5] = 0.61;   // x'2-
    SystErr[0] = 0;      // RD+
    SystErr[1] = 0;      // y'+
    SystErr[2] = 0;      // x'2+
    SystErr[3] = 0;      // RD-
    SystErr[4] = 0;      // y'-
    SystErr[5] = 0;      // x'2-
  } else if (c.EqualTo("LHCb_Prompt_2011_2016")) {
    obsErrSource = "https://inspirehep.net/literature/1642234";
    StatErr[0] = 0.0045;  // RD+
    StatErr[1] = 0.074;   // y'+
    StatErr[2] = 0.37;    // x'2+
    StatErr[3] = 0.0045;  // RD-
    StatErr[4] = 0.074;   // y'-
    StatErr[5] = 0.39;    // x'2-
    SystErr[0] = 0;       // RD+
    SystErr[1] = 0;       // y'+
    SystErr[2] = 0;       // x'2+
    SystErr[3] = 0;       // RD-
    SystErr[4] = 0;       // y'-
    SystErr[5] = 0;       // x'2-
  } else if (c.EqualTo("LHCb_Prompt_Run12_sec9")) {
    obsErrSource = "https://indico.cern.ch/event/1355805/";
    StatErr[0] = 0.0019;  // RD
    StatErr[1] = 0.033;   // c
    StatErr[2] = 0.035;   // c'
    StatErr[3] = 0.57;    // AD
    StatErr[4] = 0.034;   // dc
    StatErr[5] = 0.036;   // dc'
    SystErr[0] = 0;       // RD
    SystErr[1] = 0;       // c
    SystErr[2] = 0;       // c'
    SystErr[3] = 0;       // AD
    SystErr[4] = 0;       // dc
    SystErr[5] = 0;       // dc'
  } else if (c.EqualTo("LHCb_Prompt_Run12_appB")) {
    obsErrSource = "https://indico.cern.ch/event/1355805/";
    StatErr[0] = 0.0019;  // RD
    StatErr[1] = 0.033;   // c
    StatErr[2] = 0.035;   // c'
    StatErr[3] = 2.0;     // AD
    StatErr[4] = 0.10;    // dc
    StatErr[5] = 0.098;   // dc'
    StatErr[6] = 0.59;    // ADt
    StatErr[7] = 0.036;   // dc~
    StatErr[8] = 0.038;   // dc'~
    SystErr[0] = 0;       // RD
    SystErr[1] = 0;       // c
    SystErr[2] = 0;       // c'
    SystErr[3] = 0;       // AD
    SystErr[4] = 0;       // dc
    SystErr[5] = 0;       // dc'
    SystErr[6] = 0;       // ADt
    SystErr[7] = 0;       // dc~
    SystErr[8] = 0;       // dc'~
  } else if (c.EqualTo("LHCb_DT_Run2")) {
    obsErrSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    StatErr[0] = 0.008;  // RD+
    StatErr[1] = 0.225;  // y'+
    StatErr[2] = 1.623;  // x'2+
    StatErr[3] = 0.008;  // RD-
    StatErr[4] = 0.236;  // y'-
    StatErr[5] = 1.859;  // x'2-
    for (int i = 0; i < nObs; ++i) SystErr[i] = 0.;
  } else if (c.EqualTo("LHCb_DT_Run12")) {
    obsErrSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    StatErr[0] = 0.007;  // RD+
    StatErr[1] = 0.204;  // y'+
    StatErr[2] = 1.522;  // x'2+
    StatErr[3] = 0.007;  // RD-
    StatErr[4] = 0.211;  // y'-
    StatErr[5] = 1.665;  // x'2-
    for (int i = 0; i < nObs; ++i) SystErr[i] = 0.;
  } else {
    std::cout << "PDF_WS::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("BaBar")) {  // TODO
    corSource = "https://hflav-eos.web.cern.ch/hflav-eos/charm/CKM23/results_mix_cpv.html";
    std::vector<double> data = {
        // clang-format off
        // RD+   y'+    x'2+   RD-  y'-   x'2-
           1.,  -0.87,  0.77,  0.,  0.,   0.,    // RD+
                 1.,   -0.94,  0.,  0.,   0.,    // y'+
                        1.,    0.,  0.,   0.,    // x'2+
                               1., -0.87, 0.77,  // RD-
                                    1.,  -0.94,  // y'-
                                          1.     // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("Belle")) {
    corSource = "http://belle.kek.jp/belle/theses/doctor/lmzhang06/phd-mix-400.ps.gz";
    std::vector<double> data = {
        // clang-format off
        // RD+  y'+    x'2+   RD-  y'-    x'2-
        1.,    -0.834, 0.655, 0.,  0.,    0.,     // RD+
                1.,   -0.909, 0.,  0.,    0.,     // y'+
                       1.,    0.,  0.,    0.,     // x'2+
                              1., -0.834, 0.655,  // RD-
                                   1.,   -0.909,  // y'-
                                          1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_DT_Run1")) {
    corSource = "https://inspirehep.net/literature/1499047";
    std::vector<double> data = {
        // clang-format off
        // RD+  y'+    x'2+    RD-    y'-    x'2-
        1.,    -0.732, 0.625, -0.008, 0.,    0.,     // RD+
                1.,   -0.963,  0.,    0.,    0.,     // y'+
                       1.,     0.,    0.,    0.,     // x'2+
                               1.,   -0.707, 0.602,  // RD-
                                      1.,   -0.958,  // y'-
                                             1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_Run1")) {
    corSource = "https://inspirehep.net/literature/1499047";
    std::vector<double> data = {
        // clang-format off
        //  RD+     y'+    x'2+     RD-     y'-    x'2-
        1., -0.920, 0.823, -0.007, -0.010,  0.008,  // RD+
             1.,   -0.962, -0.011,  0.000, -0.002,  // y'+
                    1.,     0.009, -0.002,  0.004,  // x'2+
                            1.,    -0.918,  0.812,  // RD-
                                    1.,    -0.956,  // y'-
                                            1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_Prompt_2011_2016")) {
    corSource = "https://inspirehep.net/literature/1642234";
    std::vector<double> data = {
        // clang-format off
        //  RD+     y'+    x'2+     RD-     y'-    x'2-
        1., -0.935, 0.843, -0.012, -0.003, -0.002,  // RD+
             1.,   -0.963, -0.003,  0.004, -0.003,  // y'+
                    1.,     0.002, -0.003,  0.003,  // x'2+
                            1.,    -0.935,  0.846,  // RD-
                                    1.,    -0.964,  // y'-
                                            1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_Prompt_Run12_sec9")) {
    corSource = "https://indico.cern.ch/event/1355805/";
    std::vector<double> data = {
        // clang-format off
        // RD  c      c'      AD      c       c'
        1.,   -0.927, 0.803,  0.009, -0.007,  0.002,  // RD
               1.,   -0.942, -0.013,  0.012, -0.007,  // c
                      1.,     0.007, -0.007,  0.002,  // c'
                              1.,    -0.919,  0.797,  // AD
                                      1.,    -0.941,  // dc
                                              1.      // dc'
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_Prompt_Run12_appB")) {
    corSource = "https://indico.cern.ch/event/1355805/";
    std::vector<double> data = {
        // clang-format off
        // RD  c      c'      AD      c       c'      ADt     ct      c't
        1.,   -0.927, 0.803,  0.003, -0.002,  0.002,  0.008, -0.007,  0.000,  // RD
               1.,   -0.943, -0.005,  0.004, -0.004, -0.014,  0.013, -0.006,  // c
                      1.,     0.003, -0.003,  0.003,  0.007, -0.006,  0.000,  // c'
                              1.,    -0.938,  0.811,  0.,     0.,     0.,     // AD
                                      1.,    -0.943,  0.,     0.,     0.,     // dc
                                              1.,     0.,     0.,     0.,     // dc'
                                                      1.,    -0.934,  0.810,  // ADt
                                                              1.,    -0.943,  // dc~
                                                                      1.      // dc'~
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_DT_Run2")) {
    corSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    std::vector<double> data = {
        // clang-format off
        // RD+  y'+    x'2+    RD-     y'-    x'2-
        1.,    -0.768, 0.639, -0.015, -0.001, 0.,     // RD+
                1.,   -0.940,  0.,     0.,    0.,     // y'+
                       1.,     0.,     0.,    0.,     // x'2+
                               1.,    -0.769, 0.653,  // RD-
                                       1.,   -0.950,  // y'-
                                              1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else if (c.EqualTo("LHCb_DT_Run12")) {
    corSource = "https://indico.cern.ch/event/1423686/contributions/6139348/, LHCb-PAPER-2024-044";
    std::vector<double> data = {
        // clang-format off
        //  RD+  y'+    x'2+    RD-    y'-    x'2-
        1.,     -0.759, 0.631, -0.013, 0.,    0.,     // RD+
                 1.,   -0.945,  0.,    0.,    0.,     // y'+
                        1.,     0.,    0.,    0.,     // x'2+
                                1.,   -0.756, 0.637,  // RD-
                                       1.,   -0.949,  // y'-
                                              1.      // x'2-
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, data);
  } else {
    std::cout << "PDF_WS::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_WS::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
