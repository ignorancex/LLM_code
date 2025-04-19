#include <string>
#include <vector>

#include <GammaComboEngine.h>

#include <CharmUtils.h>
#include <PDF_AcpHH_LHCb_Run12.h>
#include <PDF_BES_CLEO_K3pi_Kpipi0.h>
#include <PDF_BES_Kpi.h>
#include <PDF_BES_Kpi_1d.h>
#include <PDF_BinFlip.h>
#include <PDF_Cleo.h>
#include <PDF_DY.h>
#include <PDF_DY_RS.h>
#include <PDF_DY_pipipi0.h>
#include <PDF_Fp_pipipi0.h>
#include <PDF_K3pi.h>
#include <PDF_Kpipi0.h>
#include <PDF_Kshh.h>
#include <PDF_RM.h>
#include <PDF_WS.h>
#include <PDF_WS_NoCPV.h>
#include <PDF_XY.h>
#include <PDF_scan_DY_RS.h>
#include <PDF_yCP.h>
#include <PDF_yCP_minus_yCP_KP.h>
#include <PDF_yCP_minus_yCP_RS.h>
#include <PDF_yCP_plus_yCP_RS.h>

namespace {
  /**
   * Clone all combinations to enable the comparison of fits with and without allowing for CPV in DCS decays.
   */
  void add_no_dcs_cpv_combiner(GammaComboEngine& gc, const int icomb) {
    const auto combiner = gc.getCombiner(icomb);
    const auto name = combiner->getName() + "NoDcsCPV";
    const auto title = combiner->getTitle() + " (A_{CP}(K#pi) = 0)";
    gc.cloneCombiner(1000 + icomb, icomb, name, title);
    return;
  }

  std::vector<int> get_lhcb_pdfs(const std::string run, const FSC fsc) {
    if (run == "run12") {
      std::vector<int> list = {
          3,   // XY               KSpipi Prompt Run 1
          11,  // RM               K3pi Run 1
          21,  // BinFlip          Run 1
          24,  // BinFlip          Run 2
          35,  // WS               DT Run 1
          39,  // WS               Prompt Run 1+2
          61,  // yCP_minus_yCP_RS WA2020          TODO
          64,  // yCP_minus_yCP_RS 2022 prompt
          // 80,  // DY_RS   2021
          85,  // DY_pipipi0       Run 2
      };
      switch (fsc) {
      // DY average, AcpHH
      case FSC::none:
        list.push_back(74);
        list.push_back(90);
        break;
      case FSC::partial:
        list.push_back(75);
        list.push_back(91);
        break;
      case FSC::full:
        list.push_back(76);
        list.push_back(92);
        break;
      }
      return list;
    } else {
      std::cerr << "The list of the LHCb results from the period `" << run << "` is not supported. Exiting..."
                << std::endl;
      exit(1);
    }
  }
}  // namespace

int main(int argc, char* argv[]) {

  // Get the parametrisation and the parameter to prevent CPV in DCS decays
  theory_config th_cfg = theory_config::theoretical;
  std::string combiner_name;

  int iparam = -1;
  int ifsc = -1;
  for (int i = 1; i < argc - 1; ++i) {
    if (!strcmp(argv[i], "--param")) {
      iparam = i;
      if (!strcmp(argv[i + 1], "phenomenological")) {
        th_cfg = theory_config::phenomenological;
        combiner_name = "phenomenological";
      } else if (!strcmp(argv[i + 1], "theoretical")) {
        th_cfg = theory_config::theoretical;
        combiner_name = "theoretical";
      } else if (!strcmp(argv[i + 1], "superweak")) {
        th_cfg = theory_config::superweak;
        combiner_name = "superweak";
      } else {
        std::cerr << "Option " << argv[i + 1] << " is not supported by --param" << std::endl;
        exit(1);
      }
    }
  }
  std::vector<char*> combiner_argv = {};
  for (int i = 0; i < argc; ++i) {
    if (i != iparam && i != iparam + 1) combiner_argv.emplace_back(argv[i]);
  }
  std::cout << "INFO: The fitter will be run with the following configuration:\n"
            << "      Parametrisation: " << th_cfg << "\n";

  // Define the combiner
  GammaComboEngine gc(combiner_name, combiner_argv.size(), &combiner_argv[0]);

  ///////////////////////////////////////////////////
  //
  // define PDFs
  //
  ///////////////////////////////////////////////////

  // clang-format off
  gc.addPdf(1, new PDF_XY("BaBar_Kshh", th_cfg),                                           "XY KShh      BaBar                          ");
  gc.addPdf(2, new PDF_XY("BaBar_pipipi0", th_cfg),                                        "XY pipipi0   BaBar                          ");
  gc.addPdf(3, new PDF_XY("LHCb_KSpipi", th_cfg),                                          "XY KSpipi    LHCb     2011     [D* -> D0 pi]");
  gc.addPdf(4, new PDF_Kpipi0("BaBar", th_cfg),                                            "Kpipi0       BaBar                          ");
  gc.addPdf(5, new PDF_K3pi("LHCb-run1", th_cfg),                                          "K3pi         LHCb     Run 1                 ");
  gc.addPdf(6, new PDF_XY("Belle_Belle2", th_cfg),                                         "XY KSpipi    Belle+Belle2 (951+408 fb-1)    ");

  gc.addPdf(10, new PDF_RM("HFLAV2016", th_cfg),                                           "R_M          HFLAV    2016                  ");
  gc.addPdf(11, new PDF_RM("LHCb_K3pi_Run1", th_cfg),                                      "R_M K3pi     LHCb                           ");

  gc.addPdf(20, new PDF_Kshh("Belle", th_cfg),                                             "KShh         Belle                          ");
  gc.addPdf(21, new PDF_BinFlip("LHCb_Run1", th_cfg),                                      "Bin-flip     LHCb     Run 1                 ");
  gc.addPdf(22, new PDF_BinFlip("LHCb_Run2_prompt", th_cfg),                               "Bin-flip     LHCb     Run 2    [D* -> D0 pi]");
  gc.addPdf(23, new PDF_BinFlip("LHCb_Run2_sl", th_cfg),                                   "Bin-flip     LHCb     Run 2    [B -> D0 mu] ");
  gc.addPdf(24, new PDF_BinFlip("LHCb_Run2", th_cfg),                                      "Bin-flip     LHCb     Run 2                 ");

  gc.addPdf(30, new PDF_WS_NoCPV("CDF", th_cfg),                                           "WS/RS        CDF                            ");
  gc.addPdf(31, new PDF_WS_NoCPV("BaBar", th_cfg),                                         "WS/RS        BaBar    no CPV                ");
  gc.addPdf(32, new PDF_WS_NoCPV("Belle", th_cfg),                                         "WS/RS        Belle    no CPV                ");
  gc.addPdf(33, new PDF_WS_NoCPV("BaBar", th_cfg),                                         "WS/RS        BaBar                          ");
  gc.addPdf(34, new PDF_WS_NoCPV("Belle", th_cfg),                                         "WS/RS        Belle                          ");
  gc.addPdf(35, new PDF_WS("LHCb_DT_Run1", th_cfg),                                        "WS/RS        LHCb     Run 1    [B -> D* mu] ");
  gc.addPdf(36, new PDF_WS("LHCb_Run1", th_cfg),                                           "WS/RS        LHCb     Run 1                 ");
  gc.addPdf(37, new PDF_WS("LHCb_Prompt_2011_2016", th_cfg),                               "WS/RS        LHCb     2011-6   [D* -> D0 pi]");
  gc.addPdf(38, new PDF_WS("LHCb_Prompt_Run12_sec9", th_cfg, WS_parametrisation::ccprime), "WS/RS        LHCb     Run 1+2  [D* -> D0 pi]");
  gc.addPdf(39, new PDF_WS("LHCb_Prompt_Run12_appB", th_cfg, WS_parametrisation::ccprime), "WS/RS        LHCb     Run 1+2  [D* -> D0 pi]");
  gc.addPdf(40, new PDF_WS("LHCb_DT_Run2", th_cfg),                                        "WS/RS        LHCb     Run 2    [B -> D* mu] ");
  gc.addPdf(41, new PDF_WS("LHCb_DT_Run12", th_cfg),                                       "WS/RS        LHCb     Run 1-2  [B -> D* mu] ");

  gc.addPdf(50, new PDF_Cleo("Cleo-c", th_cfg),                                            "Delta_Kpi    Cleo-c                         ");
  gc.addPdf(51, new PDF_BES_Kpi_1d(th_cfg),                                                "Delta_Kpi    BES      3fb      [A_kpi only] ");
  gc.addPdf(52, new PDF_BES_Kpi(th_cfg),                                                   "Delta_Kpi    BES      3fb                   ");
  gc.addPdf(53, new PDF_Fp_pipipi0("Cleo-c"),                                              "Fpipipi0     Cleo-c                         ");
  gc.addPdf(54, new PDF_BES_CLEO_K3pi_Kpipi0("BES3-CLEO"),                                 "K3pi-Kpipi0  BES3 + Cleo                    ");
  gc.addPdf(55, new PDF_Fp_pipipi0("BESIII"),                                              "Fpipipi0     BES3                           ");

  gc.addPdf(60, new PDF_yCP("WA2020", th_cfg),                                             "yCP          WA       2020                  ");
  gc.addPdf(61, new PDF_yCP_minus_yCP_RS("WA2020", th_cfg),                                "yCP-yCP(RS)  WA       2020                  ");
  gc.addPdf(62, new PDF_yCP_minus_yCP_KP("WA2020", th_cfg),                                "yCP-yCP(KP)  WA       2020                  ");
  gc.addPdf(63, new PDF_yCP_plus_yCP_RS("WA2020", th_cfg),                                 "yCP+yCP(RS)  WA       2020                  ");
  gc.addPdf(64, new PDF_yCP_minus_yCP_RS("LHCb2022", th_cfg),                              "yCP-yCP(RS)  LHCb     2022                  ");

  gc.addPdf(70, new PDF_DY("WA2019", th_cfg, FSC::none),                                   "DY           WA       2019     [no FSC]     ");
  gc.addPdf(71, new PDF_DY("WA2020", th_cfg, FSC::none),                                   "DY           WA       2020     [no FSC]     ");
  gc.addPdf(72, new PDF_DY("WA2021", th_cfg, FSC::none),                                   "DY           WA       2021     [no FSC]     ");
  gc.addPdf(73, new PDF_DY("Belle&BaBar", th_cfg, FSC::none),                              "DY           B-factories       [no FSC]     ");
  gc.addPdf(74, new PDF_DY("WA2020", th_cfg, FSC::partial),                                "DY           WA       2020     [partial FSC]");
  gc.addPdf(75, new PDF_DY("WA2021", th_cfg, FSC::partial),                                "DY           WA       2021     [partial FSC]");
  gc.addPdf(76, new PDF_DY("WA2021", th_cfg, FSC::full),                                   "DY           WA       2021     [full FSC]   ");

  gc.addPdf(80, new PDF_DY_RS("LHCb2021", th_cfg),                                         "DY(RS)       LHCb     2021                  ");

  gc.addPdf(85, new PDF_DY_pipipi0("LHCb-R2", th_cfg),                                     "DY(pipipi0)  LHCb     Run2                  ");

  gc.addPdf(90, new PDF_AcpHH_LHCb_Run12(th_cfg, FSC::none),                               "ACP(KK/PP)   LHCb     Run1+2   [no FSC]     ");
  gc.addPdf(91, new PDF_AcpHH_LHCb_Run12(th_cfg, FSC::partial),                            "ACP(KK/PP)   LHCb     Run1+2   [partial FSC]");
  gc.addPdf(92, new PDF_AcpHH_LHCb_Run12(th_cfg, FSC::full),                               "ACP(KK/PP)   LHCb     Run1+2   [full FSC]   ");
  gc.addSubsetPdf(93, new PDF_AcpHH_LHCb_Run12(th_cfg, FSC::none), 0, 1, 4, 5,             "ACP(KK/PP)   LHCb     Run1     [no FSC]");

  gc.addPdf(100, new PDF_scan_DY_RS(th_cfg),                                               "ScanDYRS     This is just a nuisance parameter");

  gc.addPdf(110, new PDF_yCP("WA2020_biased", th_cfg),                                     "yCP          WA       2020     [biased]     ");
  gc.addPdf(111, new PDF_yCP("LHCb2022_biased", th_cfg),                                   "yCP-yCP(RS)  LHCb     2022     [biased]     ");
  // clang-format on

  ///////////////////////////////////////////////////
  //
  // Define combinations
  //
  ///////////////////////////////////////////////////

  gc.newCombiner(0, "emtpy", "empty");

  // WA 2020
  gc.newCombiner(1, "WA2020", "World average (Dec 2020)", 1, 2, 3, 4, 10, 11, 20, 21, 30, 31, 32, 35, 37, 50, 51);
  gc.getCombiner(1)->addPdf(gc[60]);
  gc.getCombiner(1)->addPdf(gc[61]);
  gc.getCombiner(1)->addPdf(gc[62]);
  gc.getCombiner(1)->addPdf(gc[63]);
  gc.getCombiner(1)->addPdf(gc[71]);

  // WA June 2021
  gc.cloneCombiner(20, 1, "WA2021", "World average (June 2021)");
  gc.getCombiner(20)->addPdf(gc[22]);  // bin-flip run 2
  gc.getCombiner(20)->delPdf(gc[71]);  // DY WA 2020
  gc.getCombiner(20)->addPdf(gc[72]);  // DY WA 2021
  gc.getCombiner(20)->delPdf(gc[11]);  // LHCb K3pi (x2 + y2)/4
  gc.getCombiner(20)->addPdf(gc[5]);   // LHCb K3pi full
  gc.getCombiner(20)->addPdf(gc[54]);  // BES3 + CLEO K3pi, Kpipi0

  // WA after LHCb 2022 yCP measurement
  gc.cloneCombiner(30, 20, "WAFeb2022", "World average (Feb 2022)");
  gc.getCombiner(30)->addPdf(gc[64]);  // yCP LHCb 2022

  // WA after LHCb 2022 yCP measurement - biased
  gc.cloneCombiner(31, 30, "WAFeb2022_biased",
                   "World average (Feb 2022) #minus no #it{y}_{#it{CP}}^{#it{K^{#minus}#pi^{+}}} correction");
  gc.getCombiner(31)->delPdf(gc[60]);
  gc.getCombiner(31)->delPdf(gc[61]);
  gc.getCombiner(31)->delPdf(gc[62]);
  gc.getCombiner(31)->delPdf(gc[63]);
  gc.getCombiner(31)->delPdf(gc[64]);
  gc.getCombiner(31)->addPdf(gc[110]);
  gc.getCombiner(31)->addPdf(gc[111]);

  // WA September 2022 - no FSC
  gc.cloneCombiner(40, 30, "WASept2022NoFSC", "World average (Sept 2022)");
  gc.getCombiner(40)->delPdf(gc[51]);  // old BESIII measurement of delta_Kpi
  gc.getCombiner(40)->addPdf(gc[52]);  // new BESIII measurement of delta_Kpi
  gc.getCombiner(40)->addPdf(gc[53]);  // F+_pipipi0
  gc.getCombiner(40)->delPdf(gc[22]);  // bin-flip LHCb Run 2 prompt
  gc.getCombiner(40)->addPdf(gc[24]);  // bin-flip LHCb Run 2
  gc.getCombiner(40)->addPdf(gc[90]);  // ACP(KK) + DeltaACP LHCb Run 1+2

  // WA September 2022 - partial FSC
  gc.cloneCombiner(41, 40, "WASept2022PartialFSC", "World average (Sept 2022)");
  gc.getCombiner(41)->delPdf(gc[72]);  // DY WA 2021 (no FSC)
  gc.getCombiner(41)->addPdf(gc[75]);  // DY WA 2021 (partial FSC)
  gc.getCombiner(41)->addPdf(gc[73]);  // B-factories provide KK + PP only and are not included in 75
                                       // when fsr != FSC::none.
  gc.getCombiner(41)->delPdf(gc[90]);  // ACP(KK) + DeltaACP LHCb Run 1+2 (no FSC)
  gc.getCombiner(41)->addPdf(gc[91]);  // ACP(KK) + DeltaACP LHCb Run 1+2 (partial FSC)

  // WA September 2022 - full FSC
  gc.cloneCombiner(42, 41, "WASept2022FullFSC", "World average (Sept 2022)");
  gc.getCombiner(42)->delPdf(gc[75]);  // DY WA 2021 (partial FSC)
  gc.getCombiner(42)->addPdf(gc[76]);  // DY WA 2021 (partial FSC)
  gc.getCombiner(42)->delPdf(gc[91]);  // ACP(KK) + DeltaACP LHCb Run 1+2 (partial FSC)
  gc.getCombiner(42)->addPdf(gc[92]);  // ACP(KK) + DeltaACP LHCb Run 1+2 (full FSC)

  // WA March 2024 March before WS/RS
  gc.cloneCombiner(49, 40, "WAFeb2024NoFSC", "Previous WA");
  gc.getCombiner(49)->addPdf(gc[85]);  // DY(pi+ pi- pi0) from LHCb Run 2

  // WA March 2024 - no FSC
  gc.cloneCombiner(50, 40, "WAMar2024NoFSC", "World average (March 2024)");
  gc.getCombiner(50)->delPdf(gc[37]);  // WS/RS in D0 -> Kpi from LHCb 2011-2016
  gc.getCombiner(50)->addPdf(gc[39]);  // WS/RS in D0 -> Kpi from LHCb Run 1+2
  gc.getCombiner(50)->addPdf(gc[85]);  // DY(pi+ pi- pi0) from LHCb Run 2

  // WA March 2024 with parametrisation of prompt LHCb WS/RS decays from Sec. 9 - no FSC
  gc.cloneCombiner(51, 50, "WAMar2024NoFSC_WSsec9", "World average (March 2024, prompt WS/RS from Sec. 9)");
  gc.getCombiner(51)->delPdf(gc[39]);  // WS/RS in D0 -> Kpi from LHCb Run 1+2
  gc.getCombiner(51)->addPdf(gc[38]);  // WS/RS in D0 -> Kpi from LHCb Run 1+2

  // WA March 2024 without measurement of CPV in the decay (to test the sensitivity of WS/RS to ACP(KK)
  gc.cloneCombiner(52, 50, "WAMar2024NoFSC_noDcsCpv", "No direct measurements");
  gc.getCombiner(52)->delPdf(gc[90]);  // Delta_ACP and ACP(KK)

  // WA Sept 2024 (new BESIII F+(pi+pi-pi0))
  gc.cloneCombiner(53, 50, "WASep2024NoFSC", "World average (Sep 2024)");
  gc.getCombiner(53)->addPdf(gc[55]);  // BESIII measurement of Fp_pipipi0

  // WA October 2024 (new WS/RS with DT Run 2 data)
  gc.cloneCombiner(54, 53, "WAOct2024NoFSC", "World average (October 2024)");
  gc.getCombiner(54)->delPdf(gc[35]);  // WS/RS in D0 -> Kpi from LHCb Run 1 DT
  gc.getCombiner(54)->addPdf(gc[41]);  // WS/RS in D0 -> Kpi from LHCb Run 1+2 DT

  // WA November 2024 (new BinFlip from Belle + Belle 2)
  gc.cloneCombiner(55, 54, "WANov2024NoFSC", "World average (November 2024)");
  gc.getCombiner(55)->delPdf(gc[20]);  // D0 -> KS hh from Belle
  gc.getCombiner(55)->addPdf(gc[6]);   // D0 -> KS pi pi BinFlip Belle + Belle 2

  // LHCb-only averages ----------------------------------------------------------------------------------------------

  gc.newCombiner(300, "LHCbMar2024NoFSC", "LHCb average (May 2024)");
  for (const auto imeas : get_lhcb_pdfs("run12", FSC::none)) gc.getCombiner(300)->addPdf(gc[imeas]);

  // LHCb-only + charm factories averages ----------------------------------------------------------------------------

  gc.cloneCombiner(400, 300, "LHCbCFMar2024NoFSC", "LHCb + Charm factories average (May 2024)");
  for (auto imeas : {50, 52, 53}) gc.getCombiner(400)->addPdf(gc[imeas]);

  // Impact of LHCb upgrades -----------------------------------------------------------------------------------------

  // WA before LHCb Run 2
  gc.newCombiner(500, "LHCb_Run1", "World average before LHCb Run 2");
  for (auto imeas : {1, 2, 3, 4, 10, 11, 20, 21, 30, 31, 32, 35, 36, 50, 52, 60, 61, 62, 63, 70, 93})
    gc.getCombiner(500)->addPdf(gc[imeas]);

  // WA after LHCb Run 2
  gc.cloneCombiner(501, 50, "LHCb_Run2", "World average after LHCb Run 2");

  // Clone all combiners for no DCS CPV hypothesis -------------------------------------------------------------------

  for (const auto id : gc.getCombinersIds()) add_no_dcs_cpv_combiner(gc, id);

  ///////////////////////////////////////////////////
  //
  // Run
  //
  ///////////////////////////////////////////////////

  gc.run();

  return 0;
}
