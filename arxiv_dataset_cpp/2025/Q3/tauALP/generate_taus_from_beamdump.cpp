#include "Pythia8/Pythia.h"
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace Pythia8;

// For biasing the generation of high-pT events
// class HighPtBiasHook : public Pythia8::UserHooks {

// public:
//   // keep the factor so we can hand back the compensation weight
//   double selBias = 1.0;

//   // tell Pythia that we will bias the phase-space pick
//   bool canBiasSelection() override { return true; }

//   // multiplicative bias factor in the hard-process pick
//   double biasSelectionBy( const SigmaProcess* /*sigma*/,
//                           const PhaseSpace*  phaseSpace,
//                           bool /*inEvent*/) override {

//     double pT = phaseSpace->pTHat();     // hard-process pT̂
//     selBias   = pT*pT;                 // quadratic bias
//     return selBias;                      // >1 ⇒ favour high-pT̂
//   }

//   // event weight that compensates the bias above
//   double biasedSelectionWeight() override { return 1.0 / selBias; }
// };


int main(int argc, char* argv[]) {
    
    if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " <n_events> <output_file> <beam_energy> <soft|hard|both> <pTcut> <seed> <collider|fixed>\n";
    return 1;
    }

    #ifdef PYTHIA8_VERSION
        std::cout << "Pythia version: " << PYTHIA_VERSION_INTEGER << "\n";
    #else
        std::cerr << "Warning: PYTHIA8_VERSION not defined!\n";
    #endif

    // Parse arguments
    const int nTauEvents = std::stoi(argv[1]);
    const std::string baseFilename = argv[2];
    const std::string beamEnergyStr = argv[3];
    const std::string mode = argv[4];
    const double pTcut = std::stod(argv[5]);
    const std::string seedStr = argv[6];
    std::string beamMode = argv[7];

    const std::string outputFile = "pythia8_events/" + mode + "_" + baseFilename + ".txt";

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Failed to open output file: " << outputFile << "\n";
        return 1;
    }

    // Initialize Pythia
    Pythia pythia;
    pythia.readString("Random:seed = " + seedStr);
    pythia.readString("Random:setSeed = on");
    pythia.readString("Tune:pp = 14");
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");

    if (beamMode == "collider") {
        pythia.readString("Beams:frameType = 1"); // collider
        pythia.readString("Beams:eCM = " + beamEnergyStr);
    } else if (beamMode == "fixed") {
        pythia.readString("Beams:frameType = 2"); // fixed target
        pythia.readString("Beams:eA = " + beamEnergyStr);
        pythia.readString("Beams:eB = 0.0");
    } else {
        std::cerr << "Error: Invalid beam mode. Use 'collider' or 'fixed'.\n";
        return 1;
    }

    // Enable processes
    if (mode == "soft") {
        pythia.readString("SoftQCD:all = on");
        pythia.readString("HardQCD:all = off");
        pythia.readString("Onia:all = on");
        pythia.readString("SoftQCD:nonDiffractive = on");
        pythia.readString("SoftQCD:singleDiffractive = on");
        pythia.readString("SoftQCD:doubleDiffractive = on");
        // pythia.readString("PhaseSpace:pTHatMin = 0.01");
    } else if (mode == "hard") {
        pythia.readString("SoftQCD:all = off");
        pythia.readString("HardQCD:all = on");
        pythia.readString("Onia:all = on");
        pythia.readString("HardQCD:hardccbar = on");
        pythia.readString("HardQCD:hardbbbar = on");
        pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(pTcut));
    } else if (mode == "both") {
        pythia.readString("SoftQCD:all = on");
        pythia.readString("HardQCD:all = on");
        pythia.readString("Onia:all = on");
        pythia.readString("PhaseSpace:pTHatMin = 0.01");
    } else {
        std::cerr << "Error: Invalid mode. Use 'soft', 'hard', or 'both'.\n";
        return 1;
    }

    // Tau decay settings
    const std::vector<std::string> decaySettings = {
        "ProcessLevel:all = on", "HadronLevel:Decay = on",
        "431:onMode = off", "431:onIfAny = 15",
        "-431:onMode = off", "-431:onIfAny = -15",
        "411:onMode = off", "411:onIfAny = 15",
        "-411:onMode = off", "-411:onIfAny = -15",
        "100443:onMode = off", "100443:onIfAny = 15",
        "15:onMode = off"
    };
    for (const auto& setting : decaySettings) {
        pythia.readString(setting);
    }

    // Verbosity
    const std::vector<std::string> quietSettings = {
        "Init:showProcesses = off", "Init:showMultipartonInteractions = off",
        "Init:showChangedSettings = off", "Init:showAllSettings = off",
        "Init:showChangedParticleData = off",
        "Next:numberCount = 0", "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0", "Next:numberShowEvent = 0",
        "Print:quiet = on"
    };
    for (const auto& s : quietSettings) pythia.readString(s);

    pythia.settings.flag("Check:event", false);

    // Attach the user hook
    // std::shared_ptr<UserHooks> myHook = std::make_shared<HighPtBiasHook>();
    // pythia.setUserHooksPtr(myHook);

    // Initialize
    pythia.init();

    // Write simulation configuration header with placeholders
    outFile << "# pythia: " << PYTHIA_VERSION_INTEGER << std::endl;
    outFile << "# beam_energy: " << beamEnergyStr << std::endl;
    outFile << "# ptcut: " << pTcut << std::endl;
    outFile << "# seed: " << seedStr << std::endl;
    outFile << "# mode: " << mode << std::endl;
    outFile << "# beam_mode: " << beamMode << "\n";

    // Reserve space for cross sections (fixed-width fields)
    std::streampos xsecPos = outFile.tellp();  // Remember position
    outFile << "# total_xsec_mb: ";
    outFile << std::setw(15) << std::left << "0.0000000000" << std::endl;
    outFile << "# tau_xsec_mb: ";
    outFile << std::setw(15) << std::left << "0.0000000000" << std::endl;

    // Column headers
    outFile << "# event_number particle_count pid E px py pz mother_pid E_mother px_mother py_mother pz_mother weights" << std::endl;

    // Mother weight mapping
    const std::unordered_map<int, double> motherWeights = {
        {411, 1.20e-3}, {431, 5.36e-2}, {100443, 3.1e-3}
    };

    int tauCount = 0, eventCount = 0;
    double total_weight = 0.0;
    double total_tau_weight = 0.0;

    while (tauCount < nTauEvents) {
        if (!pythia.next()) continue;

        double pTHat = pythia.info.pTHat();
        if ((mode == "soft" && pTHat > pTcut) || (mode == "hard" && pTHat < pTcut)) continue;

        
        double eventWeight = pythia.info.weight();


        for (int i = 0; i < pythia.event.size(); ++i) {
            if (std::abs(pythia.event[i].id()) != 15) continue;

            int motherIdx = pythia.event[i].mother1();
            int motherID = (motherIdx >= 0 && motherIdx < pythia.event.size())
                ? std::abs(pythia.event[motherIdx].id())
                : 0;
            double weight = motherWeights.count(motherID) ? motherWeights.at(motherID) : 1.0;

            ++tauCount;
            total_tau_weight += weight*eventWeight;

            outFile << std::scientific << std::setprecision(10);   // NEW: sci-notation

            outFile << eventWeight << " "
                    << tauCount << " "
                    << pythia.event[i].id() << " "
                    << pythia.event[i].e()  << " "
                    << pythia.event[i].px() << " "
                    << pythia.event[i].py() << " "
                    << pythia.event[i].pz() << " ";

            if (motherIdx >= 0 && motherIdx < pythia.event.size()) {
                const auto& mom = pythia.event[motherIdx];
                outFile << mom.id() << " "
                        << mom.e()  << " "
                        << mom.px() << " "
                        << mom.py() << " "
                        << mom.pz();
            } else {
                outFile << "0 0 0 0 0";
            }
            
            // Adding weight from BRs to the output
            outFile << " " << weight*eventWeight << std::endl;
        }

        ++eventCount;
        total_weight += eventWeight;
    }

    // Final cross sections
    double totalCrossSection = pythia.info.sigmaGen();
    double tauCrossSection = total_tau_weight / total_weight * totalCrossSection;

    // Go back and update the reserved header space
    outFile.seekp(xsecPos);
    outFile << "# total_xsec_mb: ";
    outFile << std::setw(15) << std::left << std::fixed << std::setprecision(10) << totalCrossSection << std::endl;;
    outFile << "# tau_xsec_mb: ";
    outFile << std::setw(15) << std::left << std::fixed << std::setprecision(10) << tauCrossSection << std::endl;

    outFile.close();  // Now safe to close

    return 0;
}