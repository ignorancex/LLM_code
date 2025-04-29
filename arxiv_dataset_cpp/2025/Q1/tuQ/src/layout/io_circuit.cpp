//
// input JSON-format
//
#include "io_circuit.hpp"


void h_check_non_adjacent_gate(const nlohmann::json &, bool &, unsigned long &);
void h_check_qbit(const nlohmann::json &, bool &, unsigned long &);

nlohmann::json cirq_to_ionq_schema(const nlohmann::json & cirq_circuit) {
//   json: adapt cirq schema json to Etch-accepted json (after ionQ schema),
//      https://docs.ionq.com/api-reference/v0.3/writing-quantum-programs
//   pre-condition:
//      - json input is well-formed
//      - circuit gates configuration is up -> down, left -> right
//      - (operations) "gate"; "qubits" of CnotPowGate/CXPowGate and SwapPowGate
//         are arranged with control qbit(s): first, target qbit: last
//   post-condition: immediately followed by check of non-adjacent gate(s) in
//      received circuit of ionQ json format
   using json= nlohmann::json;

   unsigned long qbits_count {0};
   // compliance checks
   // *** these restrictions may be removed at a future date ***
   bool other_qbit {false};
   // compliance check 1: 'LineQubit' is the only (json) [qubits] type
   h_check_qbit(cirq_circuit, other_qbit, qbits_count);
   if (other_qbit){
      printf("\nprocess aborted: only \"cirq_type\" : \"LineQubit\" is accepted\n");
      exit (0);
   }

   bool non_adjacent_gate {false};
   unsigned long gate_moment_counter {0};
   // compliance check 2: only proximately adjacent gates
   h_check_non_adjacent_gate(cirq_circuit, non_adjacent_gate, gate_moment_counter);
   if (non_adjacent_gate){
      printf("\nprocess aborted: a non-adjacent gate at \"moments\" %lu\n"
            ,gate_moment_counter);
      exit (0);
   }


   // adapt cirq- to Etch-schema
   json ionq_schema;
   ionq_schema["qubits"]= qbits_count + 1;   // qbits_count + 1 -> (ionQ) "qubits"
   ionq_schema["moments"];   // a proxy count of graph columns
   ionq_schema["circuit"]= json::array();

   json object_gate= json::object({
      {"moment", 0}
      ,{"gate", ""}
      ,{"control", json::array()}
      ,{"target", 0}
   });

   json moments= cirq_circuit.at("moments");
   // mode_simulator requires a non-zero based counter
   unsigned long moment_counter {1};
   bool other_gate {false};

   for (json moment : moments) {
      json operations= moment.at("operations");
      std::string gate_unrecognised;

      for (json operation : operations) {
         json gate_type= operation.at("gate").at("cirq_type");

         json target= operation["qubits"][0]["x"];
         json exponent= operation.at("gate").at("exponent");

         if (gate_type == "CnotPowGate"
         || (gate_type == "CXPowGate" && exponent == 1.0)){
            unsigned long cnot_target= operation["qubits"][1]["x"].get<unsigned long>();
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "cnot";
            // assigning 'target' to object_gate["control"] is not an error
            object_gate["control"].push_back(target);
            object_gate["target"]= cnot_target;
         }
         else if (gate_type == "HPowGate"){
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "h";
            object_gate["target"]= target;
         }
         else if (gate_type == "PhasedXPowGate" || gate_type == "PhasedXZGate"){
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "rx";
            object_gate["target"]= target;
         }
         else if (gate_type == "S"){
            // 'vanilla' S gate, Cf. 'ZPowGate' below; see
            // https://quantumai.google/cirq/build/gates#single_qubit_gates
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "s";
            object_gate["target"]= target;
         }/*
         else if (gate_type == "SwapPowGate"){
            unsigned long swap_target= operation["qubits"][1]["x"].get<unsigned long>();
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "swap";
            // assigning 'target' to object_gate["control"] is not an error
            object_gate["control"].push_back(target);
            object_gate["target"]= swap_target;
         }*/
         else if (gate_type == "T"){
            // 'vanilla' T gate, Cf. 'ZPowGate' below; see
            // https://quantumai.google/cirq/build/gates#single_qubit_gates
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "t";
            object_gate["target"]= target;
         }
         else if (gate_type == "X"){
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "x";
            object_gate["target"]= target;
         }
         else if (gate_type == "XPowGate"){
            object_gate["moment"]= moment_counter;

            if (exponent == 1)
               object_gate["gate"]= "x";
            else
               object_gate["gate"]= "rx";

            object_gate["target"]= target;
         }
         else if (gate_type == "Y"){
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "y";
            object_gate["target"]= target;
         }
         else if (gate_type == "YPowGate"){
            object_gate["moment"]= moment_counter;

            if (exponent == 1)
               object_gate["gate"]= "y";
            else
               object_gate["gate"]= "ry";

            object_gate["target"]= target;
         }
         else if (gate_type == "Z"){
            object_gate["moment"]= moment_counter;
            object_gate["gate"]= "z";
            object_gate["target"]= target;
         }
         else if (gate_type == "ZPowGate"){
            object_gate["moment"]= moment_counter;

            if (exponent == 0.25)
               object_gate["gate"]= "t";
            else if (exponent == 0.5)
               object_gate["gate"]= "s";
            else if (exponent == 1)
               object_gate["gate"]= "z";
            else
               object_gate["gate"]= "rz";

            object_gate["target"]= target;
         }
         else {
            object_gate.clear();
            other_gate= true;
            gate_unrecognised= operation.at("gate").at("cirq_type").get<std::string>();
            break ;
         }

         ionq_schema["circuit"].push_back(object_gate);
         object_gate["control"].clear();
      }
      if (other_gate){
         std::cout << "\nprocess aborted: gate " << gate_unrecognised
         << " at \"moments\" " << moment_counter - 1 << " is unrecognised."
         << std::endl;

         exit (0);
      }
      moment_counter += 1;
   }
   ionq_schema["moments"]= moment_counter;

   return ionq_schema;
}

void h_check_non_adjacent_gate(const nlohmann::json & cirq_schema
                               , bool & non_adjacent_gate
                               , unsigned long & gate_moment_counter){
//   void: helper function, confirm all two-qbits gates are proximately adjacent
//   in the cirq schema input
   using json= nlohmann::json;

   json moments= cirq_schema.at("moments");
   for (json moment : moments) {
      json operations= moment.at("operations");
      for (json operation: operations) {
         json gate_type= operation.at("gate").at("cirq_type");

         if (gate_type == "CnotPowGate"
             || gate_type == "CXPowGate"
             || gate_type == "SwapPowGate") {
            unsigned long x_check{operation["qubits"][0]["x"].get<unsigned long>()};

            for (int qbit_idx= 1; qbit_idx < operation.at("qubits").size(); ++qbit_idx) {
               auto next_x= operation["qubits"][qbit_idx]["x"];
               if (next_x > x_check + 1){
                  non_adjacent_gate= true;
                  break;
               }
               else
                  x_check= operation["qubits"][qbit_idx]["x"].get<unsigned long>();
            }
         }
      }
      gate_moment_counter +=1;
   }
}

void h_check_qbit(const nlohmann::json & cirq_schema, bool & other_qbit
                  , unsigned long & qbits_count){
//   void: helper function, confirm 'LineQubit' is the only qbit type in the
//   cirq schema input as per function, 'parse_cirq_qubits'
//   (https://github.com/QSI-BAQS/Jabalizer.jl/blob/main/src/cirq_io.jl)
   using json= nlohmann::json;

   json moments= cirq_schema.at("moments");
   for (json moment : moments) {
      json operations= moment.at("operations");
      for (json operation : operations) {
         json qbits= operation.at("qubits");
         for (json qbit : qbits) {
            if (qbit.at("cirq_type") != "LineQubit"){
               other_qbit= true;
               break ;
            }
            if (qbit.at("x") > qbits_count)
               qbits_count= qbit.at("x");
         }
      }
   }
}