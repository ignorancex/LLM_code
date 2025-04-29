//
// specify a cluster state by measurement dimensions
//
#include "gates_lattice_set.hpp"

#include <fstream>

// prototype
bool h_is_contiguous_cnot(std::vector<Gate_SE_SW *>, const nlohmann::json &, unsigned long);

unsigned long cols_n(std::vector<Gate_SE_SW> & c_columns, const nlohmann::json & as_cols) {
   // unsigned long: generate address-by-column of gate operations in a cluster
   //    state to derive the COUNT of columns.  The conditional of 'floating',
   //    as features here, is a gate preceded by an even number of
   //    x-measurements, which serve to project state top-to-bottom/left-to-
   //    right ('wire qbits').
   // pre-condition: input (json) circuit prohibits non-adjacent cnot/swap
   // post-condition: rows_m, cols_n specify dimensions of the cluster state

   unsigned long qbits {as_cols["qubits"].get<unsigned long>()};
   unsigned long column_width[qbits];

   for (unsigned long i= 0; i < qbits; ++i) {
      column_width[i]= 0;   // abstract row addresses
   }

   unsigned long left_edge {0};
   unsigned long gtc_counter {0};
   unsigned long previous_target_index {0};

   // collection: cnot components of the input circuit
   std::vector<Gate_SE_SW *> v_cnot_path;

   // cols_n provisions for neither a (circuit-level) input nor readout column
   for (auto & gtc : as_cols["circuit"]) {
      if (gtc["gate"] == "cnot"/* || gtc["gate"] == "swap"*/){
         // review indexing upon adopting swap gates
         // account for zero-based indexing
         if (column_width[gtc["control"][0]] == 0){
            column_width[gtc["control"][0]]= 5;
            column_width[gtc["target"]]= 5;
         }
         // index is 1+
         else {
            if (column_width[gtc["target"]] == left_edge)
               column_width[gtc["target"]]= left_edge + 6;
            else if (left_edge > column_width[gtc["target"]]){
               unsigned short adjust_col_address {0};

               if (gtc["target"] <= previous_target_index || v_cnot_path.empty())
                  adjust_col_address= 3;   // (1st kind) 'floating' cnot OR first cnot
               else if (h_is_contiguous_cnot(v_cnot_path, gtc["control"][0]
                                             , left_edge + 6))
                  adjust_col_address= 5;   // contiguous cnots
               else if (gtc["target"] > previous_target_index)
                  adjust_col_address= 7;   // (2nd kind) 'floating' cnot

               if (adjust_col_address % 2 == 1){
                  column_width[gtc["target"]]= left_edge + 6;
                  adjust_col_address= 0;
               }
               else
                  column_width[gtc["target"]]= left_edge;
            }

         column_width[gtc["control"][0]]= column_width[gtc["target"]];
         }

         c_columns[gtc_counter].sw_column= column_width[gtc["target"]];
         // add cnot statistics to (vector) h_cnot_path
         struct Gate_SE_SW * p_cnot= & (c_columns[gtc_counter]);
         v_cnot_path.push_back(p_cnot);
      }
      else {
         // non-cnot/-swap gate, account for zero-based indexing
         if (column_width[gtc["target"]] == 0)
            column_width[gtc["target"]]= 3;
         // non-cnot/-swap gate, index is 1+
         else if (column_width[gtc["target"]] == left_edge)
            column_width[gtc["target"]]= left_edge + 4;
         else if (left_edge > column_width[gtc["target"]]){
            // condition: 'floating' gate
            if (gtc["target"] <= previous_target_index)
               column_width[gtc["target"]]= left_edge + 4;
            else
               column_width[gtc["target"]]= left_edge;
         }

         c_columns[gtc_counter].sw_column= column_width[gtc["target"]];
      }

      if (column_width[gtc["target"]] > left_edge)
         left_edge= column_width[gtc["target"]];
      previous_target_index= gtc["target"].get<unsigned long>();
      
      gtc_counter += 1;
   }
   // derive count from zero-based address
   return c_columns.back().sw_column + 1;
}

bool non_adjacent_gate(const nlohmann::json & ionq_circuit) {
   // bool: detect any non-adjacent cnot/swap of input circuit in the ionQ json
   //     format, by which control and target wires are distance >= 2 apart
   // pre-condition: input (json) circuit:
   //   - json is well-formed, &
   //   - >= 1 gate, &
   //   - a proper subset of Etch gates.
   // post-condition: runtime flag, proceed with preparing a cluster state?
   bool non_adjacent= false;

   // check for non-adjacent cnot/swap
   for (auto & gate : ionq_circuit["circuit"]) {
      if (gate["gate"] == "cnot" || gate["gate"] == "swap"){
         for (unsigned long c : gate["control"]) {
            if (gate["target"].get<unsigned long>() - c > 1){
               non_adjacent= true;
               break;
            }
         }
      }
   }

   return non_adjacent;
}

unsigned long rows_m(std::vector<Gate_SE_SW> & c_rows, const nlohmann::json & as_rows) {
   // unsigned long: generate address-by-row of gate operations in a cluster
   //    state to derive the COUNT of rows
   // pre-condition:
   //    - input (json) circuit prohibits non-adjacent cnot/swap; and
   //    - circuit gates configuration is up -> down, left -> right
   // post-condition: call to cols_n immediately follows this procedure
   unsigned long qbits {as_rows["qubits"].get<unsigned long>()};
   unsigned long row_address[qbits];

   for (unsigned long i= 0; i < qbits; ++i) {
      row_address[i]= i;   // mutable row addresses
   }

   unsigned long outer_edge {0};
   for (auto & gtc : as_rows["circuit"]) {
      Gate_SE_SW gate_row;  // create element of gate_by_address

      if (gtc["gate"] == "cnot"){
         // review in eventuality of non-adjacent gate operations
         if (gtc["target"] <= 1 && row_address[1] != 2){
            // "control" : [0], "target" : 1 OR  "control" : [1], "target" : 0
            row_address[1]= 2;

            // json value is sufficient for this test
            if (outer_edge < 1)
               outer_edge= gtc["target"].get<unsigned long>();

            // 'push out' subsequent row addresses
            for (unsigned long i= 2; i < qbits ; ++i) {
               row_address[i] += 1;   // mutable row addresses
            }
         }

         // [gate_target] == circuit row; gate_target == lattice row
         unsigned long gate_target {gtc["target"].get<unsigned long>()};
         bool adjust_row_address {false};

         if (gate_target > 1){
            if (gate_target > outer_edge){
               row_address[gate_target] += 1;
               outer_edge= gate_target;
               adjust_row_address= true;
            }
            else if (row_address[gate_target] == gate_target + 1){
               row_address[gate_target] += 1;
               adjust_row_address= true;
            }
         }
         // case: the first cnot of the circuit has gate_target of, say, 3; OR
         // 3 * cnots as a substitute for a swap gate
         else if (gate_target >= 1 && gtc["control"][0] > 1
         && ((row_address[gtc["control"][0]] + 2) < row_address[gate_target])){
            row_address[gate_target]= gate_target + 2;
            adjust_row_address= true;
         }

         // 'push out' subsequent row addresses
         if (adjust_row_address && (gate_target + 1 < qbits)){
            unsigned long adjustment= gate_target + 1;
            for (unsigned long i= adjustment; i < qbits ; ++i) {
               row_address[i] += 1;   // mutable row addresses
            }
         }

         gate_row.gate= "cnot";
         c_rows.push_back(gate_row);
      }
      else {
         // recall: struct elements, once set, cannot be reset...
         gate_row.gate= gtc["gate"].get<std::string>();

         c_rows.push_back(gate_row);
      }
   }
   // setting c_rows.se_row according to cnot positions
   unsigned long circuit_address;
   for (unsigned long i= 0; i < as_rows["circuit"].size(); ++i) {
      circuit_address= as_rows["circuit"][i]["target"].get<unsigned long>();
      c_rows.at(i).se_row= row_address[circuit_address];   // row address
   }
   // derive count from zero-based address
   return c_rows.back().se_row + 1;
}

bool h_is_contiguous_cnot(std::vector<Gate_SE_SW *> cnot_path
                          , const nlohmann::json & cnot_control
                          , unsigned long indicative_left_edge) {
   // helper function, does cnot abut a previous cnot?
   bool contiguous_cnot {false};

   // reverse iteration over cnot_path; 'imm' is ** Gate_SE_SW
   auto imm= cnot_path.rbegin();
   while (imm != cnot_path.rend()) {
      if ((indicative_left_edge - (*imm)->sw_column == 6)
      && ((*imm)->se_row - 1 == cnot_control)){
          contiguous_cnot= true;
          break;
      }
      imm++;
   }

   return contiguous_cnot;
}