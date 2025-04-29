#include "gates_lattice_set.hpp"
#include "interactive_io.hpp"
#include "io_circuit.hpp"

#include <fstream>

int main()
{
   printf("key in /path/to/json/circuit then, press Enter:\n");

   std::string icircuit;
   std::cin >> icircuit;
   std::ifstream json_circuit (icircuit);

   if (json_circuit.is_open()){
      using json= nlohmann::json;

      json parse_circuit= json::parse(json_circuit);   // create json object
      auto cirq_check= parse_circuit.find("cirq_type");

      if (cirq_check != parse_circuit.end()){
         // format of input circuit json -> cirq
         json ionq_schema= cirq_to_ionq_schema(parse_circuit);
         etch_circuit= ionq_schema;
      }
      // format of input circuit json -> ionQ
      else {
         if (non_adjacent_gate(parse_circuit)){
            printf("process aborted: non-adjacent gate in circuit.\n");
            return 0;
         }
         etch_circuit= parse_circuit;
      }

      unsigned long cluster_state_rows= rows_m(gate_by_address, etch_circuit);
      unsigned long cluster_state_columns= cols_n(gate_by_address, etch_circuit);

      printf("\nThe input circuit specifies a [%lu, %lu] cluster state.\n\n", cluster_state_rows,
             cluster_state_columns);

      print_gates(gate_by_address);
      user_view(gate_by_address, etch_circuit);

      return 0;
   }
   else {
      printf("That file is not opening, it may be,\n  - the file is already"
             " in use,\n  - the file path is incorrect, or\n  - the file name is"
             " incorrect.\nCheck then, try again.\n");

      return 0;
   }
}