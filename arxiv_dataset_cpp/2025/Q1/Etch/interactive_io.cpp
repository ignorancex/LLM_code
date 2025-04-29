//
// I/O operations for non-integrated Etch
//

#include "interactive_io.hpp"

#include <cctype>
#include <cstdio>
#include <string>


std::string h_branching(const std::string & prompt) {
   // string: helper function to handle type-specific std::cin
   // pre-condition: call to function, 'user_view'
   // post-condition: subsequent calls may be prompt-specific

   std::cout << prompt << std::endl;

   std::string user_input;
   std::cin >> user_input;

   std::cin.clear();

   char c1st {user_input.front()};

   if (c1st == 'q' || c1st == 'Q')
      return "quit";
   else {
      // check the string
      short counter {0};
      for (char e : user_input) {
         if (!std::isdigit(e)){
            counter += 1;
            break;
         }
      }
      if (counter > 0)
         return "";   // string is invalid input to function, 'user_view'
      else
         return user_input;   // string is a number representation
   }
}

void h_quitting () {
   // void: helper function, human-friendly output to console
   std::cout << "Quitting." << std::endl;
}

void print_gates(const std::vector<Gate_SE_SW> & v_gates) {
   // void: human-friendly output to console: gate type & resource cluster coordinates
   // pre-condition: call to functions, 'cluster_state_*'
   // post-condition: none

   std::cout  << "gate -> [SE, SW] coordinates:" << std::endl;
   for (const Gate_SE_SW & c : v_gates) {
      std::cout << "   " << c.gate << " -> [" << c.se_row << "," << c.sw_column
      << "]" << std::endl;
   }
}

void user_view(const std::vector<Gate_SE_SW> & cstats, const nlohmann::json & circuit_stats) {
   // void: human-friendly output to console: resource cluster column statistics
   // pre-condition: call to functions, 'cluster_state_*'
   // post-condition: none

   std::string branch {h_branching(prompt_1)};

   if (branch == "quit"){
      h_quitting();
      return ;
   }

   while (branch != "quit") {
      if (branch.empty()){
         branch= h_branching(prompt_2 + prompt_1);

         if (branch == "quit"){
            h_quitting();
            break;
         }
         else
            continue;
      }

      unsigned long column_lu {std::stoul(branch)};

      if (column_lu <= cstats.back().sw_column + 1){
         Lattice_data print_output= column_statistics(column_lu, cstats, circuit_stats);

         printf("pauli/non-pauli: %lu/%lu -> %g, z/(pauli + non-pauli): %lu/%lu -> %g\n"
                ,print_output[0]   // pauli-X, -Y
                ,print_output[1]   // non-pauli
                ,(double) print_output[0] / (double) print_output[1]   // pauli-X, -Y / non-pauli
                ,print_output[2]   // pauli-Z
                ,(print_output[0] + print_output[1])   // pauli-X, -Y + non-pauli
                // pauli-Z / (pauli-X, -Y + non-pauli)
                ,(double) print_output[2] / (double) (print_output[0] + print_output[1])
                );

         branch= h_branching(prompt_3);

         if (branch == "quit")
            h_quitting();
      }
      else {
         branch= h_branching(prompt_4);

         if (branch == "quit")
            h_quitting();
      }
   }
}