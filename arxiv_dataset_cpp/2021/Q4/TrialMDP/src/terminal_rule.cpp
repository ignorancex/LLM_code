

#include "terminal_rule.h"
#include <iostream>

////////////////////////
// Factory method
////////////////////////

TerminalRule* TerminalRule::make_terminal_rule(std::string test_stat, float failure_cost){
    if (test_stat == "wald"){
      return new WaldFailureTerminalRule(failure_cost);
    }
    else if (test_stat == "scaled_cmh"){
      return new RescaledFailureTerminalRule(failure_cost);
    }
    else{
      std::cerr << test_stat << " not a valid value for test statistic." << std::endl;
      throw(1);
    }
}
