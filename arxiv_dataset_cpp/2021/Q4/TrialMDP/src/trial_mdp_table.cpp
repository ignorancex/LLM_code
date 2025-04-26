// trial_mdp_table.cpp
// (c) 2020 David Merrell
//

#include "trial_mdp_table.h"
#include "contingency_table.h"
#include "state_result.h"
#include <iostream>

std::vector<int> build_n_vec(unsigned int n_max, unsigned int min_size, unsigned int n_incr){
    std::vector<int> result;
    result.push_back(0);
    for(unsigned int i = n_incr; i <= n_max - min_size; i += n_incr){
        if(i >= min_size){
            result.push_back(i);
        }
    }
    result.push_back(n_max);
    // If n_incr doesn't divide n_max neatly, then
    // set the last entry to n_max -- its distance
    // from the previous entry will be somewhat
    // farther than n_incr.
    return result;
}


TrialMDPTable::TrialMDPTable(int n_max, int min_size, int n_incr){
    n_vec = build_n_vec(n_max, min_size, n_incr);
    results = std::vector< std::unordered_map<ContingencyTable, StateResult, CTHash>* >();
    
    for(unsigned int i = 0; i < n_vec.size(); i++){
        results.push_back( new std::unordered_map<ContingencyTable, StateResult, CTHash> ); 
    }

}


