// trial_mdp.cpp
// (c) 2020 David Merrell
//
// Implementation of TrialMDP class

#include "trial_mdp.h"
#include "trial_mdp_table.h"
#include "state_result.h"
#include "transition_iterator.h"
#include "transition_dist.h"
#include "state_iterator.h"
#include "terminal_rule.h"
#include <iostream>
#include <cstring>
#include <string>
#include <sqlite3.h>
#include <cmath>
#include <limits>



/**
 * For a given state, find the action that maximizes
 * expected reward. Return the result (including the 
 * value of the maximized reward, and the terms of the
 * objective function)
 */
StateResult TrialMDP::max_expected_reward(int cur_idx, ContingencyTable ct){

    float FLOAT_NEG_INF = -std::numeric_limits<float>::infinity();
    int rwd_idx = n_attr - 1;
 
    // Track the best action we've seen thus far
    StateResult best_choice = StateResult(n_attr); 
    best_choice.values[rwd_idx] = FLOAT_NEG_INF;

    // Use this StateResult to represent the expected
    // value of the current action
    StateResult expected_values = StateResult(n_attr);

    // Iterate through the possible actions
    action_iterator->reset(cur_idx);
    while (action_iterator->not_finished()){

        // Get the size index of the resulting contingency table
	int result_size_idx = action_iterator->get_next_size_idx();

        float prob = 0.0;
        // Initialize expected reward (and other values):
        for (unsigned int i=0; i < n_attr; ++i){
            expected_values.values[i] = 0.0;
        }

	// Compute the expected reward for this action,
	// w.r.t. the randomness of the transition
        int a_A = action_iterator->action_a();
        int a_B = action_iterator->action_b();

	TransitionIterator tr_it = TransitionIterator(ct, a_A, a_B);
	transition_dist->set_state_action(ct, a_A, a_B);
        while(tr_it.not_finished()){
            
            int n_A = tr_it.get_a_counter();
            int n_B = tr_it.get_b_counter();

	    // Get the result struct associated with this state
            StateResult tr_res = (*results_table)(result_size_idx, tr_it.value());

	    // Get the probability of this transition
	    // and update the expected values:
	    prob = transition_dist->prob(n_A, n_B);
            result_interpreter.compute_lookaheads(ct, a_A, a_B, n_A, n_B, tr_res);
            for(unsigned int i=0; i < n_attr; ++i){
                expected_values.values[i] += (prob * result_interpreter.look_ahead(i));
            } 
            result_interpreter.clear_lookaheads();

            tr_it.advance();
        }


	// Compare expected reward vs. best_choice
	if (expected_values.values[rwd_idx] > best_choice.values[rwd_idx]){

            best_choice.block_size = action_iterator->get_block_size();
	    best_choice.a_allocation = action_iterator->action_a();

            for(unsigned int i=0; i < n_attr; ++i){
                best_choice.values[i] = expected_values.values[i];
            }
	}

        action_iterator->advance();
    
    }
    // Return the best action (and corresponding results)
    return best_choice;
}


// Constructor
TrialMDP::TrialMDP(int n_patients, float failure_cost, float block_cost,
                         int min_size, int block_incr, 
                         float prior_a0, float prior_a1,
                         float prior_b0, float prior_b1,
                         std::string tr_dist,
                         std::string test_statistic,
                         float act_l, float act_u, int act_n){

    result_interpreter = ResultInterpreter(test_statistic, failure_cost, block_cost, n_patients);

    n_attr = result_interpreter.get_n_attr();

    results_table = new TrialMDPTable(n_patients, min_size, block_incr); 
    state_iterator = new StateIterator(*(results_table)); 
    action_iterator = new ActionIterator(act_l, act_u, act_n, 
		                         results_table->get_n_vec(),
                                         min_size,
                                         0);

    transition_dist = TransitionDist::make_transition_dist(tr_dist,
                                                           prior_a0, prior_a1,
                                                           prior_b0, prior_b1);

    terminal_rule = TerminalRule::make_terminal_rule(test_statistic, failure_cost);


}


void TrialMDP::solve(){

    // Iterate through the terminal states;
    // set the terminal rewards
    int terminal_idx = state_iterator->get_cur_idx();
    int cur_idx = terminal_idx;
    ContingencyTable cur_table; 
    cur_table = state_iterator->value();


    while(cur_idx == terminal_idx){
	
	(*results_table)(terminal_idx, cur_table) = (*terminal_rule)(result_interpreter, cur_table);
	
	state_iterator->advance();
	cur_idx = state_iterator->get_cur_idx();
	cur_table = state_iterator->value();

    }
    
    // Move on to the earlier states. 
    // compute the maximal action for each one.
    while(state_iterator->not_finished()){

        cur_table = state_iterator->value();
        (*results_table)(cur_idx, cur_table) = max_expected_reward(cur_idx, cur_table);

        state_iterator->advance();
	cur_idx = state_iterator->get_cur_idx();
    }

    StateResult first_move = (*results_table)(0, cur_table);

    std::cout << result_interpreter.pretty_print_result(first_move);

}



void TrialMDP::to_sqlite(char* db_fname, int chunk_size=10000){

    // Make database connection pointer
    sqlite3* db;

    try{
        // Connect to database
        int conn_result = 1;
        conn_result = sqlite3_open(db_fname, &db);
        if(conn_result != 0){ throw 1; }

      
        // Drop the table if it already exists. 
        std::string droptable_str = "DROP TABLE IF EXISTS RESULTS;";
        int drop_result = 1;
        drop_result = sqlite3_exec(db, droptable_str.c_str(), NULL, NULL, NULL);

        // Build table in database
	std::string build_expr = result_interpreter.sql_create_table();
        int build_result = 1;
        build_result = sqlite3_exec(db, build_expr.c_str(), NULL, NULL, NULL);
        if (build_result != 0){ throw 2; }

	// Beginning and end of each chunk
	std::string transaction_start = "BEGIN TRANSACTION;\n";
        std::string transaction_end = "COMMIT;";

	// Iterate over the results
	StateIterator result_iter = StateIterator(*(results_table));
        while(result_iter.not_finished()){

	    // Prepare a chunk of INSERTs...
	    std::string insert_expr = transaction_start;
            int chunk_idx = 0;

	    while(result_iter.not_finished() && chunk_idx < chunk_size){

		// Get the contingency table and corresponding results
                ContingencyTable cur_table = result_iter.value();
	        int cur_idx = result_iter.get_cur_idx();
                StateResult cur_results = (*results_table)(cur_idx, cur_table);
	       
                // add a line for this result to the SQL query         
                insert_expr += result_interpreter.sql_insert_tuple(cur_results, cur_table);
		
                // Move to the next result
                result_iter.advance();

	    }
	    insert_expr += transaction_end;

	    // INSERT the chunk
            int insert_result = 1;
            insert_result = sqlite3_exec(db, insert_expr.c_str(), NULL, NULL, NULL);
            if (insert_result != 0){ throw 3; }
	}
        // Close connection
        sqlite3_close(db);

    }
    catch(int code){ 
        switch(code){
	    case 1:
	        std::cerr << "`to_sqlite`: failed to connect to SQLite database at location " << db_fname << std::endl;
		break;
	    case 2:
		std::cerr << "`to_sqlite`: failed to build table RESULTS in database." << std::endl; 
		break;
	    case 3: 
		std::cerr << "`to_sqlite`: failed to insert rows into table RESULTS." << std::endl; 
		break;
	    default: 
		std::cerr << "`to_sqlite`: method failed." << std::endl; 
		break;
        }
    }

}
    

TrialMDP::~TrialMDP(){
    delete state_iterator;
    delete action_iterator;
    delete results_table;
    delete terminal_rule;
    delete transition_dist;
}
