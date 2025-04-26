// action_iterator.cpp
// (c) 2020 David Merrell
//


#include <vector>
#include <cmath>
#include "action_iterator.h"
#include <iostream>

std::vector<int> build_alloc_vec(std::vector<float> ratio_vec, int block_size){

    std::vector<int> result;
    result.push_back(int(round(ratio_vec[0]*block_size)));

    for(unsigned int r_idx = 1; r_idx < ratio_vec.size(); r_idx++){
        int cv = int(round(ratio_vec[r_idx]*block_size));
	if (cv != result[result.size()-1]){
            result.push_back(cv);
	}
    }

    return result;
}


ActionIterator::ActionIterator(float min_ratio, float max_ratio, int n_ratios,
                               std::vector<int> n_vec, int min_s,
	                       int n_idx){
   
    // Populate the vector of allocation ratios
    ratio_vec = std::vector<float>(n_ratios, 0.0);
    float ratio_incr = (max_ratio - min_ratio)/(n_ratios - 1.0);
    for(int i=0; i < n_ratios; i++){
        ratio_vec[i] = min_ratio + (i*ratio_incr);
    }

    // Initiate block sizes
    size_vec = n_vec;
    cur_size_idx = n_idx;

    // Set "next_size_idx" to the first position
    // satisfying block_size >= min_size 
    min_size = min_s;
    next_size_idx = cur_size_idx + 1;
    block_size = size_vec[next_size_idx] - size_vec[cur_size_idx];
    while( block_size < min_size ){
        next_size_idx++;
        block_size = size_vec[next_size_idx] - size_vec[cur_size_idx];
    }

    // Initiate allocations
    alloc_vec = build_alloc_vec(ratio_vec, block_size); 
    alloc_idx = 0;
   
    act_a = alloc_vec[0];
    act_b = block_size - act_a; 
    
}


void ActionIterator::reset(int n_idx){

    alloc_idx = 0;
    cur_size_idx = n_idx;

    // Set "next_size_idx" to the first position
    // satisfying block_size >= min_size 
    next_size_idx = cur_size_idx + 1;
    block_size = size_vec[next_size_idx] - size_vec[cur_size_idx];
    while( block_size < min_size ){
        next_size_idx++;
        block_size = size_vec[next_size_idx] - size_vec[cur_size_idx];
    }

    alloc_vec = build_alloc_vec(ratio_vec, block_size);
    act_a = alloc_vec[alloc_idx];
    act_b = block_size - act_a;
}


bool ActionIterator::not_finished(){
    return (next_size_idx < size_vec.size());
}


void ActionIterator::advance(){
    if(alloc_idx < alloc_vec.size() - 1){
        alloc_idx++;
	act_a = alloc_vec[alloc_idx];
	act_b = block_size - act_a;
    } else{
        alloc_idx = 0;
        next_size_idx++;
	block_size = size_vec[next_size_idx] - size_vec[cur_size_idx];
	alloc_vec = build_alloc_vec(ratio_vec, block_size);
	act_a = alloc_vec[alloc_idx];
        act_b = block_size - act_a;	
    }
}


int ActionIterator::get_block_size(){
    return block_size;
}


int ActionIterator::action_a(){
    return act_a;
}


int ActionIterator::action_b(){
    return act_b;
}


int ActionIterator::get_next_size_idx(){
    return next_size_idx;
}
