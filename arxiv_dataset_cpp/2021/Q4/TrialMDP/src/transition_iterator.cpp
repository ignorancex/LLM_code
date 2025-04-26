// transition_iterator.cpp
// (c) 2020 David Merrell
//

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#include "contingency_iterator.h"
#include "contingency_table.h"
#include "transition_iterator.h"
#include <cmath>


TransitionIterator::TransitionIterator(ContingencyTable ct, int size_a, int size_b){

    cur_table = ct;

    a_size = size_a;
    b_size = size_b;

    a_counter = 0;
    b_counter = 0;

}


ContingencyTable TransitionIterator::value(){
    short unsigned int a0 = cur_table.a0 + a_size - a_counter;
    short unsigned int a1 = cur_table.a1 + a_counter;
    short unsigned int b0 = cur_table.b0 + b_size - b_counter;
    short unsigned int b1 = cur_table.b1 + b_counter;
    return ContingencyTable(a0, a1, b0, b1);

}


bool TransitionIterator::not_finished(){
    return (a_counter <= a_size);
}


void TransitionIterator::advance(){
    if (b_counter < b_size){
        b_counter++;
    }
    else{
        a_counter++;
        b_counter = 0;
    }
}


